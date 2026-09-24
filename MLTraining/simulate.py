"""Patient-level simulator for the recovery classifier.

Everything the model is trained and evaluated on comes from this generative
process, and every parameter is listed in SimConfig with its source:
  [L] taken from the literature (see literature.py),
  [C] calibrated by calibrate() so the simulation reproduces a published number,
  [A] an assumption, stated so it can be argued with.

The process, per patient:

  traits ──┬─> natural pain course (exponential decay towards a floor) ──┐
           │                                                             ├─> latent daily pain ─┬─> per-session pain logs
           ├─> daily exercise (habit, motivation decay, dropout,  ───────┤                      └─> week-6 NRS ─> label
           │   pain as a barrier) ─> exercise benefit (builds with       │
           │   sessions, fades without them) ────────────────────────────┘
           └─> logging behaviour ─> ExerciseLog rows ─> features.py ─> model inputs

The model never sees a sampled feature. Features are computed from simulated
logs by the same algorithm the app runs (features.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd

import features as F
import literature as lit

EPOCH_2024 = 1_704_067_200  # 2024-01-01T00:00:00Z
DAY = F.SECONDS_PER_DAY

REGION_EXERCISES = {
    "knee": ("knee-extension", "heel-slide"),
    "back": ("pelvic-tilt", "glute-bridge"),
    "shoulder": ("shoulder-flexion", "wall-slide"),
    "hip": ("glute-bridge", "clamshell"),
    "ankle": ("ankle-circles",),
}
REGIONS = tuple(REGION_EXERCISES)

# Weekly check-ins: features are computed at the end of plan day 7k - 1.
SNAPSHOT_WEEKS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class SimConfig:
    # Cohort and calendar
    n_patients: int = 3000                    # [A]
    enrollment_days: int = 730                # [A] enrolments from 2024-01-01 to 2025-12-30
    plan_days: int = 42                       # [A] 6-week daily home programme; outcome on day 42
    persistent_share_start: float = 0.25      # [A] case-mix drift: the share of patients with
    persistent_share_end: float = 0.45        #     persistent pain rises across enrolment
    region_probs: tuple = (0.30, 0.25, 0.20, 0.15, 0.10)  # [A] same order as REGIONS

    # Baseline pain, 0-10 NRS
    baseline_pain_mean: float = 5.9           # [L] Wallwork 2024 pooled baselines 56-63/100
    baseline_pain_sd: float = 1.3             # [A]
    baseline_pain_min: float = 3.0            # [A] inclusion criterion: NRS >= 3
    baseline_pain_max: float = 9.5            # [A]

    # Natural course: pain decays exponentially towards a patient-specific floor
    halflife_acute_days: float = 14.0         # [A]
    halflife_persistent_days: float = 30.0    # [A]
    halflife_log_sd: float = 0.3              # [A] between-patient spread
    floor_acute: float = 0.40                 # [C] Wallwork 2024 acute: week 6 = 26/56 of baseline
    floor_persistent: float = 0.80            # [C] Wallwork 2024 persistent: week 6 = 48/56
    floor_concentration: float = 10.0         # [A] Beta concentration of the floor fraction

    # Flares: transient spikes that decay back
    flare_daily_prob: float = 1 / 40          # [A]
    flare_size_min: float = 1.0               # [A] NRS points
    flare_size_max: float = 3.0               # [A]
    flare_halflife_days: float = 3.0          # [A]

    # Exercise behaviour
    adherence_logit_mean: float = 0.0         # [C] half of patients exercise on < half of days
    adherence_logit_sd: float = 1.0           # [A]
    habit_logit: float = 1.0                  # [A] exercising yesterday makes today more likely
    pain_barrier_logit: float = 0.25          # [L: direction, Jack 2010; A: size] per NRS point above 5
    motivation_decay_per_week: float = 0.15   # [L: direction, Pisters 2010; A: size] mean logit drop
    dropout_hazard_per_day: float = 0.008     # [A] mean daily hazard of stopping for good

    # Effect of exercise on pain
    exercise_itt_points: float = lit.EXERCISE_EFFECT_POINTS  # [L] Fransen 2015, Hayden 2021
    trial_adherence: float = 0.70             # [A] adherence in the trials behind that effect
    training_halflife_days: float = 14.0      # [A] benefit builds with sessions, fades without
    exercise_effect_multiplier: float = 1.0   # sensitivity analysis: 0, 0.5, 1, 2

    # Logging: how performed sessions become ExerciseLog rows
    log_capture: float = 1.0                  # sensitivity analysis: share of sessions logged
    exercise_log_prob: float = 0.9            # [A] each prescribed exercise logged per session
    default_pain_prob: float = 0.15           # [A] pain slider left at its default of 3
    default_reps_prob: float = 0.30           # [A] reps stepper left at the target
    pain_log_noise_sd: float = 0.8            # [A]
    exercise_pain_offset_sd: float = 0.5      # [A] some exercises hurt more than others
    capacity_mean: float = 0.8                # [A] reps completed / target at NRS 4
    capacity_sd: float = 0.1                  # [A]

    # Outcome
    assessment_noise_sd: float = 0.5          # [A] clinic NRS at enrolment and at week 6
    responder_reduction: float = lit.RESPONDER_REDUCTION  # [L] Farrar 2001

    def exercise_benefit_full_adherence(self) -> float:
        """Steady-state pain reduction (NRS points) from exercising every day.

        Chosen so that exercising on `trial_adherence` of days for the whole
        plan lowers week-6 pain by `exercise_itt_points`, the published
        exercise-vs-control difference.
        """
        rho = math.log(2) / self.training_halflife_days
        build_up = 1 - math.exp(-rho * self.plan_days)
        return self.exercise_effect_multiplier * self.exercise_itt_points / (self.trial_adherence * build_up)


@dataclass
class Cohort:
    config: SimConfig
    patients: pd.DataFrame
    pain: np.ndarray       # (n, horizon + 1) latent pain at the start of each day, flares included
    flare: np.ndarray      # (n, horizon + 1) flare component of pain
    benefit: np.ndarray    # (n, horizon + 1) accumulated exercise benefit
    exercised: np.ndarray  # (n, horizon) performed a session that day
    engaged: np.ndarray    # (n, horizon) not yet dropped out
    logs: Optional[pd.DataFrame]


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _truncated_normal(rng, mean, sd, low, high, n):
    values = rng.normal(mean, sd, n)
    bad = (values < low) | (values > high)
    while bad.any():
        values[bad] = rng.normal(mean, sd, bad.sum())
        bad = (values < low) | (values > high)
    return values


def simulate(
    cfg: SimConfig,
    seed: int,
    horizon_days: Optional[int] = None,
    persistent_share: Optional[float] = None,
    with_logs: bool = True,
) -> Cohort:
    """Simulate `cfg.n_patients` patients.

    Randomness comes from three independent streams (traits, daily dynamics,
    logging), so changing a logging parameter leaves every patient's latent
    course unchanged, and a longer horizon extends the same patients.
    """
    traits_rng, daily_rng, log_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3))
    n = cfg.n_patients
    horizon = cfg.plan_days if horizon_days is None else horizon_days

    enroll_day = traits_rng.integers(0, cfg.enrollment_days, n)
    if persistent_share is None:
        drift = enroll_day / (cfg.enrollment_days - 1)
        share = cfg.persistent_share_start + (cfg.persistent_share_end - cfg.persistent_share_start) * drift
    else:
        share = np.full(n, persistent_share)
    persistent = traits_rng.random(n) < share
    region = traits_rng.choice(len(REGIONS), n, p=np.array(cfg.region_probs))
    baseline = _truncated_normal(
        traits_rng, cfg.baseline_pain_mean, cfg.baseline_pain_sd, cfg.baseline_pain_min, cfg.baseline_pain_max, n
    )
    halflife = np.where(persistent, cfg.halflife_persistent_days, cfg.halflife_acute_days) * np.exp(
        traits_rng.normal(0, cfg.halflife_log_sd, n)
    )
    kappa = math.log(2) / halflife
    floor_mean = np.where(persistent, cfg.floor_persistent, cfg.floor_acute)
    floor = traits_rng.beta(floor_mean * cfg.floor_concentration, (1 - floor_mean) * cfg.floor_concentration)
    adherence_logit = traits_rng.normal(cfg.adherence_logit_mean, cfg.adherence_logit_sd, n)
    decay = traits_rng.gamma(2.0, cfg.motivation_decay_per_week / 2.0, n)
    hazard = cfg.dropout_hazard_per_day * traits_rng.lognormal(-0.125, 0.5, n)  # mean-preserving spread
    capacity = traits_rng.normal(cfg.capacity_mean, cfg.capacity_sd, n)
    preferred_hour = traits_rng.uniform(7, 20, n)
    pain_offset = traits_rng.normal(0, cfg.exercise_pain_offset_sd, (n, 2))
    baseline_nrs = np.clip(np.rint(baseline + traits_rng.normal(0, cfg.assessment_noise_sd, n)), 1, 10)
    discharge_noise = traits_rng.normal(0, cfg.assessment_noise_sd, n)

    rho = math.log(2) / cfg.training_halflife_days
    gain = cfg.exercise_benefit_full_adherence() * (1 - math.exp(-rho))
    flare_keep = math.exp(-math.log(2) / cfg.flare_halflife_days)

    pain = np.empty((n, horizon + 1))
    flare = np.zeros((n, horizon + 1))
    benefit = np.zeros((n, horizon + 1))
    exercised = np.zeros((n, horizon), dtype=bool)
    engaged = np.zeros((n, horizon), dtype=bool)
    still_engaged = np.ones(n, dtype=bool)
    exercised_yesterday = np.zeros(n, dtype=bool)
    x = np.zeros(n)
    f = np.zeros(n)
    for t in range(horizon + 1):
        # Same draws every day regardless of state, so streams stay aligned.
        onset = daily_rng.random(n) < cfg.flare_daily_prob
        size = daily_rng.uniform(cfg.flare_size_min, cfg.flare_size_max, n)
        u_dropout = daily_rng.random(n)
        u_exercise = daily_rng.random(n)
        if t > 0:
            f = f * flare_keep + onset * size
        natural = baseline * (floor + (1 - floor) * np.exp(-kappa * t))
        p = np.clip(natural - x + f, 0, 10)
        pain[:, t], flare[:, t], benefit[:, t] = p, f, x
        if t == horizon:
            break
        still_engaged &= u_dropout >= hazard
        logit = (
            adherence_logit
            - decay * t / 7
            + cfg.habit_logit * exercised_yesterday
            - cfg.pain_barrier_logit * (p - 5)
        )
        today = still_engaged & (u_exercise < _sigmoid(logit))
        exercised[:, t], engaged[:, t] = today, still_engaged
        x = x * math.exp(-rho) + gain * today
        exercised_yesterday = today

    patients = pd.DataFrame(
        {
            "patient_id": np.arange(n),
            "enroll_day": enroll_day,
            "persistent": persistent,
            "region": np.array(REGIONS)[region],
            "baseline_pain": baseline,
            "halflife_days": halflife,
            "floor_fraction": floor,
            "adherence_logit": adherence_logit,
            "motivation_decay": decay,
            "dropout_hazard": hazard,
            "capacity": capacity,
            "baseline_nrs": baseline_nrs.astype(int),
        }
    )
    plan = min(cfg.plan_days, horizon)
    patients["days_exercised"] = exercised[:, :plan].sum(axis=1)
    if horizon >= cfg.plan_days:
        week6 = np.clip(np.rint(pain[:, cfg.plan_days] + discharge_noise), 0, 10).astype(int)
        reduction = (baseline_nrs - week6) / baseline_nrs
        patients["week6_nrs"] = week6
        patients["reduction"] = reduction
        patients["on_track"] = (reduction >= cfg.responder_reduction - 1e-9).astype(int)
        patients["dropped_out"] = ~engaged[:, cfg.plan_days - 1]

    logs = None
    if with_logs:
        logs = _logs(cfg, log_rng, enroll_day, region, pain, exercised, pain_offset, capacity, preferred_hour)
    return Cohort(cfg, patients, pain, flare, benefit, exercised, engaged, logs)


def _logs(cfg, rng, enroll_day, region, pain, exercised, pain_offset, capacity, preferred_hour) -> pd.DataFrame:
    """Turn performed sessions into ExerciseLog rows as LogSessionView would record them."""
    n, horizon = exercised.shape
    logged = exercised & (rng.random((n, horizon)) < cfg.log_capture)
    sessions_before = np.cumsum(exercised, axis=1) - exercised

    pid, day = np.nonzero(logged)
    m = len(pid)
    hour = np.clip(preferred_hour[pid] + rng.normal(0, 1.5, m), 6, 22)
    session_start = EPOCH_2024 + (enroll_day[pid] + day) * DAY + np.floor(hour * 3600).astype(np.int64)
    n_exercises = np.array([len(REGION_EXERCISES[r]) for r in REGIONS])[region[pid]]

    include = rng.random((m, 2)) < cfg.exercise_log_prob
    include[:, 1] &= n_exercises > 1
    include[:, 0] |= ~include[:, 1]  # every session logs at least one exercise
    jitter = rng.integers(0, 60, (m, 2))
    noise = rng.normal(0, cfg.pain_log_noise_sd, (m, 2))
    default_pain = rng.random((m, 2)) < cfg.default_pain_prob
    reps_noise = rng.normal(0, 0.07, (m, 2))
    default_reps = rng.random((m, 2)) < cfg.default_reps_prob

    latent = pain[pid, day]
    training = 0.15 * np.minimum(1.0, sessions_before[pid, day] / 20)
    frames = []
    for slot in (0, 1):
        keep = include[:, slot]
        ids = np.array([REGION_EXERCISES[r][slot] if slot < len(REGION_EXERCISES[r]) else "" for r in REGIONS])
        exercise_id = ids[region[pid]]
        target = np.array([F.TARGET_REPS.get(e, 1) for e in exercise_id])
        logged_pain = np.clip(np.rint(latent + pain_offset[pid, slot] + noise[:, slot]), 0, 10)
        logged_pain = np.where(default_pain[:, slot], 3, logged_pain)
        completion = np.clip(capacity[pid] - 0.05 * (latent - 4) + training + reps_noise[:, slot], 0.3, 1.3)
        reps = np.clip(np.rint(target * completion), 1, 50)
        reps = np.where(default_reps[:, slot], target, reps)
        frames.append(
            pd.DataFrame(
                {
                    "patient_id": pid[keep],
                    "plan_day": day[keep],
                    "timestamp": session_start[keep] + slot * 180 + jitter[keep, slot],
                    "exercise_id": exercise_id[keep],
                    "reps": reps[keep].astype(int),
                    "pain": logged_pain[keep].astype(int),
                }
            )
        )
    return pd.concat(frames).sort_values(["patient_id", "timestamp"], kind="stable").reset_index(drop=True)


def snapshot_time(enroll_day: int, week: int) -> int:
    """End of the last day of plan week `week` (23:59:59 UTC)."""
    return EPOCH_2024 + (enroll_day + 7 * week) * DAY - 1


def build_snapshots(cohort: Cohort) -> pd.DataFrame:
    """One row per patient per weekly check-in: app features plus ground truth.

    Features use only logs up to the check-in. Rows with fewer than MIN_LOGS
    logs are kept with scored=False: the app shows no prediction for them, and
    the error analysis needs to know who they are.
    """
    logs, patients = cohort.logs, cohort.patients
    bounds = np.searchsorted(logs["patient_id"].to_numpy(), np.arange(len(patients) + 1))
    timestamps = logs["timestamp"].to_numpy()
    records = list(zip(timestamps, logs["exercise_id"], logs["reps"], logs["pain"]))
    plan = cohort.config.plan_days

    rows = []
    for patient in patients.itertuples(index=False):
        i = patient.patient_id
        start, stop = bounds[i], bounds[i + 1]
        history = [F.Log(float(ts), ex, int(reps), int(pain)) for ts, ex, reps, pain in records[start:stop]]
        patient_ts = timestamps[start:stop]
        for week in SNAPSHOT_WEEKS:
            today = snapshot_time(patient.enroll_day, week)
            count = int(np.searchsorted(patient_ts, today, side="right"))
            values = F.compute_features(history[:count], float(today))
            day = 7 * week - 1
            row = {
                "patient_id": i,
                "week": week,
                "enroll_day": patient.enroll_day,
                "n_logs": count,
                "scored": values is not None,
                "on_track": patient.on_track,
                "off_track": 1 - patient.on_track,
                "persistent": patient.persistent,
                "reduction": patient.reduction,
                "baseline_pain": patient.baseline_pain,
                "floor_fraction": patient.floor_fraction,
                "halflife_days": patient.halflife_days,
                "adherence_logit": patient.adherence_logit,
                "latent_pain": cohort.pain[i, day],
                "latent_benefit": cohort.benefit[i, day],
                "flare_now": cohort.flare[i, day],
                "exercised_share": cohort.exercised[i, : day + 1].mean(),
                "engaged_now": bool(cohort.engaged[i, day]),
                "future_dropout": bool(cohort.engaged[i, day] and not cohort.engaged[i, plan - 1]),
                "flare_at_discharge": cohort.flare[i, plan],
            }
            row.update(values or {name: np.nan for name in F.FEATURE_NAMES})
            rows.append(row)
    return pd.DataFrame(rows)


# --- Calibration ---------------------------------------------------------------

CALIBRATION_SEED = 7
CALIBRATION_PATIENTS = 4000


def _bisect(f: Callable[[float], float], low: float, high: float, iterations: int = 22) -> float:
    f_low = f(low)
    for _ in range(iterations):
        mid = (low + high) / 2
        f_mid = f(mid)
        if (f_mid > 0) == (f_low > 0):
            low, f_low = mid, f_mid
        else:
            high = mid
    return (low + high) / 2


def share_low_adherence(cfg: SimConfig, seed: int = CALIBRATION_SEED) -> float:
    cohort = simulate(replace(cfg, n_patients=CALIBRATION_PATIENTS), seed, with_logs=False)
    return float((cohort.patients["days_exercised"] < cfg.plan_days / 2).mean())


def six_week_ratio(cfg: SimConfig, persistent: bool, seed: int = CALIBRATION_SEED) -> float:
    cohort = simulate(
        replace(cfg, n_patients=CALIBRATION_PATIENTS), seed, persistent_share=float(persistent), with_logs=False
    )
    return float(cohort.pain[:, cfg.plan_days].mean() / cohort.pain[:, 0].mean())


def calibrate(cfg: SimConfig, rounds: int = 3) -> SimConfig:
    """Solve for the [C] parameters so the simulation reproduces its targets.

    Adherence and pain are coupled (pain is a barrier to exercise; exercise
    lowers pain), so the three parameters are solved in alternation.
    """
    for _ in range(rounds):
        cfg = replace(
            cfg,
            adherence_logit_mean=_bisect(
                lambda v: share_low_adherence(replace(cfg, adherence_logit_mean=v)) - lit.SHARE_LOW_ADHERENCE, -4, 4
            ),
        )
        cfg = replace(
            cfg,
            floor_acute=_bisect(
                lambda v: six_week_ratio(replace(cfg, floor_acute=v), False) - lit.SIX_WEEK_RATIO_ACUTE, 0.01, 0.99
            ),
        )
        cfg = replace(
            cfg,
            floor_persistent=_bisect(
                lambda v: six_week_ratio(replace(cfg, floor_persistent=v), True) - lit.SIX_WEEK_RATIO_PERSISTENT,
                0.01,
                0.99,
            ),
        )
    return cfg
