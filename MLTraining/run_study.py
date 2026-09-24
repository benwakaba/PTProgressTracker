"""Run the simulation study end to end.

    python3 run_study.py           # full run, a few minutes on a laptop
    python3 run_study.py --quick   # fewer bootstrap draws and replicates, for iterating

Writes (committed, so the README's numbers can be checked without rerunning):
    results/REPORT.md          every table, generated
    results/metrics.json       the headline numbers, machine-readable
    results/figures/*.png
and, for export_coreml.py (not committed):
    artifacts/                 final model, calibration, test-set predictions

Every modelling decision (model family, calibration, threshold, warm-up) is
made on the development cohort. The temporal test cohort is scored once.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

import evaluation as E
import literature as lit
import models as M
import plots
import simulate as S
import splits

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = RESULTS / "figures"
ARTIFACTS = HERE / "artifacts"

STUDY_SEED = 2026
# A missed off-track patient is judged twice as costly as a false alarm (README,
# "Choosing the threshold"). With calibrated probabilities, expected cost is
# minimised by flagging when P(off track) >= 1 / (1 + cost ratio).
COST_RATIO = 2.0
THRESHOLD = 1 / (1 + COST_RATIO)
# The card waits until a patient has a week of logging history: week-1
# predictions are weak, and the patients they would catch are caught later.
WARMUP_DAYS = 7
# Sensitivity analysis: strength of the exercise effect x share of sessions logged.
SCENARIOS = [(effect, capture) for effect in (0.0, 0.5, 1.0, 2.0) for capture in (1.0, 0.7)]


# --- report helpers -------------------------------------------------------------


def md_table(frame: pd.DataFrame, formats: dict | None = None) -> str:
    formats = formats or {}
    columns = list(frame.columns)
    numeric = [pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c]) for c in columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---:" if is_num else "---" for is_num in numeric) + "|",
    ]
    for i in range(len(frame)):
        cells = []
        for column in columns:
            value = frame[column].iloc[i]
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                cells.append("")
            elif column in formats:
                cells.append(formats[column].format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


PCT = "{:.1%}"
F3 = "{:.3f}"
F2 = "{:.2f}"


class Report:
    def __init__(self):
        self.parts = []

    def h(self, text: str, level: int = 2):
        self.parts.append(f"{'#' * level} {text}\n")

    def p(self, text: str):
        self.parts.append(text.strip() + "\n")

    def table(self, frame: pd.DataFrame, formats: dict | None = None):
        self.parts.append(md_table(frame, formats) + "\n")

    def write(self, path: Path):
        path.write_text("\n".join(self.parts))


def arrays(frame: pd.DataFrame, columns=M.FEATURES):
    return frame[columns].to_numpy(dtype=float), frame["off_track"].to_numpy()


def shown(frame: pd.DataFrame) -> np.ndarray:
    """Check-ins where the app shows a card: >= 3 logs and a week of history."""
    return (frame["daysSinceStart"] >= WARMUP_DAYS).to_numpy()


# --- analyses -----------------------------------------------------------------


def decision_row(name, y, flag) -> dict:
    t = E.threshold_metrics(y, flag)
    return {
        "model": name,
        "accuracy": t["accuracy"],
        "balanced acc.": t["balanced_accuracy"],
        "sensitivity": t["sensitivity"],
        "specificity": t["specificity"],
        "PPV": t["ppv"],
        "NPV": t["npv"],
        "flagged": t["flag_rate"],
    }


def probability_row(name, y, p) -> dict:
    pm = E.probability_metrics(y, p)
    return {
        "model": name,
        "AUROC": pm["auroc"],
        "AUPRC": pm["auprc"],
        "Brier skill": pm["brier_skill"],
        "log loss": pm["log_loss"],
        "ECE": pm["ece"],
        "cal. intercept": pm["calibration_intercept"],
        "cal. slope": pm["calibration_slope"],
    }


def patient_summary(frame: pd.DataFrame, flag: np.ndarray, warmup: int = WARMUP_DAYS) -> dict:
    """Weekly flags rolled up to patients. `frame` holds every check-in, scored or not.

    'Dangerous misses' counts off-track patients the app could score (>= 3 logs
    at some check-in) but never flagged, and who improved less than 10%. The
    denominator is all off-track patients, so policies compare fairly.
    """
    f = frame.assign(flag=flag, visible=(frame["daysSinceStart"] >= warmup).to_numpy())
    per = f.groupby("patient_id").agg(
        off_track=("off_track", "first"), reduction=("reduction", "first"), scored=("scored", "max"),
        shown=("visible", "max"), ever=("flag", "max"), flag_weeks=("flag", "sum"),
    )
    per["first_week"] = f[f["flag"]].groupby("patient_id")["week"].min().reindex(per.index)
    off, on = per[per["off_track"] == 1], per[per["off_track"] == 0]
    quiet = off[off["scored"] & ~off["ever"]]
    return {
        "per_patient": per,
        "off track, flagged by week 3": (off["first_week"] <= 3).mean(),
        "off track, ever flagged": off["ever"].mean(),
        "dangerous misses": (quiet["reduction"] < 0.1).sum() / len(off),
        "on track, >= 1 false alarm": on["ever"].mean(),
        "false alarms per on-track patient": on["flag_weeks"].mean(),
    }


def patient_funnel(per: pd.DataFrame) -> pd.DataFrame:
    """Every patient lands in exactly one row of their group."""
    off, on = per[per["off_track"] == 1], per[per["off_track"] == 0]
    quiet = off["shown"] & ~off["ever"]
    rows = [
        ("off track", "flagged by week 3", off["first_week"] <= 3),
        ("off track", "first flagged at week 4-5", off["first_week"] >= 4),
        ("off track", "shown a card, never flagged, improved < 10%", quiet & (off["reduction"] < 0.1)),
        ("off track", "shown a card, never flagged, improved 10-19%", quiet & off["reduction"].between(0.1, 0.2 - 1e-9)),
        ("off track", "shown a card, never flagged, improved 20-29% (near miss)", quiet & (off["reduction"] >= 0.2)),
        ("off track", "scored only in the warm-up week (no card)", off["scored"] & ~off["shown"]),
        ("off track", "never scored (< 3 logs at every check-in)", ~off["scored"]),
        ("on track", "shown a card, never flagged", on["shown"] & ~on["ever"]),
        ("on track", "1 false alarm", on["flag_weeks"] == 1),
        ("on track", "2+ false alarms", on["flag_weeks"] >= 2),
        ("on track", "no card (too few logs or warm-up only)", ~on["shown"]),
    ]
    out = []
    for group, label, mask in rows:
        base = off if group == "off track" else on
        out.append({"patients": group, "outcome": label, "n": int(mask.sum()), "share": mask.sum() / len(base)})
    return pd.DataFrame(out)


def error_profile(frame: pd.DataFrame, flag: np.ndarray) -> pd.DataFrame:
    """Who lands in each cell of the confusion matrix, described with ground truth."""
    y = frame["off_track"].to_numpy().astype(bool)
    cells = {
        "caught (TP)": flag & y,
        "missed (FN)": ~flag & y,
        "false alarm (FP)": flag & ~y,
        "correctly quiet (TN)": ~flag & ~y,
    }
    rows = []
    for name, mask in cells.items():
        part = frame[mask]
        rows.append(
            {
                "cell": name,
                "check-ins": int(mask.sum()),
                "week 2": (part["week"] == 2).mean(),
                "persistent pain": part["persistent"].mean(),
                "exercised so far": part["exercised_share"].mean(),
                "pain trend": part["painTrend"].mean(),
                "flare now": (part["flare_now"] >= 1).mean(),
                "drops out later": part["future_dropout"].mean(),
                "final reduction": part["reduction"].mean(),
                "improved 20-29%": part["reduction"].between(0.2, 0.3 - 1e-9).mean(),
                "improved < 10%": (part["reduction"] < 0.1).mean(),
            }
        )
    return pd.DataFrame(rows)


PROFILE_FORMATS = {
    "week 2": PCT, "persistent pain": PCT, "exercised so far": PCT, "pain trend": "{:+.2f}", "flare now": PCT,
    "drops out later": PCT, "final reduction": PCT, "improved 20-29%": PCT, "improved < 10%": PCT,
}


def bootstrap_fn(frame: pd.DataFrame) -> dict:
    y, p = frame["off_track"].to_numpy(), frame["p"].to_numpy()
    prevalence = y.mean()
    s = frame[frame["shown"]]
    ys, flag, rule = s["off_track"].to_numpy(), s["flag"].to_numpy(), s["rule"].to_numpy()
    m, r = E.threshold_metrics(ys, flag), E.threshold_metrics(ys, rule)
    shown_prevalence = ys.mean()
    return {
        "AUROC (all scored)": roc_auc_score(y, p),
        "Brier skill (all scored)": 1 - np.mean((p - y) ** 2) / (prevalence * (1 - prevalence)),
        "accuracy": m["accuracy"],
        "balanced accuracy": m["balanced_accuracy"],
        "sensitivity": m["sensitivity"],
        "specificity": m["specificity"],
        "PPV": m["ppv"],
        "NPV": m["npv"],
        "flagged": m["flag_rate"],
        "accuracy minus majority class": m["accuracy"] - max(shown_prevalence, 1 - shown_prevalence),
        "balanced accuracy minus rule": m["balanced_accuracy"] - r["balanced_accuracy"],
        "net benefit minus rule": E.rule_net_benefit(ys, flag, THRESHOLD) - E.rule_net_benefit(ys, rule, THRESHOLD),
    }


def _leakage_fit(factory, X, y, train, test):
    model = factory().fit(X[train], y[train])
    p = model.predict_proba(X[test])[:, 1]
    return roc_auc_score(y[test], p), float(((p >= 0.5) == y[test]).mean())


def leakage_experiment(frame: pd.DataFrame, factories: dict, repeats: int) -> pd.DataFrame:
    """Refit each model under random-row and random-patient 80/20 splits."""
    X, y = arrays(frame)
    groups = frame["patient_id"].to_numpy()
    jobs, keys = [], []
    for name, factory in factories.items():
        for r in range(repeats):
            for design, (train, test) in (
                ("random rows", splits.random_row_split(y, r)),
                ("random patients", splits.random_patient_split(groups, r)),
            ):
                jobs.append(delayed(_leakage_fit)(factory, X, y, train, test))
                keys.append((name, design, r))
    results = Parallel(n_jobs=-1)(jobs)
    return pd.DataFrame(
        [{"model": k[0], "design": k[1], "repeat": k[2], "auroc": a, "accuracy": acc} for k, (a, acc) in zip(keys, results)]
    )


def _replicate(cfg, seed, factory) -> dict:
    cohort = S.simulate(cfg, seed)
    snaps = S.build_snapshots(cohort)
    split = splits.temporal_split(cohort.patients, cfg.plan_days)
    dev = snaps[snaps.patient_id.isin(split.dev_ids) & snaps.scored]
    test = snaps[snaps.patient_id.isin(split.test_ids) & snaps.scored]
    X_dev, y_dev = arrays(dev)
    X_test, y_test = arrays(test)
    folds = splits.patient_folds(dev["patient_id"].to_numpy(), y_dev)
    oof, _ = M.oof_predictions(factory(), X_dev, y_dev, folds)
    platt = M.Platt().fit(oof, y_dev)
    p = platt(factory().fit(X_dev, y_dev).predict_proba(X_test)[:, 1])
    pm = E.probability_metrics(y_test, p)
    visible = shown(test)
    tm = E.threshold_metrics(y_test[visible], p[visible] >= THRESHOLD)
    ablation = {}
    for label, columns in (("pain_only", M.PAIN_FEATURES), ("adherence_only", M.ADHERENCE_FEATURES)):
        Xd, _ = arrays(dev, columns)
        Xt, _ = arrays(test, columns)
        ablation[label] = roc_auc_score(y_test, factory().fit(Xd, y_dev).predict_proba(Xt)[:, 1])
    return {
        "effect": cfg.exercise_effect_multiplier,
        "capture": cfg.log_capture,
        "seed": seed,
        "prevalence": pm["prevalence"],
        "auroc": pm["auroc"],
        "auroc_pain_only": ablation["pain_only"],
        "auroc_adherence_only": ablation["adherence_only"],
        "brier_skill": pm["brier_skill"],
        "calibration_intercept": pm["calibration_intercept"],
        "sensitivity": tm["sensitivity"],
        "specificity": tm["specificity"],
        "ppv": tm["ppv"],
        "flagged": tm["flag_rate"],
    }


def long_course(cfg: S.SimConfig, persistent_share: float, n: int = 4000) -> np.ndarray:
    """Daily latent pain for 52 weeks. Behaviour after week 6 follows the same process."""
    return S.simulate(replace(cfg, n_patients=n), STUDY_SEED, horizon_days=364, persistent_share=persistent_share, with_logs=False).pain


def course_statistics(cfg: S.SimConfig, persistent_share: float) -> dict:
    stats = {}
    for name, share in (("acute", 0.0), ("persistent", 1.0)):
        pain = long_course(cfg, share)
        for week in (6, 26, 52):
            stats[f"{name} week {week}"] = pain[:, 7 * week].mean() / pain[:, 0].mean()
    pain = long_course(cfg, persistent_share)
    improvement = pain[:, :1] - pain
    stats["Artus week 6 / week 13"] = improvement[:, 42].mean() / improvement[:, 91].mean()
    return stats


def trajectory_validation(cfg: S.SimConfig, persistent_share: float):
    stats = course_statistics(cfg, persistent_share)
    curves, published, rows = {}, {"acute": {}, "persistent": {}}, []
    for name, share in (("acute", 0.0), ("persistent", 1.0)):
        pain = long_course(cfg, share)
        days = np.arange(0, 365, 7)
        relative = pain[:, days] / pain[:, :1]
        curves[name] = (days / 7, pain[:, days].mean(axis=0) / pain[:, 0].mean(), np.quantile(relative, 0.25, axis=0), np.quantile(relative, 0.75, axis=0))
        for week in (6, 26, 52):
            kind = "calibration" if week == 6 else "validation"
            m, lo, hi = lit.wallwork_ratio(name, week)
            published[name][week] = (m, lo, hi, kind)
            value = stats[f"{name} week {week}"]
            rows.append({"target": f"Wallwork 2024, {name}, week {week} / baseline", "used for": kind, "published": m, "95% CI": f"{lo:.2f}-{hi:.2f}", "simulated": value, "inside CI": "yes" if lo <= value <= hi else "no"})
    m, lo, hi = lit.artus_6_over_13()
    value = stats["Artus week 6 / week 13"]
    rows.append({"target": "Artus 2010, improvement at week 6 / week 13", "used for": "validation", "published": m, "95% CI": f"{lo:.2f}-{hi:.2f} (crude)", "simulated": value, "inside CI": "yes" if lo <= value <= hi else "no"})
    return pd.DataFrame(rows), curves, published


# --- main -----------------------------------------------------------------------


def main(quick: bool):
    started = time.time()
    for directory in (RESULTS, FIGURES, ARTIFACTS):
        directory.mkdir(exist_ok=True)
    n_boot, repeats, replicates = (200, 5, 2) if quick else (1000, 20, 5)
    report, metrics = Report(), {}

    # 1. Data-generating mechanism ------------------------------------------------
    cfg = S.calibrate(S.SimConfig())
    cohort = S.simulate(cfg, STUDY_SEED)
    snaps = S.build_snapshots(cohort)
    split = splits.temporal_split(cohort.patients, cfg.plan_days)
    snaps["cohort"] = np.select(
        [snaps.patient_id.isin(split.dev_ids), snaps.patient_id.isin(split.test_ids)], ["development", "test"], "embargoed"
    )
    dev_all = snaps[snaps.cohort == "development"].reset_index(drop=True)
    test_all = snaps[snaps.cohort == "test"].reset_index(drop=True)
    dev = dev_all[dev_all.scored].reset_index(drop=True)
    test = test_all[test_all.scored].reset_index(drop=True)
    X_dev, y_dev = arrays(dev)
    X_test, y_test = arrays(test)
    g_dev = dev["patient_id"].to_numpy()
    shown_dev, shown_test = shown(dev), shown(test)

    report.h("Simulation study report", 1)
    report.p(
        f"Generated by `run_study.py` (seed {STUDY_SEED}{', quick mode' if quick else ''}). Do not edit by hand. "
        "Positive class = off track: the patient will not reach a 30% pain reduction by week 6."
    )

    report.h("Calibration targets")
    report.p("Solved by `simulate.calibrate()`. Agreement here is by construction, not evidence.")
    report.table(
        pd.DataFrame(
            [
                {"target": "patients exercising on < half of plan days", "published": "0.50 (non-adherence 'up to' 50-70%)", "simulated": S.share_low_adherence(cfg), "solved parameter": f"adherence_logit_mean = {cfg.adherence_logit_mean:.3f}"},
                {"target": "acute pain at week 6 / baseline", "published": f"{lit.SIX_WEEK_RATIO_ACUTE:.3f}", "simulated": S.six_week_ratio(cfg, False), "solved parameter": f"floor_acute = {cfg.floor_acute:.3f}"},
                {"target": "persistent pain at week 6 / baseline", "published": f"{lit.SIX_WEEK_RATIO_PERSISTENT:.3f}", "simulated": S.six_week_ratio(cfg, True), "solved parameter": f"floor_persistent = {cfg.floor_persistent:.3f}"},
            ]
        ),
        {"simulated": F3},
    )
    report.p(
        f"Exercise benefit at full adherence: {cfg.exercise_benefit_full_adherence():.2f} NRS points, chosen so that "
        f"{cfg.trial_adherence:.0%} adherence gives the published {cfg.exercise_itt_points:.2f}-point exercise-vs-control "
        "difference at week 6."
    )
    metrics["config"] = asdict(cfg)

    report.h("Cohorts")
    cohort_rows = []
    for name in ("development", "embargoed", "test"):
        part = snaps[snaps.cohort == name]
        people = cohort.patients[cohort.patients.patient_id.isin(part.patient_id.unique())]
        scored = part[part.scored]
        cohort_rows.append(
            {
                "cohort": name,
                "patients": len(people),
                "enrolled (day)": f"{people.enroll_day.min()}-{people.enroll_day.max()}",
                "persistent pain": people.persistent.mean(),
                "off track (patients)": 1 - people.on_track.mean(),
                "check-ins": len(part),
                "scored (>= 3 logs)": part.scored.mean(),
                "card shown": (part.daysSinceStart >= WARMUP_DAYS).mean(),
                "off track (scored check-ins)": scored.off_track.mean(),
            }
        )
    cohort_table = pd.DataFrame(cohort_rows)
    report.p(
        f"Cutoff day {splits.CUTOFF_DAY} (2025-07-01). Development = week-6 outcome known before the cutoff; "
        "test = enrolled on or after it; embargoed = mid-plan at the cutoff, used by neither. "
        "The share of persistent-pain patients drifts upward across enrolment by design."
    )
    report.table(cohort_table, {c: PCT for c in ("persistent pain", "off track (patients)", "scored (>= 3 logs)", "card shown", "off track (scored check-ins)")})
    metrics["cohorts"] = cohort_table.to_dict(orient="records")

    # 2. Model selection (development cohort, patient-grouped CV) -----------------
    folds = splits.patient_folds(g_dev, y_dev, seed=0)
    for train, val in folds:
        assert not set(g_dev[train]) & set(g_dev[val]), "a patient is on both sides of a fold"
    selection = M.select(X_dev, y_dev, folds)
    best_logistic = selection.table[selection.table.family == "logistic"].sort_values("log_loss").iloc[0]["candidate"]
    best_boosted = selection.table[selection.table.family == "boosted"].sort_values("log_loss").iloc[0]["candidate"]
    chosen = selection.chosen
    factory = M.CANDIDATES[chosen][1]
    report.h("Model selection")
    report.p(
        "Five-fold cross-validation on all scored development check-ins, folds grouped by patient and stratified by "
        "label, scored by log loss. The simplest family within one standard error of the best wins. "
        f"Best: `{selection.best}`. Chosen: `{chosen}`."
    )
    report.table(selection.table.rename(columns={"log_loss": "CV log loss", "se": "SE"}), {"CV log loss": "{:.4f}", "SE": "{:.4f}"})
    metrics["selection"] = {"best": selection.best, "chosen": chosen, "table": selection.table.to_dict(orient="records")}

    # 3. Calibration (fitted on out-of-fold predictions) --------------------------
    oof_raw = selection.oof[chosen]
    platt = M.Platt().fit(oof_raw, y_dev)
    oof = platt(oof_raw)
    report.h("Calibration")
    report.p(
        f"Platt recalibration fitted on out-of-fold predictions: p' = sigmoid({platt.a:.3f} * logit(p) "
        f"{'-' if platt.b < 0 else '+'} {abs(platt.b):.3f}). "
        "a = 1 and b = 0 would be a no-op."
    )
    cal_rows = []
    for label, p in (("before recalibration", oof_raw), ("after recalibration", oof)):
        pm = E.probability_metrics(y_dev, p)
        cal_rows.append({"development, out-of-fold": label, "mean predicted": pm["mean_predicted"], "observed": pm["prevalence"], "cal. intercept": pm["calibration_intercept"], "cal. slope": pm["calibration_slope"], "ECE": pm["ece"], "log loss": pm["log_loss"]})
    report.table(pd.DataFrame(cal_rows), {c: F3 for c in ("mean predicted", "observed", "cal. intercept", "cal. slope", "ECE", "log loss")})

    # 4. Decision policy (development cohort only) --------------------------------
    report.h("Decision policy (development cohort, out-of-fold)")
    week_dev = []
    for week in S.SNAPSHOT_WEEKS:
        mask = dev.week.to_numpy() == week
        week_dev.append({"week": week, "scored check-ins": int(mask.sum()), "off track": y_dev[mask].mean(), "AUROC": roc_auc_score(y_dev[mask], oof[mask])})
    report.p("How well the model ranks patients at each weekly check-in:")
    report.table(pd.DataFrame(week_dev), {"off track": PCT, "AUROC": F3})

    policy_rows = []
    dev_all_p = np.full(len(dev_all), np.nan)
    dev_all_p[dev_all.scored.to_numpy()] = oof
    for ratio in (1, 2, 3, 4):
        t = 1 / (1 + ratio)
        for warmup in (0, WARMUP_DAYS):
            flag = (dev_all.daysSinceStart >= warmup).to_numpy() & (np.nan_to_num(dev_all_p, nan=0) >= t) & dev_all.scored.to_numpy()
            summary = patient_summary(dev_all, flag, warmup)
            visible = (dev.daysSinceStart >= warmup).to_numpy()
            m = E.threshold_metrics(y_dev[visible], oof[visible] >= t)
            policy_rows.append(
                {
                    "cost ratio": ratio,
                    "threshold": t,
                    "warm-up": f"{warmup} days",
                    "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"],
                    "PPV": m["ppv"],
                    "off track, flagged by wk 3": summary["off track, flagged by week 3"],
                    "off track, ever flagged": summary["off track, ever flagged"],
                    "dangerous misses": summary["dangerous misses"],
                    "on track, >= 1 false alarm": summary["on track, >= 1 false alarm"],
                    "false alarms / on-track patient": summary["false alarms per on-track patient"],
                }
            )
    policy_table = pd.DataFrame(policy_rows)
    report.p(
        "Per-check-in metrics are over check-ins where a card would be shown. Patient columns count every development "
        "patient, including those never scored. 'Dangerous misses' = off-track patients who were scored but never "
        "flagged and improved less than 10%, as a share of all off-track patients."
    )
    fmt = {c: PCT for c in policy_table.columns if c not in ("cost ratio", "threshold", "warm-up", "false alarms / on-track patient")}
    fmt.update({"threshold": F3, "false alarms / on-track patient": F2})
    report.table(policy_table, fmt)
    report.p(
        f"Shipped policy: cost ratio {COST_RATIO:.0f} (threshold {THRESHOLD:.3f}) with a {WARMUP_DAYS}-day warm-up. "
        "Disclosure: the warm-up was considered after a debugging run had printed the test cohort's by-week table; "
        "the decision rests on the development numbers above, which show the same pattern."
    )
    dev_flag = shown_dev & (oof >= THRESHOLD)
    dev_profile = error_profile(dev[shown_dev], dev_flag[shown_dev])
    report.p("Error profile on the development cohort at the shipped policy:")
    report.table(dev_profile, PROFILE_FORMATS)
    metrics["policy"] = {"cost_ratio": COST_RATIO, "threshold": THRESHOLD, "warmup_days": WARMUP_DAYS, "dev_by_week": week_dev, "dev_table": policy_table.to_dict(orient="records")}

    # 5. Final fit on all development data; score the test cohort once -------------
    final = factory().fit(X_dev, y_dev)
    p_test_raw = final.predict_proba(X_test)[:, 1]
    p_test = platt(p_test_raw)
    flag = shown_test & (p_test >= THRESHOLD)
    rule = M.rule_flags(test) & shown_test
    test = test.assign(p=p_test, flag=flag, rule=rule, shown=shown_test)

    raw = {name: M.CANDIDATES[name][1]().fit(X_dev, y_dev).predict_proba(X_test)[:, 1] for name in (best_logistic, best_boosted)}
    oracle_X_dev, _ = arrays(dev, M.ORACLE_FEATURES)
    oracle_X_test, _ = arrays(test, M.ORACLE_FEATURES)
    p_oracle = M.CANDIDATES[best_boosted][1]().fit(oracle_X_dev, y_dev).predict_proba(oracle_X_test)[:, 1]
    duration_X_dev, _ = arrays(dev, M.FEATURES_WITH_DURATION)
    duration_X_test, _ = arrays(test, M.FEATURES_WITH_DURATION)
    duration_oof, _ = M.oof_predictions(factory(), duration_X_dev, y_dev, folds)
    duration_platt = M.Platt().fit(duration_oof, y_dev)
    p_duration = duration_platt(factory().fit(duration_X_dev, y_dev).predict_proba(duration_X_test)[:, 1])
    ablations = {}
    for label, columns in (("pain_only", M.PAIN_FEATURES), ("adherence_only", M.ADHERENCE_FEATURES)):
        Xd, _ = arrays(dev, columns)
        Xt, _ = arrays(test, columns)
        ablations[label] = factory().fit(Xd, y_dev).predict_proba(Xt)[:, 1]

    constant = np.full(len(y_test), y_dev.mean())
    report.h("Test cohort: model vs baselines")
    ys = y_test[shown_test]
    report.p(
        f"{len(test)} scored check-ins ({shown_test.sum()} with a card shown) from {test.patient_id.nunique()} patients "
        f"enrolled after the cutoff. Off-track rate on shown check-ins: {ys.mean():.1%}, so always answering "
        f"'on track' is {1 - ys.mean():.1%} accurate."
    )
    report.h("Ranking and calibration (all scored check-ins)", 3)
    probability_table = pd.DataFrame(
        [
            probability_row("Majority class (development base rate for everyone)", y_test, constant),
            probability_row(f"Logistic regression, {best_logistic.split()[-1]}, uncalibrated", y_test, raw[best_logistic]),
            probability_row(f"Boosted trees, {best_boosted.split()[-1]}, uncalibrated", y_test, raw[best_boosted]),
            probability_row(f"Shipped: {chosen} + Platt", y_test, p_test),
            probability_row("Shipped features + pain duration at intake (hypothetical)", y_test, p_duration),
            probability_row("Oracle: sees the latent state (ceiling, not deployable)", y_test, p_oracle),
        ]
    )
    report.table(probability_table, {c: F3 for c in probability_table.columns if c != "model"})
    report.h(f"Decisions at the shipped policy (check-ins where a card is shown, threshold {THRESHOLD:.3f})", 3)
    decision_table = pd.DataFrame(
        [
            decision_row("Majority class: always 'on track'", ys, np.zeros(len(ys), bool)),
            decision_row("Rule: adherence < 50% or pain rising", ys, rule[shown_test]),
            decision_row(f"Logistic regression, {best_logistic.split()[-1]}, uncalibrated", ys, raw[best_logistic][shown_test] >= THRESHOLD),
            decision_row(f"Boosted trees, {best_boosted.split()[-1]}, uncalibrated", ys, raw[best_boosted][shown_test] >= THRESHOLD),
            decision_row(f"Shipped: {chosen} + Platt", ys, flag[shown_test]),
            decision_row("Shipped features + pain duration at intake (hypothetical)", ys, p_duration[shown_test] >= THRESHOLD),
            decision_row("Oracle (ceiling)", ys, p_oracle[shown_test] >= THRESHOLD),
        ]
    )
    report.table(decision_table, {c: PCT for c in decision_table.columns if c != "model"})
    metrics["test_probability"] = probability_table.to_dict(orient="records")
    metrics["test_decisions"] = decision_table.to_dict(orient="records")

    boot = E.patient_bootstrap(test, bootstrap_fn, n_boot=n_boot, seed=1)
    report.h("Shipped model with 95% patient-bootstrap intervals", 3)
    report.p(f"{n_boot} resamples of patients, not rows. Decision metrics are over check-ins where a card is shown.")
    boot_table = boot.reset_index().rename(columns={"index": "metric"})
    report.table(boot_table, {"estimate": F3, "low": F3, "high": F3})
    metrics["test_bootstrap"] = boot_table.to_dict(orient="records")

    cm = E.threshold_metrics(ys, flag[shown_test])
    report.h("Confusion matrix (check-ins where a card is shown)", 3)
    report.table(
        pd.DataFrame(
            [
                {"": "actually off track", "flagged": cm["tp"], "not flagged": cm["fn"], "total": cm["tp"] + cm["fn"]},
                {"": "actually on track", "flagged": cm["fp"], "not flagged": cm["tn"], "total": cm["fp"] + cm["tn"]},
                {"": "total", "flagged": cm["tp"] + cm["fp"], "not flagged": cm["fn"] + cm["tn"], "total": len(ys)},
            ]
        )
    )
    per_class = E.per_class_table(ys, flag[shown_test])
    report.h("Per-class precision and recall", 3)
    report.table(per_class, {"precision": PCT, "recall": PCT, "f1": F3})
    metrics["confusion_matrix"] = {k: cm[k] for k in ("tp", "fp", "fn", "tn")}
    metrics["per_class"] = per_class.to_dict(orient="records")

    bins = E.reliability_bins(y_test, p_test)
    plots.reliability(bins, p_test, FIGURES / "reliability.png", "Reliability, test cohort")
    report.h("Reliability (all scored test check-ins)", 3)
    report.p("Equal-count bins of predicted risk with Wilson 95% intervals. Figure: `figures/reliability.png`.")
    report.table(bins.rename(columns={"low": "95% low", "high": "95% high"}), {"predicted": F3, "observed": F3, "n": "{:.0f}", "95% low": F3, "95% high": F3})

    subgroup_rows = []
    persistent = test.persistent.to_numpy()
    week = test.week.to_numpy()
    for label, mask in (
        ("all", np.ones(len(test), bool)),
        ("acute pain", ~persistent),
        ("persistent pain", persistent),
        ("week 2", week == 2),
        ("week 5", week == 5),
    ):
        for model_name, p in (("shipped", p_test), ("+ pain duration", p_duration)):
            pm = E.probability_metrics(y_test[mask], p[mask])
            subgroup_rows.append({"test check-ins": label, "model": model_name, "n": int(mask.sum()), "mean predicted": pm["mean_predicted"], "observed": pm["prevalence"], "cal. intercept": pm["calibration_intercept"], "cal. slope": pm["calibration_slope"], "AUROC": pm["auroc"]})
    dev_persistent = dev.persistent.to_numpy()
    for i, (label, mask) in enumerate(
        (("all", np.ones(len(dev), bool)), ("acute pain", ~dev_persistent), ("persistent pain", dev_persistent))
    ):
        pm = E.probability_metrics(y_dev[mask], oof[mask])
        subgroup_rows.insert(i, {"test check-ins": f"development out-of-fold (reference), {label}", "model": "shipped", "n": int(mask.sum()), "mean predicted": pm["mean_predicted"], "observed": pm["prevalence"], "cal. intercept": pm["calibration_intercept"], "cal. slope": pm["calibration_slope"], "AUROC": pm["auroc"]})
    report.h("Calibration by subgroup", 3)
    report.p("Case mix drifts toward persistent pain after the cutoff; the app's features only partly reveal it.")
    report.table(pd.DataFrame(subgroup_rows), {c: F3 for c in ("mean predicted", "observed", "cal. intercept", "cal. slope", "AUROC")})
    metrics["calibration_subgroups"] = subgroup_rows

    grid = np.round(np.arange(0.05, 0.71, 0.01), 2)
    curve = E.decision_curve(ys, p_test[shown_test], grid)
    rule_point = (THRESHOLD, E.rule_net_benefit(ys, rule[shown_test], THRESHOLD))
    plots.decision_curve(curve, rule_point, THRESHOLD, FIGURES / "decision_curve.png")
    report.h("Decision curve (check-ins where a card is shown)", 3)
    report.p("Net benefit = TP/N - FP/N * t/(1-t). Figure: `figures/decision_curve.png`.")
    report.table(curve[curve.threshold.isin([0.1, 0.2, 0.25, 0.3, round(THRESHOLD, 2), 0.4, 0.5, 0.6])], {"threshold": F2, "model": "{:.4f}", "flag_everyone": "{:.4f}", "flag_no_one": "{:.4f}"})
    report.p(f"Rule at threshold {THRESHOLD:.3f}: net benefit {rule_point[1]:.4f}.")

    week_rows = []
    for w in S.SNAPSHOT_WEEKS:
        mask = week == w
        visible = mask & shown_test
        m = E.threshold_metrics(y_test[visible], flag[visible]) if visible.any() else None
        all_week = test_all[test_all.week == w]
        week_rows.append(
            {
                "week": w,
                "scored": all_week.scored.mean(),
                "card shown": (all_week.daysSinceStart >= WARMUP_DAYS).mean(),
                "off track": y_test[mask].mean(),
                "model": roc_auc_score(y_test[mask], p_test[mask]),
                "pain_only": roc_auc_score(y_test[mask], ablations["pain_only"][mask]),
                "adherence_only": roc_auc_score(y_test[mask], ablations["adherence_only"][mask]),
                "oracle": roc_auc_score(y_test[mask], p_oracle[mask]),
                "sensitivity": m["sensitivity"] if m else np.nan,
                "specificity": m["specificity"] if m else np.nan,
            }
        )
    week_table = pd.DataFrame(week_rows)
    plots.by_week(week_table, FIGURES / "by_week.png")
    overall = {
        "model": roc_auc_score(y_test, p_test),
        "pain_only": roc_auc_score(y_test, ablations["pain_only"]),
        "adherence_only": roc_auc_score(y_test, ablations["adherence_only"]),
        "pain_duration": roc_auc_score(y_test, p_duration),
        "oracle": roc_auc_score(y_test, p_oracle),
    }
    report.h("By weekly check-in (all scored test check-ins)", 3)
    report.p(
        "AUROC for the shipped model, pain features only, adherence features only, and the latent-state oracle. "
        "Sensitivity and specificity are for check-ins where a card is shown (none in week 1). Figure: `figures/by_week.png`."
    )
    report.table(week_table, {"scored": PCT, "card shown": PCT, "off track": PCT, "model": F3, "pain_only": F3, "adherence_only": F3, "oracle": F3, "sensitivity": PCT, "specificity": PCT})
    report.p("Overall AUROC: " + ", ".join(f"{k} {v:.3f}" for k, v in overall.items()) + ".")
    metrics["by_week"] = week_rows
    metrics["ablation_auroc"] = overall

    # 6. Error analysis ------------------------------------------------------------
    report.h("Error analysis (test cohort)")
    profile = error_profile(test[shown_test], flag[shown_test])
    report.p(
        "Check-ins where a card is shown, by confusion-matrix cell, described with ground truth only the simulation "
        "knows. 'Exercised so far' is the true share of plan days exercised up to the check-in; 'drops out later' "
        "means still exercising at the check-in but stopped for good before week 6; 'flare now' is a transient "
        "spike of at least 1 NRS point."
    )
    report.table(profile, PROFILE_FORMATS)
    all_flags = np.zeros(len(test_all), bool)
    all_flags[test_all.scored.to_numpy()] = flag
    summary = patient_summary(test_all, all_flags)
    funnel = patient_funnel(summary.pop("per_patient"))
    report.h("Patient level: what happens to each patient over five check-ins", 3)
    report.table(funnel, {"share": PCT})
    blind = {
        "check-ins with < 3 logs (no card)": test_all[~test_all.scored],
        "check-ins in the warm-up week (no card)": test_all[test_all.scored & (test_all.daysSinceStart < WARMUP_DAYS)],
        "check-ins with a card": test_all[test_all.scored & (test_all.daysSinceStart >= WARMUP_DAYS)],
    }
    blind_table = pd.DataFrame(
        [{"check-ins": k, "n": len(v), "share": len(v) / len(test_all), "off track": v.off_track.mean()} for k, v in blind.items()]
    )
    report.h("Where the app is silent", 3)
    report.table(blind_table, {"share": PCT, "off track": PCT})
    metrics["error_analysis"] = {
        "profile": profile.to_dict(orient="records"),
        "patients": summary,
        "funnel": funnel.to_dict(orient="records"),
        "silent": blind_table.to_dict(orient="records"),
    }

    # 7. Why split by patient: the leakage experiment ------------------------------
    all_scored = snaps[snaps.scored].reset_index(drop=True)
    factories = {
        "Logistic regression": M.CANDIDATES[best_logistic][1],
        "Boosted trees (tuned)": M.CANDIDATES[best_boosted][1],
        "Boosted trees (CreateML-like defaults)": M.createml_like,
    }
    leak = leakage_experiment(all_scored, factories, repeats)
    leak_summary = (
        leak.groupby(["model", "design"], sort=False)
        .agg(mean=("auroc", "mean"), low=("auroc", lambda a: a.quantile(0.025)), high=("auroc", lambda a: a.quantile(0.975)), accuracy=("accuracy", "mean"))
        .reset_index()
    )
    temporal_rows = []
    for name, f in factories.items():
        p = f().fit(X_dev, y_dev).predict_proba(X_test)[:, 1]
        auc_boot = E.patient_bootstrap(test.assign(q=p), lambda fr: {"auroc": roc_auc_score(fr["off_track"], fr["q"])}, n_boot=n_boot // 5, seed=2)
        temporal_rows.append({"model": name, "design": "temporal cohorts", "mean": roc_auc_score(y_test, p), "low": auc_boot.loc["auroc", "low"], "high": auc_boot.loc["auroc", "high"], "accuracy": float(((p >= 0.5) == y_test).mean())})
    leak_summary = pd.concat([leak_summary, pd.DataFrame(temporal_rows)], ignore_index=True)
    plots.leakage(leak_summary, FIGURES / "leakage.png")
    paired = leak.pivot_table(index=["model", "repeat"], columns="design", values="auroc").reset_index()
    paired["inflation"] = paired["random rows"] - paired["random patients"]
    inflation = paired.groupby("model", sort=False)["inflation"].agg(["mean", "sem", "min", "max"]).reset_index()
    inflation = inflation.rename(columns={"sem": "SE"})
    ranks = leak_summary.pivot(index="model", columns="design", values="mean")
    report.h("Why split by patient (leakage experiment)")
    report.p(
        f"All scored check-ins, 80/20 splits repeated {repeats} times. 'random rows' is what the original pipeline did "
        "(MLDataTable.randomSplit), which puts the same patient's check-ins on both sides. Intervals: 2.5-97.5 "
        "percentiles across repeats; for 'temporal cohorts' (one split) a patient-bootstrap interval. Accuracy at 0.5. "
        "Figure: `figures/leakage.png`."
    )
    report.table(leak_summary, {"mean": F3, "low": F3, "high": F3, "accuracy": PCT})
    report.p("AUROC inflation from leaking patients (random rows minus random patients, paired by repeat):")
    report.table(inflation, {"mean": "{:+.3f}", "SE": "{:.3f}", "min": "{:+.3f}", "max": "{:+.3f}"})
    report.p(
        "Ranking by random-row AUROC: " + " > ".join(ranks["random rows"].sort_values(ascending=False).index)
        + ". Ranking by random-patient AUROC: " + " > ".join(ranks["random patients"].sort_values(ascending=False).index) + "."
    )
    metrics["leakage"] = {"summary": leak_summary.to_dict(orient="records"), "inflation": inflation.to_dict(orient="records")}

    # 8. Sensitivity analysis over the assumptions that drive the result -----------
    scenario_configs = {}
    for effect, capture in SCENARIOS:
        if (effect, 1.0) in scenario_configs:  # logging does not change the latent course
            scenario_configs[(effect, capture)] = replace(scenario_configs[(effect, 1.0)], log_capture=capture)
        else:
            scenario_configs[(effect, capture)] = S.calibrate(replace(S.SimConfig(), exercise_effect_multiplier=effect, log_capture=capture))
    jobs = [delayed(_replicate)(scenario_configs[key], 100 + r, factory) for key in SCENARIOS for r in range(replicates)]
    sensitivity = pd.DataFrame(Parallel(n_jobs=-1)(jobs))
    sens_summary = (
        sensitivity.groupby(["effect", "capture"])
        .agg(
            prevalence=("prevalence", "mean"),
            auroc=("auroc", "mean"),
            auroc_sd=("auroc", "std"),
            pain_only=("auroc_pain_only", "mean"),
            adherence_only=("auroc_adherence_only", "mean"),
            brier_skill=("brier_skill", "mean"),
            cal_intercept=("calibration_intercept", "mean"),
            sensitivity=("sensitivity", "mean"),
            specificity=("specificity", "mean"),
            ppv=("ppv", "mean"),
            flagged=("flagged", "mean"),
        )
        .reset_index()
        .sort_values(["capture", "effect"], ascending=[False, True])
    )
    report.h("Sensitivity analysis")
    report.p(
        "Two assumptions varied: the exercise effect (a multiplier on the literature-calibrated effect) and the share of "
        f"performed sessions that get logged. {replicates} replicate cohorts per scenario (seeds 100+), each scenario "
        "recalibrated so the average 6-week pain course still matches Wallwork 2024. Same model spec "
        f"(`{chosen}`) and policy throughout. Means across replicates; auroc_sd is the SD across replicates."
    )
    report.table(sens_summary, {"effect": "{:.1f}x", "capture": PCT, "prevalence": PCT, "auroc": F3, "auroc_sd": F3, "pain_only": F3, "adherence_only": F3, "brier_skill": F3, "cal_intercept": "{:+.3f}", "sensitivity": PCT, "specificity": PCT, "ppv": PCT, "flagged": PCT})
    metrics["sensitivity"] = sens_summary.to_dict(orient="records")

    # 9. Validating one sub-component against published data -----------------------
    share = float(cohort.patients.persistent.mean())
    validation, curves, published = trajectory_validation(cfg, share)
    plots.trajectory(curves, published, FIGURES / "trajectory_validation.png")
    report.h("Pain-trajectory validation")
    report.p(
        "The same simulated patients run out to 52 weeks. After week 6 behaviour follows the same process: motivation "
        "keeps decaying, dropouts stay out, and the exercise benefit fades without sessions. The week-6 values were "
        "calibration targets; the week-26/52 values and the Artus ratio were not used to set anything. Published "
        "intervals are the paper's 95% CI divided by its baseline point estimate. "
        "Figure: `figures/trajectory_validation.png`."
    )
    report.table(validation, {"published": F3, "simulated": F3})
    sweep_rows = []
    for effect in (0.0, 0.5, 1.0, 2.0):
        stats = course_statistics(scenario_configs[(effect, 1.0)], share)
        sweep_rows.append({"exercise effect": f"{effect:.1f}x", **{k: v for k, v in stats.items() if "week 6" not in k or "Artus" in k}})
    published_row = {"exercise effect": "published"}
    for name in ("acute", "persistent"):
        for w in (26, 52):
            published_row[f"{name} week {w}"] = lit.wallwork_ratio(name, w)[0]
    published_row["Artus week 6 / week 13"] = lit.artus_6_over_13()[0]
    sweep = pd.DataFrame([published_row] + sweep_rows)
    report.p(
        "The same held-out statistics under each exercise-effect scenario (post hoc: how the long-term data bears on "
        "the key assumption; nothing was tuned to it):"
    )
    report.table(sweep, {c: F3 for c in sweep.columns if c != "exercise effect"})
    metrics["validation"] = {"table": validation.to_dict(orient="records"), "by_scenario": sweep.to_dict(orient="records")}

    # 10. Artifacts for export_coreml.py -------------------------------------------------
    joblib.dump(final, ARTIFACTS / "final_model.joblib")
    (ARTIFACTS / "calibration.json").write_text(
        json.dumps(
            {
                "candidate": chosen,
                "features": M.FEATURES,
                "platt_a": platt.a,
                "platt_b": platt.b,
                "threshold": THRESHOLD,
                "cost_ratio": COST_RATIO,
                "warmup_days": WARMUP_DAYS,
                "sklearn": sklearn.__version__,
                "seed": STUDY_SEED,
                "test_auroc": overall["model"],
            },
            indent=2,
        )
    )
    test[M.FEATURES].assign(p_raw=p_test_raw, p=p_test).to_csv(ARTIFACTS / "test_predictions.csv", index=False)

    # Runtime goes to stdout only, so an unchanged rerun leaves results/ byte-identical.
    report.write(RESULTS / "REPORT.md")
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))
    print(f"Done in {time.time() - started:.1f} s. Chosen {chosen}; test AUROC {overall['model']:.3f}.")


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return round(float(value), 6)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="fewer bootstrap draws and replicates")
    main(parser.parse_args().quick)
