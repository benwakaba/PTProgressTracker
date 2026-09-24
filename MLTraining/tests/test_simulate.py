from dataclasses import replace

import numpy as np
import pytest

import features as F
import literature as lit
import simulate as S


@pytest.fixture(scope="module")
def cohort():
    return S.simulate(S.SimConfig(n_patients=800), seed=3)


def test_same_seed_same_cohort():
    a = S.simulate(S.SimConfig(n_patients=200), seed=5)
    b = S.simulate(S.SimConfig(n_patients=200), seed=5)
    assert a.patients.equals(b.patients)
    assert a.logs.equals(b.logs)


def test_label_is_a_thirty_percent_reduction_in_assessed_pain(cohort):
    p = cohort.patients
    expected = (10 * (p.baseline_nrs - p.week6_nrs) >= 3 * p.baseline_nrs).astype(int)
    assert (p.on_track == expected).all()


def test_logs_only_on_days_a_session_was_performed(cohort):
    logs = cohort.logs
    assert cohort.exercised[logs.patient_id, logs.plan_day].all()
    assert logs.pain.between(0, 10).all() and logs.reps.between(1, 50).all()


def test_nobody_logs_after_dropping_out(cohort):
    logs = cohort.logs
    assert cohort.engaged[logs.patient_id, logs.plan_day].all()


def test_log_timestamps_are_unique_and_fall_on_their_plan_day(cohort):
    logs = cohort.logs.merge(cohort.patients[["patient_id", "enroll_day"]], on="patient_id")
    day = (logs.timestamp - S.EPOCH_2024) // F.SECONDS_PER_DAY
    assert (day == logs.enroll_day + logs.plan_day).all()
    assert not logs.duplicated(["patient_id", "timestamp"]).any()


def test_check_in_features_ignore_later_logs(cohort):
    """No look-ahead: a week-2 check-in is unchanged by anything logged after it."""
    snaps = S.build_snapshots(cohort)
    patient = int(snaps[(snaps.week == 2) & snaps.scored].patient_id.iloc[0])
    enroll = int(cohort.patients.enroll_day[patient])
    history = cohort.logs[cohort.logs.patient_id == patient]
    today = S.snapshot_time(enroll, 2)
    before = [F.Log(float(r.timestamp), r.exercise_id, int(r.reps), int(r.pain)) for r in history.itertuples() if r.timestamp <= today]
    expected = snaps[(snaps.patient_id == patient) & (snaps.week == 2)].iloc[0]
    values = F.compute_features(before, float(today))
    for name in F.FEATURE_NAMES:
        assert values[name] == pytest.approx(expected[name])


def test_logging_parameters_do_not_change_the_latent_course():
    base = S.simulate(S.SimConfig(n_patients=300), seed=9)
    fewer_logs = S.simulate(S.SimConfig(n_patients=300, log_capture=0.5), seed=9)
    assert np.array_equal(base.pain, fewer_logs.pain)
    assert len(fewer_logs.logs) < len(base.logs)


def test_exercise_benefit_matches_the_trial_effect_it_was_derived_from():
    cfg = S.SimConfig()
    rho = np.log(2) / cfg.training_halflife_days
    # Expected benefit at week 6 for someone exercising on trial_adherence of days.
    benefit = cfg.exercise_benefit_full_adherence() * cfg.trial_adherence * (1 - np.exp(-rho * cfg.plan_days))
    assert benefit == pytest.approx(lit.EXERCISE_EFFECT_POINTS)


@pytest.mark.slow
def test_calibration_hits_its_targets_on_a_fresh_seed():
    cfg = S.calibrate(S.SimConfig())
    assert S.share_low_adherence(cfg, seed=123) == pytest.approx(lit.SHARE_LOW_ADHERENCE, abs=0.03)
    assert S.six_week_ratio(cfg, False, seed=123) == pytest.approx(lit.SIX_WEEK_RATIO_ACUTE, abs=0.015)
    assert S.six_week_ratio(cfg, True, seed=123) == pytest.approx(lit.SIX_WEEK_RATIO_PERSISTENT, abs=0.015)
