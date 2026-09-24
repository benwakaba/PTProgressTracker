import numpy as np
import pytest

import simulate as S
import splits


@pytest.fixture(scope="module")
def cohort():
    return S.simulate(S.SimConfig(n_patients=1500), seed=11)


@pytest.fixture(scope="module")
def snapshots(cohort):
    return S.build_snapshots(cohort)


def test_temporal_split_is_a_partition_with_no_overlap(cohort):
    split = splits.temporal_split(cohort.patients, cohort.config.plan_days)
    ids = [set(split.dev_ids), set(split.embargo_ids), set(split.test_ids)]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    assert set().union(*ids) == set(cohort.patients.patient_id)


def test_development_labels_were_known_before_the_cutoff(cohort):
    split = splits.temporal_split(cohort.patients, cohort.config.plan_days)
    patients = cohort.patients.set_index("patient_id")
    outcome_day = patients.loc[split.dev_ids, "enroll_day"] + cohort.config.plan_days
    assert (outcome_day < splits.CUTOFF_DAY).all()
    assert (patients.loc[split.test_ids, "enroll_day"] >= splits.CUTOFF_DAY).all()


def test_no_development_check_in_happens_after_the_cutoff(cohort, snapshots):
    split = splits.temporal_split(cohort.patients, cohort.config.plan_days)
    dev = snapshots[snapshots.patient_id.isin(split.dev_ids)]
    check_in_day = dev.enroll_day + 7 * dev.week - 1
    assert (check_in_day < splits.CUTOFF_DAY).all()


def test_patient_folds_never_split_a_patient(snapshots):
    scored = snapshots[snapshots.scored]
    groups = scored.patient_id.to_numpy()
    folds = splits.patient_folds(groups, scored.off_track.to_numpy())
    seen = set()
    for train, val in folds:
        assert not set(groups[train]) & set(groups[val])
        seen |= set(val)
    assert seen == set(range(len(scored)))


def test_random_row_split_leaks_patients_and_patient_split_does_not(snapshots):
    scored = snapshots[snapshots.scored]
    groups = scored.patient_id.to_numpy()
    train, test = splits.random_row_split(scored.off_track.to_numpy(), seed=0)
    assert set(groups[train]) & set(groups[test])  # the leak the old pipeline had
    train, test = splits.random_patient_split(groups, seed=0)
    assert not set(groups[train]) & set(groups[test])
