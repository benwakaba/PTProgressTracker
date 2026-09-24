"""Train/test splits by patient and by time.

Each patient contributes up to five weekly check-ins, all sharing that
patient's label. Splitting rows at random puts the same patient on both sides,
and a model can then score well by recognising patients rather than by
predicting outcomes. Every split here keeps each patient on one side.

The headline split is also temporal, because that is how the model would be
used: fit on past patients, applied to future ones. The development cohort is
patients whose week-6 outcome was known before the cutoff. Patients still
mid-plan at the cutoff are embargoed from both sides: their labels were not
available at training time, and none of their check-ins belong to the future.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

# 2025-07-01: 18 months of enrolment for development, the last 6 for testing.
CUTOFF_DAY = 547


@dataclass(frozen=True)
class TemporalSplit:
    dev_ids: np.ndarray
    embargo_ids: np.ndarray
    test_ids: np.ndarray


def temporal_split(patients: pd.DataFrame, plan_days: int, cutoff_day: int = CUTOFF_DAY) -> TemporalSplit:
    outcome_day = patients["enroll_day"] + plan_days
    dev = outcome_day < cutoff_day
    test = patients["enroll_day"] >= cutoff_day
    ids = patients["patient_id"]
    return TemporalSplit(ids[dev].to_numpy(), ids[~dev & ~test].to_numpy(), ids[test].to_numpy())


def patient_folds(groups: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 0) -> list:
    """Cross-validation folds that never split a patient, stratified by label."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def random_row_split(y: np.ndarray, seed: int, test_size: float = 0.2):
    """The split the original pipeline used (MLDataTable.randomSplit): leaks patients."""
    return train_test_split(np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y)


def random_patient_split(groups: np.ndarray, seed: int, test_size: float = 0.2):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(np.zeros(len(groups)), groups=groups))
