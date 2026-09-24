"""Candidate models, baselines and post-hoc calibration.

The positive class throughout is off_track = 1 (the patient will not reach a
30% pain reduction by week 6): it is the class the app acts on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import features as F

FEATURES = list(F.FEATURE_NAMES)
# Ablations: what each group of app features carries on its own.
PAIN_FEATURES = ["daysSinceStart", "avgPainLast7", "painTrend"]
ADHERENCE_FEATURES = ["daysSinceStart", "adherenceRatio", "currentStreak", "completionRatio"]
# Ground truth only a simulation has. A model given these approximates the
# ceiling for any model: what is left is randomness that hasn't happened yet.
LATENT_FEATURES = [
    "persistent",
    "baseline_pain",
    "floor_fraction",
    "halflife_days",
    "adherence_logit",
    "latent_pain",
    "latent_benefit",
    "flare_now",
    "exercised_share",
    "engaged_now",
]
ORACLE_FEATURES = FEATURES + LATENT_FEATURES
# A hypothetical intake question: has the pain lasted 12+ weeks? In the
# simulation that is exactly the `persistent` flag.
FEATURES_WITH_DURATION = FEATURES + ["persistent"]

EPS = 1e-6


def logistic(C: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=C, max_iter=2000))])


def boosted(n_estimators: int, max_depth: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=0,
    )


def createml_like() -> GradientBoostingClassifier:
    """Roughly MLBoostedTreeClassifier's defaults: 10 rounds, depth 6, step 0.3."""
    return GradientBoostingClassifier(
        n_estimators=10, max_depth=6, learning_rate=0.3, min_samples_leaf=1, random_state=0
    )


# Listed simplest family first; select() prefers earlier families on near-ties.
CANDIDATES = {
    "logistic C=0.01": ("logistic", lambda: logistic(0.01)),
    "logistic C=0.1": ("logistic", lambda: logistic(0.1)),
    "logistic C=1": ("logistic", lambda: logistic(1.0)),
    "logistic C=10": ("logistic", lambda: logistic(10.0)),
    "boosted 100x2": ("boosted", lambda: boosted(100, 2)),
    "boosted 300x2": ("boosted", lambda: boosted(300, 2)),
    "boosted 100x3": ("boosted", lambda: boosted(100, 3)),
    "boosted 300x3": ("boosted", lambda: boosted(300, 3)),
}
FAMILY_ORDER = ["logistic", "boosted"]


def rule_flags(frame: pd.DataFrame) -> np.ndarray:
    """The rule the app's rationale text already implies: low adherence or rising pain."""
    return ((frame["adherenceRatio"] < 0.5) | (frame["painTrend"] > 0)).to_numpy()


def oof_predictions(model, X: np.ndarray, y: np.ndarray, folds: list) -> tuple:
    """Out-of-fold P(off track) and per-fold log loss."""
    oof = np.empty(len(y))
    losses = []
    for train, val in folds:
        fitted = clone(model).fit(X[train], y[train])
        oof[val] = fitted.predict_proba(X[val])[:, 1]
        losses.append(log_loss(y[val], oof[val], labels=[0, 1]))
    return oof, np.array(losses)


@dataclass
class Selection:
    table: pd.DataFrame  # one row per candidate: family, mean and SE of fold log loss
    best: str            # lowest mean log loss
    chosen: str          # simplest family within one SE of the best
    oof: dict            # candidate -> out-of-fold predictions


def select(X: np.ndarray, y: np.ndarray, folds: list, candidates: dict = CANDIDATES) -> Selection:
    """Pick a model by cross-validated log loss with the one-standard-error rule.

    Log loss is a proper scoring rule: it rewards calibrated probabilities,
    which accuracy ignores. Ties within one standard error go to the simpler
    family, because a gain smaller than the fold-to-fold noise is not evidence.
    """
    rows, oof = [], {}
    for name, (family, factory) in candidates.items():
        oof[name], losses = oof_predictions(factory(), X, y, folds)
        rows.append(
            {
                "candidate": name,
                "family": family,
                "log_loss": losses.mean(),
                "se": losses.std(ddof=1) / np.sqrt(len(losses)),
            }
        )
    table = pd.DataFrame(rows)
    best = table.loc[table["log_loss"].idxmin()]
    eligible = table[table["log_loss"] <= best["log_loss"] + best["se"]].copy()
    eligible["rank"] = eligible["family"].map(FAMILY_ORDER.index)
    chosen = eligible.sort_values(["rank", "log_loss"]).iloc[0]["candidate"]
    return Selection(table, best["candidate"], chosen, oof)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class Platt:
    """Logistic recalibration: p' = sigmoid(a * logit(p) + b).

    Fitted on out-of-fold predictions so it corrects the model's behaviour on
    patients it has not seen. Two numbers, so the app applies it in Swift.
    """

    a: float = 1.0
    b: float = 0.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Platt":
        lr = LogisticRegression(penalty=None, max_iter=1000).fit(_logit(p)[:, None], y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.a * _logit(p) + self.b)))
