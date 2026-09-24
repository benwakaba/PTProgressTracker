"""Metrics, calibration, decision curves and patient-level bootstrap intervals.

Rows are weekly check-ins, several per patient, so rows are not independent.
Confidence intervals resample patients, not rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def threshold_metrics(y: np.ndarray, flag: np.ndarray) -> dict:
    """Confusion-matrix metrics with off_track = 1 as the positive class."""
    y, flag = np.asarray(y).astype(bool), np.asarray(flag).astype(bool)
    tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
    fn, tn = int((~flag & y).sum()), int((~flag & ~y).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": tp / max(tp + fp, 1),
        "npv": tn / max(tn + fn, 1),
        "flag_rate": flag.mean(),
    }


def per_class_table(y: np.ndarray, flag: np.ndarray) -> pd.DataFrame:
    """Precision/recall/F1 for both classes, as classification_report would give."""
    m = threshold_metrics(y, flag)
    rows = []
    for name, precision, recall, support in (
        ("off track (flagged)", m["ppv"], m["sensitivity"], m["tp"] + m["fn"]),
        ("on track (not flagged)", m["npv"], m["specificity"], m["tn"] + m["fp"]),
    ):
        f1 = 2 * precision * recall / max(precision + recall, EPS)
        rows.append({"class": name, "precision": precision, "recall": recall, "f1": f1, "support": support})
    return pd.DataFrame(rows)


def calibration_fit(y: np.ndarray, p: np.ndarray) -> dict:
    """Calibration-in-the-large (intercept with the slope fixed at 1) and calibration slope.

    Perfect calibration is intercept 0 and slope 1. A slope below 1 means
    predictions are too extreme (overfit); above 1, too timid.
    """
    z = _logit(p)
    ones = np.ones((len(y), 1))
    intercept = sm.GLM(y, ones, family=sm.families.Binomial(), offset=z).fit().params[0]
    if np.ptp(z) == 0:  # a constant forecast has no slope
        return {"calibration_intercept": float(intercept), "calibration_slope": float("nan")}
    slope = sm.GLM(y, np.column_stack([ones, z]), family=sm.families.Binomial()).fit().params[1]
    return {"calibration_intercept": float(intercept), "calibration_slope": float(slope)}


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predicted risk with observed rates and Wilson 95% intervals."""
    order = np.argsort(p, kind="stable")
    rows = []
    for chunk in np.array_split(order, n_bins):
        n = len(chunk)
        observed = y[chunk].mean()
        z = 1.96
        centre = (observed + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(observed * (1 - observed) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        rows.append(
            {"predicted": p[chunk].mean(), "observed": observed, "n": n, "low": centre - half, "high": centre + half}
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = reliability_bins(y, p, n_bins)
    return float((bins["n"] * (bins["predicted"] - bins["observed"]).abs()).sum() / bins["n"].sum())


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    prevalence = y.mean()
    brier = brier_score_loss(y, p)
    return {
        "prevalence": prevalence,
        "mean_predicted": float(p.mean()),
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier,
        # Skill relative to predicting this cohort's own event rate for everyone.
        "brier_skill": 1 - brier / (prevalence * (1 - prevalence)),
        "log_loss": log_loss(y, np.clip(p, EPS, 1 - EPS)),
        "ece": expected_calibration_error(y, p),
        **calibration_fit(y, p),
    }


def patient_bootstrap(frame: pd.DataFrame, metric_fn, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """95% percentile intervals from resampling patients with replacement.

    `metric_fn(frame) -> dict` is evaluated on each resample. Returns one row
    per metric with the full-sample point estimate and interval.
    """
    rng = np.random.default_rng(seed)
    groups = frame.groupby("patient_id").indices
    members = list(groups.values())
    draws = []
    for _ in range(n_boot):
        picked = rng.integers(0, len(members), len(members))
        draws.append(metric_fn(frame.iloc[np.concatenate([members[i] for i in picked])]))
    draws = pd.DataFrame(draws)
    point = metric_fn(frame)
    return pd.DataFrame(
        {
            "estimate": pd.Series(point),
            "low": draws.quantile(0.025),
            "high": draws.quantile(0.975),
        }
    )


def decision_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """Net benefit (Vickers & Elkin 2006) of flagging at each threshold probability.

    Net benefit = TP/N - FP/N * t/(1-t): true flags minus false alarms, with
    false alarms weighted by the odds at which a clinician would be indifferent.
    """
    n = len(y)
    prevalence = y.mean()
    rows = []
    for t in thresholds:
        weight = t / (1 - t)
        flag = p >= t
        tp = (flag & (y == 1)).sum()
        fp = (flag & (y == 0)).sum()
        rows.append(
            {
                "threshold": t,
                "model": tp / n - fp / n * weight,
                "flag_everyone": prevalence - (1 - prevalence) * weight,
                "flag_no_one": 0.0,
            }
        )
    return pd.DataFrame(rows)


def rule_net_benefit(y: np.ndarray, flag: np.ndarray, t: float) -> float:
    n = len(y)
    tp = (flag & (y == 1)).sum()
    fp = (flag & (y == 0)).sum()
    return tp / n - fp / n * t / (1 - t)
