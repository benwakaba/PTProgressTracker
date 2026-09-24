"""Figures for the study report. Every figure has a table twin in REPORT.md."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots 1-3, validated all-pairs for light mode. Slot 3 (aqua) is
# below 3:1 contrast on the surface, so it is always direct-labelled.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
    }
)

MARKER = dict(marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def reliability(bins, predictions, path, title):
    """bins: reliability_bins() output; predictions: array for the histogram."""
    fig, (ax, hist) = plt.subplots(
        2, 1, figsize=(4.6, 4.8), sharex=True, gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08}
    )
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, zorder=1)
    ax.text(0.97, 0.9, "perfect calibration", color=MUTED, ha="right", fontsize=8, rotation=0)
    ax.errorbar(
        bins["predicted"],
        bins["observed"],
        yerr=[bins["observed"] - bins["low"], bins["high"] - bins["observed"]],
        color=BLUE,
        ecolor=BLUE,
        elinewidth=1,
        capsize=0,
        zorder=3,
        **MARKER,
    )
    ax.plot(bins["predicted"], bins["observed"], color=BLUE, linewidth=1.2, zorder=2)
    ax.set(xlim=(0, 1), ylim=(0, 1), ylabel="Observed share off track")
    ax.set_title(title)
    hist.hist(predictions, bins=np.linspace(0, 1, 41), color=BLUE, alpha=0.35, edgecolor=SURFACE, linewidth=0.6)
    hist.set(xlabel="Predicted probability of being off track", ylabel="Check-ins")
    hist.grid(axis="x", visible=False)
    _save(fig, path)


def decision_curve(curve, rule_point, threshold, path):
    """curve: decision_curve() output; rule_point: (threshold, net benefit) for the rule."""
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    ax.plot(curve["threshold"], curve["model"], color=BLUE, label="Model", zorder=3)
    ax.plot(curve["threshold"], curve["flag_everyone"], color=ORANGE, label="Flag everyone", zorder=2)
    ax.plot(*rule_point, color=AQUA, label="Rule (adherence < 50% or pain rising)", linestyle="none", zorder=4, **MARKER)
    top = curve["model"].max()
    ax.axvline(threshold, color=MUTED, linewidth=0.8, zorder=1)
    ax.text(threshold + 0.008, top + 0.06, f"chosen threshold {threshold:.2f}", color=INK_2, fontsize=8, va="top")
    ax.text(0.52, -0.012, "Flag no one", color=MUTED, fontsize=8, va="top")
    ax.set(xlim=(curve["threshold"].min(), curve["threshold"].max()), ylim=(-0.06, top + 0.08))
    ax.set(xlabel="Threshold probability (flag if P(off track) ≥ threshold)", ylabel="Net benefit")
    ax.set_title("Decision curve, test cohort")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, path)


def leakage(summary, path):
    """summary: rows with model, design, mean, low, high (AUROC)."""
    designs = list(dict.fromkeys(summary["design"]))
    models = list(dict.fromkeys(summary["model"]))
    colors = [BLUE, ORANGE, AQUA]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    offsets = np.linspace(-0.18, 0.18, len(models))
    for model, color, offset in zip(models, colors, offsets):
        rows = summary[summary["model"] == model].set_index("design").loc[designs]
        x = np.arange(len(designs)) + offset
        ax.errorbar(
            x,
            rows["mean"],
            yerr=[rows["mean"] - rows["low"], rows["high"] - rows["mean"]],
            color=color,
            ecolor=color,
            elinewidth=1,
            capsize=0,
            linestyle="none",
            label=model,
            **MARKER,
        )
    ax.set_xticks(np.arange(len(designs)), designs)
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="Test AUROC", xlim=(-0.6, len(designs) - 0.4))
    ax.set_title("Same data, three ways to split it")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, path)


def by_week(table, path):
    """table: week, and one AUROC column per series: model, pain_only, oracle."""
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    # The shipped model and pain-only lines nearly coincide, so no end labels
    # (they would collide); the legend and REPORT.md carry identity and values.
    series = [("oracle", MUTED, "Latent-state oracle (ceiling)"), ("pain_only", ORANGE, "Pain features only"), ("model", BLUE, "Shipped model")]
    for column, color, label in series:
        ax.plot(table["week"], table[column], color=color, label=label, **MARKER)
    ax.axhline(0.5, color=AXIS, linewidth=0.8)
    ax.text(1, 0.505, "chance", color=MUTED, fontsize=8, va="bottom")
    ax.set(xlabel="Weekly check-in", ylabel="Test AUROC", ylim=(0.45, 1.0), xlim=(0.8, table["week"].iloc[-1] + 0.5))
    ax.set_xticks(table["week"])
    ax.grid(axis="x", visible=False)
    ax.set_title("How early can it tell?")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def trajectory(curves, published, path):
    """curves: {cohort: (weeks, mean_ratio, q25, q75)}; published: {cohort: {week: (m, lo, hi, kind)}}."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    titles = {"acute": "Acute pain", "persistent": "Persistent pain"}
    for ax, cohort in zip(axes, ("acute", "persistent")):
        weeks, mean, q25, q75 = curves[cohort]
        ax.fill_between(weeks, q25, q75, color=BLUE, alpha=0.1, linewidth=0)
        ax.plot(weeks, mean, color=BLUE, label="Simulated mean (band: middle 50% of patients)")
        for week, (m, lo, hi, kind) in published[cohort].items():
            filled = kind == "validation"
            ax.errorbar(
                [week],
                [m],
                yerr=[[m - lo], [hi - m]],
                color=ORANGE,
                ecolor=ORANGE,
                elinewidth=1,
                capsize=0,
                marker="o",
                markersize=6,
                markerfacecolor=ORANGE if filled else SURFACE,
                markeredgecolor=ORANGE,
                markeredgewidth=1.4,
                linestyle="none",
                zorder=4,
            )
        ax.set_title(titles[cohort])
        ax.set(xlabel="Weeks since enrolment", xlim=(-1, 53), ylim=(0, 1.25))
        ax.set_xticks([0, 6, 13, 26, 52])
    axes[0].set_ylabel("Pain relative to baseline")
    handles = [
        plt.Line2D([], [], color=BLUE, label="Simulated mean (band: middle 50% of patients)"),
        plt.Line2D([], [], color=ORANGE, marker="o", markerfacecolor=SURFACE, markeredgewidth=1.4, linestyle="none", label="Wallwork 2024, calibration target"),
        plt.Line2D([], [], color=ORANGE, marker="o", linestyle="none", label="Wallwork 2024, held-out check"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    _save(fig, path)
