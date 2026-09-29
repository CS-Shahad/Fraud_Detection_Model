"""Charts used in the notebook and the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import precision_recall_curve

# Colorblind-safe categorical palette, assigned in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LINESTYLES = ["-", (0, (6, 2)), (0, (2, 2)), (0, (6, 2, 2, 2)), (0, (1, 1))]
LEGIT_COLOR, FRAUD_COLOR = SERIES[0], SERIES[1]
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


def apply_style() -> None:
    """Quiet axes and grid so the data carries the chart."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_MUTED,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "legend.frameon": False,
            "legend.labelcolor": TEXT,
            "lines.linewidth": 2,
            "font.size": 10,
        }
    )


def save(fig: plt.Figure, path: str | Path | None) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


def fraud_rate_by_type(df: pd.DataFrame, path=None) -> plt.Figure:
    rate = df.groupby("type", observed=True)["isFraud"].mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(rate.index.astype(str), rate.values, color=FRAUD_COLOR, height=0.6)
    for y, value in enumerate(rate.values):
        ax.text(value, y, f"  {value:.2f}%", va="center", color=TEXT, fontsize=9)
    ax.set_title("Fraud only happens in TRANSFER and CASH_OUT")
    ax.set_xlabel("Share of transactions that are fraud (%)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, rate.max() * 1.25)
    save(fig, path)
    return fig


def amount_distribution(df: pd.DataFrame, path=None) -> plt.Figure:
    bins = np.linspace(0, 8, 60)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for label, color, name in [(0, LEGIT_COLOR, "Legitimate"), (1, FRAUD_COLOR, "Fraud")]:
        amounts = np.log10(df.loc[df["isFraud"] == label, "amount"].clip(lower=1))
        ax.hist(amounts, bins=bins, density=True, histtype="step", color=color, linewidth=2, label=name)
    ax.set_title("Transaction amounts: fraud vs legitimate")
    ax.set_xlabel("Transaction amount (log10)")
    ax.set_ylabel("Density")
    ax.set_xticks(range(0, 9), [f"$10^{i}$" for i in range(0, 9)])
    ax.legend(loc="upper left")
    save(fig, path)
    return fig


def fraud_rate_by_hour(df: pd.DataFrame, path=None) -> plt.Figure:
    rate = df.groupby(df["step"] % 24)["isFraud"].mean() * 100
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(rate.index, rate.values, color=FRAUD_COLOR, marker="o", markersize=4)
    peak = rate.idxmax()
    ax.annotate(
        f"{rate[peak]:.1f}% at {peak}:00",
        (peak, rate[peak]),
        xytext=(8, -4),
        textcoords="offset points",
        color=TEXT,
        fontsize=9,
    )
    ax.set_title("Fraud rate by hour of day")
    ax.set_xlabel("Hour of day (step mod 24)")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylim(bottom=0)
    save(fig, path)
    return fig


def precision_recall_curves(y_true, probas: dict[str, np.ndarray], baseline=None, path=None) -> plt.Figure:
    """PR curve per model; ``baseline`` is an optional (recall, precision, label) point."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    # Distinct dash patterns keep curves that lie on top of each other distinguishable.
    for color, dashes, (name, proba) in zip(SERIES, LINESTYLES, probas.items()):
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ax.plot(recall, precision, color=color, linestyle=dashes, label=name)
    if baseline is not None:
        recall, precision, label = baseline
        ax.scatter([recall], [precision], s=60, color=TEXT_MUTED, zorder=3, edgecolor=SURFACE, linewidth=2)
        ax.annotate(
            label,
            (recall, precision),
            xytext=(6, -8),
            textcoords="offset points",
            color=TEXT_MUTED,
            fontsize=9,
            va="top",
        )
    ax.set_title("Precision-recall on the test set")
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    save(fig, path)
    return fig


def confusion_matrix_plot(metrics: dict, title: str, path=None) -> plt.Figure:
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    # Log scale so the small fraud cells are not washed out by the huge legit cell.
    ax.imshow(np.log10(cm + 1), cmap=BLUES)
    for (i, j), value in np.ndenumerate(cm):
        shade = np.log10(value + 1) / np.log10(cm.max() + 1)
        ax.text(
            j, i, f"{value:,}", ha="center", va="center", fontsize=12, color="white" if shade > 0.55 else TEXT
        )
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save(fig, path)
    return fig


def feature_importance(importances: pd.Series, title: str, path=None) -> plt.Figure:
    importances = importances.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(importances.index, importances.values, color=LEGIT_COLOR, height=0.6)
    ax.set_title(title)
    ax.set_xlabel("Share of total importance (gain)")
    ax.grid(axis="y", visible=False)
    save(fig, path)
    return fig
