"""Draw the README figures from the CSVs written by the analysis steps."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter

from loadData.db import OUTPUT_DIR

FIGURE_DIR = os.path.join("docs", "figures")

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
BAND_OUTER = "#cfe2f8"
BAND_INNER = "#94bfee"

SPEC_TITLES = {
    "full": "All 13 features",
    "price_only": "Price features only",
}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=10)


def save(fig, name):
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, name)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def draw_coefficients():
    """Plot Fama-MacBeth coefficients with 95% intervals, one panel per model."""
    summary = pd.read_csv(os.path.join(OUTPUT_DIR, "regression_fm_summary.csv"))
    specs = [spec for spec in SPEC_TITLES if spec in set(summary["Spec"])]
    heights = [len(summary[summary["Spec"] == spec]) for spec in specs]

    fig, axes = plt.subplots(
        len(specs), 1, figsize=(8, 0.34 * sum(heights) + 2), sharex=True,
        height_ratios=heights, facecolor=SURFACE,
    )
    for ax, spec in zip(axes, specs):
        data = summary[summary["Spec"] == spec].sort_values("MeanCoef")
        n_dates = int(data["N_Dates"].iloc[0])
        y = range(len(data))

        style_axes(ax)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1)
        ax.hlines(y, data["CI_Lower"], data["CI_Upper"], color=SERIES, linewidth=2)
        ax.scatter(data["MeanCoef"], y, s=48, color=SERIES, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.set_yticks(list(y), data["Feature"])
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(f"{SPEC_TITLES[spec]} ({n_dates} quarterly windows)", loc="left", fontsize=11, color=TEXT_PRIMARY)

    axes[-1].xaxis.set_major_locator(MultipleLocator(0.02))
    axes[-1].xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes[-1].set_xlabel("Quarterly return per 1 standard deviation of the feature", color=TEXT_SECONDARY, fontsize=10)
    excluding_zero = int(((summary["CI_Lower"] > 0) | (summary["CI_Upper"] < 0)).sum())
    fig.suptitle(
        f"{excluding_zero} of {len(summary)} coefficient intervals (95%) exclude zero",
        x=0.01, ha="left", fontsize=14, color=TEXT_PRIMARY, fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "fama_macbeth_coefficients.png")


def draw_fan_chart():
    """Plot the simulated equal-weight portfolio with its percentile bands."""
    fan = pd.read_csv(os.path.join(OUTPUT_DIR, "mc_fanchart.csv"))
    data = fan[fan["Ticker"] == "PORTFOLIO_EW"].sort_values("Day")
    last = data.iloc[-1]
    # Every path starts at 1, which the CSV doesn't include.
    start = pd.DataFrame({"Day": [0], **{f"P{p}": [1.0] for p in (5, 25, 50, 75, 95)}})
    data = pd.concat([start, data], ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style_axes(ax)
    ax.fill_between(data["Day"], data["P5"], data["P95"], color=BAND_OUTER, linewidth=0)
    ax.fill_between(data["Day"], data["P25"], data["P75"], color=BAND_INNER, linewidth=0)
    ax.plot(data["Day"], data["P50"], color=SERIES, linewidth=2)
    ax.axhline(1, color=TEXT_SECONDARY, linewidth=1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    for column, label in [("P95", "95th"), ("P75", "75th"), ("P50", "Median"), ("P25", "25th"), ("P5", "5th")]:
        ax.annotate(
            f"{label}: {last[column] - 1:+.0%}", (last["Day"], last[column]), xytext=(8, 0),
            textcoords="offset points", va="center", fontsize=10, color=TEXT_PRIMARY,
            annotation_clip=False,
        )

    ax.set_xlim(0, last["Day"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v - 1:+.0%}" if abs(v - 1) > 1e-9 else "0%"))
    ax.set_xlabel("Trading days ahead", color=TEXT_SECONDARY, fontsize=10)
    ax.set_title(
        "Simulated one-year return, equal-weight portfolio",
        loc="left", fontsize=14, color=TEXT_PRIMARY, fontweight="bold",
    )
    save(fig, "portfolio_fan_chart.png")


def make_figures():
    draw_coefficients()
    draw_fan_chart()


if __name__ == "__main__":
    make_figures()
