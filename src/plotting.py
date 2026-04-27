from __future__ import annotations

from pathlib import Path

import matplotlib as mpl


PALETTE = {
    "target": "#C0392B",
    "comparator": "#2471A3",
    "persistent_glp1": "#1A5276",
    "glp1_stop_after_12m": "#E67E22",
    "continued_comparator": "#27AE60",
    "before": "#4C78A8",
    "after": "#F28E2B",
    "kernel": "#2E4057",
}


def panel_label(ax: mpl.axes.Axes, letter: str, *, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def despine(ax: mpl.axes.Axes, *, left: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)


def save_figure(fig: mpl.figure.Figure, outdir: Path, stem: str) -> None:
    fig.savefig(outdir / f"{stem}.png", dpi=300)
    fig.savefig(outdir / f"{stem}.svg")


def _boxed_text(
    ax: mpl.axes.Axes,
    x: float,
    y: float,
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = "#666666",
    fontsize: float = 7.5,
    ha: str = "center",
    va: str = "center",
) -> None:
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": facecolor,
            "edgecolor": edgecolor,
            "linewidth": 0.8,
        },
        transform=ax.transAxes,
    )


def _arrow(ax: mpl.axes.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={"arrowstyle": "->", "color": "#666666", "linewidth": 1.0},
    )
