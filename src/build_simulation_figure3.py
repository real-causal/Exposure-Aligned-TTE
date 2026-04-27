from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exposure_aligned_tte.plotting import despine, panel_label, save_figure
from exposure_aligned_tte.simulation_manuscript import (
    KERNEL_SHAPE_LABEL,
    METHOD_COLOR,
    METHOD_SHORT_LABEL,
    SUPPLEMENT_METHOD_ORDER,
    simulation_style,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_csv(base: Path, name: str) -> pd.DataFrame:
    path = base / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def make_figure3(base_dir: Path, outdir: Path) -> None:
    accumulation = load_csv(base_dir, "simulation_sensitivity_accumulation_scale.csv").sort_values("x_value").reset_index(drop=True)
    kernel_shape = load_csv(base_dir, "simulation_sensitivity_kernel_shape_misspecification.csv").reset_index(drop=True)
    primary = load_csv(base_dir, "simulation_method_summary_metrics.csv")
    intermittent = load_csv(base_dir, "simulation_sensitivity_intermittent_method_metrics.csv")
    primary_w = load_csv(base_dir, "simulation_weight_summary_primary.csv")
    intermittent_w = load_csv(base_dir, "simulation_weight_summary_intermittent.csv")

    method_order = [m for m in SUPPLEMENT_METHOD_ORDER if m in set(primary["method"]) and m in set(intermittent["method"])]
    primary = primary.set_index("method").loc[method_order].reset_index()
    intermittent = intermittent.set_index("method").loc[method_order].reset_index()

    treated_primary = primary_w[primary_w["arm"] == "treated"].copy()
    treated_primary["scenario"] = "Primary"
    treated_intermittent = intermittent_w[intermittent_w["arm"] == "treated"].copy()
    treated_intermittent["scenario"] = "Intermittent"
    weight_df = pd.concat([treated_primary, treated_intermittent], ignore_index=True)

    with simulation_style():
        fig = plt.figure(figsize=(10.8, 8.2))
        gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.26, left=0.08, right=0.98, top=0.95, bottom=0.08)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])

        y_min = 100.0 * min(accumulation["mc_interval_low"].min(), accumulation["true_rd_60m"].min())
        y_max = 100.0 * max(accumulation["mc_interval_high"].max(), accumulation["true_rd_60m"].max())
        y_pad = max(1.0, 0.08 * (y_max - y_min))
        ax_a.plot(
            accumulation["x_value"],
            100.0 * accumulation["true_rd_60m"],
            color="#2C2C2A",
            linestyle="--",
            linewidth=1.2,
            marker="",
            label="True 5-year RD",
        )
        for _, row in accumulation.iterrows():
            x = float(row["x_value"])
            ax_a.plot(
                [x, x],
                [100.0 * float(row["mc_interval_low"]), 100.0 * float(row["mc_interval_high"])],
                color=METHOD_COLOR["lag_aware_weighted_tte"],
                linewidth=1.5,
            )
            ax_a.plot(
                x,
                100.0 * float(row["mean_estimate_60m"]),
                marker="D",
                color=METHOD_COLOR["lag_aware_weighted_tte"],
                ms=5.6,
                zorder=3,
                lw=0,
            )
        ax_a.set_xticks(accumulation["x_value"])
        ax_a.set_xticklabels([f"\u03bb={int(v)}" for v in accumulation["x_value"]])
        ax_a.set_xlabel("Assumed accumulation scale")
        ax_a.set_ylabel("5-year risk difference (%)")
        ax_a.set_title("Accumulation-scale sensitivity", pad=4)
        ax_a.set_ylim(y_min - y_pad, y_max + y_pad)
        ax_a.legend(loc="upper right", frameon=False)
        despine(ax_a)
        panel_label(ax_a, "A")

        x_shape = np.arange(len(kernel_shape), dtype=float)
        truth_line = float(kernel_shape["true_rd_60m"].iloc[0])
        shape_y_min = 100.0 * min(kernel_shape["mc_interval_low"].min(), truth_line)
        shape_y_max = 100.0 * max(kernel_shape["mc_interval_high"].max(), truth_line)
        shape_y_pad = max(1.0, 0.08 * (shape_y_max - shape_y_min))
        ax_b.axhline(100.0 * truth_line, color="#2C2C2A", linestyle="--", linewidth=1.2, label="True 5-year RD")
        for idx, row in kernel_shape.iterrows():
            marker = "D" if str(row["shape_name"]) == "soft_accumulation" else "o"
            ms = 5.6 if marker == "D" else 5.0
            ax_b.plot(
                [x_shape[idx], x_shape[idx]],
                [100.0 * float(row["mc_interval_low"]), 100.0 * float(row["mc_interval_high"])],
                color=METHOD_COLOR["lag_aware_weighted_tte"],
                linewidth=1.5,
            )
            ax_b.plot(
                x_shape[idx],
                100.0 * float(row["mean_estimate_60m"]),
                marker=marker,
                color=METHOD_COLOR["lag_aware_weighted_tte"],
                ms=ms,
                zorder=3,
                lw=0,
            )
        shape_labels = [KERNEL_SHAPE_LABEL.get(str(s), str(s)) for s in kernel_shape["shape_name"]]
        ax_b.set_xticks(x_shape)
        ax_b.set_xticklabels(shape_labels)
        ax_b.set_xlabel("Assumed exposure-history weighting shape")
        ax_b.set_ylabel("5-year risk difference (%)")
        ax_b.set_title("Exposure-history weighting-shape sensitivity", pad=4)
        ax_b.set_ylim(shape_y_min - shape_y_pad, shape_y_max + shape_y_pad)
        ax_b.legend(loc="upper right", frameon=False)
        despine(ax_b)
        panel_label(ax_b, "B")

        y_pos = np.arange(len(method_order))[::-1].astype(float)
        y_primary = y_pos + 0.14
        y_intermittent = y_pos - 0.14
        truth_primary = 100.0 * float(primary["true_rd_60m"].iloc[0])
        truth_intermittent = 100.0 * float(intermittent["true_rd_60m"].iloc[0])
        x_vals = np.concatenate(
            [
                100.0 * primary["mc_interval_low"].to_numpy(dtype=float),
                100.0 * primary["mc_interval_high"].to_numpy(dtype=float),
                100.0 * intermittent["mc_interval_low"].to_numpy(dtype=float),
                100.0 * intermittent["mc_interval_high"].to_numpy(dtype=float),
                np.array([truth_primary, truth_intermittent]),
            ]
        )
        x_pad = max(1.0, 0.08 * (float(np.nanmax(x_vals)) - float(np.nanmin(x_vals))))
        primary_color = METHOD_COLOR["lag_aware_weighted_tte"]
        intermittent_color = "#D95F5F"
        for y, row in zip(y_primary, primary.itertuples(index=False)):
            lo = 100.0 * float(row.mc_interval_low)
            hi = 100.0 * float(row.mc_interval_high)
            est = 100.0 * float(row.mean_estimate_60m)
            ax_c.plot([lo, hi], [y, y], color=primary_color, lw=1.4, solid_capstyle="round", alpha=0.9)
            ax_c.plot(est, y, marker="D", color=primary_color, ms=5.2, zorder=3, lw=0)
        for y, row in zip(y_intermittent, intermittent.itertuples(index=False)):
            lo = 100.0 * float(row.mc_interval_low)
            hi = 100.0 * float(row.mc_interval_high)
            est = 100.0 * float(row.mean_estimate_60m)
            ax_c.plot([lo, hi], [y, y], color=intermittent_color, lw=1.4, solid_capstyle="round", alpha=0.9)
            ax_c.plot(est, y, marker="o", color=intermittent_color, ms=5.0, zorder=3, lw=0)
        ax_c.axvline(truth_primary, color=primary_color, lw=1.1, ls="--", alpha=0.95, label="Primary truth")
        ax_c.axvline(truth_intermittent, color=intermittent_color, lw=1.1, ls="--", alpha=0.95, label="Intermittent truth")
        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels([METHOD_SHORT_LABEL[m] for m in method_order])
        ax_c.tick_params(axis="y", length=0)
        ax_c.set_xlim(float(np.nanmin(x_vals)) - x_pad, float(np.nanmax(x_vals)) + x_pad)
        ax_c.set_xlabel("5-year risk difference (%)")
        ax_c.set_title("Decomposition analysis across simulation settings", pad=4)
        ax_c.legend(loc="lower right", frameon=False)
        ax_c.xaxis.grid(True)
        ax_c.yaxis.grid(False)
        despine(ax_c)
        panel_label(ax_c, "C")

        metrics = [("p99_weight", "99th percentile"), ("max_weight", "Maximum")]
        scenario_order = ["Primary", "Intermittent"]
        scenario_colors = {"Primary": primary_color, "Intermittent": intermittent_color}
        base_positions = np.arange(len(metrics), dtype=float)
        offsets = {"Primary": -0.13, "Intermittent": 0.13}
        for metric_idx, (metric_col, metric_label) in enumerate(metrics):
            for scenario_name in scenario_order:
                sub = weight_df[weight_df["scenario"] == scenario_name]
                vals = sub[metric_col].to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                center = base_positions[metric_idx] + offsets[scenario_name]
                jitter = np.linspace(-0.04, 0.04, vals.size) if vals.size > 1 else np.array([0.0])
                ax_d.scatter(
                    np.full(vals.size, center) + jitter,
                    vals,
                    s=20,
                    alpha=0.65,
                    color=scenario_colors[scenario_name],
                    edgecolor="none",
                    zorder=2,
                )
                ax_d.plot(
                    center,
                    float(np.mean(vals)),
                    marker="D",
                    ms=5.0,
                    color="#222222",
                    zorder=3,
                    lw=0,
                )
        ax_d.set_xticks(base_positions)
        ax_d.set_xticklabels([label for _, label in metrics])
        ax_d.set_ylabel("Weight value")
        ax_d.set_title("Simulation weight diagnostics (treated arm)", pad=4)
        ax_d.set_ylim(0.0, max(21.0, float(np.nanmax(weight_df[["p99_weight", "max_weight"]].to_numpy(dtype=float))) * 1.08))
        legend_handles = [
            mpl.lines.Line2D([], [], linestyle="", marker="o", color=scenario_colors["Primary"], label="Primary scenario"),
            mpl.lines.Line2D([], [], linestyle="", marker="o", color=scenario_colors["Intermittent"], label="Intermittent scenario"),
            mpl.lines.Line2D([], [], linestyle="", marker="D", color="#222222", label="Replicate mean"),
        ]
        ax_d.legend(handles=legend_handles, loc="upper left", frameon=False)
        ax_d.yaxis.grid(True)
        ax_d.xaxis.grid(False)
        despine(ax_d)
        panel_label(ax_d, "D")

        save_figure(fig, outdir, "figure3_simulation_sensitivity_composite")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Figure 3 multi-panel simulation sensitivity figure from existing simulation outputs.")
    parser.add_argument(
        "--base-dir",
        default=str(REPO_ROOT / "results" / "final_200rep" / "source_csv"),
        help="Directory containing source simulation CSV files.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for the figure; defaults to results/final_200rep/figures.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else REPO_ROOT / "results" / "final_200rep" / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    make_figure3(base_dir=base_dir, outdir=outdir)


if __name__ == "__main__":
    main()
