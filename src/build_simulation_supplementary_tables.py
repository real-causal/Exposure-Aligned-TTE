from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


METHOD_LABELS = {
    "initiation_only": "Initiation-only",
    "naive_persistence": "Naive persistent-user",
    "unweighted_cumulative": "Unweighted cumulative exposure",
    "unweighted_lagged": "Unweighted lagged exposure",
    "weighted_cumulative": "Weighted cumulative exposure",
    "lag_aware_weighted_tte": "Lag-aware weighted TTE",
}

METHOD_ORDER = {name: i for i, name in enumerate(METHOD_LABELS)}

HISTORY_LABELS = {
    "persistent_treatment": "Persistent treatment",
    "stop_after_12m": "Stopped after 12 months",
    "comparator_reference": "Comparator reference",
}

HISTORY_ORDER = {name: i for i, name in enumerate(HISTORY_LABELS)}

TIMING_SCENARIO_LABELS = {
    "monotone": "Simple discontinuation",
    "intermittent": "Intermittent use",
}

WEIGHT_SCENARIO_LABELS = {
    "primary": "Primary scenario",
    "intermittent_restart": "Intermittent restart scenario",
}

ARM_LABELS = {
    "treated": "Treated",
    "reference": "Reference",
    "all": "Overall",
}


def load_csv(base: Path, name: str) -> pd.DataFrame:
    path = base / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def add_method_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method_label"] = out["method"].map(lambda x: METHOD_LABELS.get(str(x), str(x)))
    out["method_order"] = out["method"].map(lambda x: METHOD_ORDER.get(str(x), 999))
    return out


def build_table_s1(base: Path) -> pd.DataFrame:
    monotone = add_method_labels(load_csv(base, "simulation_timing_monotone_method_metrics.csv"))
    monotone["timing_scenario"] = "Simple discontinuation"
    intermittent = add_method_labels(load_csv(base, "simulation_timing_intermittent_method_metrics.csv"))
    intermittent["timing_scenario"] = "Intermittent use"
    scenario_order = {"Simple discontinuation": 0, "Intermittent use": 1}
    out = pd.concat([monotone, intermittent], ignore_index=True)
    out = out.rename(
        columns={
            "true_rd_60m": "true_5y_risk_difference",
            "mean_estimate_60m": "mean_estimated_5y_risk_difference",
            "mc_interval_low": "monte_carlo_interval_low",
            "mc_interval_high": "monte_carlo_interval_high",
            "abs_bias": "absolute_bias",
        }
    )
    out = out[
        [
            "timing_scenario",
            "method",
            "method_label",
            "true_5y_risk_difference",
            "mean_estimated_5y_risk_difference",
            "monte_carlo_interval_low",
            "monte_carlo_interval_high",
            "bias",
            "absolute_bias",
            "rmse",
            "coverage",
            "coverage_pct",
        ]
    ]
    out["scenario_order"] = out["timing_scenario"].map(lambda x: scenario_order.get(str(x), 999))
    out["method_order"] = out["method"].map(lambda x: METHOD_ORDER.get(str(x), 999))
    out = out.sort_values(["scenario_order", "method_order"], kind="stable").drop(columns=["scenario_order", "method_order"])
    return out.reset_index(drop=True)


def build_table_s2(base: Path) -> pd.DataFrame:
    truth = load_csv(base, "simulation_timing_intermittent_truth_curves.csv").copy()
    truth["history_label"] = truth["history_name"].map(lambda x: HISTORY_LABELS.get(str(x), str(x)))
    curves = load_csv(base, "simulation_timing_intermittent_curve_summary.csv").copy()
    curves = add_method_labels(curves)
    curves["history_label"] = curves["history_name"].map(lambda x: HISTORY_LABELS.get(str(x), str(x)))
    out = curves.merge(
        truth[["month", "history_name", "true_risk"]],
        on=["month", "history_name"],
        how="left",
        validate="many_to_one",
    )
    out["timing_scenario"] = "Intermittent use"
    out = out.rename(
        columns={
            "estimated_risk_mean": "mean_estimated_risk",
            "estimated_risk_median": "median_estimated_risk",
        }
    )
    out = out[
        [
            "timing_scenario",
            "method",
            "method_label",
            "history_name",
            "history_label",
            "month",
            "true_risk",
            "mean_estimated_risk",
            "median_estimated_risk",
        ]
    ]
    out["method_order"] = out["method"].map(lambda x: METHOD_ORDER.get(str(x), 999))
    out["history_order"] = out["history_name"].map(lambda x: HISTORY_ORDER.get(str(x), 999))
    out = out.sort_values(["method_order", "history_order", "month"], kind="stable").drop(columns=["method_order", "history_order"])
    return out.reset_index(drop=True)


def build_table_s3(base: Path) -> pd.DataFrame:
    corr = load_csv(base, "simulation_timing_exposure_correlation.csv").copy()
    corr["timing_scenario"] = corr["scenario_label"].map(lambda x: TIMING_SCENARIO_LABELS.get(str(x), str(x)))
    summary = (
        corr.groupby(["scenario_label", "timing_scenario"], as_index=False)
        .agg(
            n_replicates=("replicate", "size"),
            mean_correlation=("corr_cumulative_vs_lagged", "mean"),
            sd_correlation=("corr_cumulative_vs_lagged", "std"),
            median_correlation=("corr_cumulative_vs_lagged", "median"),
            min_correlation=("corr_cumulative_vs_lagged", "min"),
            max_correlation=("corr_cumulative_vs_lagged", "max"),
        )
    )
    scenario_order = {"Simple discontinuation": 0, "Intermittent use": 1}
    out = corr.merge(summary, on=["scenario_label", "timing_scenario"], how="left", validate="many_to_one")
    out = out.rename(columns={"corr_cumulative_vs_lagged": "replicate_correlation"})
    out = out[
        [
            "timing_scenario",
            "scenario_label",
            "replicate",
            "replicate_correlation",
            "n_replicates",
            "mean_correlation",
            "sd_correlation",
            "median_correlation",
            "min_correlation",
            "max_correlation",
        ]
    ]
    out["scenario_order"] = out["timing_scenario"].map(lambda x: scenario_order.get(str(x), 999))
    out = out.sort_values(["scenario_order", "replicate"], kind="stable").drop(columns=["scenario_order"])
    return out.reset_index(drop=True)


def _harmonize_scale_sensitivity(df: pd.DataFrame, family: str, parameter_name: str) -> pd.DataFrame:
    out = df.copy()
    out["sensitivity_family"] = family
    out["parameter_name"] = parameter_name
    out["parameter_value"] = out["x_value"]
    out["bias"] = out.get("bias", out["mean_estimate_60m"] - out["true_rd_60m"])
    out["absolute_bias"] = np.abs(out["bias"])
    if "rmse" not in out.columns:
        out["rmse"] = np.nan
    out["delay_months"] = np.nan
    out["max_lag_months"] = np.nan
    return out[
        [
            "sensitivity_family",
            "parameter_name",
            "parameter_value",
            "true_rd_60m",
            "mean_estimate_60m",
            "mc_interval_low",
            "mc_interval_high",
            "bias",
            "absolute_bias",
            "rmse",
            "delay_months",
            "max_lag_months",
        ]
    ]


def build_table_s4(base: Path) -> pd.DataFrame:
    accumulation = _harmonize_scale_sensitivity(
        load_csv(base, "simulation_sensitivity_accumulation_scale.csv"),
        family="Accumulation scale",
        parameter_name="Assumed accumulation scale",
    )
    lag = _harmonize_scale_sensitivity(
        load_csv(base, "simulation_sensitivity_lag_misspecification.csv"),
        family="Lag misspecification",
        parameter_name="Assumed lag months",
    )
    shape = load_csv(base, "simulation_sensitivity_kernel_shape_misspecification.csv").copy()
    shape["sensitivity_family"] = "Kernel shape"
    shape["parameter_name"] = "Kernel shape"
    shape["parameter_value"] = shape["shape_label"]
    shape["absolute_bias"] = np.abs(shape["bias"])
    shape = shape[
        [
            "sensitivity_family",
            "parameter_name",
            "parameter_value",
            "true_rd_60m",
            "mean_estimate_60m",
            "mc_interval_low",
            "mc_interval_high",
            "bias",
            "absolute_bias",
            "rmse",
            "delay_months",
            "max_lag_months",
        ]
    ]
    out = pd.concat([accumulation, lag, shape], ignore_index=True)
    out = out.rename(
        columns={
            "true_rd_60m": "true_5y_risk_difference",
            "mean_estimate_60m": "mean_estimated_5y_risk_difference",
            "mc_interval_low": "monte_carlo_interval_low",
            "mc_interval_high": "monte_carlo_interval_high",
        }
    )
    family_order = {"Accumulation scale": 0, "Lag misspecification": 1, "Kernel shape": 2}
    out["family_order"] = out["sensitivity_family"].map(lambda x: family_order.get(str(x), 999))
    out = out.sort_values(["family_order", "parameter_value"], kind="stable").drop(columns=["family_order"])
    return out.reset_index(drop=True)


def _harmonize_method_metrics(df: pd.DataFrame, scenario_label: str) -> pd.DataFrame:
    out = add_method_labels(df)
    out["panel"] = "Decomposition performance"
    out["analysis_context"] = scenario_label
    out = out.rename(
        columns={
            "true_rd_60m": "true_5y_risk_difference",
            "mean_estimate_60m": "mean_estimated_5y_risk_difference",
            "mc_interval_low": "monte_carlo_interval_low",
            "mc_interval_high": "monte_carlo_interval_high",
            "abs_bias": "absolute_bias",
        }
    )
    out["arm"] = pd.NA
    out["arm_label"] = pd.NA
    out["n_rows"] = pd.NA
    out["mean_weight"] = pd.NA
    out["median_weight"] = pd.NA
    out["p90_weight"] = pd.NA
    out["p95_weight"] = pd.NA
    out["p99_weight"] = pd.NA
    out["max_weight"] = pd.NA
    return out[
        [
            "panel",
            "analysis_context",
            "method",
            "method_label",
            "arm",
            "arm_label",
            "n_rows",
            "true_5y_risk_difference",
            "mean_estimated_5y_risk_difference",
            "monte_carlo_interval_low",
            "monte_carlo_interval_high",
            "bias",
            "absolute_bias",
            "rmse",
            "coverage",
            "coverage_pct",
            "mean_weight",
            "median_weight",
            "p90_weight",
            "p95_weight",
            "p99_weight",
            "max_weight",
        ]
    ]


def _harmonize_weight_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["panel"] = "Weight diagnostics"
    out["analysis_context"] = out["scenario_name"].map(lambda x: WEIGHT_SCENARIO_LABELS.get(str(x), str(x)))
    out["method"] = pd.NA
    out["method_label"] = pd.NA
    out["arm_label"] = out["arm"].map(lambda x: ARM_LABELS.get(str(x), str(x)))
    out["true_5y_risk_difference"] = pd.NA
    out["mean_estimated_5y_risk_difference"] = pd.NA
    out["monte_carlo_interval_low"] = pd.NA
    out["monte_carlo_interval_high"] = pd.NA
    out["bias"] = pd.NA
    out["absolute_bias"] = pd.NA
    out["rmse"] = pd.NA
    out["coverage"] = pd.NA
    out["coverage_pct"] = pd.NA
    return out[
        [
            "panel",
            "analysis_context",
            "method",
            "method_label",
            "arm",
            "arm_label",
            "n_rows",
            "true_5y_risk_difference",
            "mean_estimated_5y_risk_difference",
            "monte_carlo_interval_low",
            "monte_carlo_interval_high",
            "bias",
            "absolute_bias",
            "rmse",
            "coverage",
            "coverage_pct",
            "mean_weight",
            "median_weight",
            "p90_weight",
            "p95_weight",
            "p99_weight",
            "max_weight",
        ]
    ]


def build_table_s5(base: Path) -> pd.DataFrame:
    primary_methods = _harmonize_method_metrics(
        load_csv(base, "simulation_method_summary_metrics.csv"),
        scenario_label="Primary decomposition scenario",
    )
    intermittent_methods = _harmonize_method_metrics(
        load_csv(base, "simulation_sensitivity_intermittent_method_metrics.csv"),
        scenario_label="Intermittent restart decomposition scenario",
    )
    primary_weights = _harmonize_weight_metrics(load_csv(base, "simulation_weight_summary_primary.csv"))
    intermittent_weights = _harmonize_weight_metrics(load_csv(base, "simulation_weight_summary_intermittent.csv"))
    out = pd.concat([primary_methods, intermittent_methods, primary_weights, intermittent_weights], ignore_index=True)
    out["panel_order"] = out["panel"].map({"Decomposition performance": 0, "Weight diagnostics": 1})
    context_order = {
        "Primary decomposition scenario": 0,
        "Intermittent restart decomposition scenario": 1,
        "Primary scenario": 2,
        "Intermittent restart scenario": 3,
    }
    out["context_order"] = out["analysis_context"].map(lambda x: context_order.get(str(x), 999))
    out["method_order"] = out["method"].map(lambda x: METHOD_ORDER.get(str(x), 999) if pd.notna(x) else 999)
    arm_order = {"treated": 0, "reference": 1, "all": 2}
    out["arm_order"] = out["arm"].map(lambda x: arm_order.get(str(x), 999) if pd.notna(x) else 999)
    out = out.sort_values(["panel_order", "context_order", "method_order", "arm_order"], kind="stable").drop(
        columns=["panel_order", "context_order", "method_order", "arm_order"]
    )
    return out.reset_index(drop=True)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build simulation supplementary tables S1-S5 from manuscript CSV outputs.")
    parser.add_argument(
        "--base-dir",
        default=str(REPO_ROOT / "results" / "final_200rep" / "source_csv"),
        help="Directory containing source simulation CSV files.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for supplementary tables; defaults to results/final_200rep/tables.",
    )
    args = parser.parse_args()

    base = Path(args.base_dir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else REPO_ROOT / "results" / "final_200rep" / "tables"
    outdir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "tableS1_simulation_method_performance_timing_scenarios.csv": build_table_s1(base),
        "tableS2_simulation_truth_and_estimated_representative_risk_curves.csv": build_table_s2(base),
        "tableS3_simulation_correlation_cumulative_vs_exposure_history_burden.csv": build_table_s3(base),
        "tableS4_simulation_weighting_function_and_accumulation_scale_sensitivity.csv": build_table_s4(base),
        "tableS5_simulation_decomposition_and_weight_diagnostics.csv": build_table_s5(base),
    }

    manifest = {"base_dir": _display_path(base), "outdir": _display_path(outdir), "generated": []}
    for filename, df in outputs.items():
        path = outdir / filename
        df.to_csv(path, index=False)
        manifest["generated"].append({"file": filename, "rows": int(len(df)), "columns": df.columns.tolist()})

    (outdir / "supplementary_tables_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
