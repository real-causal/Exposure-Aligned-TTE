from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results" / "final_200rep"
SOURCE_CSV_ROOT = RESULTS_ROOT / "source_csv"
TABLE_ROOT = RESULTS_ROOT / "tables"
FIGURE_ROOT = RESULTS_ROOT / "figures"


SIMULATION_SOURCE_FILES = {
    "method_summary": "simulation_method_summary_metrics.csv",
    "intermittent_method_summary": "simulation_sensitivity_intermittent_method_metrics.csv",
    "timing_monotone_metrics": "simulation_timing_monotone_method_metrics.csv",
    "timing_intermittent_metrics": "simulation_timing_intermittent_method_metrics.csv",
    "timing_truth_curves": "simulation_timing_intermittent_truth_curves.csv",
    "timing_curve_summary": "simulation_timing_intermittent_curve_summary.csv",
    "timing_exposure_correlation": "simulation_timing_exposure_correlation.csv",
    "timing_example_histories": "simulation_timing_example_histories.csv",
    "accumulation_scale_sensitivity": "simulation_sensitivity_accumulation_scale.csv",
    "lag_misspecification_sensitivity": "simulation_sensitivity_lag_misspecification.csv",
    "kernel_shape_sensitivity": "simulation_sensitivity_kernel_shape_misspecification.csv",
    "weight_summary_primary": "simulation_weight_summary_primary.csv",
    "weight_summary_intermittent": "simulation_weight_summary_intermittent.csv",
}


SUPPLEMENTARY_TABLE_FILES = {
    "table_s1": "tableS1_simulation_method_performance_timing_scenarios.csv",
    "table_s2": "tableS2_simulation_truth_and_estimated_representative_risk_curves.csv",
    "table_s3": "tableS3_simulation_correlation_cumulative_vs_exposure_history_burden.csv",
    "table_s4": "tableS4_simulation_weighting_function_and_accumulation_scale_sensitivity.csv",
    "table_s5": "tableS5_simulation_decomposition_and_weight_diagnostics.csv",
}


FIGURE_FILES = {
    "figure1_png": "figure1_timing_sensitive_exposure_histories.png",
    "figure1_svg": "figure1_timing_sensitive_exposure_histories.svg",
    "figure2_png": "figure2_composite_timing_aware_framework.png",
    "figure2_svg": "figure2_composite_timing_aware_framework.svg",
    "figure3_png": "figure3_simulation_sensitivity_composite.png",
    "figure3_svg": "figure3_simulation_sensitivity_composite.svg",
}


def simulation_source_path(key: str) -> Path:
    return SOURCE_CSV_ROOT / SIMULATION_SOURCE_FILES[key]


def supplementary_table_path(key: str) -> Path:
    return TABLE_ROOT / SUPPLEMENTARY_TABLE_FILES[key]


def figure_path(key: str) -> Path:
    return FIGURE_ROOT / FIGURE_FILES[key]


def load_simulation_source(key: str) -> pd.DataFrame:
    return pd.read_csv(simulation_source_path(key))


def load_supplementary_table(key: str) -> pd.DataFrame:
    return pd.read_csv(supplementary_table_path(key))
