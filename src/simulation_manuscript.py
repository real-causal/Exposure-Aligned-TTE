from __future__ import annotations

import argparse
import json
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from exposure_aligned_tte.plotting import (
    PALETTE,
    _arrow,
    _boxed_text,
    despine,
    panel_label,
    save_figure,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


METHOD_ORDER = [
    "initiation_only",
    "naive_persistence",
    "unweighted_cumulative",
    "lag_aware_weighted_tte",
]

SUPPLEMENT_METHOD_ORDER = [
    "initiation_only",
    "naive_persistence",
    "unweighted_cumulative",
    "unweighted_lagged",
    "weighted_cumulative",
    "lag_aware_weighted_tte",
]


METHOD_LABEL = {
    "initiation_only": "Initiation-only\nITT-like TTE",
    "naive_persistence": "Naive persistent-user\ncomparison",
    "unweighted_cumulative": "Unweighted cumulative\nexposure model",
    "unweighted_lagged": "Unweighted lagged\nexposure model",
    "weighted_cumulative": "Weighted cumulative\nexposure model",
    "lag_aware_weighted_tte": "Lag-aware weighted\nTTE",
}


METHOD_SHORT_LABEL = {
    "initiation_only": "ITT-like",
    "naive_persistence": "Naive pers.",
    "unweighted_cumulative": "Unweighted\ncumulative",
    "unweighted_lagged": "Unweighted\nlagged",
    "weighted_cumulative": "Weighted\ncumulative",
    "lag_aware_weighted_tte": "Lag-aware\nweighted",
}


METHOD_COLOR = {
    "initiation_only": "#9AA0A6",
    "naive_persistence": "#D95F5F",
    "unweighted_cumulative": "#F2A65A",
    "unweighted_lagged": "#D28A2F",
    "weighted_cumulative": "#8EC1EA",
    "lag_aware_weighted_tte": "#0072B2",
}


HISTORY_ORDER = ["persistent_treatment", "stop12", "comparator_reference"]


HISTORY_LABEL = {
    "persistent_treatment": "Persistent treatment",
    "stop12": "Stop after 12 months",
    "comparator_reference": "Comparator reference",
}


HISTORY_COLOR = {
    "persistent_treatment": PALETTE["persistent_glp1"],
    "stop12": PALETTE["glp1_stop_after_12m"],
    "comparator_reference": PALETTE["continued_comparator"],
}


KERNEL_SHAPE_LABEL = {
    "soft_accumulation": "True soft\naccumulation",
    "hard_threshold": "Hard threshold\nafter 12 mo",
    "front_loaded": "Front-\nloaded",
}


SIMULATION_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.35,
    "grid.color": "#E0E0E0",
    "grid.alpha": 1.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.dpi": 150,
}


@contextmanager
def simulation_style():
    with mpl.rc_context(SIMULATION_RC):
        yield


@dataclass(frozen=True)
class KernelSpec:
    baseline_weight: float = 0.20
    accumulation_scale: float = 10.0
    max_lag_months: int = 36
    shape: str = "soft_accumulation"
    delay_months: int = 12


@dataclass(frozen=True)
class Scenario:
    months: int = 60
    baseline_tx_intercept: float = 0.10
    baseline_tx_x: float = 0.20
    baseline_l_x: float = 0.90
    baseline_l_sd: float = 0.80
    continuation_intercept: float = 2.65
    continuation_x: float = 0.05
    continuation_l: float = -1.35
    continuation_time: float = 0.04
    continuation_z: float = 0.00
    allow_restart: bool = False
    restart_intercept: float = -3.40
    restart_x: float = 0.05
    restart_l: float = -0.35
    restart_z: float = 0.00
    outcome_intercept: float = -8.18
    outcome_time: float = 0.024
    outcome_x: float = 0.28
    outcome_l: float = 1.25
    outcome_h: float = -0.22
    clinical_ar: float = 0.80
    clinical_x: float = 0.28
    clinical_e: float = -0.04
    clinical_h: float = -0.08
    clinical_sd: float = 0.55


def expit(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=float)
    out = np.empty_like(x_arr, dtype=float)
    pos = x_arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    if np.isscalar(x):
        return float(out)
    return out


def build_kernel(spec: KernelSpec) -> np.ndarray:
    raw_weights = np.zeros(spec.max_lag_months + 1, dtype=float)
    for lag in range(spec.max_lag_months + 1):
        if spec.shape == "soft_accumulation":
            weight = spec.baseline_weight + (1.0 - spec.baseline_weight) * (
                1.0 - np.exp(-lag / max(spec.accumulation_scale, 1e-6))
            )
        elif spec.shape == "hard_threshold":
            weight = 0.0 if lag < spec.delay_months else 1.0
        elif spec.shape == "front_loaded":
            weight = spec.baseline_weight + (1.0 - spec.baseline_weight) * np.exp(
                -lag / max(spec.accumulation_scale, 1e-6)
            )
        else:
            raise ValueError(f"Unsupported kernel shape: {spec.shape}")
        raw_weights[lag] = weight
    denom = float(raw_weights.sum())
    if denom <= 0:
        raise ValueError("Kernel weights must sum to a positive value.")
    return raw_weights / denom


def kernel_table(spec: KernelSpec) -> pd.DataFrame:
    kernel = build_kernel(spec)
    raw_weights = np.zeros(spec.max_lag_months + 1, dtype=float)
    for lag in range(spec.max_lag_months + 1):
        if spec.shape == "soft_accumulation":
            raw_weights[lag] = spec.baseline_weight + (1.0 - spec.baseline_weight) * (
                1.0 - np.exp(-lag / max(spec.accumulation_scale, 1e-6))
            )
        elif spec.shape == "hard_threshold":
            raw_weights[lag] = 0.0 if lag < spec.delay_months else 1.0
        elif spec.shape == "front_loaded":
            raw_weights[lag] = spec.baseline_weight + (1.0 - spec.baseline_weight) * np.exp(
                -lag / max(spec.accumulation_scale, 1e-6)
            )
    kernel_scale = raw_weights / max(float(np.nanmax(raw_weights)), 1e-8)
    return pd.DataFrame(
        {
            "lag_month": np.arange(len(kernel), dtype=int),
            "weight": kernel,
            "relative_contribution": kernel_scale,
            "baseline_weight": spec.baseline_weight,
            "accumulation_scale": spec.accumulation_scale,
            "delay_months": spec.delay_months,
            "max_lag_months": spec.max_lag_months,
            "shape": spec.shape,
            "shape_label": KERNEL_SHAPE_LABEL.get(spec.shape, spec.shape),
        }
    )


def history_matrix(history_name: str, n: int, months: int) -> np.ndarray:
    e = np.zeros((n, months), dtype=float)
    if history_name == "persistent_treatment":
        e[:, :] = 1.0
    elif history_name == "stop12":
        e[:, :12] = 1.0
    elif history_name == "comparator_reference":
        pass
    else:
        raise ValueError(f"Unknown history_name={history_name}")
    return e


def history_assignment(history_name: str, n: int, months: int) -> tuple[np.ndarray, np.ndarray]:
    if history_name == "persistent_treatment":
        z = np.ones(n, dtype=float)
        e = np.ones((n, months), dtype=float)
    elif history_name == "stop12":
        z = np.ones(n, dtype=float)
        e = np.zeros((n, months), dtype=float)
        e[:, :12] = 1.0
    elif history_name == "comparator_reference":
        z = np.zeros(n, dtype=float)
        e = np.zeros((n, months), dtype=float)
    else:
        raise ValueError(f"Unknown history_name={history_name}")
    return z, e


def lagged_summary(exposure: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    n, months = exposure.shape
    out = np.zeros((n, months), dtype=float)
    max_lag = len(kernel) - 1
    for month in range(months):
        for lag in range(min(max_lag, month) + 1):
            out[:, month] += kernel[lag] * exposure[:, month - lag]
    return out


def cumulative_exposure(exposure: np.ndarray) -> np.ndarray:
    denom = np.arange(1, exposure.shape[1] + 1, dtype=float)
    return np.cumsum(exposure, axis=1) / denom


def simulate_regime_truth(
    n: int,
    scenario: Scenario,
    history_name: str,
    kernel_true: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z_hist, exposure = history_assignment(history_name, n, scenario.months)
    l_state = scenario.baseline_l_x * x + rng.normal(scale=scenario.baseline_l_sd, size=n)
    h_mat = lagged_summary(exposure, kernel_true)
    event = np.zeros((n, scenario.months), dtype=int)
    alive = np.ones(n, dtype=bool)

    for month in range(scenario.months):
        lp = (
            scenario.outcome_intercept
            + scenario.outcome_time * month
            + scenario.outcome_x * x
            + scenario.outcome_l * l_state
            + scenario.outcome_h * h_mat[:, month]
        )
        p_event = expit(lp)
        draw = rng.binomial(1, p_event, size=n)
        draw = np.where(alive, draw, 0)
        event[:, month] = draw
        alive &= draw == 0
        l_state = np.where(
            alive,
            scenario.clinical_ar * l_state
            + scenario.clinical_x * x
            + scenario.clinical_e * exposure[:, month]
            + scenario.clinical_h * h_mat[:, month]
            + rng.normal(scale=scenario.clinical_sd, size=n),
            l_state,
        )

    cumulative_risk = (np.cumsum(event, axis=1) > 0).mean(axis=0)
    return pd.DataFrame(
        {
            "month": np.arange(1, scenario.months + 1, dtype=int),
            "history_name": history_name,
            "true_risk": cumulative_risk,
        }
    )


def simulate_observed_panel(
    n: int,
    scenario: Scenario,
    kernel_true: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.binomial(1, expit(scenario.baseline_tx_intercept + scenario.baseline_tx_x * x), size=n)
    l_state = scenario.baseline_l_x * x + rng.normal(scale=scenario.baseline_l_sd, size=n)
    exposure = np.zeros((n, scenario.months), dtype=int)
    clinical = np.zeros((n, scenario.months), dtype=float)
    event = np.zeros((n, scenario.months), dtype=int)
    alive = np.ones(n, dtype=bool)
    max_lag = len(kernel_true) - 1

    for month in range(scenario.months):
        clinical[:, month] = l_state
        if month == 0:
            exposure[:, 0] = z
        else:
            prev_e = exposure[:, month - 1]
            lp_continue = (
                scenario.continuation_intercept
                + scenario.continuation_x * x
                + scenario.continuation_l * l_state
                + scenario.continuation_time * (month / 12.0)
                + scenario.continuation_z * z
            )
            p_continue = expit(lp_continue)
            continued = rng.binomial(1, p_continue, size=n)
            if scenario.allow_restart:
                lp_restart = (
                    scenario.restart_intercept
                    + scenario.restart_x * x
                    + scenario.restart_l * l_state
                    + scenario.restart_z * z
                )
                p_restart = expit(lp_restart)
                restarted = rng.binomial(1, p_restart, size=n)
                exposure[:, month] = np.where(prev_e == 1, continued, restarted * z)
            else:
                exposure[:, month] = np.where(prev_e == 1, continued, 0)

        h_now = np.zeros(n, dtype=float)
        for lag in range(min(max_lag, month) + 1):
            h_now += kernel_true[lag] * exposure[:, month - lag]
        lp_event = (
            scenario.outcome_intercept
            + scenario.outcome_time * month
            + scenario.outcome_x * x
            + scenario.outcome_l * l_state
            + scenario.outcome_h * h_now
        )
        p_event = expit(lp_event)
        draw = rng.binomial(1, p_event, size=n)
        draw = np.where(alive, draw, 0)
        event[:, month] = draw
        alive &= draw == 0
        l_state = np.where(
            alive,
            scenario.clinical_ar * l_state
            + scenario.clinical_x * x
            + scenario.clinical_e * exposure[:, month]
            + scenario.clinical_h * h_now
            + rng.normal(scale=scenario.clinical_sd, size=n),
            l_state,
        )

    lagged_true = lagged_summary(exposure.astype(float), kernel_true)
    cumulative = cumulative_exposure(exposure.astype(float))
    prev_exposure = np.concatenate([z[:, None], exposure[:, :-1]], axis=1)
    event_any = event.any(axis=1)
    event_month = np.where(event_any, event.argmax(axis=1), scenario.months - 1)
    observed_mask = np.arange(scenario.months)[None, :] <= event_month[:, None]

    id_long = np.repeat(np.arange(n, dtype=int), scenario.months)[observed_mask.ravel()]
    month_index = np.tile(np.arange(scenario.months, dtype=int), n)[observed_mask.ravel()]
    month = month_index + 1
    panel = pd.DataFrame(
        {
            "id": id_long,
            "month_index": month_index,
            "month": month,
            "month_c": month_index / 12.0,
            "month_sq": (month_index / 12.0) ** 2,
            "x": np.repeat(x, scenario.months)[observed_mask.ravel()],
            "z": np.repeat(z, scenario.months)[observed_mask.ravel()],
            "l_t": clinical.ravel()[observed_mask.ravel()],
            "e_t": exposure.ravel()[observed_mask.ravel()],
            "h_true": lagged_true.ravel()[observed_mask.ravel()],
            "cumulative_e": cumulative.ravel()[observed_mask.ravel()],
            "event": event.ravel()[observed_mask.ravel()],
            "prev_e": prev_exposure.ravel()[observed_mask.ravel()],
        }
    )
    return panel


def build_prediction_panel(
    x_values: np.ndarray,
    months: int,
    history_name: str,
    kernel_model: np.ndarray,
) -> pd.DataFrame:
    n = len(x_values)
    z_hist, exposure = history_assignment(history_name, n, months)
    lagged_model = lagged_summary(exposure, kernel_model)
    cumulative = cumulative_exposure(exposure)
    month_index = np.tile(np.arange(months, dtype=int), n)
    history_e = exposure.ravel()
    return pd.DataFrame(
        {
            "id": np.repeat(np.arange(n, dtype=int), months),
            "month_index": month_index,
            "month": month_index + 1,
            "month_c": month_index / 12.0,
            "month_sq": (month_index / 12.0) ** 2,
            "x": np.repeat(x_values, months),
            "z": np.repeat(z_hist, months),
            "e_t": history_e,
            "lagged_h": lagged_model.ravel(),
            "cumulative_e": cumulative.ravel(),
        }
    )


def _fit_glm(
    panel: pd.DataFrame,
    columns: list[str],
    *,
    weight_col: str | None = None,
) -> sm.GLM:
    exog = sm.add_constant(panel[columns], has_constant="add")
    weights = panel[weight_col] if weight_col is not None else None
    model = sm.GLM(panel["event"], exog, family=sm.families.Binomial(), freq_weights=weights)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerfectSeparationWarning)
        warnings.filterwarnings("ignore", message="overflow encountered in exp", category=RuntimeWarning)
        return model.fit(maxiter=100, disp=0)


def _nearest_psd(matrix: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, jitter, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def fit_weight_models(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["id", "month"]).copy()
    baseline = (
        panel.groupby("id", as_index=False)
        .agg(z=("z", "first"), x=("x", "first"))
        .sort_values("id")
    )
    ps_model = _fit_glm(baseline.assign(event=baseline["z"]), ["x"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow encountered in exp", category=RuntimeWarning)
        p_treat = float(np.clip(baseline["z"].mean(), 1e-3, 1.0 - 1e-3))
        baseline["ps"] = np.clip(
            ps_model.predict(sm.add_constant(baseline[["x"]], has_constant="add")),
            1e-3,
            1.0 - 1e-3,
        )
    baseline["baseline_weight"] = np.where(
        baseline["z"] == 1,
        p_treat / baseline["ps"],
        (1.0 - p_treat) / (1.0 - baseline["ps"]),
    )
    panel = panel.merge(baseline[["id", "baseline_weight"]], on="id", how="left")

    has_restart_transition = bool(((panel["month"] > 1) & (panel["prev_e"] == 0) & (panel["e_t"] == 1)).any())
    if has_restart_transition:
        rows = panel[panel["month"] > 1].copy()
    else:
        rows = panel[(panel["month"] > 1) & (panel["prev_e"] == 1)].copy()
    if rows.empty:
        panel["analysis_weight"] = panel["baseline_weight"].fillna(1.0)
        return panel

    denom_cols = ["month_c", "month_sq", "x", "z", "prev_e", "l_t"]
    numer_cols = ["month_c", "month_sq", "x", "z", "prev_e"]
    denom = _fit_glm(rows, denom_cols)
    numer = _fit_glm(rows, numer_cols)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow encountered in exp", category=RuntimeWarning)
        rows["p_den"] = np.clip(
            denom.predict(sm.add_constant(rows[denom_cols], has_constant="add")),
            1e-3,
            1.0 - 1e-3,
        )
        rows["p_num"] = np.clip(
            numer.predict(sm.add_constant(rows[numer_cols], has_constant="add")),
            1e-3,
            1.0 - 1e-3,
        )
    rows["step_weight"] = np.where(
        rows["e_t"] == 1,
        rows["p_num"] / rows["p_den"],
        (1.0 - rows["p_num"]) / (1.0 - rows["p_den"]),
    )
    rows["analysis_weight"] = rows.groupby("id")["step_weight"].cumprod().clip(upper=20.0)
    panel = panel.merge(rows[["id", "month", "analysis_weight"]], on=["id", "month"], how="left")
    panel["analysis_weight"] = panel.groupby("id")["analysis_weight"].ffill().fillna(1.0)
    panel["analysis_weight"] = panel["analysis_weight"] * panel["baseline_weight"].fillna(1.0)
    return panel


def _risk_curve_from_result(
    result: sm.GLM,
    columns: list[str],
    prediction_panel: pd.DataFrame,
) -> pd.DataFrame:
    pred = prediction_panel.copy()
    exog = sm.add_constant(pred[columns], has_constant="add")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="overflow encountered in exp", category=RuntimeWarning)
        hazard = np.clip(result.predict(exog), 1e-6, 1.0 - 1e-6)
    pred["hazard"] = hazard
    pred["survival"] = 1.0 - pred["hazard"]
    pred["survival"] = pred.groupby("id")["survival"].cumprod()
    pred["risk"] = 1.0 - pred["survival"]
    out = pred.groupby("month", as_index=False)["risk"].mean()
    return out.rename(columns={"risk": "estimated_risk"})


def _rd_ci_from_draws(
    result: sm.GLM,
    columns: list[str],
    prediction_panels: dict[str, pd.DataFrame],
    rng: np.random.Generator,
    n_draws: int,
) -> tuple[float, float]:
    params = np.asarray(result.params, dtype=float)
    cov = np.asarray(result.cov_params(), dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = _nearest_psd(cov)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="covariance is not symmetric positive-semidefinite", category=RuntimeWarning)
        draws = rng.multivariate_normal(params, cov, size=n_draws, check_valid="ignore")
    rds = []
    for draw in draws:
        risks = {}
        for history_name, panel in prediction_panels.items():
            exog = sm.add_constant(panel[columns], has_constant="add").to_numpy(dtype=float)
            linpred = exog @ draw
            hazard = np.clip(expit(linpred), 1e-6, 1.0 - 1e-6).reshape(-1)
            tmp = panel[["id", "month"]].copy()
            tmp["hazard"] = hazard
            tmp["survival"] = 1.0 - tmp["hazard"]
            tmp["survival"] = tmp.groupby("id")["survival"].cumprod()
            tmp["risk"] = 1.0 - tmp["survival"]
            risk_60 = float(tmp.loc[tmp["month"] == tmp["month"].max(), "risk"].mean())
            risks[history_name] = risk_60
        rds.append(risks["persistent_treatment"] - risks["comparator_reference"])
    return float(np.nanpercentile(rds, 2.5)), float(np.nanpercentile(rds, 97.5))


def evaluate_methods(
    panel: pd.DataFrame,
    x_values: np.ndarray,
    kernel_model: np.ndarray,
    rng: np.random.Generator,
    n_ci_draws: int,
    weighted_panel: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if weighted_panel is None:
        weighted_panel = fit_weight_models(panel)
    prediction_panels = {
        history_name: build_prediction_panel(x_values, int(panel["month"].max()), history_name, kernel_model)
        for history_name in HISTORY_ORDER
    }

    method_specs = {
        "initiation_only": {
            "columns": ["month_c", "month_sq", "x", "z", "z_month"],
            "weight_col": None,
            "transform": lambda df: df.assign(z_month=df["z"] * df["month_c"]),
            "pred_transform": lambda df: df.assign(z_month=df["z"] * df["month_c"]),
        },
        "naive_persistence": {
            "columns": ["month_c", "month_sq", "x", "z", "z_e", "z_e_month"],
            "weight_col": None,
            "transform": lambda df: df.assign(
                z_e=df["z"] * df["e_t"],
                z_e_month=df["z"] * df["e_t"] * df["month_c"],
            ),
            "pred_transform": lambda df: df.assign(
                z_e=df["z"] * df["e_t"],
                z_e_month=df["z"] * df["e_t"] * df["month_c"],
            ),
        },
        "unweighted_cumulative": {
            "columns": ["month_c", "month_sq", "x", "cumulative_e"],
            "weight_col": None,
            "transform": lambda df: df,
            "pred_transform": lambda df: df,
        },
        "unweighted_lagged": {
            "columns": ["month_c", "month_sq", "x", "lagged_h"],
            "weight_col": None,
            "transform": lambda df: df.rename(columns={"h_true": "lagged_h"}),
            "pred_transform": lambda df: df,
        },
        "weighted_cumulative": {
            "columns": ["month_c", "month_sq", "x", "cumulative_e"],
            "weight_col": "analysis_weight",
            "transform": lambda df: df,
            "pred_transform": lambda df: df,
        },
        "lag_aware_weighted_tte": {
            "columns": ["month_c", "month_sq", "x", "lagged_h"],
            "weight_col": "analysis_weight",
            "transform": lambda df: df.rename(columns={"h_true": "lagged_h"}),
            "pred_transform": lambda df: df,
        },
    }

    estimate_rows: list[dict] = []
    curve_rows: list[dict] = []

    for method_name in SUPPLEMENT_METHOD_ORDER:
        spec = method_specs[method_name]
        analysis_panel = spec["transform"](weighted_panel.copy())
        pred_panels = {
            history_name: spec["pred_transform"](prediction_panels[history_name].copy())
            for history_name in HISTORY_ORDER
        }
        result = _fit_glm(analysis_panel, spec["columns"], weight_col=spec["weight_col"])
        risk_60 = {}
        for history_name, pred_panel in pred_panels.items():
            curve = _risk_curve_from_result(result, spec["columns"], pred_panel)
            curve["method"] = method_name
            curve["history_name"] = history_name
            curve_rows.extend(curve.to_dict(orient="records"))
            risk_60[history_name] = float(curve.loc[curve["month"] == curve["month"].max(), "estimated_risk"].iloc[0])

        ci_low, ci_high = _rd_ci_from_draws(result, spec["columns"], pred_panels, rng, n_ci_draws)
        estimate_rows.append(
            {
                "method": method_name,
                "persistent_risk_60m": risk_60["persistent_treatment"],
                "stop12_risk_60m": risk_60["stop12"],
                "comparator_risk_60m": risk_60["comparator_reference"],
                "rd_persistent_vs_comparator_60m": risk_60["persistent_treatment"] - risk_60["comparator_reference"],
                "rd_persistent_vs_stop12_60m": risk_60["persistent_treatment"] - risk_60["stop12"],
                "ci_lower_60m": ci_low,
                "ci_upper_60m": ci_high,
            }
        )

    return pd.DataFrame(estimate_rows), pd.DataFrame(curve_rows)


def summarize_weight_distribution(
    weighted_panel: pd.DataFrame,
    *,
    scenario_name: str,
    replicate: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for arm_value, arm_label in [(1, "treated"), (0, "reference"), (None, "all")]:
        if arm_value is None:
            grp = weighted_panel
        else:
            grp = weighted_panel[weighted_panel["z"] == arm_value]
        if grp.empty:
            continue
        weights = pd.to_numeric(grp["analysis_weight"], errors="coerce").dropna().to_numpy(dtype=float)
        if weights.size == 0:
            continue
        rows.append(
            {
                "scenario_name": scenario_name,
                "replicate": replicate,
                "arm": arm_label,
                "n_rows": int(weights.size),
                "mean_weight": float(np.mean(weights)),
                "median_weight": float(np.median(weights)),
                "p90_weight": float(np.nanpercentile(weights, 90)),
                "p95_weight": float(np.nanpercentile(weights, 95)),
                "p99_weight": float(np.nanpercentile(weights, 99)),
                "max_weight": float(np.max(weights)),
            }
        )
    return pd.DataFrame(rows)


def summarize_exposure_history_correlation(
    *,
    scenario: Scenario,
    kernel_true_spec: KernelSpec,
    reps: int,
    n_main: int,
    seed: int,
    scenario_label: str,
) -> pd.DataFrame:
    kernel_true = build_kernel(kernel_true_spec)
    rows: list[dict] = []
    for rep in range(reps):
        panel = simulate_observed_panel(n_main, scenario, kernel_true, seed + rep)
        treated = panel[(panel["z"] == 1) & (panel["month"] > 1)].copy()
        corr = np.nan
        if len(treated) >= 3:
            corr = float(treated["cumulative_e"].corr(treated["h_true"]))
        rows.append(
            {
                "scenario_label": scenario_label,
                "replicate": rep + 1,
                "corr_cumulative_vs_lagged": corr,
            }
        )
    return pd.DataFrame(rows)


def build_timing_example_histories(
    *,
    kernel_true_spec: KernelSpec,
    months: int,
) -> pd.DataFrame:
    kernel_true = build_kernel(kernel_true_spec)
    examples = {
        "Early exposure only": np.concatenate([np.ones(12, dtype=float), np.zeros(months - 12, dtype=float)]),
        "Recent sustained exposure": np.concatenate([np.zeros(months - 12, dtype=float), np.ones(12, dtype=float)]),
    }
    rows: list[dict] = []
    for example_name, exposure in examples.items():
        exposure_mat = exposure[None, :]
        cumulative = cumulative_exposure(exposure_mat).ravel()
        lagged = lagged_summary(exposure_mat, kernel_true).ravel()
        for month_idx in range(months):
            rows.append(
                {
                    "example_name": example_name,
                    "month": month_idx + 1,
                    "exposure": float(exposure[month_idx]),
                    "cumulative_e": float(cumulative[month_idx]),
                    "lagged_h": float(lagged[month_idx]),
                    "cumulative_60m": float(cumulative[-1]),
                    "lagged_60m": float(lagged[-1]),
                }
            )
    return pd.DataFrame(rows)


def build_timing_scenarios(base_scenario: Scenario) -> tuple[Scenario, Scenario]:
    monotone = replace(
        base_scenario,
        continuation_intercept=2.25,
        continuation_l=-1.20,
        outcome_l=1.15,
        allow_restart=False,
        restart_intercept=-2.40,
        restart_x=0.05,
        restart_l=-0.35,
    )
    intermittent = replace(monotone, allow_restart=True)
    return monotone, intermittent


def compute_timing_figure_inputs(
    outdir: Path,
    *,
    scenario: Scenario,
    kernel_true_spec: KernelSpec,
    kernel_model_spec: KernelSpec,
    reps: int,
    n_main: int,
    truth_n: int,
    sensitivity_reps: int,
    n_sensitivity: int,
    n_ci_draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timing_monotone_scenario, timing_intermittent_scenario = build_timing_scenarios(scenario)
    timing_monotone_results = compute_main_results(
        outdir,
        reps=reps,
        n_main=n_main,
        truth_n=truth_n,
        kernel_true_spec=kernel_true_spec,
        kernel_model_spec=kernel_model_spec,
        scenario=timing_monotone_scenario,
        seed=seed + 520000,
        n_ci_draws=n_ci_draws,
        scenario_name="timing_monotone",
    )
    timing_intermittent_results = compute_main_results(
        outdir,
        reps=reps,
        n_main=n_main,
        truth_n=truth_n,
        kernel_true_spec=kernel_true_spec,
        kernel_model_spec=kernel_model_spec,
        scenario=timing_intermittent_scenario,
        seed=seed + 540000,
        n_ci_draws=n_ci_draws,
        scenario_name="timing_intermittent",
    )
    timing_correlation = pd.concat(
        [
            summarize_exposure_history_correlation(
                scenario=timing_monotone_scenario,
                kernel_true_spec=kernel_true_spec,
                reps=sensitivity_reps,
                n_main=n_sensitivity,
                seed=seed + 560000,
                scenario_label="monotone",
            ),
            summarize_exposure_history_correlation(
                scenario=timing_intermittent_scenario,
                kernel_true_spec=kernel_true_spec,
                reps=sensitivity_reps,
                n_main=n_sensitivity,
                seed=seed + 580000,
                scenario_label="intermittent",
            ),
        ],
        ignore_index=True,
    )
    timing_example_histories = build_timing_example_histories(
        kernel_true_spec=kernel_true_spec,
        months=scenario.months,
    )
    return (
        timing_monotone_results["estimate_metrics"],
        timing_intermittent_results["estimate_metrics"],
        timing_intermittent_results["truth_curves"],
        timing_intermittent_results["curve_summary"],
        timing_correlation,
        timing_example_histories,
    )


def summarize_persistence(panel: pd.DataFrame) -> pd.DataFrame:
    initiators = panel.groupby("id", as_index=False).agg(z=("z", "first"))
    n_initiators = int((initiators["z"] == 1).sum())
    initiator_panel = panel[panel["z"] == 1].copy()
    if initiator_panel.empty or n_initiators == 0:
        return pd.DataFrame(columns=["month", "p_on_treatment"])
    out = (
        initiator_panel.groupby("month", as_index=False)["e_t"]
        .sum()
        .rename(columns={"e_t": "n_on_treatment"})
    )
    out["n_initiators"] = n_initiators
    out["p_on_treatment"] = out["n_on_treatment"] / float(n_initiators)
    return out


def compute_main_results(
    outdir: Path,
    *,
    reps: int,
    n_main: int,
    truth_n: int,
    kernel_true_spec: KernelSpec,
    kernel_model_spec: KernelSpec,
    scenario: Scenario,
    seed: int,
    n_ci_draws: int,
    scenario_name: str = "primary",
) -> dict[str, pd.DataFrame]:
    kernel_true = build_kernel(kernel_true_spec)
    kernel_model = build_kernel(kernel_model_spec)
    truth_curves = pd.concat(
        [
            simulate_regime_truth(truth_n, scenario, history_name, kernel_true, seed + 8000 + i)
            for i, history_name in enumerate(HISTORY_ORDER)
        ],
        ignore_index=True,
    )
    truth_60 = truth_curves[truth_curves["month"] == scenario.months].set_index("history_name")["true_risk"].to_dict()
    true_rd = float(truth_60["persistent_treatment"] - truth_60["comparator_reference"])

    all_estimates = []
    all_curves = []
    all_persistence = []
    all_weights = []

    for rep in range(reps):
        rep_seed = seed + rep
        panel = simulate_observed_panel(n_main, scenario, kernel_true, rep_seed)
        weighted_panel = fit_weight_models(panel)
        x_values = (
            panel.groupby("id", as_index=False)
            .agg(x=("x", "first"))
            .sort_values("id")["x"]
            .to_numpy(dtype=float)
        )
        rng = np.random.default_rng(rep_seed + 500000)
        estimates, curves = evaluate_methods(panel, x_values, kernel_model, rng, n_ci_draws, weighted_panel=weighted_panel)
        estimates["replicate"] = rep + 1
        curves["replicate"] = rep + 1
        persistence = summarize_persistence(panel)
        persistence["replicate"] = rep + 1
        weight_summary = summarize_weight_distribution(weighted_panel, scenario_name=scenario_name, replicate=rep + 1)
        all_estimates.append(estimates)
        all_curves.append(curves)
        all_persistence.append(persistence)
        all_weights.append(weight_summary)

    estimates = pd.concat(all_estimates, ignore_index=True)
    curves = pd.concat(all_curves, ignore_index=True)
    persistence = pd.concat(all_persistence, ignore_index=True)
    weight_summary = pd.concat(all_weights, ignore_index=True)

    metrics = []
    for method_name, grp in estimates.groupby("method", sort=False):
        covered = (
            (pd.to_numeric(grp["ci_lower_60m"], errors="coerce") <= true_rd)
            & (pd.to_numeric(grp["ci_upper_60m"], errors="coerce") >= true_rd)
        )
        rd_hat = pd.to_numeric(grp["rd_persistent_vs_comparator_60m"], errors="coerce")
        bias = float((rd_hat - true_rd).mean())
        rmse = float(np.sqrt(np.mean(np.square(rd_hat - true_rd))))
        metrics.append(
            {
                "method": method_name,
                "true_rd_60m": true_rd,
                "mean_estimate_60m": float(rd_hat.mean()),
                "mc_interval_low": float(np.nanpercentile(rd_hat, 2.5)),
                "mc_interval_high": float(np.nanpercentile(rd_hat, 97.5)),
                "bias": bias,
                "abs_bias": float(abs(bias)),
                "rmse": rmse,
                "coverage": float(covered.mean()),
                "coverage_pct": float(100.0 * covered.mean()),
            }
        )

    curve_summary = (
        curves.groupby(["method", "history_name", "month"], as_index=False)["estimated_risk"]
        .agg(["mean", "median"])
        .reset_index()
        .rename(columns={"mean": "estimated_risk_mean", "median": "estimated_risk_median"})
    )
    persistence_summary = (
        persistence.groupby("month", as_index=False)["p_on_treatment"]
        .agg(["mean", "median"])
        .reset_index()
        .rename(columns={"mean": "p_on_treatment_mean", "median": "p_on_treatment_median"})
    )

    return {
        "truth_curves": truth_curves,
        "estimates": estimates,
        "estimate_metrics": pd.DataFrame(metrics),
        "curve_estimates": curves,
        "curve_summary": curve_summary,
        "persistence_summary": persistence_summary,
        "weight_summary": weight_summary,
        "kernel_true": kernel_table(kernel_true_spec),
    }


def run_bias_sensitivity(
    *,
    scenario_grid: list[tuple[str, float, Scenario, KernelSpec]],
    methods_for_plot: list[str],
    reps: int,
    n_main: int,
    truth_n: int,
    kernel_true_spec: KernelSpec,
    seed: int,
    n_ci_draws: int,
) -> pd.DataFrame:
    rows = []
    for setting_index, (setting_name, x_value, scenario, kernel_model_spec) in enumerate(scenario_grid):
        kernel_true = build_kernel(kernel_true_spec)
        kernel_model = build_kernel(kernel_model_spec)
        truth_curves = pd.concat(
            [
                simulate_regime_truth(truth_n, scenario, history_name, kernel_true, seed + 90000 + setting_index * 10 + i)
                for i, history_name in enumerate(HISTORY_ORDER)
            ],
            ignore_index=True,
        )
        truth_60 = truth_curves[truth_curves["month"] == scenario.months].set_index("history_name")["true_risk"].to_dict()
        true_rd = float(truth_60["persistent_treatment"] - truth_60["comparator_reference"])
        rep_rows = []
        discont_rate = []
        for rep in range(reps):
            rep_seed = seed + setting_index * 1000 + rep
            panel = simulate_observed_panel(n_main, scenario, kernel_true, rep_seed)
            x_values = (
                panel.groupby("id", as_index=False)
                .agg(x=("x", "first"))
                .sort_values("id")["x"]
                .to_numpy(dtype=float)
            )
            rng = np.random.default_rng(rep_seed + 700000)
            estimates, _ = evaluate_methods(panel, x_values, kernel_model, rng, n_ci_draws)
            estimates["replicate"] = rep + 1
            rep_rows.append(estimates)
            persistence = summarize_persistence(panel)
            if not persistence.empty:
                sel = persistence[persistence["month"] == 12]
                if not sel.empty:
                    discont_rate.append(float(1.0 - sel["p_on_treatment"].iloc[0]))
        setting_estimates = pd.concat(rep_rows, ignore_index=True)
        early_discontinuation = float(np.mean(discont_rate)) if discont_rate else np.nan
        for method_name in methods_for_plot:
            grp = setting_estimates[setting_estimates["method"] == method_name]
            rd = pd.to_numeric(grp["rd_persistent_vs_comparator_60m"], errors="coerce")
            rows.append(
                {
                    "setting_name": setting_name,
                    "x_value": x_value,
                    "method": method_name,
                    "true_rd_60m": true_rd,
                    "mean_estimate_60m": float(rd.mean()),
                    "bias": float((rd - true_rd).mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(rd - true_rd)))),
                    "early_discontinuation_rate": early_discontinuation,
                }
            )
    return pd.DataFrame(rows)


def run_kernel_shape_sensitivity(
    *,
    scenario: Scenario,
    kernel_true_spec: KernelSpec,
    kernel_model_specs: list[KernelSpec],
    reps: int,
    n_main: int,
    truth_n: int,
    seed: int,
    n_ci_draws: int,
) -> pd.DataFrame:
    rows = []
    kernel_true = build_kernel(kernel_true_spec)
    truth_curves = pd.concat(
        [
            simulate_regime_truth(truth_n, scenario, history_name, kernel_true, seed + 95000 + i)
            for i, history_name in enumerate(HISTORY_ORDER)
        ],
        ignore_index=True,
    )
    truth_60 = truth_curves[truth_curves["month"] == scenario.months].set_index("history_name")["true_risk"].to_dict()
    true_rd = float(truth_60["persistent_treatment"] - truth_60["comparator_reference"])

    for shape_index, kernel_model_spec in enumerate(kernel_model_specs):
        rep_rows = []
        for rep in range(reps):
            rep_seed = seed + shape_index * 1000 + rep
            panel = simulate_observed_panel(n_main, scenario, kernel_true, rep_seed)
            x_values = (
                panel.groupby("id", as_index=False)
                .agg(x=("x", "first"))
                .sort_values("id")["x"]
                .to_numpy(dtype=float)
            )
            rng = np.random.default_rng(rep_seed + 810000)
            estimates, _ = evaluate_methods(panel, x_values, build_kernel(kernel_model_spec), rng, n_ci_draws)
            rep_rows.append(estimates[estimates["method"] == "lag_aware_weighted_tte"].copy())
        shape_estimates = pd.concat(rep_rows, ignore_index=True)
        rd = pd.to_numeric(shape_estimates["rd_persistent_vs_comparator_60m"], errors="coerce")
        rows.append(
            {
                "shape_name": kernel_model_spec.shape,
                "shape_label": KERNEL_SHAPE_LABEL.get(kernel_model_spec.shape, kernel_model_spec.shape),
                "mean_estimate_60m": float(rd.mean()),
                "mc_interval_low": float(np.nanpercentile(rd, 2.5)),
                "mc_interval_high": float(np.nanpercentile(rd, 97.5)),
                "bias": float((rd - true_rd).mean()),
                "rmse": float(np.sqrt(np.mean(np.square(rd - true_rd)))),
                "true_rd_60m": true_rd,
                "delay_months": kernel_model_spec.delay_months,
                "max_lag_months": kernel_model_spec.max_lag_months,
            }
        )
    return pd.DataFrame(rows)


def make_figure5_simulation_design(
    outdir: Path,
    kernel_true_df: pd.DataFrame,
    persistence_df: pd.DataFrame,
) -> None:
    with simulation_style():
        fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8))

        ax = axes[0, 0]
        ax.set_axis_off()
        _boxed_text(ax, 0.10, 0.78, "$X$\nBaseline risk")
        _boxed_text(ax, 0.28, 0.78, "$Z$\nBaseline initiation")
        _boxed_text(ax, 0.50, 0.78, "$E_{t-1}$\nPrior exposure")
        _boxed_text(ax, 0.50, 0.49, "$L_t$\nClinical status")
        _boxed_text(ax, 0.74, 0.78, "$E_{t+1}$\nFuture persistence")
        _boxed_text(ax, 0.87, 0.49, "$Y_t$\nOutcome")
        _boxed_text(ax, 0.33, 0.20, "$H_t = \\sum_s w(s) E_{t-s}$\nAccumulated exposure history")
        _arrow(ax, (0.15, 0.78), (0.23, 0.78))
        _arrow(ax, (0.34, 0.78), (0.44, 0.78))
        _arrow(ax, (0.50, 0.70), (0.50, 0.56))
        _arrow(ax, (0.56, 0.49), (0.68, 0.73))
        _arrow(ax, (0.56, 0.49), (0.81, 0.49))
        _arrow(ax, (0.42, 0.24), (0.80, 0.45))
        ax.annotate(
            "",
            xy=(0.39, 0.24),
            xytext=(0.49, 0.72),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "color": "#666666", "linewidth": 0.9, "linestyle": ":"},
        )
        ax.text(
            0.03,
            0.33,
            "Treatment-confounder feedback:\n$E_{t-1} \\rightarrow L_t \\rightarrow E_{t+1}$",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.2,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#DDDDDD"},
        )
        ax.text(
            0.02,
            0.05,
            "Treatment persistence depends on evolving clinical history;\n"
            "outcome risk depends on gradually accumulated exposure history.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.3,
        )
        panel_label(ax, "A", x=-0.02)

        ax = axes[0, 1]
        ax.plot(kernel_true_df["lag_month"], kernel_true_df["weight"], color=PALETTE["kernel"], linewidth=1.8)
        ax.axvline(36, color="#999999", linestyle=":", linewidth=0.8)
        for ref_month in [0, 6, 12, 24]:
            ref_row = kernel_true_df.loc[kernel_true_df["lag_month"] == ref_month]
            if not ref_row.empty:
                ax.scatter(
                    ref_month,
                    float(ref_row["weight"].iloc[0]),
                    color=PALETTE["kernel"],
                    s=18,
                    zorder=3,
                )
        ax.set_xlabel("Elapsed months since prior exposure")
        ax.set_ylabel("Accumulation kernel weight")
        ax.set_xlim(0, 36)
        ax.set_ylim(0, max(0.06, float(kernel_true_df["weight"].max()) * 1.1))
        rel = kernel_true_df.set_index("lag_month")["relative_contribution"]
        ax.text(
            0.03,
            0.96,
            "Early exposure contributes partially;\naccumulated exposure contributes more fully over time.\n"
            f"Relative contribution: {rel.get(0, np.nan):.0%} at month 0, {rel.get(12, np.nan):.0%} by month 12;\n"
            "maximum lookback = 36 months",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.3,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#CCCCCC"},
        )
        despine(ax)
        panel_label(ax, "B")

        ax = axes[1, 0]
        ax.plot(
            persistence_df["month"],
            persistence_df["p_on_treatment_mean"],
            color=PALETTE["target"],
            linewidth=1.8,
        )
        ax.set_xlabel("Month")
        ax.set_ylabel("Proportion still exposed")
        ax.set_xlim(1, 60)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 12, 24, 36, 48, 60])
        ax.axvline(12, color="#999999", linestyle="--", linewidth=0.8)
        month12 = persistence_df.loc[persistence_df["month"] == 12, "p_on_treatment_mean"]
        if not month12.empty:
            early_disc = 1.0 - float(month12.iloc[0])
            ax.text(
                0.03,
                0.96,
                f"{early_disc:.0%} discontinued by 12 months",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.3,
                bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#CCCCCC"},
            )
        ax.text(
            0.03,
            0.06,
            "Initiators quickly diverge into heterogeneous exposure histories.",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.3,
        )
        despine(ax)
        panel_label(ax, "C")

        ax = axes[1, 1]
        ax.set_axis_off()
        methods = [
            [
                "Initiation-only\nITT-like TTE",
                "Baseline\ninitiation",
                "Mixes persistent and\nnonpersistent exposure",
            ],
            [
                "Naive persistent-user\ncomparison",
                "Observed\npersistence",
                "Ignores treatment-\nconfounder feedback",
            ],
            [
                "Unweighted cumulative\nexposure model",
                "Cumulative\nexposure",
                "Ignores time-varying\nconfounding",
            ],
            [
                "Lag-aware\nweighted TTE",
                "Lagged exposure\n+ IPW",
                "Proposed\nframework",
            ],
        ]
        table = ax.table(
            cellText=methods,
            colLabels=["Method", "What It Uses", "Main Limitation"],
            cellLoc="left",
            colLoc="left",
            colWidths=[0.42, 0.23, 0.35],
            bbox=[0.01, 0.06, 0.98, 0.88],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.0)
        table.scale(1.0, 1.55)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#D5D5D5")
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor("#F4F6F6")
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor("white")
                if col == 0:
                    method_name = METHOD_ORDER[row - 1]
                    cell.get_text().set_color(METHOD_COLOR[method_name])
                    cell.get_text().set_fontweight("semibold")
        panel_label(ax, "D", x=-0.02)

        save_figure(fig, outdir, "figure5_simulation_design_delayed_cumulative_effects")
        plt.close(fig)


def _metric_cell_color(metric: str, value: float, values: np.ndarray) -> tuple[float, float, float, float]:
    if metric == "coverage":
        score = 1.0 - min(abs(value - 0.95) / 0.50, 1.0)
        return mpl.cm.Greens(0.25 + 0.65 * score)
    abs_values = np.abs(values)
    scale = float(np.nanmax(abs_values)) if np.isfinite(abs_values).any() else 1.0
    scale = max(scale, 1e-8)
    score = 1.0 - min(abs(value) / scale, 1.0)
    return mpl.cm.Blues(0.25 + 0.65 * score)


def make_figure6_simulation_results(
    outdir: Path,
    truth_curves: pd.DataFrame,
    estimate_metrics: pd.DataFrame,
    curve_summary: pd.DataFrame,
    discontinuation_sensitivity: pd.DataFrame,
    confounding_sensitivity: pd.DataFrame,
    lag_sensitivity: pd.DataFrame,
    kernel_shape_sensitivity: pd.DataFrame,
) -> None:
    truth_line = float(estimate_metrics["true_rd_60m"].iloc[0])

    with simulation_style():
        fig = plt.figure(figsize=(11.8, 10.2))
        gs = fig.add_gridspec(
            3,
            2,
            height_ratios=[1.0, 1.0, 1.12],
            hspace=0.38,
            wspace=0.26,
            left=0.06,
            right=0.97,
            top=0.95,
            bottom=0.06,
        )

        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])
        ax_e = fig.add_subplot(gs[2, :])

        plot_df = estimate_metrics.set_index("method").loc[METHOD_ORDER].reset_index()

        y_pos = np.arange(len(METHOD_ORDER))[::-1]
        for y, row in zip(y_pos, plot_df.itertuples(index=False)):
            method_name = str(row.method)
            color = METHOD_COLOR[method_name]
            is_proposed = method_name == "lag_aware_weighted_tte"
            marker = "o"
            ms = 7.2 if is_proposed else 5.3
            alpha = 1.0 if is_proposed else 0.70
            est = 100.0 * float(row.mean_estimate_60m)
            ax_a.plot(est, y, marker=marker, color=color, ms=ms, zorder=3, lw=0, alpha=alpha)
            ax_a.annotate(
                f"{est:.1f}",
                xy=(est, y),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.0,
                color=color,
                alpha=alpha,
            )
        ax_a.axvline(100.0 * truth_line, color="#2C2C2A", lw=1.3, ls="--", label="True 5-year RD")
        ax_a.set_yticks(y_pos)
        ax_a.set_yticklabels([METHOD_LABEL[m] for m in METHOD_ORDER])
        ax_a.set_xlabel("5-year risk difference (%)")
        ax_a.set_title("Estimated 5-year risk difference", pad=4)
        ax_a.tick_params(axis="y", length=0)
        legend_handles = [
            mpl.lines.Line2D([0], [0], color="#2C2C2A", lw=1.3, ls="--", label="True 5-year RD"),
            mpl.lines.Line2D([0], [0], color="#222222", marker="o", lw=0, label="Mean estimate"),
        ]
        ax_a.legend(handles=legend_handles, loc="lower right", frameon=False)
        ax_a.text(
            0.02,
            0.03,
            "Negative values favor persistent treatment",
            transform=ax_a.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.8,
            color="#444444",
        )
        ax_a.xaxis.grid(True)
        ax_a.yaxis.grid(False)
        despine(ax_a)
        panel_label(ax_a, "A")

        bias_plot_df = plot_df.copy()
        bias_plot_df["signed_bias_pct"] = 100.0 * bias_plot_df["bias"]
        bias_max = float(np.max(np.abs(bias_plot_df["signed_bias_pct"])))
        xpad = max(0.8, 0.18 * bias_max)
        for y, row in zip(y_pos, bias_plot_df.itertuples(index=False)):
            method_name = str(row.method)
            color = METHOD_COLOR[method_name]
            is_proposed = method_name == "lag_aware_weighted_tte"
            alpha = 1.0 if is_proposed else 0.70
            bias_val = float(row.signed_bias_pct)
            ax_b.barh(
                y,
                bias_val,
                height=0.42,
                color=mpl.colors.to_rgba(color, 0.90 if is_proposed else 0.60),
                edgecolor=color,
                linewidth=1.2 if is_proposed else 0.9,
                zorder=2,
                alpha=alpha,
            )
            ha = "left" if bias_val >= 0 else "right"
            x_text = bias_val + (0.28 if bias_val >= 0 else -0.28)
            ax_b.text(
                x_text,
                y,
                f"{bias_val:+.1f}",
                ha=ha,
                va="center",
                fontsize=7.1,
                color=color,
                alpha=alpha,
            )
        ax_b.set_yticks(y_pos)
        ax_b.set_yticklabels([METHOD_LABEL[m] for m in METHOD_ORDER])
        ax_b.tick_params(axis="y", length=0)
        ax_b.axvline(0.0, color="#2C2C2A", lw=1.1, ls="--")
        ax_b.set_xlim(-(bias_max + xpad), bias_max + xpad)
        ax_b.set_xlabel("Bias (%)")
        ax_b.set_title("Bias in 5-year risk difference", pad=4)
        ax_b.text(
            0.02,
            0.97,
            "Exaggerated benefit",
            transform=ax_b.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color="#444444",
        )
        ax_b.text(
            0.98,
            0.97,
            "Toward null",
            transform=ax_b.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color="#444444",
        )
        ax_b.xaxis.grid(True)
        ax_b.yaxis.grid(False)
        despine(ax_b)
        panel_label(ax_b, "B")

        disc_df = (
            discontinuation_sensitivity[discontinuation_sensitivity["method"] == "initiation_only"]
            .copy()
            .sort_values("early_discontinuation_rate")
        )
        disc_x = 100.0 * disc_df["early_discontinuation_rate"].to_numpy(dtype=float)
        disc_y = 100.0 * disc_df["bias"].to_numpy(dtype=float)
        ax_c.scatter(disc_x, disc_y, color=METHOD_COLOR["initiation_only"], s=28, zorder=3)
        if len(disc_df) >= 3:
            slope, intercept = np.polyfit(disc_x, disc_y, deg=1)
            xfit = np.linspace(float(disc_x.min()), float(disc_x.max()), 100)
            ax_c.plot(xfit, intercept + slope * xfit, color="#777777", lw=1.0, ls=":")
        ax_c.axhline(0.0, color="#222222", lw=0.7, ls="--", alpha=0.5)
        ax_c.set_xlabel("Discontinued by 12 months (%)")
        ax_c.set_ylabel("Bias toward the null in initiation-only RD (%)")
        ax_c.set_title("Initiation-only dilution increased with early discontinuation", pad=4)
        ax_c.text(
            0.98,
            0.95,
            "More discontinuation,\ngreater dilution",
            transform=ax_c.transAxes,
            ha="right",
            va="top",
            fontsize=6.7,
            color="#666666",
        )
        ax_c.xaxis.grid(True)
        ax_c.yaxis.grid(True)
        despine(ax_c)
        panel_label(ax_c, "C")

        conf_df = (
            confounding_sensitivity[confounding_sensitivity["method"] == "naive_persistence"]
            .copy()
            .sort_values("x_value")
        )
        conf_y = 100.0 * np.abs(conf_df["bias"].to_numpy(dtype=float))
        conf_x = conf_df["x_value"].to_numpy(dtype=float)
        ax_d.scatter(conf_x, conf_y, color=METHOD_COLOR["naive_persistence"], s=28, zorder=3)
        ax_d.axhline(0.0, color="#222222", lw=0.7, ls="--", alpha=0.5)
        ax_d.set_xticks(conf_x)
        ax_d.set_xticklabels([f"{x:.2g}x" for x in conf_x])
        ax_d.set_xlabel("Feedback strength")
        ax_d.set_ylabel("Absolute bias of naive persistence RD (%)")
        ax_d.set_title("Naive persistence bias increased with treatment-confounder feedback", pad=4)
        ax_d.xaxis.grid(True)
        ax_d.yaxis.grid(True)
        despine(ax_d)
        panel_label(ax_d, "D")

        truth_lookup = truth_curves.rename(columns={"true_risk": "risk"})
        lag_aware = curve_summary[curve_summary["method"] == "lag_aware_weighted_tte"].rename(
            columns={"estimated_risk_mean": "risk"}
        )
        for history_name in HISTORY_ORDER:
            true_hist = truth_lookup[truth_lookup["history_name"] == history_name]
            est_hist = lag_aware[lag_aware["history_name"] == history_name]
            color = HISTORY_COLOR[history_name]
            ax_e.plot(true_hist["month"], 100.0 * true_hist["risk"], ls="--", color=color, lw=1.5)
            ax_e.plot(est_hist["month"], 100.0 * est_hist["risk"], ls="-", color=color, lw=2.0)
        ax_e.axvline(12, color="#888888", lw=0.7, ls=":")
        ax_e.text(12.7, 1.25, "Stop at 12 months", fontsize=6.8, color="#666666")
        ax_e.set_xlim(0, 60)
        ax_e.set_xticks([0, 12, 24, 36, 48, 60])
        ax_e.set_xlabel("Month")
        ax_e.set_ylabel("Cumulative risk (%)")
        ax_e.set_title("Recovery of representative risk curves", pad=4)
        style_handles = [
            mpl.lines.Line2D([0], [0], color="#333333", lw=1.5, ls="--", label="Truth"),
            mpl.lines.Line2D([0], [0], color="#333333", lw=2.0, ls="-", label="Lag-aware estimate"),
        ]
        history_handles = [
            mpl.lines.Line2D([0], [0], color=HISTORY_COLOR[h], lw=2.2, label=HISTORY_LABEL[h])
            for h in HISTORY_ORDER
        ]
        leg1 = ax_e.legend(handles=style_handles, loc="upper left", fontsize=6.9, frameon=False)
        ax_e.add_artist(leg1)
        ax_e.legend(handles=history_handles, loc="lower right", fontsize=6.9, frameon=False, title="History")
        ax_e.xaxis.grid(True)
        ax_e.yaxis.grid(True)
        despine(ax_e)
        panel_label(ax_e, "E")

        save_figure(fig, outdir, "figure6_simulation_results_delayed_exposure_history_contrasts")
        plt.close(fig)


def make_figureS5_simulation_sensitivity(
    outdir: Path,
    estimate_metrics: pd.DataFrame,
    lag_sensitivity: pd.DataFrame,
    kernel_shape_sensitivity: pd.DataFrame,
) -> None:
    truth_line = float(estimate_metrics["true_rd_60m"].iloc[0])
    lag_df = lag_sensitivity.copy().sort_values("x_value")
    shape_df = kernel_shape_sensitivity.copy().reset_index(drop=True)

    with simulation_style():
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
        ax_e, ax_f = axes

        sens_ymin = 100.0 * min(
            lag_df["mc_interval_low"].min(),
            shape_df["mc_interval_low"].min(),
            truth_line,
        )
        sens_ymax = 100.0 * max(
            lag_df["mc_interval_high"].max(),
            shape_df["mc_interval_high"].max(),
            truth_line,
        )
        sens_pad = max(1.0, 0.06 * (sens_ymax - sens_ymin))

        ax_e.axhline(100.0 * truth_line, color="#2C2C2A", lw=1.1, ls="--", label="True 5-year RD")
        for _, row in lag_df.iterrows():
            ax_e.plot(
                [row["x_value"], row["x_value"]],
                [100.0 * row["mc_interval_low"], 100.0 * row["mc_interval_high"]],
                color=METHOD_COLOR["lag_aware_weighted_tte"],
                lw=1.5,
            )
            ax_e.plot(
                row["x_value"],
                100.0 * row["mean_estimate_60m"],
                marker="D",
                color=METHOD_COLOR["lag_aware_weighted_tte"],
                ms=5.6,
                zorder=3,
                lw=0,
            )
        ax_e.set_xticks(lag_df["x_value"])
        ax_e.set_xticklabels([f"\u03bb={int(v)}" for v in lag_df["x_value"]])
        ax_e.set_xlabel("Assumed accumulation scale")
        ax_e.set_ylabel("5-year risk difference (%)")
        ax_e.set_title("Accumulation-speed sensitivity", pad=4)
        ax_e.set_ylim(sens_ymin - sens_pad, sens_ymax + sens_pad)
        ax_e.legend(loc="upper right")
        despine(ax_e)
        panel_label(ax_e, "A")

        x_shape = np.arange(len(shape_df), dtype=float)
        ax_f.axhline(100.0 * truth_line, color="#2C2C2A", lw=1.1, ls="--", label="True 5-year RD")
        for idx, row in shape_df.iterrows():
            marker = "D" if row["shape_name"] == "soft_accumulation" else "o"
            ms = 5.6 if marker == "D" else 5.0
            ax_f.plot(
                [x_shape[idx], x_shape[idx]],
                [100.0 * row["mc_interval_low"], 100.0 * row["mc_interval_high"]],
                color=METHOD_COLOR["lag_aware_weighted_tte"],
                lw=1.5,
            )
            ax_f.plot(
                x_shape[idx],
                100.0 * row["mean_estimate_60m"],
                marker=marker,
                color=METHOD_COLOR["lag_aware_weighted_tte"],
                ms=ms,
                zorder=3,
                lw=0,
            )
        ax_f.set_xticks(x_shape)
        ax_f.set_xticklabels(shape_df["shape_label"])
        ax_f.set_xlabel("Assumed kernel shape")
        ax_f.set_ylabel("5-year risk difference (%)")
        ax_f.set_title("Kernel-shape sensitivity", pad=4)
        ax_f.set_ylim(sens_ymin - sens_pad, sens_ymax + sens_pad)
        ax_f.legend(loc="upper right")
        despine(ax_f)
        panel_label(ax_f, "B")

        save_figure(fig, outdir, "figureS5_simulation_kernel_sensitivity")
        plt.close(fig)


def make_figureS4_simulation_coverage(
    outdir: Path,
    estimate_metrics: pd.DataFrame,
) -> None:
    method_order = [m for m in SUPPLEMENT_METHOD_ORDER if m in set(estimate_metrics["method"])]
    plot_df = estimate_metrics.set_index("method").loc[method_order].reset_index()
    with simulation_style():
        fig, ax = plt.subplots(figsize=(5.8, 2.8))
        ax.set_axis_off()
        table_rows = [
            [METHOD_LABEL[m].replace("\n", " "), f"{float(row.coverage_pct):.1f}%"]
            for m, row in zip(method_order, plot_df.itertuples(index=False))
        ]
        table = ax.table(
            cellText=table_rows,
            colLabels=["Method", "Coverage"],
            cellLoc="left",
            colLoc="left",
            loc="center",
            bbox=[0.02, 0.04, 0.96, 0.90],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.0)
        for (row_idx, col_idx), cell in table.get_celld().items():
            cell.set_edgecolor("#DDDDDD")
            cell.set_linewidth(0.6)
            if row_idx == 0:
                cell.set_facecolor("#F4F6F6")
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor("white")
                if col_idx == 0:
                    method_name = method_order[row_idx - 1]
                    cell.get_text().set_color(METHOD_COLOR[method_name])
        ax.text(
            0.02,
            0.98,
            "Supplementary coverage summary",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.6,
            fontweight="semibold",
        )
        save_figure(fig, outdir, "figureS4_simulation_coverage_medium_scale")
        plt.close(fig)


def make_figureS6_simulation_decomposition(
    outdir: Path,
    primary_metrics: pd.DataFrame,
    intermittent_metrics: pd.DataFrame,
) -> None:
    method_order = [m for m in SUPPLEMENT_METHOD_ORDER if m in set(primary_metrics["method"])]
    with simulation_style():
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), sharey=True)
        panels = [
            ("A", "Primary monotone-discontinuation scenario", primary_metrics),
            ("B", "Intermittent restart scenario", intermittent_metrics),
        ]
        for ax, (label, title, metrics_df) in zip(axes, panels):
            plot_df = metrics_df.set_index("method").loc[method_order].reset_index()
            truth_line = float(plot_df["true_rd_60m"].iloc[0])
            y_pos = np.arange(len(method_order))[::-1]
            for y, row in zip(y_pos, plot_df.itertuples(index=False)):
                method_name = str(row.method)
                color = METHOD_COLOR[method_name]
                marker = "D" if method_name == "lag_aware_weighted_tte" else "o"
                lo = 100.0 * float(row.mc_interval_low)
                hi = 100.0 * float(row.mc_interval_high)
                est = 100.0 * float(row.mean_estimate_60m)
                ax.plot([lo, hi], [y, y], color=color, lw=1.4, solid_capstyle="round")
                ax.plot(est, y, marker=marker, color=color, ms=5.3, zorder=3, lw=0)
            ax.axvline(100.0 * truth_line, color="#2C2C2A", lw=1.2, ls="--")
            ax.set_yticks(y_pos)
            ax.set_yticklabels([METHOD_LABEL[m] for m in method_order])
            ax.set_xlabel("5-year risk difference (%)")
            ax.set_title(title, pad=4)
            ax.tick_params(axis="y", length=0)
            ax.xaxis.grid(True)
            ax.yaxis.grid(False)
            despine(ax)
            panel_label(ax, label)
        save_figure(fig, outdir, "figureS6_simulation_method_decomposition")
        plt.close(fig)


def make_figureS7_weight_diagnostics(
    outdir: Path,
    weight_summary: pd.DataFrame,
) -> None:
    plot_df = weight_summary[weight_summary["arm"] == "treated"].copy()
    if plot_df.empty:
        return
    scenario_order = [s for s in ["primary", "intermittent_restart"] if s in set(plot_df["scenario_name"])]
    label_map = {
        "primary": "Primary",
        "intermittent_restart": "Intermittent",
    }
    with simulation_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6), sharex=True)
        metrics = [
            ("A", "p99_weight", "99th percentile weight"),
            ("B", "max_weight", "Maximum weight"),
        ]
        x_pos = np.arange(len(scenario_order), dtype=float)
        for ax, (label, metric_col, ylabel) in zip(axes, metrics):
            for idx, scenario_name in enumerate(scenario_order):
                vals = plot_df.loc[plot_df["scenario_name"] == scenario_name, metric_col].to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                jitter = np.linspace(-0.06, 0.06, vals.size) if vals.size > 1 else np.array([0.0])
                ax.scatter(
                    np.full(vals.size, x_pos[idx]) + jitter,
                    vals,
                    color=METHOD_COLOR["lag_aware_weighted_tte"],
                    s=22,
                    alpha=0.75,
                    zorder=2,
                )
                ax.plot(
                    x_pos[idx],
                    float(np.mean(vals)),
                    marker="D",
                    color="#222222",
                    ms=5.0,
                    zorder=3,
                    lw=0,
                )
            ax.set_xticks(x_pos)
            ax.set_xticklabels([label_map[s] for s in scenario_order])
            ax.set_ylabel(ylabel)
            ax.yaxis.grid(True)
            ax.xaxis.grid(False)
            despine(ax)
            panel_label(ax, label)
        save_figure(fig, outdir, "figureS7_simulation_weight_diagnostics")
        plt.close(fig)


def make_figure1_timing_sensitive_exposure_histories(
    outdir: Path,
    monotone_metrics: pd.DataFrame,
    intermittent_metrics: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    example_histories: pd.DataFrame,
) -> None:
    method_order = METHOD_ORDER
    mono_df = monotone_metrics.set_index("method").loc[method_order].reset_index()
    int_df = intermittent_metrics.set_index("method").loc[method_order].reset_index()
    mono_truth = float(mono_df["true_rd_60m"].iloc[0])
    int_truth = float(int_df["true_rd_60m"].iloc[0])
    x_values = np.concatenate(
        [
            100.0 * mono_df["mean_estimate_60m"].to_numpy(dtype=float),
            100.0 * int_df["mean_estimate_60m"].to_numpy(dtype=float),
            np.array([100.0 * mono_truth, 100.0 * int_truth]),
        ]
    )
    xpad = max(1.0, 0.10 * (float(x_values.max()) - float(x_values.min())))
    corr_lookup = (
        correlation_summary.groupby("scenario_label", as_index=False)["corr_cumulative_vs_lagged"]
        .mean()
        .set_index("scenario_label")["corr_cumulative_vs_lagged"]
        .to_dict()
    )

    with simulation_style():
        fig = plt.figure(figsize=(11.8, 8.8))
        gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28, left=0.07, right=0.97, top=0.95, bottom=0.08)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])

        for ax, plot_df, truth_line, title, corr_key, panel in [
            (ax_a, mono_df, mono_truth, "Monotone discontinuation scenario", "monotone", "A"),
            (ax_b, int_df, int_truth, "Intermittent exposure with restarts", "intermittent", "B"),
        ]:
            y_pos = np.arange(len(method_order))[::-1]
            for y, row in zip(y_pos, plot_df.itertuples(index=False)):
                method_name = str(row.method)
                color = METHOD_COLOR[method_name]
                is_proposed = method_name == "lag_aware_weighted_tte"
                alpha = 1.0 if is_proposed else 0.70
                est = 100.0 * float(row.mean_estimate_60m)
                ax.plot(est, y, marker="o", color=color, ms=7.2 if is_proposed else 5.3, zorder=3, lw=0, alpha=alpha)
                ax.annotate(
                    f"{est:.1f}",
                    xy=(est, y),
                    xytext=(6, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=7.0,
                    color=color,
                    alpha=alpha,
                )
            ax.axvline(100.0 * truth_line, color="#2C2C2A", lw=1.2, ls="--")
            ax.set_yticks(y_pos)
            ax.set_yticklabels([METHOD_LABEL[m] for m in method_order])
            ax.tick_params(axis="y", length=0)
            ax.set_xlim(float(x_values.min()) - xpad, float(x_values.max()) + xpad)
            ax.set_xlabel("5-year risk difference (%)")
            ax.set_title(title, pad=4)
            ax.text(
                0.98,
                0.04,
                f"mean corr(cumulative, H) = {corr_lookup.get(corr_key, np.nan):.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6.7,
                color="#555555",
            )
            ax.xaxis.grid(True)
            ax.yaxis.grid(False)
            despine(ax)
            panel_label(ax, panel)

        bias_matrix = np.column_stack(
            [
                100.0 * mono_df["abs_bias"].to_numpy(dtype=float),
                100.0 * int_df["abs_bias"].to_numpy(dtype=float),
            ]
        )
        im = ax_c.imshow(bias_matrix, cmap=mpl.cm.Blues, aspect="auto")
        ax_c.set_xticks([0, 1])
        ax_c.set_xticklabels(["Monotone", "Intermittent"])
        ax_c.set_yticks(np.arange(len(method_order), dtype=float))
        ax_c.set_yticklabels([METHOD_LABEL[m] for m in method_order])
        ax_c.tick_params(axis="y", length=0)
        ax_c.set_title("Absolute bias by method across scenarios", pad=4)
        for row_idx in range(bias_matrix.shape[0]):
            for col_idx in range(bias_matrix.shape[1]):
                val = bias_matrix[row_idx, col_idx]
                ax_c.text(
                    col_idx,
                    row_idx,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="#0F172A" if val < np.nanmax(bias_matrix) * 0.60 else "white",
                )
        cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.03)
        cbar.set_label("Absolute bias (%)", fontsize=7)
        cbar.ax.tick_params(labelsize=7)
        panel_label(ax_c, "C")

        example_order = ["Early exposure only", "Recent sustained exposure"]
        y_positions = [1.0, 0.0]
        example_colors = {
            "Early exposure only": "#B8C0C8",
            "Recent sustained exposure": METHOD_COLOR["lag_aware_weighted_tte"],
        }
        for y, example_name in zip(y_positions, example_order):
            ex_df = example_histories[example_histories["example_name"] == example_name].copy()
            on_months = ex_df.loc[ex_df["exposure"] > 0.5, "month"].to_numpy(dtype=float)
            for month in on_months:
                ax_d.barh(y, width=0.92, left=month - 0.46, height=0.28, color=example_colors[example_name], edgecolor="none")
            cum60 = 100.0 * float(ex_df["cumulative_60m"].iloc[0])
            h60 = float(ex_df["lagged_60m"].iloc[0])
            ax_d.text(
                61.5,
                y,
                f"Cum. @60 = {cum60:.0f}%\nH_60 = {h60:.2f}",
                ha="left",
                va="center",
                fontsize=7.0,
                color="#333333",
            )
        ax_d.set_xlim(0.5, 72.0)
        ax_d.set_ylim(-0.6, 1.6)
        ax_d.set_xticks([1, 12, 24, 36, 48, 60])
        ax_d.set_xlabel("Month")
        ax_d.set_yticks(y_positions)
        ax_d.set_yticklabels(example_order)
        ax_d.tick_params(axis="y", length=0)
        ax_d.set_title("Same cumulative exposure, different lagged burden", pad=4)
        ax_d.text(
            0.02,
            0.04,
            "Both histories include 12 exposed months by month 60,\n"
            "but remote exposure falls outside most of the 36-month lookback window.",
            transform=ax_d.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.7,
            color="#555555",
        )
        ax_d.xaxis.grid(True)
        ax_d.yaxis.grid(False)
        despine(ax_d)
        panel_label(ax_d, "D")

        save_figure(fig, outdir, "figure1_timing_sensitive_exposure_histories")
        plt.close(fig)


def make_figure2_composite_timing_aware_framework(
    outdir: Path,
    primary_metrics: pd.DataFrame,
    intermittent_metrics: pd.DataFrame,
    intermittent_truth_curves: pd.DataFrame,
    intermittent_curve_summary: pd.DataFrame,
    example_histories: pd.DataFrame,
) -> None:
    method_order = METHOD_ORDER
    primary_df = primary_metrics.set_index("method").loc[method_order].reset_index()
    intermittent_df = intermittent_metrics.set_index("method").loc[method_order].reset_index()
    truth_line_inter = float(intermittent_df["true_rd_60m"].iloc[0])

    with simulation_style():
        fig = plt.figure(figsize=(11.8, 10.0))
        gs = fig.add_gridspec(
            3,
            2,
            height_ratios=[0.95, 1.0, 1.12],
            hspace=0.38,
            wspace=0.28,
            left=0.07,
            right=0.97,
            top=0.95,
            bottom=0.06,
        )
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[1, 0])
        ax_d = fig.add_subplot(gs[1, 1])
        ax_e = fig.add_subplot(gs[2, :])

        example_order = ["Early exposure only", "Recent sustained exposure"]
        y_positions = [1.0, 0.0]
        example_colors = {
            "Early exposure only": "#B8C0C8",
            "Recent sustained exposure": METHOD_COLOR["lag_aware_weighted_tte"],
        }
        lookback_start = 24.5
        lookback_end = 60.5
        ax_a.axvspan(lookback_start, lookback_end, color="#EAF3FB", alpha=0.75, zorder=0)
        ax_a.text(
            42.5,
            1.47,
            "36-month lookback at month 60",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color="#4C6A85",
        )
        for y, example_name in zip(y_positions, example_order):
            ex_df = example_histories[example_histories["example_name"] == example_name].copy()
            on_months = ex_df.loc[ex_df["exposure"] > 0.5, "month"].to_numpy(dtype=float)
            for month in on_months:
                ax_a.barh(y, width=0.92, left=month - 0.46, height=0.28, color=example_colors[example_name], edgecolor="none")
            cum60 = 100.0 * float(ex_df["cumulative_60m"].iloc[0])
            h60 = float(ex_df["lagged_60m"].iloc[0])
            label = (
                f"Cumulative exposure\nat 60 mo: {cum60:.0f}%\n"
                f"Lagged burden\nat 60 mo: {h60:.2f}"
            )
            ax_a.text(
                61.5,
                y,
                label,
                ha="left",
                va="center",
                fontsize=7.0,
                color="#333333",
            )
        ax_a.set_xlim(0.5, 72.0)
        ax_a.set_ylim(-0.6, 1.6)
        ax_a.set_xticks([1, 12, 24, 36, 48, 60])
        ax_a.set_xlabel("Month")
        ax_a.set_yticks(y_positions)
        ax_a.set_yticklabels(example_order)
        ax_a.tick_params(axis="y", length=0)
        ax_a.set_title("Equal cumulative exposure can imply different lagged burden", pad=4)
        ax_a.text(
            0.02,
            0.04,
            "Both histories include 12 exposed months by month 60,\n"
            "but remote exposure falls outside most of the 36-month lookback window.",
            transform=ax_a.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.7,
            color="#555555",
        )
        ax_a.xaxis.grid(True)
        ax_a.yaxis.grid(False)
        despine(ax_a)
        panel_label(ax_a, "A")

        bias_matrix = np.column_stack(
            [
                100.0 * primary_df["abs_bias"].to_numpy(dtype=float),
                100.0 * intermittent_df["abs_bias"].to_numpy(dtype=float),
            ]
        )
        light_blues = mpl.colors.LinearSegmentedColormap.from_list(
            "light_blues",
            ["#F7FBFF", "#DCEAF7", "#9ECAE1", "#3182BD"],
        )
        im = ax_b.imshow(
            bias_matrix,
            cmap=light_blues,
            aspect="auto",
            vmin=0.0,
            vmax=max(6.0, float(np.nanmax(bias_matrix))),
        )
        ax_b.set_xticks([0, 1])
        ax_b.set_xticklabels(["Simple\ndiscontinuation", "Intermittent\ngaps/restarts"])
        ax_b.set_yticks(np.arange(len(method_order), dtype=float))
        ax_b.set_yticklabels([METHOD_LABEL[m] for m in method_order])
        ax_b.tick_params(axis="y", length=0)
        ax_b.set_title("Bias depended on exposure-history complexity", pad=4)
        for row_idx in range(bias_matrix.shape[0]):
            for col_idx in range(bias_matrix.shape[1]):
                val = bias_matrix[row_idx, col_idx]
                ax_b.text(
                    col_idx,
                    row_idx,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    fontweight="semibold",
                    color="#0F172A" if val < np.nanmax(bias_matrix) * 0.60 else "white",
                )
        cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.03)
        cbar.set_label("Absolute bias (%)", fontsize=7)
        cbar.ax.tick_params(labelsize=7)
        panel_label(ax_b, "B")

        y_pos = np.arange(len(method_order))[::-1]
        x_vals = np.concatenate(
            [
                100.0 * intermittent_df["mean_estimate_60m"].to_numpy(dtype=float),
                np.array([100.0 * truth_line_inter]),
            ]
        )
        xpad = max(0.8, 0.08 * (float(x_vals.max()) - float(x_vals.min())))
        for y, row in zip(y_pos, intermittent_df.itertuples(index=False)):
            method_name = str(row.method)
            color = METHOD_COLOR[method_name]
            is_proposed = method_name == "lag_aware_weighted_tte"
            est = 100.0 * float(row.mean_estimate_60m)
            ax_c.plot(est, y, marker="o", color=color, ms=7.2 if is_proposed else 5.3, zorder=3, lw=0, alpha=1.0 if is_proposed else 0.75)
            ax_c.text(
                est + (0.20 if est >= 0 else -0.20),
                y,
                f"{est:.1f}",
                ha="left" if est >= 0 else "right",
                va="center",
                fontsize=7.1,
                color=color,
            )
        ax_c.axvline(100.0 * truth_line_inter, color="#2C2C2A", lw=1.2, ls="--")
        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels([METHOD_LABEL[m] for m in method_order])
        ax_c.tick_params(axis="y", length=0)
        ax_c.set_xlim(float(x_vals.min()) - xpad, float(x_vals.max()) + xpad)
        ax_c.set_xlabel("5-year risk difference (%)")
        ax_c.set_title("Estimated 5-year risk difference under intermittent exposure", pad=4)
        ax_c.text(
            0.02,
            0.03,
            "Negative values favor persistent treatment",
            transform=ax_c.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.8,
            color="#444444",
        )
        ax_c.xaxis.grid(True)
        ax_c.yaxis.grid(False)
        despine(ax_c)
        panel_label(ax_c, "C")

        bias_df = intermittent_df.copy()
        bias_df["signed_bias_pct"] = 100.0 * bias_df["bias"]
        bias_max = float(np.max(np.abs(bias_df["signed_bias_pct"])))
        bias_pad = max(0.8, 0.18 * bias_max)
        for y, row in zip(y_pos, bias_df.itertuples(index=False)):
            method_name = str(row.method)
            color = METHOD_COLOR[method_name]
            is_proposed = method_name == "lag_aware_weighted_tte"
            bias_val = float(row.signed_bias_pct)
            ax_d.barh(
                y,
                bias_val,
                height=0.42,
                color=mpl.colors.to_rgba(color, 0.90 if is_proposed else 0.60),
                edgecolor=color,
                linewidth=1.2 if is_proposed else 0.9,
                zorder=2,
            )
            ax_d.text(
                bias_val + (0.28 if bias_val >= 0 else -0.28),
                y,
                f"{bias_val:+.1f}",
                ha="left" if bias_val >= 0 else "right",
                va="center",
                fontsize=7.1,
                color=color,
            )
        ax_d.set_yticks(y_pos)
        ax_d.set_yticklabels([METHOD_LABEL[m] for m in method_order])
        ax_d.tick_params(axis="y", length=0)
        ax_d.axvline(0.0, color="#2C2C2A", lw=1.1, ls="--")
        ax_d.set_xlim(-(bias_max + bias_pad), bias_max + bias_pad)
        ax_d.set_xlabel("Bias (%)")
        ax_d.set_title("Signed bias under intermittent exposure", pad=4)
        ax_d.text(
            0.02,
            0.97,
            "Exaggerated benefit",
            transform=ax_d.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color="#444444",
        )
        ax_d.text(
            0.98,
            0.97,
            "Toward null",
            transform=ax_d.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color="#444444",
        )
        ax_d.xaxis.grid(True)
        ax_d.yaxis.grid(False)
        despine(ax_d)
        panel_label(ax_d, "D")

        truth_lookup = intermittent_truth_curves.rename(columns={"true_risk": "risk"})
        lag_aware = intermittent_curve_summary[intermittent_curve_summary["method"] == "lag_aware_weighted_tte"].rename(
            columns={"estimated_risk_mean": "risk"}
        )
        for history_name in HISTORY_ORDER:
            true_hist = truth_lookup[truth_lookup["history_name"] == history_name]
            est_hist = lag_aware[lag_aware["history_name"] == history_name]
            color = HISTORY_COLOR[history_name]
            ax_e.plot(true_hist["month"], 100.0 * true_hist["risk"], ls="--", color=color, lw=1.5)
            ax_e.plot(est_hist["month"], 100.0 * est_hist["risk"], ls="-", color=color, lw=2.0)
        ax_e.axvline(12, color="#888888", lw=0.7, ls=":")
        ax_e.text(12.7, 1.25, "Stop at 12 months", fontsize=6.8, color="#666666")
        ax_e.set_xlim(0, 60)
        ax_e.set_xticks([0, 12, 24, 36, 48, 60])
        ax_e.set_xlabel("Month")
        ax_e.set_ylabel("Cumulative risk (%)")
        ax_e.set_title("Recovery of representative risk curves under intermittent exposure", pad=4)
        style_handles = [
            mpl.lines.Line2D([0], [0], color="#333333", lw=1.5, ls="--", label="Truth"),
            mpl.lines.Line2D([0], [0], color="#333333", lw=2.0, ls="-", label="Lag-aware estimate"),
        ]
        history_handles = [
            mpl.lines.Line2D([0], [0], color=HISTORY_COLOR[h], lw=2.2, label=HISTORY_LABEL[h])
            for h in HISTORY_ORDER
        ]
        leg1 = ax_e.legend(handles=style_handles, loc="upper left", fontsize=6.9, frameon=False)
        ax_e.add_artist(leg1)
        ax_e.legend(handles=history_handles, loc="lower right", fontsize=6.9, frameon=False, title="History")
        ax_e.xaxis.grid(True)
        ax_e.yaxis.grid(True)
        despine(ax_e)
        panel_label(ax_e, "E")

        save_figure(fig, outdir, "figure2_composite_timing_aware_framework")
        plt.close(fig)


def write_manifest(outdir: Path, manifest: dict) -> None:
    with (outdir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manuscript-ready simulation figures for delayed cumulative effects.")
    parser.add_argument("--outdir", default=None, help="Output directory for simulation tables and figures.")
    parser.add_argument("--reps", type=int, default=80, help="Main-scenario simulation replicates.")
    parser.add_argument("--n-main", type=int, default=1800, help="Sample size per main-scenario replicate.")
    parser.add_argument("--n-sensitivity", type=int, default=1400, help="Sample size per sensitivity replicate.")
    parser.add_argument("--truth-n", type=int, default=80000, help="Monte Carlo size for truth curves.")
    parser.add_argument("--sensitivity-reps", type=int, default=28, help="Replicates per sensitivity setting.")
    parser.add_argument("--ci-draws", type=int, default=60, help="Coefficient draws per replicate for approximate confidence intervals.")
    parser.add_argument("--seed", type=int, default=20260419, help="Base random seed.")
    parser.add_argument("--figure1-only", action="store_true", help="Only generate Figure 1 and its timing-sensitive inputs.")
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve() if args.outdir else REPO_ROOT / "results" / "generated" / "simulation_manuscript"
    outdir.mkdir(parents=True, exist_ok=True)

    scenario = Scenario()
    kernel_true_spec = KernelSpec(baseline_weight=0.20, accumulation_scale=10.0, max_lag_months=36, shape="soft_accumulation")
    kernel_model_spec = KernelSpec(baseline_weight=0.20, accumulation_scale=10.0, max_lag_months=36, shape="soft_accumulation")

    if args.figure1_only:
        (
            timing_monotone_metrics,
            timing_intermittent_metrics,
            timing_intermittent_truth_curves,
            timing_intermittent_curve_summary,
            timing_correlation,
            timing_example_histories,
        ) = compute_timing_figure_inputs(
            outdir,
            scenario=scenario,
            kernel_true_spec=kernel_true_spec,
            kernel_model_spec=kernel_model_spec,
            reps=max(12, args.sensitivity_reps),
            n_main=args.n_sensitivity,
            truth_n=max(30000, args.truth_n // 3),
            sensitivity_reps=max(6, args.sensitivity_reps // 2),
            n_sensitivity=max(800, args.n_sensitivity // 2),
            n_ci_draws=max(8, args.ci_draws // 3),
            seed=args.seed,
        )
        timing_monotone_metrics.to_csv(outdir / "simulation_timing_monotone_method_metrics.csv", index=False)
        timing_intermittent_metrics.to_csv(outdir / "simulation_timing_intermittent_method_metrics.csv", index=False)
        timing_intermittent_truth_curves.to_csv(outdir / "simulation_timing_intermittent_truth_curves.csv", index=False)
        timing_intermittent_curve_summary.to_csv(outdir / "simulation_timing_intermittent_curve_summary.csv", index=False)
        timing_correlation.to_csv(outdir / "simulation_timing_exposure_correlation.csv", index=False)
        timing_example_histories.to_csv(outdir / "simulation_timing_example_histories.csv", index=False)
        make_figure1_timing_sensitive_exposure_histories(
            outdir,
            timing_monotone_metrics,
            timing_intermittent_metrics,
            timing_correlation,
            timing_example_histories,
        )
        primary_metrics_path = outdir / "simulation_method_summary_metrics.csv"
        primary_truth_path = outdir / "simulation_truth_curves.csv"
        primary_curve_path = outdir / "simulation_curve_summary.csv"
        if primary_metrics_path.exists() and primary_truth_path.exists() and primary_curve_path.exists():
            make_figure2_composite_timing_aware_framework(
                outdir,
                pd.read_csv(primary_metrics_path),
                timing_intermittent_metrics,
                timing_intermittent_truth_curves,
                timing_intermittent_curve_summary,
                timing_example_histories,
            )
        manifest = {
            "outdir": display_path(outdir),
            "seed": args.seed,
            "figure1_only": True,
            "timing_reps": max(12, args.sensitivity_reps),
            "timing_n": args.n_sensitivity,
            "timing_truth_n": max(30000, args.truth_n // 3),
            "generated": sorted(p.name for p in outdir.glob("*")),
        }
        write_manifest(outdir, manifest)
        return

    main_results = compute_main_results(
        outdir,
        reps=args.reps,
        n_main=args.n_main,
        truth_n=args.truth_n,
        kernel_true_spec=kernel_true_spec,
        kernel_model_spec=kernel_model_spec,
        scenario=scenario,
        seed=args.seed,
        n_ci_draws=args.ci_draws,
        scenario_name="primary",
    )

    intermittent_scenario = replace(
        scenario,
        allow_restart=True,
        restart_intercept=-3.2,
        restart_l=-0.4,
    )
    intermittent_results = compute_main_results(
        outdir,
        reps=max(8, args.sensitivity_reps),
        n_main=args.n_sensitivity,
        truth_n=max(30000, args.truth_n // 3),
        kernel_true_spec=kernel_true_spec,
        kernel_model_spec=kernel_model_spec,
        scenario=intermittent_scenario,
        seed=args.seed + 250000,
        n_ci_draws=max(30, args.ci_draws // 2),
        scenario_name="intermittent_restart",
    )

    (
        timing_monotone_metrics,
        timing_intermittent_metrics,
        timing_intermittent_truth_curves,
        timing_intermittent_curve_summary,
        timing_correlation,
        timing_example_histories,
    ) = compute_timing_figure_inputs(
        outdir,
        scenario=scenario,
        kernel_true_spec=kernel_true_spec,
        kernel_model_spec=kernel_model_spec,
        reps=max(12, args.sensitivity_reps),
        n_main=args.n_sensitivity,
        truth_n=max(30000, args.truth_n // 3),
        sensitivity_reps=max(6, args.sensitivity_reps // 2),
        n_sensitivity=max(800, args.n_sensitivity // 2),
        n_ci_draws=max(8, args.ci_draws // 3),
        seed=args.seed,
    )

    discontinuation_grid = [
        (f"disc_{idx+1}", float(intercept), replace(scenario, continuation_intercept=intercept), kernel_model_spec)
        for idx, intercept in enumerate([3.5, 3.3, 3.1, 2.9, 2.7, 2.5, 2.3])
    ]
    discontinuation_sensitivity = run_bias_sensitivity(
        scenario_grid=discontinuation_grid,
        methods_for_plot=METHOD_ORDER,
        reps=args.sensitivity_reps,
        n_main=args.n_sensitivity,
        truth_n=max(40000, args.truth_n // 2),
        kernel_true_spec=kernel_true_spec,
        seed=args.seed + 300000,
        n_ci_draws=max(30, args.ci_draws // 2),
    )

    base_cont_l = float(scenario.continuation_l)
    base_out_l = float(scenario.outcome_l)
    confounding_grid = [
        (
            f"feedback_{scale:.2f}x",
            scale,
            replace(
                scenario,
                continuation_l=base_cont_l * scale,
                outcome_l=base_out_l * scale,
            ),
            kernel_model_spec,
        )
        for scale in [0.50, 0.75, 1.00, 1.25, 1.50]
    ]
    confounding_sensitivity = run_bias_sensitivity(
        scenario_grid=confounding_grid,
        methods_for_plot=METHOD_ORDER,
        reps=args.sensitivity_reps,
        n_main=args.n_sensitivity,
        truth_n=max(40000, args.truth_n // 2),
        kernel_true_spec=kernel_true_spec,
        seed=args.seed + 400000,
        n_ci_draws=max(30, args.ci_draws // 2),
    )

    lag_grid = [
        ("lambda6", 6.0, scenario, KernelSpec(baseline_weight=0.20, accumulation_scale=6.0, max_lag_months=36, shape="soft_accumulation")),
        ("lambda10", 10.0, scenario, KernelSpec(baseline_weight=0.20, accumulation_scale=10.0, max_lag_months=36, shape="soft_accumulation")),
        ("lambda18", 18.0, scenario, KernelSpec(baseline_weight=0.20, accumulation_scale=18.0, max_lag_months=36, shape="soft_accumulation")),
    ]
    lag_metrics = []
    for _, lag_value, _, kernel_spec in lag_grid:
        results = compute_main_results(
            outdir,
            reps=max(8, args.sensitivity_reps),
            n_main=args.n_sensitivity,
            truth_n=max(30000, args.truth_n // 3),
            kernel_true_spec=kernel_true_spec,
            kernel_model_spec=kernel_spec,
            scenario=scenario,
            seed=args.seed + 600000 + int(lag_value),
            n_ci_draws=max(30, args.ci_draws // 2),
        )
        row = results["estimate_metrics"].loc[
            results["estimate_metrics"]["method"] == "lag_aware_weighted_tte"
        ].iloc[0]
        lag_metrics.append(
            {
                "x_value": lag_value,
                "mean_estimate_60m": float(row["mean_estimate_60m"]),
                "mc_interval_low": float(row["mc_interval_low"]),
                "mc_interval_high": float(row["mc_interval_high"]),
                "true_rd_60m": float(row["true_rd_60m"]),
            }
        )
    lag_plot_df = pd.DataFrame(lag_metrics)

    kernel_shape_sensitivity = run_kernel_shape_sensitivity(
        scenario=scenario,
        kernel_true_spec=kernel_true_spec,
        kernel_model_specs=[
            KernelSpec(baseline_weight=0.20, accumulation_scale=10.0, max_lag_months=36, shape="hard_threshold", delay_months=12),
            KernelSpec(baseline_weight=0.20, accumulation_scale=10.0, max_lag_months=36, shape="soft_accumulation"),
            KernelSpec(baseline_weight=0.20, accumulation_scale=10.0, max_lag_months=36, shape="front_loaded"),
        ],
        reps=max(8, args.sensitivity_reps),
        n_main=args.n_sensitivity,
        truth_n=max(30000, args.truth_n // 3),
        seed=args.seed + 700000,
        n_ci_draws=max(30, args.ci_draws // 2),
    )

    main_results["truth_curves"].to_csv(outdir / "simulation_truth_curves.csv", index=False)
    main_results["estimates"].to_csv(outdir / "simulation_method_estimates_by_replicate.csv", index=False)
    main_results["estimate_metrics"].to_csv(outdir / "simulation_method_summary_metrics.csv", index=False)
    main_results["curve_estimates"].to_csv(outdir / "simulation_curve_estimates_by_replicate.csv", index=False)
    main_results["curve_summary"].to_csv(outdir / "simulation_curve_summary.csv", index=False)
    main_results["persistence_summary"].to_csv(outdir / "simulation_persistence_summary.csv", index=False)
    main_results["weight_summary"].to_csv(outdir / "simulation_weight_summary_primary.csv", index=False)
    main_results["kernel_true"].to_csv(outdir / "simulation_true_lag_kernel.csv", index=False)
    main_results["estimate_metrics"][["method", "coverage_pct"]].to_csv(
        outdir / "tableS4_simulation_coverage.csv",
        index=False,
    )
    intermittent_results["estimate_metrics"].to_csv(outdir / "simulation_sensitivity_intermittent_method_metrics.csv", index=False)
    intermittent_results["weight_summary"].to_csv(outdir / "simulation_weight_summary_intermittent.csv", index=False)
    timing_monotone_metrics.to_csv(outdir / "simulation_timing_monotone_method_metrics.csv", index=False)
    timing_intermittent_metrics.to_csv(outdir / "simulation_timing_intermittent_method_metrics.csv", index=False)
    timing_intermittent_truth_curves.to_csv(outdir / "simulation_timing_intermittent_truth_curves.csv", index=False)
    timing_intermittent_curve_summary.to_csv(outdir / "simulation_timing_intermittent_curve_summary.csv", index=False)
    timing_correlation.to_csv(outdir / "simulation_timing_exposure_correlation.csv", index=False)
    timing_example_histories.to_csv(outdir / "simulation_timing_example_histories.csv", index=False)
    discontinuation_sensitivity.to_csv(outdir / "simulation_sensitivity_discontinuation.csv", index=False)
    confounding_sensitivity.to_csv(outdir / "simulation_sensitivity_confounding.csv", index=False)
    lag_plot_df.to_csv(outdir / "simulation_sensitivity_accumulation_scale.csv", index=False)
    lag_plot_df.to_csv(outdir / "simulation_sensitivity_lag_misspecification.csv", index=False)
    kernel_shape_sensitivity.to_csv(outdir / "simulation_sensitivity_kernel_shape_misspecification.csv", index=False)

    make_figure5_simulation_design(
        outdir,
        main_results["kernel_true"],
        main_results["persistence_summary"],
    )
    make_figure6_simulation_results(
        outdir,
        main_results["truth_curves"],
        main_results["estimate_metrics"],
        main_results["curve_summary"],
        discontinuation_sensitivity,
        confounding_sensitivity,
        lag_plot_df,
        kernel_shape_sensitivity,
    )
    make_figureS4_simulation_coverage(outdir, main_results["estimate_metrics"])
    make_figureS5_simulation_sensitivity(
        outdir,
        main_results["estimate_metrics"],
        lag_plot_df,
        kernel_shape_sensitivity,
    )
    make_figureS6_simulation_decomposition(
        outdir,
        main_results["estimate_metrics"],
        intermittent_results["estimate_metrics"],
    )
    make_figureS7_weight_diagnostics(
        outdir,
        pd.concat(
            [main_results["weight_summary"], intermittent_results["weight_summary"]],
            ignore_index=True,
        ),
    )
    make_figure1_timing_sensitive_exposure_histories(
        outdir,
        timing_monotone_metrics,
        timing_intermittent_metrics,
        timing_correlation,
        timing_example_histories,
    )
    make_figure2_composite_timing_aware_framework(
        outdir,
        main_results["estimate_metrics"],
        timing_intermittent_metrics,
        timing_intermittent_truth_curves,
        timing_intermittent_curve_summary,
        timing_example_histories,
    )

    manifest = {
        "outdir": display_path(outdir),
        "seed": args.seed,
        "main_reps": args.reps,
        "main_n": args.n_main,
        "sensitivity_reps": args.sensitivity_reps,
        "sensitivity_n": args.n_sensitivity,
        "truth_n": args.truth_n,
        "kernel_true": kernel_true_spec.__dict__,
        "generated": sorted(p.name for p in outdir.glob("*")),
    }
    write_manifest(outdir, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
