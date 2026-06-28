"""Metricas de avaliacao por faixas de propriedades observacionais."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from redshift.data.gate_err import (
    ERROR_NORMALIZED_COLUMNS,
    MAGNITUDE_SCALED_COLUMNS,
)
from redshift.utils.modeling import (
    ERR_FEATURES,
    MAG_FEATURES,
    normalized_median_abs_deviation,
)


DEFAULT_REDSHIFT_EDGES: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4, 0.6, np.inf)
DEFAULT_N_QUANTILES = 4
MAGNITUDE_SLICE_COLUMN = "r"


def build_slice_metrics(
    X_eval_processed: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    split_name: str,
    dataset_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Calcula metricas por redshift, magnitude r e erro fotometrico medio.

    As features salvas em data/processed estao escaladas. Para criar faixas
    interpretaveis de magnitude e erro fotometrico, as colunas sao restauradas
    para a escala original usando preprocessors.joblib.
    """

    X_eval_original = restore_original_features(X_eval_processed, dataset_dir)
    y_true_redshift = np.asarray(y_true)
    y_pred_redshift = np.asarray(y_pred)
    photometric_error = X_eval_original.loc[:, ERR_FEATURES].mean(axis=1)

    return {
        "by_redshift": evaluate_fixed_bins(
            values=y_true_redshift,
            y_true=y_true_redshift,
            y_pred=y_pred_redshift,
            split_name=split_name,
            slice_name="redshift",
            value_label="z",
            edges=DEFAULT_REDSHIFT_EDGES,
        ),
        "by_r_magnitude": evaluate_quantile_bins(
            values=X_eval_original[MAGNITUDE_SLICE_COLUMN],
            y_true=y_true_redshift,
            y_pred=y_pred_redshift,
            split_name=split_name,
            slice_name="r_magnitude",
        ),
        "by_photometric_error": evaluate_quantile_bins(
            values=photometric_error,
            y_true=y_true_redshift,
            y_pred=y_pred_redshift,
            split_name=split_name,
            slice_name="mean_photometric_error",
        ),
    }


def restore_original_features(
    X_processed: pd.DataFrame,
    dataset_dir: Path,
) -> pd.DataFrame:
    """Desfaz o pre-processamento de features para diagnosticos por faixa."""

    preprocessors = joblib.load(dataset_dir / "preprocessors.joblib")
    magnitude_scaler = preprocessors["magnitude_scaler"]

    if preprocessors.get("processed_feature_format") == "gate_err_intermediate":
        X_original = pd.DataFrame(index=X_processed.index)
        X_original.loc[:, MAG_FEATURES] = magnitude_scaler.inverse_transform(
            X_processed.loc[:, MAGNITUDE_SCALED_COLUMNS]
        )
        log_error_values = preprocessors[
            "error_gate_minmax_scaler"
        ].inverse_transform(
            X_processed.loc[:, ERROR_NORMALIZED_COLUMNS].to_numpy()
        )
        X_original.loc[:, ERR_FEATURES] = np.expm1(log_error_values)
        return X_original

    magnitude_error_scaler = preprocessors["magnitude_error_scaler"]
    X_original = X_processed.copy()
    X_original.loc[:, MAG_FEATURES] = magnitude_scaler.inverse_transform(
        X_original.loc[:, MAG_FEATURES]
    )
    log_error_values = magnitude_error_scaler.inverse_transform(
        X_original.loc[:, ERR_FEATURES]
    )
    X_original.loc[:, ERR_FEATURES] = np.expm1(log_error_values)

    return X_original


def evaluate_fixed_bins(
    values: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    split_name: str,
    slice_name: str,
    value_label: str,
    edges: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Calcula metricas em faixas fixas."""

    labels = [
        f"{value_label} {format_interval_start(edges[i])}..{format_interval_end(edges[i + 1])}"
        for i in range(len(edges) - 1)
    ]
    bins = pd.cut(
        np.asarray(values),
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    return evaluate_binned_metrics(bins, y_true, y_pred, split_name, slice_name)


def evaluate_quantile_bins(
    values: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    split_name: str,
    slice_name: str,
    n_quantiles: int = DEFAULT_N_QUANTILES,
) -> list[dict[str, Any]]:
    """Calcula metricas por quantis de uma variavel observacional."""

    bins = pd.qcut(values, q=n_quantiles, duplicates="drop")
    return evaluate_binned_metrics(bins, y_true, y_pred, split_name, slice_name)


def evaluate_binned_metrics(
    bins: pd.Series | pd.Categorical,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    split_name: str,
    slice_name: str,
) -> list[dict[str, Any]]:
    """Calcula as metricas pedidas para cada faixa nao vazia."""

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    bins_array = np.asarray(bins)
    rows: list[dict[str, Any]] = []

    for bin_order, bin_label in enumerate(pd.Categorical(bins).categories):
        mask = bins_array == bin_label
        if not np.any(mask):
            continue

        rows.append(
            {
                "slice": slice_name,
                "bin": str(bin_label),
                "bin_order": bin_order,
                f"{split_name}_n": int(np.sum(mask)),
                **compute_slice_metrics(
                    y_true_array[mask],
                    y_pred_array[mask],
                    split_name,
                ),
            }
        )

    return rows


def compute_slice_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str,
) -> dict[str, float | int]:
    """Calcula MAE, RMSE, bias, predicoes negativas e NMAD em uma faixa."""

    residuals = y_pred - y_true
    delta_z = residuals / (1 + y_true)

    return {
        f"{split_name}_mae": float(np.mean(np.abs(residuals))),
        f"{split_name}_rmse": float(np.sqrt(np.mean(residuals**2))),
        f"{split_name}_bias": float(np.mean(residuals)),
        f"{split_name}_negative_redshift_predictions": int(np.sum(y_pred < 0)),
        f"{split_name}_nmad": normalized_median_abs_deviation(delta_z),
    }


def format_interval_start(value: float) -> str:
    """Formata o limite inferior de uma faixa."""

    if np.isneginf(value):
        return "-inf"
    return f"{value:g}"


def format_interval_end(value: float) -> str:
    """Formata o limite superior de uma faixa."""

    if np.isposinf(value):
        return "inf"
    return f"{value:g}"
