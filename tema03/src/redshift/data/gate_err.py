"""Pre-processamento das features do baseline GATE-ERR manual."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


MAGNITUDE_COLUMNS: tuple[str, ...] = ("u", "g", "r", "i", "z")
MAGNITUDE_ERROR_COLUMNS: tuple[str, ...] = (
    "uErr",
    "gErr",
    "rErr",
    "iErr",
    "zErr",
)
GATE_FEATURE_COLUMNS: tuple[str, ...] = (
    "u_gate",
    "g_gate",
    "r_gate",
    "i_gate",
    "z_gate",
)
MAGNITUDE_SCALED_COLUMNS: tuple[str, ...] = tuple(
    f"{column}_scaled" for column in MAGNITUDE_COLUMNS
)
ERROR_NORMALIZED_COLUMNS: tuple[str, ...] = tuple(
    f"{column}_norm" for column in MAGNITUDE_ERROR_COLUMNS
)
GATE_ERROR_PERCENTILE = 99.0


def make_gate_err_features(
    magnitudes_scaled: pd.DataFrame | np.ndarray,
    errors_norm: pd.DataFrame | np.ndarray,
    gate_strength: float,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Combina magnitudes escaladas e pesos derivados dos erros."""

    if gate_strength < 0:
        raise ValueError("gate_strength deve ser maior ou igual a zero.")

    magnitude_values = np.asarray(magnitudes_scaled, dtype=float)
    error_values = np.asarray(errors_norm, dtype=float)
    if magnitude_values.shape != error_values.shape:
        raise ValueError(
            "Magnitudes e erros normalizados devem ter o mesmo formato no GATE-ERR."
        )

    error_values = np.clip(error_values, 0.0, 1.0)
    weights = np.exp(-gate_strength * error_values)
    gate_values = magnitude_values * weights

    if index is None and isinstance(magnitudes_scaled, pd.DataFrame):
        index = magnitudes_scaled.index

    return pd.DataFrame(
        gate_values,
        columns=GATE_FEATURE_COLUMNS,
        index=index,
    )


def fit_gate_err_preprocess_train(
    X_train: pd.DataFrame,
    gate_strength: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ajusta os transformadores do GATE-ERR exclusivamente no treino."""

    _validate_gate_input_columns(X_train)
    _validate_photometric_errors(X_train.loc[:, MAGNITUDE_ERROR_COLUMNS])

    magnitude_scaler = StandardScaler()
    magnitude_values = magnitude_scaler.fit_transform(
        X_train.loc[:, MAGNITUDE_COLUMNS]
    )
    magnitudes_scaled = pd.DataFrame(
        magnitude_values,
        columns=MAGNITUDE_SCALED_COLUMNS,
        index=X_train.index,
    )

    errors_log = np.log1p(X_train.loc[:, MAGNITUDE_ERROR_COLUMNS])
    error_gate_p99 = errors_log.quantile(GATE_ERROR_PERCENTILE / 100.0)
    errors_log_clipped = errors_log.clip(upper=error_gate_p99, axis="columns")

    error_gate_minmax_scaler = MinMaxScaler()
    errors_norm = pd.DataFrame(
        np.clip(
            error_gate_minmax_scaler.fit_transform(errors_log_clipped),
            0.0,
            1.0,
        ),
        columns=ERROR_NORMALIZED_COLUMNS,
        index=X_train.index,
    )

    preprocessors: dict[str, Any] = {
        "magnitude_scaler": magnitude_scaler,
        "error_gate_minmax_scaler": error_gate_minmax_scaler,
        "error_gate_p99": error_gate_p99,
        "gate_strength": gate_strength,
        "gate_feature_columns": GATE_FEATURE_COLUMNS,
        "magnitude_columns": MAGNITUDE_COLUMNS,
        "magnitude_error_columns": MAGNITUDE_ERROR_COLUMNS,
        "magnitude_scaled_columns": MAGNITUDE_SCALED_COLUMNS,
        "error_normalized_columns": ERROR_NORMALIZED_COLUMNS,
        "processed_feature_format": "gate_err_intermediate",
    }
    intermediate = pd.concat([magnitudes_scaled, errors_norm], axis="columns")
    return intermediate, preprocessors


def transform_gate_err_intermediate(
    X: pd.DataFrame,
    preprocessors: dict[str, Any],
) -> pd.DataFrame:
    """Transforma dados brutos nos componentes anteriores ao gate."""

    _validate_gate_input_columns(X)
    _validate_photometric_errors(X.loc[:, MAGNITUDE_ERROR_COLUMNS])
    _validate_fitted_gate_preprocessors(preprocessors)

    magnitudes_scaled = pd.DataFrame(
        preprocessors["magnitude_scaler"].transform(
            X.loc[:, MAGNITUDE_COLUMNS]
        ),
        columns=MAGNITUDE_SCALED_COLUMNS,
        index=X.index,
    )
    errors_log = np.log1p(X.loc[:, MAGNITUDE_ERROR_COLUMNS])
    error_gate_p99 = pd.Series(
        preprocessors["error_gate_p99"],
        index=MAGNITUDE_ERROR_COLUMNS,
    )
    errors_log_clipped = errors_log.clip(upper=error_gate_p99, axis="columns")
    errors_norm = pd.DataFrame(
        np.clip(
            preprocessors["error_gate_minmax_scaler"].transform(
                errors_log_clipped
            ),
            0.0,
            1.0,
        ),
        columns=ERROR_NORMALIZED_COLUMNS,
        index=X.index,
    )
    return pd.concat([magnitudes_scaled, errors_norm], axis="columns")


def transform_gate_err_features(
    X: pd.DataFrame,
    preprocessors: dict[str, Any],
    gate_strength: float | None = None,
) -> pd.DataFrame:
    """Transforma features brutas com parâmetros ajustados no treino."""

    resolved_gate_strength = _resolve_gate_strength(
        preprocessors,
        gate_strength,
    )
    intermediate = transform_gate_err_intermediate(X, preprocessors)
    return make_gate_err_features_from_processed(
        X_processed=intermediate,
        preprocessors=preprocessors,
        gate_strength=resolved_gate_strength,
    )


def make_gate_err_features_from_processed(
    X_processed: pd.DataFrame,
    preprocessors: dict[str, Any],
    gate_strength: float,
) -> pd.DataFrame:
    """Cria o gate diretamente a partir da base intermediaria GATE-ERR."""

    _validate_intermediate_columns(X_processed)
    _validate_fitted_gate_preprocessors(preprocessors)

    return make_gate_err_features(
        magnitudes_scaled=X_processed.loc[:, MAGNITUDE_SCALED_COLUMNS],
        errors_norm=X_processed.loc[:, ERROR_NORMALIZED_COLUMNS],
        gate_strength=gate_strength,
    )


def gate_err_preprocessing_for_artifact(
    preprocessors: dict[str, Any],
    gate_strength: float,
) -> dict[str, Any]:
    """Seleciona os parâmetros necessários para reproduzir o GATE-ERR."""

    _validate_fitted_gate_preprocessors(preprocessors)
    return {
        "magnitude_scaler": preprocessors["magnitude_scaler"],
        "error_gate_minmax_scaler": preprocessors["error_gate_minmax_scaler"],
        "error_gate_p99": preprocessors["error_gate_p99"],
        "gate_strength": float(gate_strength),
        "gate_feature_columns": tuple(preprocessors["gate_feature_columns"]),
        "magnitude_columns": tuple(preprocessors["magnitude_columns"]),
        "magnitude_error_columns": tuple(
            preprocessors["magnitude_error_columns"]
        ),
        "magnitude_scaled_columns": tuple(
            preprocessors["magnitude_scaled_columns"]
        ),
        "error_normalized_columns": tuple(
            preprocessors["error_normalized_columns"]
        ),
    }


def _resolve_gate_strength(
    preprocessors: dict[str, Any],
    gate_strength: float | None,
) -> float:
    resolved = gate_strength
    if resolved is None:
        resolved = preprocessors.get("gate_strength")
    if resolved is None:
        # Compatibilidade com preprocessors gerados antes da renomeacao.
        resolved = preprocessors.get("alpha")
    if resolved is None:
        raise ValueError(
            "Informe gate_strength para construir as features GATE-ERR."
        )
    return float(resolved)


def _validate_gate_input_columns(X: pd.DataFrame) -> None:
    required = MAGNITUDE_COLUMNS + MAGNITUDE_ERROR_COLUMNS
    missing = [column for column in required if column not in X.columns]
    if missing:
        raise ValueError(f"Colunas ausentes para GATE-ERR: {', '.join(missing)}")


def _validate_photometric_errors(errors: pd.DataFrame) -> None:
    values = errors.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Erros fotometricos devem conter apenas valores finitos.")
    if (values < 0).any():
        raise ValueError("Erros fotometricos negativos nao sao validos no GATE-ERR.")


def _validate_intermediate_columns(X: pd.DataFrame) -> None:
    required = MAGNITUDE_SCALED_COLUMNS + ERROR_NORMALIZED_COLUMNS
    missing = [column for column in required if column not in X.columns]
    if missing:
        raise ValueError(
            f"Colunas intermediarias GATE-ERR ausentes: {', '.join(missing)}"
        )


def _validate_fitted_gate_preprocessors(preprocessors: dict[str, Any]) -> None:
    required = {
        "magnitude_scaler",
        "error_gate_minmax_scaler",
        "error_gate_p99",
        "gate_feature_columns",
        "magnitude_columns",
        "magnitude_error_columns",
        "magnitude_scaled_columns",
        "error_normalized_columns",
    }
    missing = sorted(required.difference(preprocessors))
    if missing:
        raise ValueError(
            "Preprocessamento GATE-ERR ausente. Rode preprocess_gate_err.py. "
            f"Chaves ausentes: {', '.join(missing)}"
        )
