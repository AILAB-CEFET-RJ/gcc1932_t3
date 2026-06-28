"""Funcoes compartilhadas entre scripts de modelos."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from redshift.data.gate_err import (
    GATE_FEATURE_COLUMNS,
    gate_err_preprocessing_for_artifact,
    make_gate_err_features_from_processed,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
GATE_ERR_PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed Gate-Err"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

MAG_FEATURES = ["u", "g", "r", "i", "z"]
ERR_FEATURES = ["uErr", "gErr", "rErr", "iErr", "zErr"]
FEATURE_SETS = {
    "mag": MAG_FEATURES,
    "mag_err": MAG_FEATURES + ERR_FEATURES,
    "gate_err_manual": list(GATE_FEATURE_COLUMNS),
}
TEST_SETS = ("B", "C", "D")
TEST_SET_CHOICES = (*TEST_SETS, "all")
EVAL_SPLIT_PURPOSES = {
    "val": "model_selection",
    "test": "final_evaluation",
}
EVAL_SPLIT_METRICS_DIRS = {
    "val": "Model Selection",
    "test": "Tests",
}
EVAL_SPLIT_PROCESSED_DIRS = {
    "val": "Val",
    "test": "Test",
}
DATASET_METRICS_DIRS = {
    "happyT": "happy",
    "teddyT": "teddy",
}
TARGET_TRANSFORM = "redshift"
CATASTROPHIC_OUTLIER_THRESHOLD = 0.15


def model_artifacts_dir(model_name: str) -> Path:
    """Retorna a pasta raiz para artefatos de modelos.

    O argumento e mantido para usar a mesma assinatura dos scripts de modelo,
    mas a organizacao final fica:
    models/Model Selection ou Tests/dataset/feature_set/modelo.
    """

    return MODELS_DIR


def model_metrics_dir(model_name: str) -> Path:
    """Retorna a pasta raiz das metricas.

    O argumento e mantido para usar a mesma assinatura dos scripts de modelo,
    mas a organizacao final fica:
    metrics/Model Selection ou Tests/dataset/feature_set/model.
    """

    return METRICS_DIR


def model_figures_dir(model_name: str) -> Path:
    """Retorna a pasta raiz das figuras.

    O argumento e mantido para usar a mesma assinatura dos scripts de modelo,
    mas a organizacao final fica:
    figures/Model Selection ou Tests/dataset/feature_set/model.
    """

    return FIGURES_DIR


def dataset_metrics_dir_name(dataset: str) -> str:
    """Retorna o nome da pasta de metricas para um dataset."""

    return DATASET_METRICS_DIRS.get(dataset, dataset)


def metrics_output_path(
    metrics_dir: Path,
    eval_split: str,
    feature_set: str,
    model_name: str,
    dataset: str,
    filename: str,
) -> Path:
    """Monta o caminho padrao de metricas para um experimento."""

    if eval_split not in EVAL_SPLIT_METRICS_DIRS:
        available = ", ".join(EVAL_SPLIT_METRICS_DIRS)
        raise ValueError(f"Split de avaliacao invalido: {eval_split}. Opcoes: {available}")

    return (
        metrics_dir
        / EVAL_SPLIT_METRICS_DIRS[eval_split]
        / dataset_metrics_dir_name(dataset)
        / feature_set
        / model_name
        / filename
    )


def table_output_path(
    tables_dir: Path,
    eval_split: str,
    feature_set: str,
    model_name: str,
    dataset: str,
    filename: str,
) -> Path:
    """Monta o caminho padrao de tabelas para uma busca/experimento."""

    if eval_split not in EVAL_SPLIT_METRICS_DIRS:
        available = ", ".join(EVAL_SPLIT_METRICS_DIRS)
        raise ValueError(f"Split de avaliacao invalido: {eval_split}. Opcoes: {available}")

    return (
        tables_dir
        / EVAL_SPLIT_METRICS_DIRS[eval_split]
        / dataset_metrics_dir_name(dataset)
        / feature_set
        / model_name
        / filename
    )


def model_output_path(
    models_dir: Path,
    eval_split: str,
    feature_set: str,
    model_name: str,
    dataset: str,
    filename: str,
) -> Path:
    """Monta o caminho padrao de artefatos para um experimento."""

    if eval_split not in EVAL_SPLIT_METRICS_DIRS:
        available = ", ".join(EVAL_SPLIT_METRICS_DIRS)
        raise ValueError(f"Split de avaliacao invalido: {eval_split}. Opcoes: {available}")

    return (
        models_dir
        / EVAL_SPLIT_METRICS_DIRS[eval_split]
        / dataset_metrics_dir_name(dataset)
        / feature_set
        / model_name
        / filename
    )


def figure_output_path(
    figures_dir: Path,
    eval_split: str,
    feature_set: str,
    model_name: str,
    dataset: str,
    filename: str,
) -> Path:
    """Monta o caminho padrao de figuras para um experimento."""

    if eval_split not in EVAL_SPLIT_METRICS_DIRS:
        available = ", ".join(EVAL_SPLIT_METRICS_DIRS)
        raise ValueError(f"Split de avaliacao invalido: {eval_split}. Opcoes: {available}")

    return (
        figures_dir
        / EVAL_SPLIT_METRICS_DIRS[eval_split]
        / dataset_metrics_dir_name(dataset)
        / feature_set
        / model_name
        / filename
    )


def processed_dataset_dir(
    processed_data_dir: Path,
    eval_split: str,
    dataset: str,
) -> Path:
    """Retorna a pasta processada correta para val ou test."""

    if eval_split not in EVAL_SPLIT_PROCESSED_DIRS:
        available = ", ".join(EVAL_SPLIT_PROCESSED_DIRS)
        raise ValueError(f"Split de avaliacao invalido: {eval_split}. Opcoes: {available}")

    return processed_data_dir / EVAL_SPLIT_PROCESSED_DIRS[eval_split] / dataset


def resolve_processed_data_dir(
    feature_set: str,
    processed_data_dir: Path,
) -> Path:
    """Usa a raiz isolada do GATE-ERR quando nenhuma raiz customizada foi dada."""

    if (
        feature_set == "gate_err_manual"
        and Path(processed_data_dir) == PROCESSED_DATA_DIR
    ):
        return GATE_ERR_PROCESSED_DATA_DIR

    return Path(processed_data_dir)


def read_optional_csv(path: Path) -> pd.DataFrame:
    """Le um CSV se existir; caso contrario retorna um dataframe vazio."""

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def read_optional_series(path: Path) -> pd.Series:
    """Le uma serie CSV se existir e nao estiver vazia."""

    if not path.exists():
        return pd.Series(dtype=float, name="redshift")

    try:
        return pd.read_csv(path).squeeze("columns")
    except EmptyDataError:
        return pd.Series(dtype=float, name="redshift")


def validate_test_set(test_set: str | None) -> str:
    """Valida o conjunto externo usado na avaliacao final."""

    if test_set not in TEST_SETS:
        available = ", ".join(TEST_SETS)
        raise ValueError(f"Test set invalido: {test_set}. Opcoes: {available}")

    return test_set


def load_processed_split(
    dataset_dir: Path,
    test_set: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Carrega os splits processados de um dataset."""

    X_train = pd.read_csv(dataset_dir / "X_train.csv")
    X_val = read_optional_csv(dataset_dir / "X_val.csv")
    if test_set is None:
        X_test = read_optional_csv(dataset_dir / "X_test.csv")
    else:
        test_set = validate_test_set(test_set)
        X_test = read_optional_csv(dataset_dir / f"X_test_{test_set}.csv")
    y_train = pd.read_csv(dataset_dir / "y_train.csv").squeeze("columns")
    y_val = read_optional_series(dataset_dir / "y_val.csv")
    if test_set is None:
        y_test = read_optional_series(dataset_dir / "y_test.csv")
    else:
        y_test = read_optional_series(dataset_dir / f"y_test_{test_set}.csv")

    return X_train, X_val, X_test, y_train, y_val, y_test


def validate_feature_columns(X: pd.DataFrame, feature_cols: list[str]) -> None:
    """Garante que as features esperadas existem no dataframe."""

    missing = [col for col in feature_cols if col not in X.columns]
    if missing:
        raise ValueError(f"Features ausentes: {', '.join(missing)}")


def get_feature_cols(feature_set: str) -> list[str]:
    """Retorna as colunas do conjunto de features escolhido."""

    if feature_set not in FEATURE_SETS:
        available = ", ".join(FEATURE_SETS)
        raise ValueError(f"Feature set invalido: {feature_set}. Opcoes: {available}")

    return FEATURE_SETS[feature_set]


def prepare_model_features(
    X_processed: pd.DataFrame,
    feature_set: str,
    dataset_dir: Path,
    gate_strength: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Prepara a matriz final recebida pelo estimador."""

    feature_cols = get_feature_cols(feature_set)
    if feature_set != "gate_err_manual":
        validate_feature_columns(X_processed, feature_cols)
        return X_processed.loc[:, feature_cols].copy(), None

    if gate_strength is None:
        raise ValueError(
            "--gate-strength e obrigatorio para o feature set gate_err_manual."
        )

    preprocessors = joblib.load(dataset_dir / "preprocessors.joblib")
    X_gate = make_gate_err_features_from_processed(
        X_processed=X_processed,
        preprocessors=preprocessors,
        gate_strength=gate_strength,
    )
    gate_preprocessing = gate_err_preprocessing_for_artifact(
        preprocessors=preprocessors,
        gate_strength=gate_strength,
    )
    return X_gate, gate_preprocessing


def get_eval_data(
    eval_split: str,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Seleciona o split usado para avaliacao."""

    eval_data = {
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    if eval_split not in eval_data:
        available = ", ".join(eval_data)
        raise ValueError(f"Split de avaliacao invalido: {eval_split}. Opcoes: {available}")

    X_eval, y_eval = eval_data[eval_split]
    if X_eval.empty or y_eval.empty:
        raise ValueError(
            f"Split de avaliacao '{eval_split}' esta vazio ou incompleto. "
            "Verifique os arquivos processados desse dataset."
        )

    return X_eval, y_eval


def normalized_median_abs_deviation(delta_z: np.ndarray) -> float:
    """NMAD = 1.4826 * mediana(|dz - mediana(dz)|).

    Dispersao robusta usada em photo-z. O fator 1.4826 faz o NMAD coincidir com
    o desvio-padrao caso os dados fossem Gaussianos, mas sem ser afetado pelas
    caudas pesadas / outliers catastroficos de dz.
    """

    delta_z = np.asarray(delta_z)
    median = np.median(delta_z)
    return float(1.4826 * np.median(np.abs(delta_z - median)))


def bootstrap_standard_error(
    values: np.ndarray,
    statistic,
    n_boot: int = 500,
    seed: int = 42,
) -> float:
    """Erro-padrao de uma estatistica via bootstrap (reamostragem com reposicao).

    Usado para o NMAD, cujo erro-padrao nao tem formula fechada simples sob a
    distribuicao de cauda pesada de dz. Seed fixa para reprodutibilidade.
    """

    values = np.asarray(values)
    n = len(values)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        estimates[b] = statistic(sample)

    return float(np.std(estimates, ddof=1))


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    prefix: str,
) -> dict[str, float | int]:
    """Calcula metricas basicas de regressao."""

    y_true_array = np.asarray(y_true)
    residuals = y_pred - y_true_array

    return {
        f"{prefix}_mae": float(mean_absolute_error(y_true_array, y_pred)),
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred))),
        f"{prefix}_r2": float(r2_score(y_true_array, y_pred)),
        f"{prefix}_bias": float(np.mean(residuals)),
        f"{prefix}_n": int(len(y_true_array)),
    }


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    split_name: str,
) -> dict[str, float | int]:
    """Avalia predicoes na escala original de redshift."""

    y_true_redshift = np.asarray(y_true)
    y_pred_redshift = np.asarray(y_pred)
    metrics = regression_metrics(y_true_redshift, y_pred_redshift, split_name)

    delta_z = (y_pred_redshift - y_true_redshift) / (1 + y_true_redshift)
    normalized_abs_error = np.abs(delta_z)
    catastrophic_outliers = (
        normalized_abs_error > CATASTROPHIC_OUTLIER_THRESHOLD
    )
    metrics[f"{split_name}_negative_redshift_predictions"] = int(np.sum(y_pred_redshift < 0))
    metrics[f"{split_name}_catastrophic_outlier_fraction"] = float(
        np.mean(catastrophic_outliers)
    )
    metrics[f"{split_name}_catastrophic_outlier_threshold"] = (
        CATASTROPHIC_OUTLIER_THRESHOLD
    )
    metrics[f"{split_name}_nmad"] = normalized_median_abs_deviation(delta_z)
    metrics[f"{split_name}_nmad_se"] = bootstrap_standard_error(
        delta_z, normalized_median_abs_deviation
    )

    return metrics


def build_metadata(
    experiment_name: str,
    dataset: str,
    model_label: str,
    feature_set: str,
    eval_split: str,
    feature_cols: list[str],
    model_params: dict[str, Any],
    test_set: str | None = None,
) -> dict[str, Any]:
    """Monta metadados reprodutiveis do experimento."""

    metadata = {
        "experiment_name": experiment_name,
        "dataset": dataset,
        "model": model_label,
        "feature_set": feature_set,
        "eval_split": eval_split,
        "evaluation_purpose": EVAL_SPLIT_PURPOSES[eval_split],
        "target": TARGET_TRANSFORM,
        "features": feature_cols,
        "n_features": len(feature_cols),
        "model_params": model_params,
        "created_at_utc": datetime.now(timezone.utc).strftime("%H:%M"),
    }
    if test_set is not None:
        metadata["test_set"] = validate_test_set(test_set)

    return metadata


def eval_artifact_suffix(eval_split: str, test_set: str | None = None) -> str:
    """Retorna o sufixo de arquivo para validacao ou teste externo."""

    if eval_split != "test":
        return eval_split

    if test_set is None:
        available = ", ".join(TEST_SETS)
        raise ValueError(
            f"--test-set e obrigatorio quando --eval-split test. Opcoes: {available}"
        )

    return f"test_{validate_test_set(test_set)}"


def save_json(data: dict[str, Any], path: Path) -> None:
    """Salva dicionario em JSON com indentacao."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_model_artifact(
    model: Any,
    feature_cols: list[str],
    metadata: dict[str, Any],
    path: Path,
    preprocessing: dict[str, Any] | None = None,
) -> None:
    """Salva o modelo junto com colunas e metadados."""

    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target_transform": TARGET_TRANSFORM,
        "metadata": metadata,
    }
    if preprocessing is not None:
        artifact["preprocessing"] = preprocessing

    joblib.dump(artifact, path)


def print_regression_summary(metrics: dict[str, Any]) -> None:
    """Mostra um resumo curto do experimento no terminal."""

    eval_split = metrics["eval_split"]
    mae_key = f"{eval_split}_mae"
    rmse_key = f"{eval_split}_rmse"

    print(f"Experimento: {metrics['experiment_name']}")
    print(f"Dataset: {metrics['dataset']}")
    print(f"Feature set: {metrics['feature_set']}")
    print(f"Eval split: {metrics['eval_split']} ({metrics['evaluation_purpose']})")
    if metrics.get("test_set") is not None:
        print(f"Test set: {metrics['test_set']}")
    print(f"Features: {', '.join(metrics['features'])}")
    print(f"{eval_split} MAE redshift: {metrics['metrics'][mae_key]:.6f}")
    print(f"{eval_split} RMSE redshift: {metrics['metrics'][rmse_key]:.6f}")
