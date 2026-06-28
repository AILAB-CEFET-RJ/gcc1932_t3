"""Modelo de Regressao Linear para redshift fotometrico."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from redshift.utils.modeling import (
    FEATURE_SETS,
    PROCESSED_DATA_DIR,
    TEST_SET_CHOICES,
    TEST_SETS,
    build_metadata,
    dataset_metrics_dir_name,
    eval_artifact_suffix,
    evaluate_predictions,
    figure_output_path,
    get_eval_data,
    get_feature_cols,
    load_processed_split,
    metrics_output_path,
    model_artifacts_dir,
    model_output_path,
    model_figures_dir,
    model_metrics_dir,
    processed_dataset_dir,
    prepare_model_features,
    print_regression_summary,
    resolve_processed_data_dir,
    save_json,
    save_model_artifact,
    validate_feature_columns,
)
from redshift.evaluation.plots import save_redshift_residual_plot
from redshift.evaluation.slices import build_slice_metrics


MODEL_NAME = "linear_regression"
MODEL_LABEL = "LinearRegression"
MODELS_DIR = model_artifacts_dir(MODEL_NAME)
FIGURES_DIR = model_figures_dir(MODEL_NAME)
METRICS_DIR = model_metrics_dir(MODEL_NAME)


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_cols: list[str],
) -> LinearRegression:
    """Treina Regressao Linear usando o conjunto de features informado."""

    validate_feature_columns(X_train, feature_cols)

    model = LinearRegression()
    model.fit(X_train.loc[:, feature_cols], y_train)

    return model


def evaluate_and_save_outputs(
    model: LinearRegression,
    X_eval: pd.DataFrame,
    X_eval_processed: pd.DataFrame,
    y_eval: pd.Series,
    feature_cols: list[str],
    dataset: str,
    feature_set: str,
    eval_split: str,
    test_set: str | None,
    dataset_dir: Path,
    models_dir: Path,
    figures_dir: Path,
    metrics_dir: Path,
    gate_strength: float | None = None,
    preprocessing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Avalia um split e salva modelo, metricas e grafico."""

    validate_feature_columns(X_eval, feature_cols)
    eval_pred = model.predict(X_eval.loc[:, feature_cols])
    artifact_suffix = eval_artifact_suffix(eval_split, test_set)
    gate_suffix = (
        f"_strength{gate_strength:g}"
        if feature_set == "gate_err_manual"
        else ""
    )
    experiment_name = f"{MODEL_NAME}_{feature_set}{gate_suffix}"
    model_params: dict[str, Any] = {"fit_intercept": bool(model.fit_intercept)}
    if feature_set == "gate_err_manual":
        model_params.update(
            {
                "gate_strength": gate_strength,
                "gate_weight": "exp(-gate_strength * error_norm)",
                "gate_error_clipping_percentile": 99,
            }
        )
    metadata = build_metadata(
        experiment_name=experiment_name,
        dataset=dataset,
        model_label=MODEL_LABEL,
        feature_set=feature_set,
        eval_split=eval_split,
        feature_cols=feature_cols,
        model_params=model_params,
        test_set=test_set,
    )
    metrics: dict[str, Any] = {
        **metadata,
        "metrics": evaluate_predictions(y_eval, eval_pred, eval_split),
    }
    if eval_split == "test":
        metrics["slice_metrics"] = build_slice_metrics(
            X_eval_processed=X_eval_processed,
            y_true=y_eval,
            y_pred=eval_pred,
            split_name=eval_split,
            dataset_dir=dataset_dir,
        )

    save_model_artifact(
        model=model,
        feature_cols=feature_cols,
        metadata=metadata,
        path=model_output_path(
            models_dir=models_dir,
            eval_split=eval_split,
            feature_set=feature_set,
            model_name=MODEL_NAME,
            dataset=dataset,
            filename=f"{experiment_name}_{artifact_suffix}.joblib",
        ),
        preprocessing=preprocessing,
    )
    save_json(
        metrics,
        metrics_output_path(
            metrics_dir=metrics_dir,
            eval_split=eval_split,
            feature_set=feature_set,
            model_name=MODEL_NAME,
            dataset=dataset,
            filename=(
                f"{MODEL_NAME}_{dataset_metrics_dir_name(dataset)}_"
                f"{feature_set}{gate_suffix}_{artifact_suffix}_metrics.json"
            ),
        ),
    )
    save_redshift_residual_plot(
        y_true=y_eval,
        y_pred=eval_pred,
        path=figure_output_path(
            figures_dir=figures_dir,
            eval_split=eval_split,
            feature_set=feature_set,
            model_name=MODEL_NAME,
            dataset=dataset,
            filename=f"{experiment_name}_{artifact_suffix}_residuals.png",
        ),
        title=f"{MODEL_LABEL} | {dataset} | {feature_set} | {artifact_suffix}",
    )

    return metrics


def run_experiment(
    dataset: str,
    feature_set: str = "mag",
    eval_split: str = "val",
    test_set: str | None = None,
    gate_strength: float | None = None,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    figures_dir: Path = FIGURES_DIR,
    metrics_dir: Path = METRICS_DIR,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Executa o experimento de Regressao Linear e salva artefatos."""

    processed_data_dir = resolve_processed_data_dir(
        feature_set,
        processed_data_dir,
    )
    dataset_dir = processed_dataset_dir(processed_data_dir, eval_split, dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Pasta de dados processados nao encontrada: {dataset_dir}")

    feature_cols = get_feature_cols(feature_set)
    if eval_split == "test" and test_set == "all":
        X_train_processed = pd.read_csv(dataset_dir / "X_train.csv")
        y_train = pd.read_csv(dataset_dir / "y_train.csv").squeeze("columns")
        X_train, preprocessing = prepare_model_features(
            X_processed=X_train_processed,
            feature_set=feature_set,
            dataset_dir=dataset_dir,
            gate_strength=gate_strength,
        )
        model = train_linear_regression(X_train, y_train, feature_cols)

        results: list[dict[str, Any]] = []
        for split_name in TEST_SETS:
            X_eval_processed = pd.read_csv(
                dataset_dir / f"X_test_{split_name}.csv"
            )
            X_eval, _ = prepare_model_features(
                X_processed=X_eval_processed,
                feature_set=feature_set,
                dataset_dir=dataset_dir,
                gate_strength=gate_strength,
            )
            y_eval = pd.read_csv(dataset_dir / f"y_test_{split_name}.csv").squeeze(
                "columns"
            )
            results.append(
                evaluate_and_save_outputs(
                    model=model,
                    X_eval=X_eval,
                    X_eval_processed=X_eval_processed,
                    y_eval=y_eval,
                    feature_cols=feature_cols,
                    dataset=dataset,
                    feature_set=feature_set,
                    eval_split=eval_split,
                    test_set=split_name,
                    dataset_dir=dataset_dir,
                    models_dir=models_dir,
                    figures_dir=figures_dir,
                    metrics_dir=metrics_dir,
                    gate_strength=gate_strength,
                    preprocessing=preprocessing,
                )
            )
        return results

    test_set_to_load = test_set if eval_split == "test" else None
    (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        y_train,
        y_val,
        y_test,
    ) = load_processed_split(
        dataset_dir,
        test_set=test_set_to_load,
    )
    X_eval_processed, y_eval = get_eval_data(
        eval_split,
        X_val_processed,
        y_val,
        X_test_processed,
        y_test,
    )
    X_train, preprocessing = prepare_model_features(
        X_processed=X_train_processed,
        feature_set=feature_set,
        dataset_dir=dataset_dir,
        gate_strength=gate_strength,
    )
    X_eval, _ = prepare_model_features(
        X_processed=X_eval_processed,
        feature_set=feature_set,
        dataset_dir=dataset_dir,
        gate_strength=gate_strength,
    )

    model = train_linear_regression(X_train, y_train, feature_cols)
    return evaluate_and_save_outputs(
        model=model,
        X_eval=X_eval,
        X_eval_processed=X_eval_processed,
        y_eval=y_eval,
        feature_cols=feature_cols,
        dataset=dataset,
        feature_set=feature_set,
        eval_split=eval_split,
        test_set=test_set_to_load,
        dataset_dir=dataset_dir,
        models_dir=models_dir,
        figures_dir=figures_dir,
        metrics_dir=metrics_dir,
        gate_strength=gate_strength,
        preprocessing=preprocessing,
    )


def parse_args() -> argparse.Namespace:
    """Define argumentos de linha de comando."""

    parser = argparse.ArgumentParser(
        description="Treina Regressao Linear para redshift fotometrico."
    )
    parser.add_argument(
        "--dataset",
        choices=["happyT", "teddyT"],
        default="happyT",
        help="Dataset processado em data/processed/Val ou data/processed/Test.",
    )
    parser.add_argument(
        "--feature-set",
        choices=list(FEATURE_SETS),
        default="mag",
        help=(
            "Conjunto de features: mag, mag_err ou gate_err_manual "
            "(magnitudes ponderadas pelos erros)."
        ),
    )
    parser.add_argument(
        "--gate-strength",
        type=float,
        default=None,
        help=(
            "Intensidade do peso exp(-gate_strength * error_norm) "
            "no gate_err_manual."
        ),
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="val",
        help="Split usado para avaliacao: val para selecao de modelo; test para avaliacao final.",
    )
    parser.add_argument(
        "--test-set",
        choices=TEST_SET_CHOICES,
        default=None,
        help="Conjunto externo usado quando --eval-split test: B, C, D ou all.",
    )
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Diretorio base com subpastas Val e Test.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Diretorio para salvar modelos.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=METRICS_DIR,
        help="Diretorio para salvar metricas.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Diretorio para salvar figuras de avaliacao.",
    )

    return parser.parse_args()


def main() -> None:
    """Executa o modelo pelo terminal."""

    args = parse_args()
    metrics_result = run_experiment(
        dataset=args.dataset,
        feature_set=args.feature_set,
        eval_split=args.eval_split,
        test_set=args.test_set,
        gate_strength=args.gate_strength,
        processed_data_dir=args.processed_data_dir,
        models_dir=args.models_dir,
        figures_dir=args.figures_dir,
        metrics_dir=args.metrics_dir,
    )
    if isinstance(metrics_result, list):
        for metrics in metrics_result:
            print_regression_summary(metrics)
    else:
        print_regression_summary(metrics_result)


if __name__ == "__main__":
    main()
