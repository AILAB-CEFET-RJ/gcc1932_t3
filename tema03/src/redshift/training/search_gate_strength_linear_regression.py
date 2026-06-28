"""Busca da intensidade do GATE-ERR para Regressao Linear em validacao."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from redshift.models.linear_regression import (  # noqa: E402
    METRICS_DIR,
    MODEL_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    run_experiment,
)
from redshift.utils.modeling import TABLES_DIR, table_output_path  # noqa: E402


FEATURE_SET = "gate_err_manual"
DEFAULT_GATE_STRENGTHS = [0.1, 0.5, 1.0, 5.0]


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Converte as metricas de uma execucao em uma linha da busca."""

    split_metrics = metrics["metrics"]
    gate_strength = metrics["model_params"]["gate_strength"]
    return {
        "experiment_name": metrics["experiment_name"],
        "dataset": metrics["dataset"],
        "feature_set": metrics["feature_set"],
        "eval_split": metrics["eval_split"],
        "gate_strength": gate_strength,
        "mae": split_metrics["val_mae"],
        "rmse": split_metrics["val_rmse"],
        "nmad": split_metrics["val_nmad"],
        "nmad_se": split_metrics["val_nmad_se"],
        "r2": split_metrics["val_r2"],
        "bias": split_metrics["val_bias"],
        "n": split_metrics["val_n"],
        "catastrophic_outlier_fraction": split_metrics[
            "val_catastrophic_outlier_fraction"
        ],
        "negative_redshift_predictions": split_metrics[
            "val_negative_redshift_predictions"
        ],
    }


def run_search(
    dataset: str,
    gate_strengths: list[float] | None = None,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    metrics_dir: Path = METRICS_DIR,
    tables_dir: Path = TABLES_DIR,
) -> pd.DataFrame:
    """Testa os alphas do gate em train e compara apenas em validation."""

    gate_strengths = gate_strengths or DEFAULT_GATE_STRENGTHS
    rows = []
    for gate_strength in gate_strengths:
        metrics = run_experiment(
            dataset=dataset,
            feature_set=FEATURE_SET,
            eval_split="val",
            gate_strength=gate_strength,
            processed_data_dir=processed_data_dir,
            models_dir=models_dir,
            metrics_dir=metrics_dir,
        )
        rows.append(summarize_metrics(metrics))

    results = pd.DataFrame(rows).sort_values(
        by=["nmad", "mae", "gate_strength"],
        ascending=[True, True, True],
    )
    results["selected"] = False
    results.loc[results.index[0], "selected"] = True
    results = results.reset_index(drop=True)

    output_path = table_output_path(
        tables_dir=tables_dir,
        eval_split="val",
        feature_set=FEATURE_SET,
        model_name=MODEL_NAME,
        dataset=dataset,
        filename=f"{MODEL_NAME}_{FEATURE_SET}_val_search.csv",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Seleciona a intensidade do GATE-ERR para Regressao Linear."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["happyT", "teddyT"],
        default="happyT",
    )
    parser.add_argument(
        "--gate-strengths",
        type=float,
        nargs="+",
        default=DEFAULT_GATE_STRENGTHS,
        help="Intensidades testadas somente no split de validacao.",
    )
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--metrics-dir", type=Path, default=METRICS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=TABLES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_search(
        dataset=args.dataset,
        gate_strengths=args.gate_strengths,
        processed_data_dir=args.processed_data_dir,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir,
        tables_dir=args.tables_dir,
    )
    selected = results.loc[results["selected"]].iloc[0]
    print(
        f"Melhor gate_strength={selected['gate_strength']:g} | "
        f"NMAD={selected['nmad']:.6f} | "
        f"RMSE={selected['rmse']:.6f} | "
        f"MAE={selected['mae']:.6f}"
    )


if __name__ == "__main__":
    main()
