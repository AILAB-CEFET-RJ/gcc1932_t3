""" Busca em grade para Polynomial Ridge no split de validacao. Faz uma busca em grade de hiperparametros (grau e alpha)
    para o modelo Polynomial Ridge, usando o split de validacao. Salva uma tabela resumo com as metricas de cada combinacao
    testada."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from redshift.models.polynomial_ridge import (  # noqa: E402
    METRICS_DIR,
    MODEL_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    run_experiment,
)
from redshift.utils.modeling import (  # noqa: E402
    FEATURE_SETS,
    TABLES_DIR,
    table_output_path,
)


DEFAULT_DEGREES = [1, 2]
DEFAULT_ALPHAS = [0.1, 0.5, 1, 2, 5, 10, 50, 100, 1000]
DEFAULT_GATE_STRENGTHS = [0.1, 0.5, 1.0, 5.0]


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Achata o JSON de metricas em uma linha tabular."""

    eval_split = metrics["eval_split"]
    model_params = metrics["model_params"]
    split_metrics = metrics["metrics"]

    return {
        "experiment_name": metrics["experiment_name"],
        "dataset": metrics["dataset"],
        "feature_set": metrics["feature_set"],
        "eval_split": eval_split,
        "degree": model_params["degree"],
        "alpha": model_params["alpha"],
        "gate_strength": model_params.get("gate_strength"),
        "mae": split_metrics[f"{eval_split}_mae"],
        "rmse": split_metrics[f"{eval_split}_rmse"],
        "nmad": split_metrics[f"{eval_split}_nmad"],
        "nmad_se": split_metrics[f"{eval_split}_nmad_se"],
        "r2": split_metrics[f"{eval_split}_r2"],
        "bias": split_metrics[f"{eval_split}_bias"],
        "n": split_metrics[f"{eval_split}_n"],
        "catastrophic_outlier_fraction": split_metrics[
            f"{eval_split}_catastrophic_outlier_fraction"
        ],
        "negative_redshift_predictions": split_metrics[
            f"{eval_split}_negative_redshift_predictions"
        ],
    }


SELECTION_METRIC = "nmad"
SELECTION_SE_COLUMN = "nmad_se"


def select_best_within_1se(
    results: pd.DataFrame,
    metric: str = SELECTION_METRIC,
    se_column: str = SELECTION_SE_COLUMN,
) -> pd.DataFrame:
    """Ordena a busca pela regra de 1 desvio-padrao (one-standard-error rule).

    A selecao usa o NMAD, dispersao robusta de photo-z, em vez do RMSE. O RMSE
    e dominado por poucos outliers catastroficos e tende a apontar sempre o maior
    degree e o menor alpha por uma margem dentro do ruido. A regra:

    1. Acha a config de menor NMAD.
    2. Usa como 1 SE o erro-padrao do NMAD dessa config, estimado por bootstrap
       (coluna `nmad_se`), sem supor normalidade da distribuicao de dz.
    3. Mantem todas as configs com NMAD <= melhor_NMAD + SE, ou seja,
       estatisticamente indistinguiveis do melhor.
    4. Dentre essas, escolhe a mais simples/regularizada: menor degree e, em
       seguida, maior alpha do Ridge. Se essas duas propriedades empatarem,
       usa o menor NMAD para escolher a intensidade do gate.

    Acrescenta as colunas selection_metric, selection_se, selection_threshold,
    within_1se e selected_1se, e ordena com a config recomendada no topo,
    seguida das demais dentro da banda e, por fim, o restante pela metrica.
    """

    results = results.copy()

    best_idx = results[metric].idxmin()
    best_value = float(results.loc[best_idx, metric])
    se = float(results.loc[best_idx, se_column])
    threshold = best_value + se

    results["selection_metric"] = metric
    results["selection_se"] = se
    results["selection_threshold"] = threshold
    results["within_1se"] = results[metric] <= threshold

    candidates = results[results["within_1se"]].sort_values(
        by=["degree", "alpha", metric],
        ascending=[True, False, True],
    )
    selected_name = candidates.iloc[0]["experiment_name"]
    results["selected_1se"] = results["experiment_name"] == selected_name

    return results.sort_values(
        by=["selected_1se", "within_1se", metric, "mae"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def run_search(
    dataset: str,
    feature_set: str = "mag",
    eval_split: str = "val",
    degrees: list[int] | None = None,
    alphas: list[float] | None = None,
    gate_strengths: list[float] | None = None,
    processed_data_dir: Path = PROCESSED_DATA_DIR,
    models_dir: Path = MODELS_DIR,
    metrics_dir: Path = METRICS_DIR,
    tables_dir: Path = TABLES_DIR,
) -> pd.DataFrame:
    """Executa a busca em grade e salva uma tabela resumo."""

    degrees = degrees or DEFAULT_DEGREES
    alphas = alphas or DEFAULT_ALPHAS
    if feature_set == "gate_err_manual":
        resolved_gate_strengths: list[float | None] = (
            gate_strengths or DEFAULT_GATE_STRENGTHS
        )
    else:
        resolved_gate_strengths = [None]

    rows = []
    for degree in degrees:
        for alpha in alphas:
            for gate_strength in resolved_gate_strengths:
                metrics = run_experiment(
                    dataset=dataset,
                    feature_set=feature_set,
                    eval_split=eval_split,
                    degree=degree,
                    alpha=alpha,
                    gate_strength=gate_strength,
                    processed_data_dir=processed_data_dir,
                    models_dir=models_dir,
                    metrics_dir=metrics_dir,
                )
                rows.append(summarize_metrics(metrics))

    results = select_best_within_1se(pd.DataFrame(rows))

    output_path = table_output_path(
        tables_dir=tables_dir,
        eval_split=eval_split,
        feature_set=feature_set,
        model_name=MODEL_NAME,
        dataset=dataset,
        filename=f"{MODEL_NAME}_{feature_set}_{eval_split}_search.csv",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    return results


def parse_args() -> argparse.Namespace:
    """Define argumentos de linha de comando."""

    parser = argparse.ArgumentParser(
        description="Busca hiperparametros de Polynomial Ridge."
    )
    parser.add_argument(
        "--dataset",
        choices=["happyT", "teddyT"],
        default="happyT",
        help="Dataset processado em data/processed/Val.",
    )
    parser.add_argument(
        "--feature-set",
        choices=list(FEATURE_SETS),
        default="mag",
        help="Conjunto de features: mag, mag_err ou gate_err_manual.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["val"],
        default="val",
        help="Split usado para avaliacao. A busca de hiperparametros deve usar apenas val.",
    )
    parser.add_argument(
        "--degrees",
        type=int,
        nargs="+",
        default=DEFAULT_DEGREES,
        help="Graus testados nas PolynomialFeatures.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHAS,
        help="Valores de alpha testados no Ridge.",
    )
    parser.add_argument(
        "--gate-strengths",
        type=float,
        nargs="+",
        default=DEFAULT_GATE_STRENGTHS,
        help=(
            "Intensidades testadas para o GATE-ERR. "
            "Usadas apenas em gate_err_manual."
        ),
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
        "--tables-dir",
        type=Path,
        default=TABLES_DIR,
        help="Diretorio para salvar tabelas resumo.",
    )

    return parser.parse_args()


def main() -> None:
    """Executa a busca pelo terminal."""

    args = parse_args()
    results = run_search(
        dataset=args.dataset,
        feature_set=args.feature_set,
        eval_split=args.eval_split,
        degrees=args.degrees,
        alphas=args.alphas,
        gate_strengths=args.gate_strengths,
        processed_data_dir=args.processed_data_dir,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir,
        tables_dir=args.tables_dir,
    )

    selected = results[results["selected_1se"]].iloc[0]
    best_nmad = results["nmad"].min()
    print(
        "Config escolhida pela regra de 1 desvio-padrao sobre o NMAD "
        f"(degree={int(selected['degree'])}, alpha={selected['alpha']:g}"
        + (
            f", gate_strength={selected['gate_strength']:g}"
            if pd.notna(selected["gate_strength"])
            else ""
        )
        + "):"
    )
    print(
        f"  NMAD={selected['nmad']:.6f} | RMSE={selected['rmse']:.6f} | "
        f"MAE={selected['mae']:.6f}"
    )
    print(
        f"  menor NMAD da busca={best_nmad:.6f} | "
        f"limite 1-SE={selected['selection_threshold']:.6f}"
    )
    print("\nConfigs dentro de 1 desvio-padrao do melhor NMAD:")
    columns = [
        "degree",
        "alpha",
        "gate_strength",
        "nmad",
        "rmse",
        "mae",
        "within_1se",
        "selected_1se",
    ]
    print(results.loc[results["within_1se"], columns].to_string(index=False))


if __name__ == "__main__":
    main()
