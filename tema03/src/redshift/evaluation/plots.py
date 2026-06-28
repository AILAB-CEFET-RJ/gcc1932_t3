"""Funcoes de visualizacao para avaliacao de modelos."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_redshift_residual_plot(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    path: Path,
    title: str,
) -> None:
    """Salva um grafico com real vs previsto e residuos na escala original.

    A geracao da figura e tratada como nao-fatal: se a renderizacao falhar (por
    exemplo, backend do matplotlib bloqueado pelo sistema operacional), o erro e
    apenas avisado e a execucao continua, preservando metricas, modelos e tabelas.
    """

    try:
        _render_redshift_residual_plot(y_true, y_pred, path, title)
    except Exception as error:  # noqa: BLE001 - figura nao deve derrubar o experimento
        print(
            f"[aviso] nao foi possivel salvar a figura {path}: {error}",
            file=sys.stderr,
        )


def _render_redshift_residual_plot(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    path: Path,
    title: str,
) -> None:
    """Renderiza e salva o grafico de avaliacao."""

    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    residuals = y_pred_array - y_true_array

    min_value = min(y_true_array.min(), y_pred_array.min())
    max_value = max(y_true_array.max(), y_pred_array.max())

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].scatter(y_true_array, y_pred_array, s=12, alpha=0.35, color="#2a6f97")
    axes[0].plot([min_value, max_value], [min_value, max_value], color="#d62828", linestyle="--", linewidth=1.5)
    axes[0].set_title("Redshift real vs previsto")
    axes[0].set_xlabel("redshift real")
    axes[0].set_ylabel("redshift previsto")

    axes[1].scatter(y_pred_array, residuals, s=12, alpha=0.35, color="#89c2d9")
    axes[1].axhline(0, color="#d62828", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuos vs previsto")
    axes[1].set_xlabel("redshift previsto")
    axes[1].set_ylabel("residuo (previsto - real)")

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
