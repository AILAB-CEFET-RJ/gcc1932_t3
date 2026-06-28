"""Pre-processamento isolado para o baseline GATE-ERR manual."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from redshift.data.gate_err import (
    fit_gate_err_preprocess_train,
    transform_gate_err_intermediate,
)
from redshift.data.preprocess import (
    RAW_DATA_DIR,
    TEST_SETS,
    get_dataset_file_path,
    load_dataset,
    load_raw_datasets,
    save_processed_splits,
    split_features_target,
    split_train_validation_from_a,
    transform_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GATE_ERR_PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed Gate-Err"
VAL_MODE_DIR = "Val"
TEST_MODE_DIR = "Test"


def fit_gate_err_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """Ajusta todo o preprocessing GATE-ERR somente no conjunto de treino."""

    X_train_intermediate, preprocessors = fit_gate_err_preprocess_train(X_train)
    return X_train_intermediate, transform_target(y_train), preprocessors


def preprocess_gate_err_for_val_mode(
    dataset_name: str,
    raw_dir: str | Path = RAW_DATA_DIR,
    validation_size: float = 0.20,
    random_state: int = 42,
    stratify_by_redshift: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, object]]:
    """Ajusta em train e transforma validation, sem acessar B, C ou D."""

    a_df = load_dataset(
        get_dataset_file_path(
            raw_dir=raw_dir,
            dataset_name=dataset_name,
            split_name="A",
        )
    )
    train_df, val_df = split_train_validation_from_a(
        a_df,
        validation_size=validation_size,
        random_state=random_state,
        stratify_by_redshift=stratify_by_redshift,
    )
    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    X_train_processed, y_train_processed, preprocessors = fit_gate_err_train(
        X_train,
        y_train,
    )
    X_val_processed = transform_gate_err_intermediate(X_val, preprocessors)
    return (
        X_train_processed,
        X_val_processed,
        y_train_processed,
        transform_target(y_val),
        preprocessors,
    )


def preprocess_gate_err_for_test_mode(
    dataset_name: str,
    raw_dir: str | Path = RAW_DATA_DIR,
    validation_size: float = 0.20,
    random_state: int = 42,
    stratify_by_redshift: bool = True,
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.Series,
    dict[str, pd.Series],
    dict[str, object],
]:
    """Ajusta em A completo e transforma B, C e D separadamente."""

    a_df, test_dfs = load_raw_datasets(
        dataset_name=dataset_name,
        raw_dir=raw_dir,
    )
    train_df, val_df = split_train_validation_from_a(
        a_df,
        validation_size=validation_size,
        random_state=random_state,
        stratify_by_redshift=stratify_by_redshift,
    )
    train_val_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    X_train, y_train = split_features_target(train_val_df)
    X_train_processed, y_train_processed, preprocessors = fit_gate_err_train(
        X_train,
        y_train,
    )

    X_tests: dict[str, pd.DataFrame] = {}
    y_tests: dict[str, pd.Series] = {}
    for split_name, test_df in test_dfs.items():
        X_test, y_test = split_features_target(test_df)
        X_tests[split_name] = transform_gate_err_intermediate(
            X_test,
            preprocessors,
        )
        y_tests[split_name] = transform_target(y_test)

    return (
        X_train_processed,
        X_tests,
        y_train_processed,
        y_tests,
        preprocessors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-processa dados isolados para o GATE-ERR manual."
    )
    parser.add_argument(
        "--dataset",
        choices=["happyT", "teddyT"],
        default="happyT",
    )
    parser.add_argument("--raw-dir", default=RAW_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        default=GATE_ERR_PROCESSED_DATA_DIR,
        help="Raiz exclusiva do GATE-ERR; nao altera data/processed.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "val", "test"],
        default="all",
    )
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-stratify", action="store_true")
    args = parser.parse_args()

    saved_dirs: list[Path] = []
    if args.mode in {"all", "val"}:
        X_train, X_val, y_train, y_val, preprocessors = (
            preprocess_gate_err_for_val_mode(
                dataset_name=args.dataset,
                raw_dir=args.raw_dir,
                validation_size=args.validation_size,
                random_state=args.random_state,
                stratify_by_redshift=not args.no_stratify,
            )
        )
        save_processed_splits(
            output_dir=Path(args.output_dir) / VAL_MODE_DIR,
            dataset_name=args.dataset,
            X_train=X_train,
            X_val=X_val,
            X_test=None,
            X_tests=None,
            y_train=y_train,
            y_val=y_val,
            y_test=None,
            y_tests=None,
            preprocessors=preprocessors,
        )
        saved_dirs.append(Path(args.output_dir) / VAL_MODE_DIR / args.dataset)
        print("GATE-ERR Val: fit em train; transform em validation")

    if args.mode in {"all", "test"}:
        X_train, X_tests, y_train, y_tests, preprocessors = (
            preprocess_gate_err_for_test_mode(
                dataset_name=args.dataset,
                raw_dir=args.raw_dir,
                validation_size=args.validation_size,
                random_state=args.random_state,
                stratify_by_redshift=not args.no_stratify,
            )
        )
        save_processed_splits(
            output_dir=Path(args.output_dir) / TEST_MODE_DIR,
            dataset_name=args.dataset,
            X_train=X_train,
            X_val=None,
            X_test=None,
            X_tests=X_tests,
            y_train=y_train,
            y_val=None,
            y_test=None,
            y_tests=y_tests,
            preprocessors=preprocessors,
        )
        saved_dirs.append(Path(args.output_dir) / TEST_MODE_DIR / args.dataset)
        print("GATE-ERR Test: fit em A completo; transform em B, C e D")

    print("Arquivos GATE-ERR salvos em:")
    for saved_dir in saved_dirs:
        print(f"  {saved_dir}")


if __name__ == "__main__":
    main()
