"""Pre-processamento procedural para modelos de redshift fotometrico."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
VAL_MODE_DIR = "Val"
TEST_MODE_DIR = "Test"

MAGNITUDE_COLUMNS: tuple[str, ...] = ("u", "g", "r", "i", "z")
MAGNITUDE_ERROR_COLUMNS: tuple[str, ...] = ("uErr", "gErr", "rErr", "iErr", "zErr")
TARGET_COLUMN = "redshift"
FEATURE_COLUMNS: tuple[str, ...] = MAGNITUDE_COLUMNS + MAGNITUDE_ERROR_COLUMNS
RAW_COLUMNS: tuple[str, ...] = (
    "id",
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
    "redshiftErr",
)
TEST_SETS: tuple[str, ...] = ("B", "C", "D")
DATASET_FILES: dict[str, dict[str, str]] = {
    "happyT": {
        "A": "happyT_A.txt",
        "B": "happyT_B.txt",
        "C": "happyT_C.txt",
        "D": "happyT_D.txt",
    },
    "teddyT": {
        "A": "teddyT_A.cat",
        "B": "teddyT_B.cat",
        "C": "teddyT_C.cat",
        "D": "teddyT_D.cat",
    },
}


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Verifica se todas as colunas obrigatorias existem no dataframe."""

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")


def get_dataset_file_path(
    raw_dir: str | Path,
    dataset_name: str,
    split_name: str,
) -> Path:
    """Retorna o caminho padrao de um dataset e split."""

    if dataset_name not in DATASET_FILES:
        available = ", ".join(DATASET_FILES)
        raise ValueError(f"Dataset invalido: {dataset_name}. Opcoes: {available}")

    if split_name not in DATASET_FILES[dataset_name]:
        available = ", ".join(DATASET_FILES[dataset_name])
        raise ValueError(f"Split invalido: {split_name}. Opcoes: {available}")

    path = Path(raw_dir) / DATASET_FILES[dataset_name][split_name]

    if not path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    return path


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Carrega um arquivo de catalogo em formato texto."""

    path = Path(path)

    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=RAW_COLUMNS,
        engine="python",
    )


def load_raw_datasets(
    dataset_name: str,
    raw_dir: str | Path = RAW_DATA_DIR,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Carrega A para treino/validacao e B, C, D para teste externo."""

    a_path = get_dataset_file_path(
        raw_dir=raw_dir,
        dataset_name=dataset_name,
        split_name="A",
    )
    train_val_df = load_dataset(a_path)

    test_dfs = {
        split_name: load_dataset(
            get_dataset_file_path(
                raw_dir=raw_dir,
                dataset_name=dataset_name,
                split_name=split_name,
            )
        )
        for split_name in TEST_SETS
    }

    return train_val_df, test_dfs


# Criar faixas de redshift para estratificar o split, mantendo a distribuicao. Pois o redshift tem uma distribuicao assimetrica e multimodal
def make_redshift_stratification_bins(
    y: pd.Series,
    n_bins: int = 10,
) -> pd.Series:
    """Cria faixas de redshift para manter a distribuicao no split."""

    return pd.qcut(y, q=n_bins, labels=False, duplicates="drop")


def split_train_validation_from_a(
    a_df: pd.DataFrame,
    validation_size: float = 0.20,
    random_state: int = 42,
    stratify_by_redshift: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide o conjunto A em treino e validacao.

    A sugestao inicial e usar 80% do A para treino e 20% para validacao. B, C
    e D ficam completamente fora desse split e sao usados apenas como teste.
    """

    validate_columns(a_df, FEATURE_COLUMNS + (TARGET_COLUMN,))

    stratify = None
    if stratify_by_redshift: 
        stratify = make_redshift_stratification_bins(a_df[TARGET_COLUMN])

    train_df, val_df = train_test_split(
        a_df,
        test_size=validation_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )

    return train_df.copy(), val_df.copy()


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as features usadas pelo modelo e o alvo redshift."""

    validate_columns(df, FEATURE_COLUMNS + (TARGET_COLUMN,))

    X = df.loc[:, FEATURE_COLUMNS].copy()
    y = df.loc[:, TARGET_COLUMN].copy()

    return X, y


def fit_preprocess_train(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """Ajusta o pre-processamento no treino e transforma treino.
    Os scalers sao ajustados apenas em X_train para evitar vazamento de dados.
    """

    validate_columns(X_train, FEATURE_COLUMNS)

    X_train_processed = X_train.loc[:, FEATURE_COLUMNS].copy()

    # 1. Aplicar log1p nos erros de magnitude para reduzir assimetria.
    X_train_processed.loc[:, MAGNITUDE_ERROR_COLUMNS] = np.log1p(
        X_train_processed.loc[:, MAGNITUDE_ERROR_COLUMNS]
    )

    # 2. Ajustar o StandardScaler somente nas magnitudes do treino.
    magnitude_scaler = StandardScaler()
    X_train_processed.loc[:, MAGNITUDE_COLUMNS] = magnitude_scaler.fit_transform(
        X_train_processed.loc[:, MAGNITUDE_COLUMNS]
    )

    # 3. Ajustar o RobustScaler somente nos erros transformados do treino.
    magnitude_error_scaler = RobustScaler()
    X_train_processed.loc[:, MAGNITUDE_ERROR_COLUMNS] = (
        magnitude_error_scaler.fit_transform(
            X_train_processed.loc[:, MAGNITUDE_ERROR_COLUMNS]
        )
    )

    # 4. Manter o alvo na escala original de redshift.
    y_train_processed = transform_target(y_train)

    preprocessors = {
        "magnitude_scaler": magnitude_scaler,
        "magnitude_error_scaler": magnitude_error_scaler,
        "feature_columns": FEATURE_COLUMNS,
        "magnitude_columns": MAGNITUDE_COLUMNS,
        "magnitude_error_columns": MAGNITUDE_ERROR_COLUMNS,
        "target_column": TARGET_COLUMN,
    }

    # 5. Retornar o treino processado e os scalers ajustados para uso posterior.
    return X_train_processed, y_train_processed, preprocessors


def transform_features(
    X: pd.DataFrame,
    preprocessors: dict[str, object],
) -> pd.DataFrame:
    """Transforma features nas bases de dados de validacao e teste com os scalers ajustados no treino."""

    feature_columns = preprocessors["feature_columns"]
    magnitude_columns = preprocessors["magnitude_columns"]
    magnitude_error_columns = preprocessors["magnitude_error_columns"]
    magnitude_scaler = preprocessors["magnitude_scaler"]
    magnitude_error_scaler = preprocessors["magnitude_error_scaler"]

    validate_columns(X, feature_columns)

    X_processed = X.loc[:, feature_columns].copy()

    # 1. Repetir a mesma transformacao log1p aplicada aos erros no treino.
    X_processed.loc[:, magnitude_error_columns] = np.log1p(
        X_processed.loc[:, magnitude_error_columns]
    )

    # 2. Usar os scalers ajustados no treino, sem chamar fit novamente.
    X_processed.loc[:, magnitude_columns] = magnitude_scaler.transform(
        X_processed.loc[:, magnitude_columns]
    )
    X_processed.loc[:, magnitude_error_columns] = magnitude_error_scaler.transform(
        X_processed.loc[:, magnitude_error_columns]
    )

    return X_processed


def transform_target(y: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Mantem o alvo redshift na escala original."""

    if isinstance(y, pd.Series):
        return y.copy()

    return np.asarray(y).copy()


def inverse_transform_target(
    y_transformed: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    """Retorna o redshift, que ja esta na escala original."""

    if isinstance(y_transformed, pd.Series):
        return y_transformed.copy()

    return np.asarray(y_transformed).copy()


def transform_new_data(
    df: pd.DataFrame,
    preprocessors: dict[str, object],
) -> pd.DataFrame:
    """Pre-processa novos dados usando os scalers ajustados no treino."""

    return transform_features(df, preprocessors)


def preprocess_train_val_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    dict[str, object],
]:
    """Pre-processa treino, validacao e teste sem vazamento de dados."""

    # 1. Separar features e alvo em cada particao.
    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    X_test, y_test = split_features_target(test_df)

    # 2. Ajustar scalers apenas no treino e transformar o proprio treino.
    X_train_processed, y_train_processed, preprocessors = fit_preprocess_train(
        X_train,
        y_train,
    )

    # 3. Transformar validacao e teste com os scalers ja ajustados no treino.
    X_val_processed = transform_features(X_val, preprocessors)
    X_test_processed = transform_features(X_test, preprocessors)

    # 4. Manter os alvos na escala original, sem ajuste estatistico.
    y_val_processed = transform_target(y_val)
    y_test_processed = transform_target(y_test)

    return (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        y_train_processed,
        y_val_processed,
        y_test_processed,
        preprocessors,
    )


def preprocess_dataset_a_b_c_d(
    dataset_name: str,
    raw_dir: str | Path = RAW_DATA_DIR,
    validation_size: float = 0.20,
    random_state: int = 42,
    stratify_by_redshift: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.Series,
    pd.Series,
    dict[str, pd.Series],
    dict[str, object],
]:
    """Pre-processa A para treino/validacao e B, C, D como testes separados.

    Retorna X_train, X_val, X_tests, y_train, y_val, y_tests e preprocessors.
    """

    # 1. Carregar A e os conjuntos externos B, C e D.
    a_df, test_dfs = load_raw_datasets(
        dataset_name=dataset_name,
        raw_dir=raw_dir,
    )

    # 2. Dividir apenas o conjunto A em treino e validacao.
    train_df, val_df = split_train_validation_from_a(
        a_df,
        validation_size=validation_size,
        random_state=random_state,
        stratify_by_redshift=stratify_by_redshift,
    )

    # 3. Separar features e alvo de treino e validacao.
    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)

    # 4. Ajustar scalers somente no treino derivado do conjunto A.
    X_train_processed, y_train_processed, preprocessors = fit_preprocess_train(
        X_train,
        y_train,
    )

    # 5. Transformar validacao com os scalers ajustados no treino.
    X_val_processed = transform_features(X_val, preprocessors)
    y_val_processed = transform_target(y_val)

    # 6. Transformar cada teste externo com os scalers ajustados no treino.
    X_test_processed_by_split: dict[str, pd.DataFrame] = {}
    y_test_processed_by_split: dict[str, pd.Series] = {}
    for split_name, test_df in test_dfs.items():
        X_test, y_test = split_features_target(test_df)
        X_test_processed_by_split[split_name] = transform_features(
            X_test,
            preprocessors,
        )
        y_test_processed_by_split[split_name] = transform_target(y_test)

    return (
        X_train_processed,
        X_val_processed,
        X_test_processed_by_split,
        y_train_processed,
        y_val_processed,
        y_test_processed_by_split,
        preprocessors,
    )


def preprocess_dataset_for_val_mode(
    dataset_name: str,
    raw_dir: str | Path = RAW_DATA_DIR,
    validation_size: float = 0.20,
    random_state: int = 42,
    stratify_by_redshift: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, object]]:
    """Pre-processa dados para model selection.

    O scaler e ajustado apenas no split de treino derivado do conjunto A. O split
    de validacao e apenas transformado. O teste externo nao e salvo neste modo.
    """

    a_df, _ = load_raw_datasets(
        dataset_name=dataset_name,
        raw_dir=raw_dir,
    )
    train_df, val_df = split_train_validation_from_a(
        a_df,
        validation_size=validation_size,
        random_state=random_state,
        stratify_by_redshift=stratify_by_redshift,
    )

    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    X_train_processed, y_train_processed, preprocessors = fit_preprocess_train(
        X_train,
        y_train,
    )
    X_val_processed = transform_features(X_val, preprocessors)
    y_val_processed = transform_target(y_val)

    return (
        X_train_processed,
        X_val_processed,
        y_train_processed,
        y_val_processed,
        preprocessors,
    )


def preprocess_dataset_for_test_mode(
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
    """Pre-processa dados para avaliacao final.

    O conjunto A e dividido do mesmo jeito do modo Val, depois treino e validacao
    sao unidos. O scaler e ajustado em train+val e aplicado separadamente aos
    testes externos B, C e D.
    """

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
    X_train_processed, y_train_processed, preprocessors = fit_preprocess_train(
        X_train,
        y_train,
    )
    X_test_processed_by_split: dict[str, pd.DataFrame] = {}
    y_test_processed_by_split: dict[str, pd.Series] = {}
    for split_name, test_df in test_dfs.items():
        X_test, y_test = split_features_target(test_df)
        X_test_processed_by_split[split_name] = transform_features(
            X_test,
            preprocessors,
        )
        y_test_processed_by_split[split_name] = transform_target(y_test)

    return (
        X_train_processed,
        X_test_processed_by_split,
        y_train_processed,
        y_test_processed_by_split,
        preprocessors,
    )


def preprocess_happy_a_b_c_d(
    raw_dir: str | Path = RAW_DATA_DIR,
    validation_size: float = 0.20,
    random_state: int = 42,
    stratify_by_redshift: bool = True,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    pd.Series,
    pd.Series,
    dict[str, pd.Series],
    dict[str, object],
]:
    """Atalho para pre-processar o dataset happyT."""

    return preprocess_dataset_a_b_c_d(
        dataset_name="happyT",
        raw_dir=raw_dir,
        validation_size=validation_size,
        random_state=random_state,
        stratify_by_redshift=stratify_by_redshift,
    )


def save_preprocessors(preprocessors: dict[str, object], path: str | Path) -> None:
    """Salva os scalers ajustados para uso posterior."""

    joblib.dump(preprocessors, path)


def load_preprocessors(path: str | Path) -> dict[str, object]:
    """Carrega scalers ajustados salvos com save_preprocessors."""

    return joblib.load(path)


def save_processed_splits(
    output_dir: str | Path,
    dataset_name: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame | None,
    X_test: pd.DataFrame | None,
    X_tests: dict[str, pd.DataFrame] | None,
    y_train: pd.Series,
    y_val: pd.Series | None,
    y_test: pd.Series | None,
    y_tests: dict[str, pd.Series] | None,
    preprocessors: dict[str, object],
) -> None:
    """Salva as bases processadas em CSV e os scalers em joblib."""

    output_dir = Path(output_dir) / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remover saidas antigas para a pasta refletir apenas o formato atual.
    old_outputs = [
        "X_train.csv",
        "X_val.csv",
        "X_test.csv",
        "X_test_B.csv",
        "X_test_C.csv",
        "X_test_D.csv",
        "y_train.csv",
        "y_val.csv",
        "y_test.csv",
        "y_test_B.csv",
        "y_test_C.csv",
        "y_test_D.csv",
        "preprocessors.joblib",
    ]
    for filename in old_outputs:
        path = output_dir / filename
        if path.exists():
            path.unlink()

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    if X_val is not None and y_val is not None:
        X_val.to_csv(output_dir / "X_val.csv", index=False)
        y_val.to_csv(output_dir / "y_val.csv", index=False)
    if X_test is not None and y_test is not None:
        X_test.to_csv(output_dir / "X_test.csv", index=False)
        y_test.to_csv(output_dir / "y_test.csv", index=False)
    if X_tests is not None and y_tests is not None:
        for split_name in TEST_SETS:
            if split_name not in X_tests or split_name not in y_tests:
                raise ValueError(f"Teste externo ausente no split {split_name}")
            X_tests[split_name].to_csv(output_dir / f"X_test_{split_name}.csv", index=False)
            y_tests[split_name].to_csv(output_dir / f"y_test_{split_name}.csv", index=False)

    save_preprocessors(preprocessors, output_dir / "preprocessors.joblib")


def main() -> None:
    """Executa o pre-processamento pelo terminal."""

    parser = argparse.ArgumentParser(
        description="Pre-processa datasets happyT ou teddyT para modelos de redshift."
    )
    parser.add_argument(
        "--dataset",
        choices=["happyT", "teddyT"],
        default="happyT",
        help="Prefixo do dataset em data/raw.",
    )
    parser.add_argument(
        "--raw-dir",
        default=RAW_DATA_DIR,
        help="Pasta com os arquivos brutos.",
    )
    parser.add_argument(
        "--output-dir",
        default=PROCESSED_DATA_DIR,
        help="Pasta base onde Val/<dataset> e Test/<dataset> serao salvos.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "val", "test"],
        default="all",
        help="Modo de pre-processamento: val, test ou all.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.20,
        help="Fracao do conjunto A usada para validacao. Exemplo: 0.20.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semente do split treino/validacao.",
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Desliga a estratificacao por faixas de redshift.",
    )

    args = parser.parse_args()

    saved_dirs: list[Path] = []
    if args.mode in {"all", "val"}:
        (
            X_train,
            X_val,
            y_train,
            y_val,
            preprocessors,
        ) = preprocess_dataset_for_val_mode(
            dataset_name=args.dataset,
            raw_dir=args.raw_dir,
            validation_size=args.validation_size,
            random_state=args.random_state,
            stratify_by_redshift=not args.no_stratify,
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

        print(f"Modo Val: scaler ajustado em train; avaliacao em val")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val: {X_val.shape}")

    if args.mode in {"all", "test"}:
        (
            X_train,
            X_tests,
            y_train,
            y_tests,
            preprocessors,
        ) = preprocess_dataset_for_test_mode(
            dataset_name=args.dataset,
            raw_dir=args.raw_dir,
            validation_size=args.validation_size,
            random_state=args.random_state,
            stratify_by_redshift=not args.no_stratify,
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

        print(f"Modo Test: scaler ajustado em train+val; avaliacao em test")
        print(f"  X_train: {X_train.shape}")
        for split_name in TEST_SETS:
            print(f"  X_test_{split_name}: {X_tests[split_name].shape}")

    print(f"Dataset: {args.dataset}")
    print(f"Validacao: {args.validation_size:.0%} do conjunto A")
    print("Arquivos salvos em:")
    for saved_dir in saved_dirs:
        print(f"  {saved_dir}")


if __name__ == "__main__":
    main()
