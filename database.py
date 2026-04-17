from typing import Any

import pandas as pd
from numpy import dtype, ndarray
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

DB_PATH = 'data/banco_completo_REGISTRO_COVID_28_08_processado_cardiopatia_sociodemographic.parquet'
ORIGINAL_OUTCOME = 'intercorrencia_3_5_6_13_16'
COLUMNS_TO_DROP = ['intercorrencia___13', 'intercorrencia___3', 'intercorrencia___16', 'intercorrencia___6',
                   'intercorrencia___5', 'direto_cti', 'dataadm', 'onda']


def open_parquet(filepath: str) -> pd.DataFrame:
    """"" Abre um arquivo .parquet e retorna o dataframe criado a partir do mesmo. """
    return pd.read_parquet(filepath)


def get_vars(df: pd.DataFrame) -> list[str]:
    """" Retorna as colunas do dataframe como lista de strings. """
    return list(df.columns)


def create_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """" Retira as colunas de desfecho e de filtragem de dados da base;
         cria uma coluna combinada de desfecho. """
    for outcome in COLUMNS_TO_DROP:
        if outcome in get_vars(df):
            df = df.drop(outcome, axis=1).copy()
    if ORIGINAL_OUTCOME in get_vars(df):
        df = df.dropna(subset=[ORIGINAL_OUTCOME]).copy()
        outcome = df[ORIGINAL_OUTCOME].copy()
        df = df.drop(ORIGINAL_OUTCOME, axis=1).copy()
        df['outcome'] = outcome

    return df


def get_data(df: pd.DataFrame) -> tuple[ndarray[tuple[int, int], dtype[Any]], ndarray[tuple[int], dtype[Any]]]:
    """" Separa as features (X) da varíavel de desfecho (y) """
    y = df['outcome'].to_numpy()
    X = df.drop('outcome', axis=1).to_numpy()
    return X, y


def impute_data(X: np.ndarray, strat='median') -> np.ndarray:
    """" Faz a imputação dos dados de feature passados, preenchendo células vazias (NaN);
         Por padrão, a estratégia utilizada é de mediana. """
    # print(f"Criando imputador de dados (estratégia: mediana)")
    imputer = SimpleImputer(strategy=strat)

    # print(f"Imputando valores de X")
    imputer.fit(X)
    # print(f'Contagem de NaN em X antes da imputação: {np.count_nonzero(np.isnan(X))}')
    imputed_X = imputer.transform(X)
    # print(f'Contagem de NaN em X depois da imputação: {np.count_nonzero(np.isnan(imputed_X))}')

    return imputed_X


def normalize_data(X: np.ndarray) -> np.ndarray:
    """" Normaliza os dados de feature passados, garantindo que a média passe a ser
         nula e a variância passe a ser unitária. """
    # print(f'X shape: {X.shape}')
    if X.shape[0] == 1:
        X = X.reshape(-1, 1)
    if X.ndim > 1 and X.shape[1] == 1:
        X = X.reshape(1, -1)

    scaler = StandardScaler()

    # print('Normalizando dados de X')
    scaler.fit(X)
    # print(f'Média e variância de X antes da normalização: {np.mean(X)}, {np.var(X)}')
    normalized_X = scaler.transform(X)
    # print(f'Média e variância de X antes da normalização: {np.mean(normalized_X)}, {np.var(normalized_X)}')

    return normalized_X
