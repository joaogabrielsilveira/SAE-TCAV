from typing import Any, Sequence

import pandas as pd
from numpy import dtype, ndarray
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from filepaths import get_env_path

TARGET_POS_LINES = 5000
TARGET_NEG_LINES = 5000
M_CANDIDATES = 500
RNG_SEED = 42

RENAL_DB_PATH = get_env_path('data/renal/tidy_event_data.feather')
COVID_DB_PATH = get_env_path('data/covid/banco_completo_REGISTRO_COVID_28_08_processado_cardiopatia_sociodemographic.parquet')
COVID_ORIGINAL_OUTCOME = 'intercorrencia_3_5_6_13_16'
COVID_COLUMNS_TO_DROP = ['intercorrencia___13', 'intercorrencia___3', 'intercorrencia___16', 'intercorrencia___6',
                   'intercorrencia___5', 'direto_cti', 'dataadm', 'onda']

def open_parquet(filepath: str) -> pd.DataFrame:
    """"" Abre um arquivo .parquet e retorna o dataframe criado a partir do mesmo. """
    return pd.read_parquet(filepath)

def open_feather(filepath: str) -> pd.DataFrame:
    return pd.read_feather(filepath)

def get_vars(df: pd.DataFrame) -> list[str]:
    """" Retorna as colunas do dataframe como lista de strings. """
    return list(df.columns)

def create_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """" Retira as colunas de desfecho e de filtragem de dados da base;
         cria uma coluna combinada de desfecho. """
    for outcome in COVID_COLUMNS_TO_DROP:
        if outcome in get_vars(df):
            df = df.drop(outcome, axis=1).copy()
    if COVID_ORIGINAL_OUTCOME in get_vars(df):
        df = df.dropna(subset=[COVID_ORIGINAL_OUTCOME]).copy()
        outcome = df[COVID_ORIGINAL_OUTCOME].copy()
        df = df.drop(COVID_ORIGINAL_OUTCOME, axis=1).copy()
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

class TabPFNPrepConfig:
    # parâmetros que definem como os dados serão tratados
    rng_seed: int = 42
    target_pos_lines: int = 5000
    target_neg_lines: int = 5000
    max_total_rows: int = 10000
    final_top_k: int = 500
    m_candidates: int = 3000
    forced_train_year_start: int = 1997
    forced_test_year_start: int = 2006

def canonicalize_event_df(df: pd.DataFrame) -> pd.DataFrame:
    # adiciona coluna de ano e regulariza os tipos
    out = df.copy()
    out['date'] = pd.to_datetime(out['date'])
    out['year'] = out['date'].dt.year.astype(int)
    out['patient_id'] = out['patient_id'].astype(str)

    return out


def infer_train_test_years(years_all: Sequence[int], forced_start: int = 1997,
                           forced_end: int = 2006) -> tuple[list[int], list[int]]:
    # converte pra int
    years_all = sorted(set(map(int, years_all)))

    # se todos anos desejados para treino estiverem no conjunto, usa eles
    forced_range = range(forced_start, forced_end + 1)
    forced_set = set(forced_range)
    if forced_set.issubset(years_all):
        train_years = list(forced_range)

    # else, usa a primeira metade
    else:
        half = len(years_all) // 2
        train_years = years_all[:half]
    test_years = sorted([y for y in years_all if y not in set(train_years)])
    return sorted(train_years), test_years

def split_patients(all_patients: Sequence[str], test_size: float=0.1, random_state: int = 42) -> tuple[list[str], list[str]]:
    train_pat, test_pat = train_test_split(all_patients, test_size=test_size, random_state=random_state)

    return sorted(map(str, train_pat)), sorted(map(str, test_pat))

def build_patient_availability_table(df_train: pd.DataFrame, candidate_train_patients: Sequence[str]) -> pd.DataFrame:
    # separação de eventos de DEATH com paciente e ano
    death_py = df_train[
        df_train['event'].str.contains('DEATH', case=False, na=False)
    ][['patient_id', 'year']].drop_duplicates()

    pos_patient_candidates = death_py['patient_id'].unique().tolist()

    # cria dict de [id -> ano do primeiro evento de DEATH]
    death_year_map = death_py.groupby('patient_id')['year'].min().to_dict()

    # lista de anos associada a cada paciente
    presence = df_train[['patient_id', 'year']].drop_duplicates()
    presence_groups = presence.groupby('patient_id')['year'].apply(lambda s: sorted(s.tolist())).to_dict()

    def available_years_for_patient(pid: str) -> list[int]:
        # seleciona anos associados ao paciente filtrando os que vêm depois de um evento de DEATH
        years = presence_groups.get(pid, [])
        dy = death_year_map.get(pid, None)
        if dy is not None:
            years = [y for y in years if y <= dy]

        return years

    rows = []
    for pid in sorted(set(candidate_train_patients)):
        years_avail = available_years_for_patient(pid)
        rows.append({
            'patient_id': pid,
            'n_avail_rows': len(years_avail),
            'years_avail': years_avail,
            'is_pos': pid in pos_patient_candidates
        })

    out = pd.DataFrame(rows)
    return out[out['n_avail_rows'] > 0].copy()

def select_equal_patients_with_line_cap(patients_df: pd.DataFrame, target_pos_lines:int = 5000, target_neg_lines:int = 5000, rng_seed: int = 42):
    # separando pacientes por desfecho
    pos_df = patients_df[patients_df['is_pos'] == True ].copy()
    neg_df = patients_df[patients_df['is_pos'] == False].copy()

    # shuffle
    pos_df = pos_df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)
    neg_df = neg_df.sample(frac=1, random_state=rng_seed+1).reset_index(drop=True)

    # coluna cumulativa que soma as linhas dos pacientes
    pos_df['cum_rows'] = pos_df['n_avail_rows'].cumsum()
    neg_df['cum_rows'] = neg_df['n_avail_rows'].cumsum()

    # itera até conseguir ficar abaixo dos limites de linhas
    # para achar o índice de cutoff para ambos grupos
    max_n_possible = min(len(pos_df), len(neg_df))
    feasible_ns = []
    for n in range(1, max_n_possible + 1):
        pos_rows = int(pos_df.loc[n - 1, 'cum_rows'])
        neg_rows = int(neg_df.loc[n - 1, 'cum_rows'])
        if pos_rows <= target_pos_lines and neg_rows <= target_neg_lines:
            feasible_ns.append((n, pos_rows, neg_rows))

    if feasible_ns:
        n_selected, pos_rows_sel, neg_rows_sel = feasible_ns[-1]
        selected_pos = pos_df['patient_id'].iloc[:n_selected].tolist()
        selected_neg = neg_df['patient_id'].iloc[:n_selected].tolist()
        return selected_pos, selected_neg, n_selected, pos_rows_sel, neg_rows_sel

    n_pos_max = int((pos_df['cum_rows'] <= target_pos_lines).sum())
    n_neg_max = int((neg_df['cum_rows'] <= target_neg_lines).sum())
    n_selected = min(n_pos_max, n_neg_max)

    if n_selected > 0:
        pos_rows_sel = int(pos_df.loc[n_selected - 1, 'cum_rows'])
        neg_rows_sel = int(neg_df.loc[n_selected - 1, 'cum_rows'])
        selected_pos = pos_df['patient_id'].iloc[:n_selected].tolist()
        selected_neg = neg_df['patient_id'].iloc[:n_selected].tolist()
        return selected_pos, selected_neg, n_selected, pos_rows_sel, neg_rows_sel

    return [], [], 0, 0, 0


def build_pivot_preserve_presence(df: pd.DataFrame, patients: Sequence[str], years: Sequence[int], events: Sequence[str]):
    df_sub = df[
        df['patient_id'].isin(patients)
        & df['year'].isin(years)
        & df['event'].isin(events)
    ].copy()

    # tabela que guarda, para cada par de [paciente, ano], o número de ocorrências de cada tipo de evento
    pivot = pd.pivot_table(
        df_sub,
        index=['patient_id', 'year'],
        columns='event',
        values='date',
        aggfunc='count',
        fill_value=0
    )

    return pivot.astype(int).sort_index()

def select_top_events_lgbm(df:pd.DataFrame, selected_train_patients: Sequence[str],
                           train_years: Sequence[int], m_candidates: int, final_top_k: int, lgb_params: dict)\
    -> list[str]:

    df_train_py_event = df[
        df['patient_id'].isin(selected_train_patients) & df['year'].isin(train_years)
    ][['patient_id', 'year', 'event']].drop_duplicates()

    # seleciona eventos candidatos (top m em frequẽncia)
    event_freq = df_train_py_event.groupby('event').size().sort_values(ascending=False)
    candidate_events = [e for e in event_freq.index if e.upper() != 'DEATH'][:m_candidates]

    events_for_pivot = set(candidate_events) | {'DEATH'}
    pivot_candidate = build_pivot_preserve_presence(df, selected_train_patients, train_years, list(events_for_pivot))
    if 'DEATH' not in pivot_candidate.columns:
        pivot_candidate['DEATH'] = 0

    # representa todos pares de [id, ano] em um vetor que indica se há DEATH associada a ele
    pivot_candidate = pivot_candidate.sort_index().astype(int)

    availbale_candidate_events = [c for c in candidate_events if c in pivot_candidate.columns]
    X_lgb = pivot_candidate[availbale_candidate_events].astype(np.float32)
    print([col for col in X_lgb.columns])
    y_lgb = (pivot_candidate['DEATH'] > 0).astype(int).values

    pos = y_lgb.sum()
    neg = len(y_lgb) - pos

    if pos == 0:
        raise RuntimeError('Nenhum paciente da classe positiva no lgbm')

    params = dict(lgb_params)

    # proporção de pacientes negativos e positivos
    params['scale_pos_weight'] = (neg / pos) if pos > 0 else 1.0

    lgb_train = lgb.Dataset(X_lgb, y_lgb)
    gbm = lgb.train(params, lgb_train, num_boost_round=200)

    importance_df = pd.DataFrame(
        {
            'feature': availbale_candidate_events,
            'importance': gbm.feature_importance(importance_type='gain')
         }
    ).sort_values('importance', ascending=False)

    return importance_df['feature'].tolist()[:final_top_k]


def trim_post_death_rows(pivot_df: pd.DataFrame):
    if 'DEATH' not in pivot_df.columns:
        return pivot_df.copy()

    death_rows = pivot_df[pivot_df['DEATH'] > 0].reset_index()[['patient_id', 'year']].drop_duplicates()

    death_year_map = death_rows.groupby("patient_id")["year"].min().to_dict()

    # frame com colunas [paciente, ano]
    idx = pivot_df.index.to_frame(index=False)
    keep_mask = []

    # pra cada linha, se o ano for maior que o de evento DEATH do paciente, não incluimos na lista final
    for _, r in idx.iterrows():
        pid = r['patient_id']
        yr = int(r['year'])
        dy = death_year_map.get(pid, None)
        keep_mask.append(False if (dy is not None and yr > dy) else True)

    return pivot_df.iloc[np.asarray(keep_mask, dtype=bool)].copy()



def build_train_test_rows(df: pd.DataFrame, selected_train_patients: Sequence[str], candidate_test_patients: Sequence[str],
                          train_years: Sequence[int], test_years: Sequence[int], top_k_events: Sequence[str]):
    events_to_keep = set(top_k_events) | {'DEATH'}
    pivot_train = build_pivot_preserve_presence(df, selected_train_patients, train_years, list(events_to_keep))
    if 'DEATH' not in pivot_train.columns:
        pivot_train['DEATH'] = 0
    pivot_train = pivot_train.sort_index().astype(int)
    train_rows = trim_post_death_rows(pivot_train).reset_index().copy()

    pivot_test = build_pivot_preserve_presence(df, candidate_test_patients, test_years, list(events_to_keep))
    if 'DEATH' not in pivot_test.columns:
        pivot_test['DEATH'] = 0
    test_rows = pivot_test.astype(int).reset_index().copy()


    missing_cols = [c for c in top_k_events if c not in test_rows.columns]
    if missing_cols:
        test_rows = pd.concat(
            [test_rows, pd.DataFrame(0, index=test_rows.index, columns=missing_cols)],
            axis=1
        )

    cols = ['patient_id', 'year'] + list(top_k_events) + ['DEATH']
    train_rows = train_rows[cols]
    test_rows = test_rows[cols]

    return train_rows, test_rows

def prepare_tabpfn_rows(df: pd.DataFrame, cfg: TabPFNPrepConfig, lgbm_params: dict) -> dict[str, object]:
    years_all = sorted(df['year'].unique())
    # separação dos anos
    train_years, test_years = infer_train_test_years(years_all=years_all, forced_start=cfg.forced_train_year_start,
                                                     forced_end=cfg.forced_test_year_start)
    all_patients = np.asarray(df['patient_id'].unique(), dtype=str)
    # separação dos pacientes
    train_patients, test_patients = split_patients(all_patients, test_size=0.1, random_state=cfg.rng_seed)

    # recriação do df com apenas pacientes e anos dentro dos conjuntos de treino
    df_train = df[
        df['patient_id'].isin(train_patients) & df['year'].isin(train_years)
    ][['patient_id', 'year', 'event', 'date']].drop_duplicates()

    patients_df = build_patient_availability_table(df_train, train_patients)

    pos_sel, neg_sel, n_each, _, _ = select_equal_patients_with_line_cap(patients_df, target_pos_lines=cfg.target_pos_lines,
                                                                         target_neg_lines=cfg.target_neg_lines, rng_seed=cfg.rng_seed)

    # se não conseguir selecionar pacientes, seleciona até o tamanho da menor classe
    if n_each == 0:
        n_fallback = min(
            len(patients_df[patients_df['is_pos'] == True]),
            len(patients_df[patients_df['is_pos'] == False])
        )

        if n_fallback == 0:
            raise RuntimeError('Sem pacientes em uma das classes')

        pos_df = (
            patients_df[patients_df['is_pos'] == True]
            .sample(frac=1, random_state=cfg.rng_seed)
            .reset_index(drop=True)
        )
        neg_df = (
            patients_df[patients_df['is_pos'] == False]
            .sample(frac=1, random_state=cfg.rng_seed + 1)
            .reset_index(drop=True)
        )

        pos_sel = pos_df['patient_id'].iloc[:n_fallback].tolist()
        neg_sel = neg_df['patient_id'].iloc[:n_fallback].tolist()

    # pacientes de treino são a união entre as classes selecionadas
    selected_train_patients = pos_sel + neg_sel

    top_k_events = select_top_events_lgbm(df, selected_train_patients, train_years,
                                          cfg.m_candidates, final_top_k=cfg.final_top_k, lgb_params=lgbm_params)

    train_rows, test_rows = build_train_test_rows(df, selected_train_patients, test_patients, train_years,
                                                  test_years, top_k_events)

    if len(train_rows) > cfg.max_total_rows:
        train_rows = train_rows.sample(n=cfg.max_total_rows, random_state=cfg.rng_seed).reset_index(drop=True)

    return {
        'train_rows': train_rows,
        'test_rows': test_rows,
        'top_k_events': top_k_events,
        'train_years': train_years,
        'test_years': test_years,
        'selected_train_patients': selected_train_patients
    }

def prepare_database(df: pd.DataFrame):
    df = canonicalize_event_df(df)
    cfg = TabPFNPrepConfig()
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": 42,
        "n_jobs": 4,
        "num_leaves": 64,
        "learning_rate": 0.1,
        "max_depth": -1,
    }

    return prepare_tabpfn_rows(df, cfg, lgb_params)

def get_tabpfn_arrays(p: dict[str, object]):
    train_rows = p["train_rows"]
    test_rows = p["test_rows"]
    top_k_events = p["top_k_events"]

    X_train_np = train_rows[top_k_events].to_numpy(dtype=np.float32, copy=True)
    y_train_np = (train_rows['DEATH'] > 0).astype(int).to_numpy(copy=True)
    years_train_np = train_rows['year'].astype(int).to_numpy(copy=True)

    X_test_np = test_rows[top_k_events].to_numpy(dtype=np.float32, copy=True)
    y_test_np = (test_rows['DEATH'] > 0).astype(int).to_numpy(copy=True)
    years_test_np = test_rows['year'].astype(int).to_numpy(copy=True)

    return {
        'X_train': X_train_np,
        'y_train': y_train_np,
        'years_train': years_train_np,
        'X_test': X_test_np,
        'y_test': y_test_np,
        'years_test': years_test_np
    }
