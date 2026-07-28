import time
from typing import Optional, Tuple, Any
import tabpfn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig
import torch
import numpy as np
import pandas as pd
from filepaths import get_env_path
from importlib import resources
from sklearn.metrics import f1_score
from pathlib import Path
import os
from pickle import dump, load
from progress_utils import progress_iter

BATCH_SIZE = 512

class TabPFNEvalConfig:
    rng_seed: int = 42
    tabpfn_model_name: str = 'tabpfn_dist_model_1'
    batch_size_predict: int = BATCH_SIZE
    run_id: str = "demo_tabpfn_step1"
    device: str = "auto"
    show_progress: bool = False

class EmbeddingExtractConfig:
    batch_size = 512
    max_extract: Optional[int] = None
    use_cache: bool = True
    show_progress: bool = False
    progress_desc: str = "Extracting embeddings"

TRAINING_EMBEDDING_FILE = get_env_path('models/tabpfn/dr_tabpfn_train_emb.npy')
TEST_EMBEDDING_FILE = get_env_path('models/tabpfn/dr_tabpfn_test_emb.npy')
PRED_BIN_FILE = get_env_path('models/tabpfn/y_pred_bin.npy')
PRED_PROB_FILE = get_env_path('models/tabpfn/y_pred_prob_')


def infer_model_additional_x_info(model):
    device = torch.device('cpu')
    example_add_shape = None
    try:
        if hasattr(model, "additional_x_") and model.additional_x_ is not None and 'dist_shift_domain' in model.additional_x_:
            v = model.additional_x_['dist_shift_domain']
            if isinstance(v, torch.Tensor):
                device = v.device
                example_add_shape = tuple(v.shape)
    except Exception:
        device = torch.device('cpu')
        example_add_shape = None

    return device, example_add_shape


def fit_dr_tabpfn(X_train: np.ndarray, y_train: np.ndarray, train_years: np.ndarray, eval_cfg: TabPFNEvalConfig):
    X_train_np = np.ascontiguousarray(X_train.astype(np.float32, copy=True))
    y_train_np = np.ascontiguousarray(y_train.astype(np.float32, copy=True))

    # permitir que o tabfpn modifique as entradas, se necessário
    X_train_np.setflags(write=True)
    y_train_np.setflags(write=True)

    train_years = np.asarray(train_years).astype(int)
    # um index (dominio) por ano
    year_to_domain_train = {y: i for i, y in enumerate(sorted(set(train_years.tolist())))}
    dist_shift_domain_train_np = np.ascontiguousarray(
        np.array([year_to_domain_train[int(y)] for y in train_years], dtype=np.int64)
    )

    dist_shift_domain_train_np.setflags(write=True)

    try:
        libpath = resources.files(tabpfn)
        model_path_config = TabPFNModelPathsConfig(
            paths=[f"{libpath}/model_cache/{eval_cfg.tabpfn_model_name}.cpkt"],
            task_type="dist_shift_multiclass",
        )

        drift_model = get_best_tabpfn(
            task_type="dist_shift_multiclass",
            model_type="single_fast",
            paths_config=model_path_config,
            debug=False,
            device=eval_cfg.device,
        )

        if hasattr(drift_model, "show_progress"):
            drift_model.show_progress = eval_cfg.show_progress
        if hasattr(drift_model, "seed"):
            drift_model.seed = eval_cfg.rng_seed
        model_source = "drift_resilient_best"
        model_resolution_error = None

    except Exception as error:
        drift_model = TabPFNClassifier(device=eval_cfg.device)
        model_source = "fallback_classifier"
        model_resolution_error = type(error).__name__

    t0 = time.perf_counter()
    drift_model = drift_model.fit(
        X_train_np,
        y_train_np,
        additional_x={"dist_shift_domain": dist_shift_domain_train_np}  # tempo incluso no domínio, e não apenas uma
                                                                        # variável tabular
    )

    fit_time_sec = time.perf_counter() - t0

    model_add_x_device, example_add_shape = infer_model_additional_x_info(drift_model)

    return {
        'model': drift_model,
        'fit_time_sec': float(fit_time_sec),
        'year_to_domain_train': year_to_domain_train,
        'dist_shift_domain_train': dist_shift_domain_train_np,
        'model_add_x_device': model_add_x_device,
        'example_add_shape': example_add_shape,
        'model_source': model_source,
        'model_resolution_error': model_resolution_error,
    }


def ensure_test_feature_columns(test_rows, top_k_events):
    out = test_rows.copy()
    missing = [c for c in top_k_events if c not in out.columns]
    if missing:
        zeros = pd.DataFrame(0, index=out.index, columns=missing)
        out = pd.concat([out, zeros], axis=1)

    return out


def make_dist_tensor(dist_dom_np, model_add_x_device, example_add_shape):
    t = torch.tensor(dist_dom_np, dtype=torch.long, device=model_add_x_device)
    if example_add_shape is not None and t.ndim == 1:
        if len(example_add_shape) >= 3:
            t = t.reshape(-1, 1, 1)
    else:
        if t.ndim == 1:
            t = t.reshape(-1, 1, 1)
    return t

def walkforward_evaluate_tabpfn(drift_model: TabPFNClassifier, test_rows: pd.DataFrame,
                                top_k_events: list[str], model_add_x_device: torch.device | str,
                                train_years: list[int], batch_size_predict:int = 512,
                                example_add_shape: Optional[Tuple[int, ...]] = None,
                                use_cache: bool = True,
                                test_years: list[int] | None = None,
                                show_progress: bool = False,
                                ) -> dict[str, Any]:
    test_rows = ensure_test_feature_columns(test_rows, top_k_events)

    combined_years = sorted(set(train_years).union(set(test_rows['year'].astype(int).unique().tolist())))
    if test_years is not None:
        print(len(test_rows))
        test_rows = test_rows[test_rows['year'].isin(test_years)]
        print(len(test_rows))
    year_to_domain_combined = {y: i for i, y in enumerate(combined_years)}
    
    results_per_year: list[dict[str, Any]] = []
    preds_per_year = []
    t_infer_total = 0.0

    eval_years = sorted(test_rows['year'].astype(int).unique())
    for eval_year in progress_iter(
        eval_years,
        enabled=show_progress,
        desc="TabPFN walk-forward",
        total=len(eval_years),
        unit="year",
    ):
        # eventos do ano de teste
        df_year = test_rows[test_rows['year'] == eval_year].reset_index(drop=True)
        #print(df_year.columns)
        if df_year.shape[0] == 0:
            continue

        n_samples = df_year.shape[0]
        y_true = (df_year['DEATH'] > 0).astype(int).values
        year_file = PRED_PROB_FILE + str(eval_year) + '.pkl'
        # label de domínio temporal para todas linhas
        dist_dom_all = np.array(
            [year_to_domain_combined[int(y)] for y in df_year['year'].astype(int)],
            dtype=np.int64
        )

        preds_list = []
        t0_year = time.perf_counter()
        if use_cache and os.path.exists(year_file):
            print('Carregando resultados do arquivo salvo')
            with open(year_file, 'rb') as f:
                results = load(f)
                preds_per_year.append((eval_year, results.pop('y_pred_proba')))
                results.pop('y_pred_bin', None)
                results_per_year.append(results)
            continue

        batch_starts = range(0, n_samples, batch_size_predict)
        for start in progress_iter(
            batch_starts,
            enabled=show_progress,
            desc=f"Predicting {eval_year}",
            total=len(batch_starts),
            unit="batch",
            leave=False,
        ):
            end = min(start+batch_size_predict, n_samples)

            # entradas e domínios temporais para o batch atual
            Xb_np = df_year[top_k_events].iloc[start:end].values.astype(np.float32)
            dist_dom_np = dist_dom_all[start:end]

            dist_dom_t = make_dist_tensor(
                dist_dom_np=dist_dom_np,
                model_add_x_device=model_add_x_device,
                example_add_shape=example_add_shape,
            )

            # predições para o batch
            if torch.device(model_add_x_device).type == "cpu":
                preds_batch = drift_model.predict_proba(
                    Xb_np,
                    additional_x={"dist_shift_domain": dist_dom_t}
                )
            else:
                # se preciso, move a entrada para a gpu
                Xb_t = torch.tensor(Xb_np, dtype=torch.float32, device=model_add_x_device)
                with torch.no_grad():
                    preds_batch = drift_model.predict_proba(
                        Xb_t,
                        additional_x={"dist_shift_domain": dist_dom_t}
                    )

            if isinstance(preds_batch, torch.Tensor):
                preds_batch = preds_batch.detach().cpu().numpy()

            preds_list.append(preds_batch)

        t_infer_year = time.perf_counter() - t0_year
        t_infer_total += t_infer_year

        preds_proba = np.vstack(preds_list)
        y_pred = np.argmax(preds_proba, axis=1)

        f1_macro = f1_score(y_true, y_pred, average='macro') if len(np.unique(y_true)) > 1 else float('nan')
        f1_pos = f1_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float('nan')
        results = {
                'y_pred_bin': y_pred,
                'y_pred_proba': preds_proba[:, 1],
                'year': int(eval_year),
                'n_samples': int(n_samples),
                'n_deaths': int(y_true.sum()),
                'f1_macro': float(f1_macro) if not np.isnan(f1_macro) else float('nan'),
                'f1_pos': float(f1_pos) if not np.isnan(f1_pos) else float('nan'),
                'infer_time_sec': float(t_infer_year)
            }
        if use_cache:
            with open(year_file, 'wb') as f:
                dump(results, f)
                for i in range(min(5, len(preds_proba))):
                    print(results['y_pred_proba'][i], preds_proba[i], y_pred[i])

                print(f'f1_macro: {results["f1_macro"]}, f1_pos: {results["f1_pos"]}')
                print(f'Resultados de {eval_year} salvos em {year_file}')
        preds_per_year.append((eval_year, results.pop('y_pred_proba')))
        results_per_year.append(results)
    return {
        'results_per_year': results_per_year,
        'total_infer_time_sec': float(t_infer_total),
        'year_to_domain_combined': year_to_domain_combined,
        'test_rows_checked': test_rows,
    }


def batch_get_embeddings(model: TabPFNClassifier, X_all: np.ndarray, dist_full: np.ndarray, batch_size: int = 512,
                         device: torch.device | str = 'cpu',
                         example_add_shape: Optional[Tuple[int, ...]] = None,
                         show_progress: bool = False,
                         progress_desc: str = "Extracting embeddings") -> tuple[np.ndarray, list]:
    out_list = []
    tensors_list = []

    n = X_all.shape[0]

    batch_starts = range(0, n, batch_size)
    for start in progress_iter(
        batch_starts,
        enabled=show_progress,
        desc=progress_desc,
        total=len(batch_starts),
        unit="batch",
        leave=False,
    ):
        end = min(start + batch_size, n)
        xb = X_all[start:end].astype(np.float32)
        dist_b = dist_full[start:end].astype(np.int64)

        dist_t = make_dist_tensor(
            dist_b,
            model_add_x_device=device,
            example_add_shape=example_add_shape
        )
        
        try:
            emb_b = model.get_embeddings(xb, additional_x={'dist_shift_domain': dist_t})
        except Exception as e:
            emb_b = model.get_embeddings(xb)

        if isinstance(emb_b, torch.Tensor):
            emb_b_np = emb_b.detach().cpu().numpy()
            emb_t = emb_b.detach().clone().requires_grad_(True)

        else:
            emb_b_np = np.asarray(emb_b)
            emb_t = torch.tensor(emb_b_np).clone().requires_grad_(True)

        out_list.append(np.asarray(emb_b_np))
        tensors_list.append(emb_t)

    return np.vstack(out_list), tensors_list

def scale_embeddings_l2(train_emb: np.ndarray, test_emb: np.ndarray):
    t_train = torch.tensor(train_emb, dtype=torch.float32)
    t_test  = torch.tensor(test_emb,  dtype=torch.float32)

    train_norm = torch.nn.functional.normalize(t_train, p=2, dim=1).numpy()
    test_norm  = torch.nn.functional.normalize(t_test,  p=2, dim=1).numpy()

    return train_norm, test_norm

def extract_embeddings_robust(model: TabPFNClassifier, X: np.ndarray, years: np.ndarray,
                               year_to_domain_map: dict[int, int], cfg: EmbeddingExtractConfig = EmbeddingExtractConfig(),
                               device: torch.device | str = 'cpu', is_train: bool = False, ctx_idx: Optional[np.ndarray] = None,
                               example_add_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    X_all = np.asarray(X, dtype=np.float32)
    years_all = np.asarray(years).astype(int)

    if cfg.max_extract is not None:
        X_all = X_all[:cfg.max_extract]
        years_all = years_all[:cfg.max_extract]

    if is_train and ctx_idx is not None:
        exclude_mask = np.ones(X_all.shape[0], dtype=bool)
        exclude_mask[ctx_idx] = False
        X_all = X_all[exclude_mask]
        years_all = years_all[exclude_mask]

    dist_vec = np.asarray([year_to_domain_map[int(y)] for y in years_all], dtype=np.int64)

    if not hasattr(model, 'get_embeddings'):
        raise RuntimeError("Modelo não tem método de extração de embeddings")

    emb, emb_t = batch_get_embeddings(
        model=model,
        X_all=X_all,
        dist_full=dist_vec,
        batch_size=cfg.batch_size,
        device=device,
        example_add_shape=example_add_shape,
        show_progress=cfg.show_progress,
        progress_desc=cfg.progress_desc,
    )

    return np.asarray(emb)


def flatten_embeddings(emb: np.ndarray):
    emb = np.asarray(emb)
    if emb.ndim == 1:
        return emb.reshape(-1, 1)
    if emb.ndim == 2:
        return emb
    if emb.ndim == 3 and emb.shape[1] == 1:
        sq = np.squeeze(emb, axis=1)
        return sq if sq.ndim == 2 else sq.reshape(-1, 1)
    return emb.reshape(emb.shape[0], -1)


def load_or_extract_embeddings(model: TabPFNClassifier, X_train_np: np.ndarray, X_test_np: np.ndarray,
                               years_train: np.ndarray,years_test: np.ndarray, year_to_domain_map: dict[int, int],
                               embeddings_dir: str | Path, cfg:EmbeddingExtractConfig = EmbeddingExtractConfig(),
                               device: torch.device | str = 'cpu', example_add_shape: Optional[Tuple[int, ...]] = None
                               ) -> dict[str, np.ndarray]:
    if embeddings_dir is not None:
        embeddings_dir = get_env_path(embeddings_dir)

    if embeddings_dir is None or embeddings_dir == '':
        p_train = TRAINING_EMBEDDING_FILE
        p_test = TEST_EMBEDDING_FILE
    else:
        train_years, test_years = np.unique(years_train), np.unique(years_test)
        if len(train_years) == 1:
            train_year_str = f'{train_years[0]}'
        else:
            train_year_str = f'{np.min(train_years)}-{np.max(train_years)}'
        
        p_train = get_env_path(f'models/tabpfn/dr_tabpfn_train_emb({train_year_str}).npy')
        
        if len(test_years) == 1:
            test_year_str = f'{test_years[0]}'
        else:
            test_year_str = f'{np.min(test_years)}-{np.max(test_years)}'
        
        p_test = get_env_path(f'models/tabpfn/dr_tabpfn_test_emb({test_year_str}).npy')

    print(p_train, p_test)
    if cfg.use_cache and os.path.exists(p_train):
        train_emb = np.load(p_train)
    elif X_train_np is not None:
        train_emb = extract_embeddings_robust(
            model=model,
            X=X_train_np,
            years=years_train,
            year_to_domain_map=year_to_domain_map,
            cfg=cfg,
            device=device,
            is_train=True,
            ctx_idx=None,
            example_add_shape=example_add_shape
        )
    else:
        train_emb = None

    if cfg.use_cache and os.path.exists(p_test):
        test_emb = np.load(p_test)
    elif X_test_np is not None:
        test_emb = extract_embeddings_robust(
            model=model,
            X=X_test_np,
            years=years_test,
            year_to_domain_map=year_to_domain_map,
            cfg=cfg,
            device=device,
            is_train=True,
            ctx_idx=None,
            example_add_shape=example_add_shape
        )
    else:
        test_emb = None
    
    if train_emb is not None:
        np.save(p_train, train_emb)
        train_emb_flat = flatten_embeddings(train_emb)
    else:
        train_emb_flat = None

    if test_emb is not None:
        np.save(p_test, test_emb)
        test_emb_flat = flatten_embeddings(test_emb)
    else:
        test_emb_flat = None

    return {
        "train_emb": train_emb,
        "test_emb": test_emb,
        "train_emb_flat": train_emb_flat,
        "test_emb_flat": test_emb_flat,
    }

def scale_embeddings(emb_train: np.ndarray, emb_test: np.ndarray, fit_test:bool = False) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    scaler.fit(emb_train)
    emb_train_new = scaler.transform(emb_train)
    if fit_test:
        scaler.fit(emb_test)
    emb_test_new = scaler.transform(emb_test)

    return emb_train_new, emb_test_new, scaler

def temporal_test_subsplits(y_test: np.ndarray, rng_seed: int = 42) -> dict[str, np.ndarray]:
    y_test = np.asarray(y_test)
    n_test = len(y_test)
    all_idx = np.arange(n_test)

    idx_discovery, idx_rest = train_test_split(
        all_idx,
        test_size=0.67,
        random_state=rng_seed,
        stratify=y_test,
    )

    idx_cav_train, idx_eval_hold = train_test_split(
        idx_rest,
        test_size=0.5,
        random_state=rng_seed,
        stratify=y_test[idx_rest],
    )

    idx_tcav_eval, idx_held_out = train_test_split(
        idx_eval_hold,
        test_size=0.5,
        random_state=rng_seed,
        stratify=y_test[idx_eval_hold],
    )

    # overlap safety checks (same notebook logic)
    assert len(set(idx_discovery) & set(idx_cav_train)) == 0, "Discovery overlaps with CAV Train!"
    assert len(set(idx_discovery) & set(idx_tcav_eval)) == 0, "Discovery overlaps with TCAV Eval!"
    assert len(set(idx_discovery) & set(idx_held_out)) == 0, "Discovery overlaps with Held-Out!"
    assert len(set(idx_cav_train) & set(idx_tcav_eval)) == 0, "CAV Train overlaps with TCAV Eval!"
    assert len(set(idx_cav_train) & set(idx_held_out)) == 0, "CAV Train overlaps with Held-Out!"
    assert len(set(idx_tcav_eval) & set(idx_held_out)) == 0, "TCAV Eval overlaps with Held-Out!"

    return {
        "idx_test_discover": np.asarray(idx_discovery),
        "idx_test_cav_train": np.asarray(idx_cav_train),
        "idx_test_tcav_eval": np.asarray(idx_tcav_eval),
        "idx_test_held_out": np.asarray(idx_held_out),
    }


def semantic_test_subsplits(
    y_test: np.ndarray,
    patient_ids: np.ndarray,
    rng_seed: int = 42,
    fractions: tuple[float, float, float, float] = (0.33, 0.335, 0.1675, 0.1675),
) -> dict[str, np.ndarray]:
    """Deterministic patient-grouped split for semantic rule experiments.

    Fractions correspond to rule fitting, rule selection, TCAV evaluation, and
    final semantic evaluation.  Entire patient histories stay in one split.
    The legacy row-level ``temporal_test_subsplits`` remains unchanged.
    """

    from semantic_splits import semantic_test_subsplits as grouped_split

    return grouped_split(y_test, patient_ids, rng_seed=rng_seed, fractions=fractions)
