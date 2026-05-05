import time
from typing import Optional, Tuple, Any

import tabpfn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig
import torch
import os
import numpy as np
import pandas as pd
from database import impute_data, normalize_data
from filepaths import get_env_path
from importlib import resources
from sklearn.metrics import f1_score
from pathlib import Path

BATCH_SIZE = 512

class TabPFNEvalConfig:
    rng_seed: int = 42
    tabpfn_model_name: str = 'tabpfn_dist_model_1'
    batch_size_predict: int = BATCH_SIZE
    run_id: str = "demo_tabpfn_step1"

class EmbeddingExtractConfig:
    batch_size = 512
    max_extract: Optional[int] = None
    use_cache: bool = True

TRAINING_EMBEDDING_FILE = get_env_path('models/tabpfn/tabpfn_emb_train.npy')
TEST_EMBEDDING_FILE = get_env_path('models/tabpfn/tabpfn_emb_test.npy')
PRED_BIN_FILE = get_env_path('models/tabpfn/y_pred_bin.npy')
PRED_PROB_FILE = get_env_path('models/tabpfn/y_pred_prob.npy')

# função antiga que recupera os embeddings do tabpfn vanilla
# def get_tabpfn_model(arrays: dict[str, np.ndarray], get_embeddings=False, get_pred=False) -> (TabPFNClassifier |
#                                                                                               tuple[
#                                                                                                   TabPFNClassifier, np.ndarray, np.ndarray,
#                                                                                                   dict[
#                                                                                                       str, np.ndarray]
#                                                                                               ]):
#     """ Cria um modelo tabpfn com os dados fornecidos.
#         Se get_embeddings for verdadeiro, retorna uma tupla com o modelo e os embeddings de
#         treino e teste, repectivamente. Se possível, os embeddings são extraídos de um arquivo salvo,
#         caso contrário são extraídos do próprio modelo. """
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     # Separação dos dados de treino e teste
#     X_train = arrays['X_train']
#     y_train = arrays['y_train']
#     X_test = arrays['X_test']
#     y_test = arrays['y_test']
#
#     # Imputação dos dados
#     X_train_imputed, y_train_imputed = impute_data(X_train), y_train
#     X_test_imputed, y_test_imputed = impute_data(X_test), y_test
#
#     # Normalização dos dados
#     X_train_normalized, y_train_normalized = normalize_data(X_train_imputed), y_train_imputed.astype(np.float64)
#     X_test_normalized, y_test_normalized = normalize_data(X_test_imputed), y_test_imputed.astype(np.float64)
#
#     clf = TabPFNClassifier(device=device, n_estimators=1)
#     clf.fit(X_train_normalized, y_train_normalized)
#     y_pred_bin = None
#     y_pred_prob = None
#
#     if get_pred:
#         if os.path.exists(PRED_BIN_FILE):
#             y_pred_bin = np.load(PRED_BIN_FILE)
#         else:
#             y_pred_list = []
#             for i in range(0, X_test_normalized.shape[0], BATCH_SIZE):
#                 print(f'Batch {i // BATCH_SIZE}: {i}-{i + BATCH_SIZE}')
#                 y_pred_list.append(clf.predict(X_test_normalized[i:i + BATCH_SIZE, :]))
#
#             y_pred_bin = np.concatenate(y_pred_list, axis=0)
#
#             with open(PRED_BIN_FILE, 'wb') as pred:
#                 np.save(pred, y_pred_bin)
#
#         if os.path.exists(PRED_PROB_FILE):
#             y_pred_prob = np.load(PRED_PROB_FILE)
#         else:
#             y_pred_list = []
#             for i in range(0, X_test_normalized.shape[0], BATCH_SIZE):
#                 print(f'Batch {i // BATCH_SIZE}: {i}-{i + BATCH_SIZE}')
#                 y_pred_list.append(clf.predict_proba(X_test_normalized[i:i + BATCH_SIZE, :])[:, 1])
#
#             y_pred_prob = np.concatenate(y_pred_list, axis=0)
#
#             with open(PRED_PROB_FILE, 'wb') as pred:
#                 np.save(pred, y_pred_prob)
#
#     if get_embeddings:
#         if os.path.exists(TRAINING_EMBEDDING_FILE) and os.path.exists(TEST_EMBEDDING_FILE):
#             print("Extraindo embeddings do arquivo")
#             train_embeddings = np.load(TRAINING_EMBEDDING_FILE)
#             test_embeddings = np.load(TEST_EMBEDDING_FILE)
#         else:
#             print("Extraindo embeddings do modelo")
#             embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=10)
#             if not os.path.exists(TRAINING_EMBEDDING_FILE):
#                 train_emb_list = []
#                 for i in range(0, X_train_normalized.shape[0], BATCH_SIZE):
#                     print(f'Batch {i // BATCH_SIZE}: {i}-{i+BATCH_SIZE}')
#                     train_embeddings = embedding_extractor.get_embeddings(X_train_normalized, y_train_normalized,
#                                                                           X_train_normalized[i:i + BATCH_SIZE, :],
#                                                                           data_source='train')
#                     if train_embeddings.ndim == 3:
#                         train_embeddings = np.mean(train_embeddings, axis=0)
#                     train_emb_list.append(train_embeddings)
#                 train_embeddings = np.concatenate(train_emb_list, axis=0)
#
#                 with open(TRAINING_EMBEDDING_FILE, 'wb') as train_emb:
#                     np.save(train_emb, train_embeddings)
#             else:
#                 train_embeddings = np.load(TRAINING_EMBEDDING_FILE)
#
#             if not os.path.exists(TEST_EMBEDDING_FILE):
#                 test_emb_list = []
#                 for i in range(0, X_test_normalized.shape[0], BATCH_SIZE):
#                     print(f'Batch {i // BATCH_SIZE}: {i}-{i + BATCH_SIZE}')
#                     test_embeddings = embedding_extractor.get_embeddings(X_train_normalized, y_train_normalized,
#                                                                          X_test_normalized[i:i + BATCH_SIZE, :],
#                                                                          data_source='test')
#                     if test_embeddings.ndim == 3:
#                         test_embeddings = np.mean(test_embeddings, axis=0)
#                     test_emb_list.append(test_embeddings)
#                 test_embeddings = np.concatenate(test_emb_list, axis=0)
#
#                 with open(TEST_EMBEDDING_FILE, 'wb') as test_emb:
#                     np.save(test_emb, test_embeddings)
#             else:
#                 test_embeddings = np.load(TEST_EMBEDDING_FILE)
#
#         return clf, train_embeddings, test_embeddings, {
#             'X_train': X_train, 'X_test': X_test,
#             'y_train': y_train, 'y_test': y_test,
#             'X_train_imputed': X_train_imputed,
#             'X_test_imputed': X_test_imputed,
#             'y_train_imputed': y_train_imputed,
#             'y_test_imputed': y_test_imputed,
#             'X_train_normalized': X_train_normalized,
#             'X_test_normalized': X_test_normalized,
#             'y_train_normalized': y_train_normalized,
#             'y_test_normalized': y_test_normalized,
#             'y_pred_bin': y_pred_bin,
#             'y_pred_prob': y_pred_prob
#         }
#
#     return clf

def infer_model_additional_x_info(model):
    device = torch.device('cpu')
    example_add_shape = None
    try:
        if hasattr(model, "additional_x_") and model.additional_x_ is not None and 'dist_shift_domain' in model.additional_x_:
            print("add x tudo certo")
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
    year_to_domain_train = {y: i for i, y in sorted(set(enumerate(train_years)))}
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
            device="auto",
        )

        if hasattr(drift_model, "show_progress"):
            drift_model.show_progress = False
        if hasattr(drift_model, "seed"):
            drift_model.seed = eval_cfg.rng_seed

    except Exception:
        drift_model = TabPFNClassifier(device='auto')

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
                                example_add_shape = Optional[Tuple[int, ...]],) -> dict[str, Any]:
    test_rows = ensure_test_feature_columns(test_rows, top_k_events)

    combined_years = sorted(set(train_years).union(set(test_rows['year'].unique().astype(int).tolist())))
    year_to_domain_combined = {y: i for i, y in enumerate(combined_years)}

    results_per_year: list[dict[str, Any]] = []
    t_infer_total = 0.0

    for eval_year in sorted(test_rows['year'].unique().astype(int)):
        # eventos do ano de teste
        df_year = test_rows[test_rows['year'] == eval_year].reset_index(drop=True)
        if df_year.shape[0] == 0:
            continue

        n_samples = df_year.shape[0]
        y_true = (df_year['DEATH'] > 0).astype(int).values

        # label de domínio temporal para todas linhas
        dist_dom_all = np.array(
            [year_to_domain_combined[int(y)] for y in df_year['year'].astype(int).tolist()],
            dtype=np.int64
        )

        preds_list = []
        t0_year = time.perf_counter()

        for start in range(0, n_samples, batch_size_predict):
            end = min(start+batch_size_predict, n_samples)

            # entradas e domínios temporais para o batch atual
            Xb_np = df_year[top_k_events].iloc[start:end].values.astype(np.float32)
            dist_dom_np = dist_dom_all[start:end]

            dist_dom_t = make_dist_tensor(
                dist_dom_np,
                model_add_x_device,
                example_add_shape,
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
        f1_pos = f1_score(y_true, y_pred, average='micro') if len(np.unique(y_true)) > 1 else float('nan')

        results_per_year.append(
            {
                'year': int(eval_year),
                'n_samples': int(n_samples),
                'n_deaths': int(y_true.sum()),
                'f1_macro': float(f1_macro) if not np.isnan(f1_macro) else float('nan'),
                'f1_pos': float(f1_pos) if not np.isnan(f1_pos) else float('nan'),
                'infer_time_sec': float(t_infer_year)
            }
        )

    return {
        'results_per_year': results_per_year,
        'total_infer_time_sec': float(t_infer_total),
        'year_to_domain_combined': year_to_domain_combined,
        'test_rows_checked': test_rows,
    }


def batch_get_embeddings(model: TabPFNClassifier, X_all: np.ndarray, dist_full: np.ndarray, batch_size: int = 512,
                         device: torch.device | str = 'cpu',
                         example_add_shape: Optional[Tuple[int, ...]] = None) -> tuple[np.ndarray, list]:
    out_list = []
    tensors_list = []

    n = X_all.shape[0]

    for start in range(0, n, batch_size):
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
        except Exception:
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

def scale_data(train_rows: pd.DataFrame):
    pass

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
        example_add_shape=example_add_shape
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
    embeddings_dir = get_env_path(embeddings_dir)

    p_train = embeddings_dir / "dr_tabpfn_train_emb.npy"
    p_test = embeddings_dir / "dr_tabpfn_test_emb.npy"
    # p_train_flat = embeddings_dir / "dr_tabpfn_train_emb_flat.npy"
    # p_test_flat = embeddings_dir / "dr_tabpfn_test_emb_flat.npy"

    if cfg.use_cache and p_train.exists():
        train_emb = np.load(p_train)
    else:
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

    if cfg.use_cache and p_train.exists():
        test_emb = np.load(p_test)
    else:
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

    np.save(p_train, train_emb)
    np.save(p_test, test_emb)

    train_emb_flat = flatten_embeddings(train_emb)
    test_emb_flat = flatten_embeddings(test_emb)

    return {
        "train_emb": train_emb,
        "test_emb": test_emb,
        "train_emb_flat": train_emb_flat,
        "test_emb_flat": test_emb_flat,
    }

def scale_embeddings(emb_train: np.ndarray, emb_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    scaler.fit(emb_train)
    emb_train_new = scaler.transform(emb_train)
    emb_test_new = scaler.transform(emb_test)

    return emb_train_new, emb_test_new

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