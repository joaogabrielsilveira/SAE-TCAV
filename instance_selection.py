from tabpfn_model import *
from sae import *
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from collections.abc import Callable
from sklearn.metrics import f1_score, roc_auc_score

def select_closest_patients_balanced(codes_infer: np.ndarray, codes_train: np.ndarray,
                            pos_idx: np.ndarray, neg_idx: np.ndarray, k: int=250):
    cos_sim_pos = cosine_similarity(X=codes_infer, Y=codes_train[pos_idx])
    most_similar_pos = np.argsort(cos_sim_pos, axis=1)[:, ::-1][:, :k]
    cos_sim_neg = cosine_similarity(X=codes_infer, Y=codes_train[neg_idx])
    most_similar_neg = np.argsort(cos_sim_neg, axis=1)[:, ::-1][:, :k]

    global_pos = pos_idx[most_similar_pos]
    global_neg = neg_idx[most_similar_neg]

    global_all = np.hstack([global_pos, global_neg])

    return global_all

def select_closest_patients(codes_infer: np.ndarray, codes_train: np.ndarray,
                            pos_idx: np.ndarray, neg_idx: np.ndarray, k: int=250):
    
    cos_sim = cosine_similarity(X=codes_infer, Y=codes_train)
    most_similar_global = np.argsort(cos_sim, axis=1)[:, ::-1][:, :(k*2)]
    
    return most_similar_global

def select_all(codes_infer: np.ndarray, codes_train: np.ndarray,
                pos_idx: np.ndarray, neg_idx: np.ndarray, k: int=250):
    idx_all = np.hstack([pos_idx, neg_idx])
    return np.tile(idx_all, (codes_infer.shape[0], 1))

def infer_with_selected_instances(base_model: TabPFNClassifier, base_sae: SAE, X_train_np: np.ndarray,
            y_train_np: np.ndarray, years_train_np: np.ndarray, train_embs: np.ndarray, X_infer_np: pd.DataFrame,
            y_infer_np: np.ndarray, years_infer_np: np.ndarray, emb_scaler: StandardScaler, 
            add_x_device: str, add_x_shape: np.ndarray, select_func: Callable = select_closest_patients) -> dict[str]:
    first_year = int(np.min(years_train_np))
    last_year =  int(max(np.max(years_infer_np), np.max(years_train_np)))

    
    all_years = range(first_year, last_year+1)
    year_to_domain_map = {y: i for i, y in enumerate(all_years)}

    infer_embs = extract_embeddings_robust(model=base_model, X=X_infer_np, year_to_domain_map=year_to_domain_map,
                                           years=years_infer_np, is_train=False).squeeze()
    infer_embs_scaled = emb_scaler.transform(X=infer_embs)

    train_embs_t = torch.tensor(train_embs, dtype=torch.float32)
    infer_embs_t = torch.tensor(infer_embs_scaled, dtype=torch.float32)

    codes_train = np.asarray(base_sae.encode(train_embs_t).detach(), dtype=np.float32)
    codes_infer = np.asarray(base_sae.encode(infer_embs_t).detach(), dtype=np.float32)

    pos_idx = np.where(y_train_np == 0)[0]
    neg_idx = np.where(y_train_np > 0)[0]

    k = 250
    indices = select_func(codes_infer, codes_train, pos_idx, neg_idx, k)

    preds = []
    for patient in range(len(X_infer_np)):
        patient_indices = indices[patient, :]
        patient_year = int(years_infer_np[patient])
        dist_fit = make_dist_tensor(dist_dom_np=np.asarray([year_to_domain_map[y] for y in years_train_np[patient_indices]]), model_add_x_device=add_x_device, example_add_shape=add_x_shape)
        dist_patient = make_dist_tensor(dist_dom_np=np.asarray([year_to_domain_map[patient_year]]), model_add_x_device=add_x_device, example_add_shape=add_x_shape)

        base_model.fit(X=X_train_np[patient_indices, :], y=y_train_np[patient_indices], additional_x={'dist_shift_domain': dist_fit})
        pred = base_model.predict_proba(X=X_infer_np[patient].reshape(1, -1), additional_x={'dist_shift_domain': dist_patient})
        preds.append(pred)

    preds_prob = np.vstack(preds)
    preds_bin = np.argmax(preds_prob, axis=1)

    f1_pos = f1_score(y_true=y_infer_np, y_pred=preds_bin)
    f1_macro = f1_score(y_true=y_infer_np, y_pred=preds_bin, average='macro')
    roc_auc = roc_auc_score(y_true=y_infer_np, y_score=preds_prob[:, 1])

    return {
        'preds_prob': preds_prob,
        'preds_bin': preds_bin,
        'f1_pos': f1_pos,
        'f1_macro': f1_macro,
        'roc_auc': roc_auc,
    }
