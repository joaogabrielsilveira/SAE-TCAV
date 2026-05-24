import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from typing_extensions import Any
from pickle import dump, load
import os
from filepaths import get_env_path
from tabpfn import TabPFNClassifier
from tabpfn_model import make_dist_tensor
from torch.func import grad, vmap
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
CAVS_FILE = get_env_path('models/tcav/cavs.pkl')
GRADS_FILE = get_env_path('models/tcav/grads.pkl')

# class LogisticRegression(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(in_features=288, out_features=1)
#         self.criterion = nn.BCELoss()
#         self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)

#     def forward(self, x):
#         return torch.sigmoid(self.linear(x)).squeeze()
    
#     def loss(self, y_pred, y_true):
#         return self.criterion(y_pred, y_true.float())

# def train_logistic_regression(model: LogisticRegression, x: np.ndarray, y: np.ndarray, epochs: int = 1000):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.to(device)
#     x_t = torch.tensor(x, dtype=torch.float32).to(device)
#     y_t = torch.tensor(y, dtype=torch.float32).to(device)

#     for epoch in range(epochs):
#         model.train()
#         y_pred = model(x_t)
#         loss = model.loss(y_pred, y_t)
#         model.optimizer.zero_grad()
#         loss.backward()
#         model.optimizer.step()

#         model.eval()
#         if epoch % 100 == 0:
#             with torch.no_grad():
#                 y_pred = model(x_t)
#                 pred_labels = (y_pred > 0.5).cpu().numpy()
#                 acc = (pred_labels == y).mean()
#                 print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')

#     return model.to('cpu')

def get_model_gradients(model: TabPFNClassifier, dist_vec: np.ndarray, X: np.ndarray) -> np.ndarray:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(GRADS_FILE):
        with open(GRADS_FILE, 'rb') as f:
            grads = load(f)
            print(f'Carregando grads de {GRADS_FILE}')
            return grads

    model_decode_layer = model.model_processed_.decoder_dict['standard'].to(device)
    BATCH_SIZE = 128
    gradients = []
    for s in range(0, X.shape[0], BATCH_SIZE):
        e = min(s + BATCH_SIZE, X.shape[0])

        print(f'Batch {s}-{e}')
        x_batch = X[s:e].astype(np.float32)
        dist_batch = dist_vec[s:e]

        dist_t = torch.tensor(dist_batch, dtype=torch.long, device='cpu').reshape(-1, 1, 1)

        with torch.enable_grad():
            emb = model.get_embeddings(x_batch, additional_x={"dist_shift_domain": dist_t})
            if emb.ndim == 3 and emb.shape[0] == 1:
                emb = emb[0]
            elif emb.ndim == 3 and emb.shape[1] == 1:
                emb = emb.squeeze(1)

            emb_in = emb.clone().detach().to(device, dtype=torch.float32).requires_grad_(True)
            emb_in.requires_grad = True
            print(emb_in.shape)

            def single_pass(x_p):
                out = model_decode_layer(x_p).unsqueeze(0)
                if out.ndim == 3:
                    out = out[0]
                return out[0, 1]

            batch_grad = vmap(grad(single_pass))(emb_in)
            gradients.append(batch_grad.detach().cpu().numpy())
            print(gradients[0].shape)
    
    with open(GRADS_FILE, 'wb') as f:
        dump(np.vstack(gradients), f)
    return np.vstack(gradients)

def calculate_tcav_score(cav: np.ndarray, model_gradients: np.ndarray) -> float:
    prod = np.dot(model_gradients, cav)
    return np.mean(prod > 0)

def get_tcav_scores(cavs: list[dict], grads: np.ndarray) -> dict[int, float]:
    scores = {}
    for cav in cavs:
        idx, v = cav['Factor'], cav['CAV']
        scores[idx] = calculate_tcav_score(v, grads)
    
    return scores

def extract_rule_conditions(path: str):
    print(path)
    conds_list = []
    conditions = path.split(' AND ')
    for condition in conditions:
        condition = condition.strip()
        # print(condition)
        op = ''
        if '>=' in condition:
            feat, thresh = condition.split(' >= ')
            op = '>='
        elif '>' in condition:
            feat, thresh = condition.split(' > ')
            op = '>'
        elif '<=' in condition:
            feat, thresh = condition.split(' <= ')
            op = '<='
        elif '<' in condition:
            feat, thresh = condition.split(' < ')
            op = '<'
        thresh = float(thresh.strip())
        conds_list.append((feat.strip(), op, thresh))
    
    # print(conds_list)
        
    return conds_list

def get_rule_mask(X_df: pd.DataFrame, conditions: list[Any], feature_cols: list[str]) -> np.ndarray:
    mask = np.ones(X_df.shape[0], dtype=bool)

    for cond in conditions:
        feat, op, thresh = cond
        # feat_idx = feature_cols.index(feat)
        values = X_df[feat].values
        if op == '>':
            mask = mask & (values > thresh)
        elif op == '>=':
            mask = mask & (values >= thresh)
        elif op == '<':
            mask = mask & (values < thresh)
        elif op == '<=':
            mask = mask & (values <= thresh)
    
    return mask

def train_cavs_from_rules(rules_per_percentile: dict[str, Any], X_cav_train_df: pd.DataFrame, 
                          cav_train_emb: np.ndarray, cav_train_emb_encoded: np.ndarray, y_cav_train: np.ndarray,
                          feature_cols: list[str], emb_scaler: StandardScaler,
                           high_quantile: float=0.5, min_pos_samples: int=50, random_state: int=42) -> dict[Any, Any]:
    rng = np.random.default_rng(random_state)
    cav_dict = {}
    cav_list = []

    for rule in rules_per_percentile:
        factor = rule['Factor']
        path = rule['Rule']

        encoded_activations = cav_train_emb_encoded[:, factor]
        n_pos_encoded_activations = (encoded_activations > 0).sum()
        if n_pos_encoded_activations == 0:
            continue

        conds = extract_rule_conditions(path)
        mask = get_rule_mask(X_cav_train_df, conds, feature_cols)
        
        # pacientes que seguem a regra
        true_idx = np.where(mask == True)[0]

        # se tiver muito poucos, é ruim de treinar uma regressão, desista
        if len(true_idx) < min_pos_samples:
            continue

        # / não seguem a regra
        false_idx = np.where(mask == False)[0]

        # ativações baixas devem ser consideravelmente abaixo do percentil que gerou a regra
        low_activation_thresh = np.quantile(encoded_activations, max(high_quantile - 0.1, 0.0))

        # idx_pos_low = [encoded_activations[true_idx] <= low_activation_thresh]
        # pacientes que nçao seguem a regra e têm ativação muito baixa
        # print(encoded_activations.shape, false_idx.shape)
        idx_neg_low = false_idx[encoded_activations[false_idx] <= low_activation_thresh]

        # o "alvo", quantidade ideal de negativos, é igual à de positivos (balanceamento de classes)
        n_neg_target = len(true_idx)

        
        # se possível, pegamops um número equivalente de amostras negativas
        if len(idx_neg_low) >= n_neg_target:
            idx_negative = rng.choice(idx_neg_low, size=n_neg_target, replace=False)
        
        # caso contrário, havendo o mínimo necessário, pegamos todas
        elif len(idx_neg_low) >= min_pos_samples:
            idx_negative = idx_neg_low
        
        # se nem o mínimo puder ser alcançado, desista
        else:
            continue

        idx_positive = true_idx

        # cria os vetores combinados de entrada e pred para a regressão
        X = np.vstack([cav_train_emb[idx_positive], cav_train_emb[idx_negative]])
        y = np.hstack([np.ones(len(idx_positive)), np.zeros(len(idx_negative))])
        
        # treina a regressão com os dados balanceados (simula fronteira que separa entradas que seguem e não seguem a regra)
        # essencialemente, ela aprende como as regras criadas a partir dos conceitos esparsos aparecem nos embeddings originais do modelo
        clf = LogisticRegression(C=0.1, solver='liblinear',penalty='l2', class_weight='balanced',
                                 max_iter=1000, random_state=random_state)
        clf.fit(X, y)
        
        # o vetor com coeficientes desta regressão, então, aponta para onde essa regra aumenta
        # essa direção é depois comparada com os gradientes do modelo
        cav = clf.coef_[0]
        nrm = np.linalg.norm(cav)
        if (nrm) > 0:
            cav /= nrm
        
        cav_list.append(cav)
        cav_dict[factor] = {
            'Factor': factor,
            'CAV': cav,
            'clf': clf,
            'Low_thresh': low_activation_thresh,
            'positive_idx': idx_positive,
            'negative_idx': idx_negative,
            'Rule': path
        }
    
    return cav_dict

def train_cav_from_subset(embs: np.ndarray, idx_pos: np.ndarray, idx_neg: np.ndarray,
                          scaler_emb: StandardScaler, sample_fraction: float=1.0, rng_seed: int=42) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)

    n_pos = len(idx_pos)
    n_sample_pos = max(2, int(n_pos * sample_fraction))
    idx_pos_sample = rng.choice(idx_pos, size=min(n_sample_pos, n_pos), replace=False)

    n_neg = len(idx_neg)
    n_target_neg = len(idx_pos_sample)
    if n_neg == 0:
        return None
    
    idx_neg_sample = rng.choice(idx_neg, size=min(n_target_neg, n_neg), replace=False)
    emb_pos = embs[idx_pos_sample]
    y_pos = np.ones(len(idx_pos_sample))
    emb_neg = embs[idx_neg_sample]
    y_neg = np.zeros(len(idx_neg_sample))

    emb_total = np.vstack([emb_pos, emb_neg])
    y_total = np.hstack([y_pos, y_neg])

    clf = LogisticRegression(C=0.1, solver='liblinear',penalty='l2', class_weight='balanced',
                                 max_iter=1000, random_state=rng_seed)
    clf.fit(emb_total, y_total)

    v = clf.coef_[0]
    nrm = np.linalg.norm(v)
    if nrm > 0:
        v /= nrm
    
    return v

def generate_random_cav(embs: np.ndarray, scaler_emb: StandardScaler, n_samples: int, rng_seed: int=42) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    idx_all = np.arange(embs.shape[0])
    rng.shuffle(idx_all)

    n = min(n_samples, embs.shape[0] // 2)
    if n < 2:
        return None

    idx_pos = idx_all[0:n]
    idx_neg = idx_all[n:2*n]

    emb_pos = embs[idx_pos]
    y_pos = np.ones(n)
    emb_neg = embs[idx_neg]
    y_neg = np.zeros(n)

    emb_total = np.vstack([emb_pos, emb_neg])
    y_total = np.hstack([y_pos, y_neg])

    clf = LogisticRegression(C=0.1, solver='liblinear',penalty='l2', class_weight='balanced',
                                 max_iter=1000, random_state=rng_seed)
    clf.fit(emb_total, y_total)

    v = clf.coef_[0]
    nrm = np.linalg.norm(v)
    if nrm > 0:
        v /= nrm
    
    return v

def robust_tcav_significance_test(concept_idx: int, embs: np.ndarray, idx_pos: np.ndarray,
                                  idx_neg: np.ndarray, model_grads: np.ndarray, scaler_emb: StandardScaler,
                                  n_runs: int=15, sample_fraction: float =1.0, rng_seed: int=42) -> dict[str, Any]:
    
    # testes com dados negativos (que não obedecem à regra) aleatórios,
    # em vez de selecionar os que mais se distanciam da outra classe

    concept_cavs = []
    concept_tcav_scores = []

    for i in range(n_runs):
        seed = rng_seed + i * 100
        new_cav = train_cav_from_subset(
            embs=embs, idx_pos=idx_pos, idx_neg=idx_neg,
            scaler_emb=scaler_emb, sample_fraction=sample_fraction,
            rng_seed=seed
        )
        if new_cav is None:
            continue

        concept_cavs.append(new_cav)

        score = calculate_tcav_score(cav=new_cav, model_gradients=model_grads)
        concept_tcav_scores.append(score)

    random_cavs = []
    random_scores = []
    
    n_random_samples = max(20, min(len(idx_pos), len(idx_neg)) // 2)
    for i in range(n_runs):
        seed = rng_seed + 10000 + i * 100
        new_cav = generate_random_cav(
            embs=embs, scaler_emb=scaler_emb,
            n_samples=n_random_samples,
            rng_seed=seed
        )

        if new_cav is None:
            continue

        random_cavs.append(new_cav)

        score = calculate_tcav_score(cav=new_cav, model_gradients=model_grads)
        random_scores.append(score)
    
    concept_tcav_scores = np.asarray(concept_tcav_scores)
    random_scores = np.asarray(random_scores)

    std_dev_concept = np.std(concept_tcav_scores)
    std_dev_random = np.std(random_scores)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        t, p_value = stats.ttest_ind(concept_tcav_scores, random_scores, nan_policy='raise')
        # input()
        # t, p_value = stats.ttest_ind(concept_tcav_scores, random_scores, nan_policy='raise')

    pooled_std_dev = np.sqrt(
                    (
                    (len(concept_tcav_scores) - 1) * std_dev_concept ** 2\
                    +(len(random_scores) - 1) * std_dev_random ** 2
                    ) / (len(concept_tcav_scores) + len(random_scores) - 2)
                )
    

    return {
        'Factor': concept_idx,
        'concept_cavs': concept_cavs,
        'random_cavs': random_cavs,
        'concept_scores': concept_tcav_scores,
        'random_scores': random_scores,
        'p_value': p_value,
        't_stat': t,
        'is_significant': (not np.isnan(p_value) or p_value < 0.05)
    }
    


