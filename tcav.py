import numpy as np
import torch
from torch import nn
import sklearn.linear_model
from typing_extensions import Any
from pickle import dump, load
import os
from filepaths import get_env_path
from tabpfn import TabPFNClassifier
from tabpfn_model import make_dist_tensor
from torch.func import grad, vmap
CAVS_FILE = get_env_path('models/tcav/cavs.pkl')
GRADS_FILE = get_env_path('models/tcav/grads.pkl')

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=288, out_features=1)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze()
    
    def loss(self, y_pred, y_true):
        return self.criterion(y_pred, y_true.float())

def train_logistic_regression(model: LogisticRegression, x: np.ndarray, y: np.ndarray, epochs: int = 1000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    x_t = torch.tensor(x, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)

    for epoch in range(epochs):
        model.train()
        y_pred = model(x_t)
        loss = model.loss(y_pred, y_t)
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

        model.eval()
        if epoch % 100 == 0:
            with torch.no_grad():
                y_pred = model(x_t)
                pred_labels = (y_pred > 0.5).cpu().numpy()
                acc = (pred_labels == y).mean()
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}')

    return model.to('cpu')

def get_model_gradients(model: TabPFNClassifier, dist_vec: np.ndarray, embs: np.ndarray) -> np.ndarray:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(GRADS_FILE):
        with open(GRADS_FILE, 'rb') as f:
            grads = load(f)
            print(f'Carregando grads de {GRADS_FILE}')
            return grads

    model_decode_layer = model.model_processed_.decoder_dict['standard'].to(device)
    BATCH_SIZE = 128
    gradients = []
    for s in range(0, embs.shape[0], BATCH_SIZE):
        e = min(s + BATCH_SIZE, embs.shape[0])

        print(f'Batch {s}-{e}')
        emb_batch = embs[s:e].astype(np.float32)
        dist_batch = dist_vec[s:e]

        dist_t = torch.tensor(dist_batch, dtype=torch.long, device='cpu').reshape(-1, 1, 1)

        with torch.enable_grad():
            emb = model.get_embeddings(emb_batch, additional_x={"dist_shift_domain": dist_t})
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

def get_cavs(valid_trees: list[dict[str, Any]], embeddings: np.ndarray) -> list[tuple[int, np.ndarray]]:
    if os.path.exists(CAVS_FILE):
        with open(CAVS_FILE, 'rb') as f:
            cavs = load(f)
            print(f'Carregando cavs de {CAVS_FILE}')
            return cavs
    cavs = []
    for tree in valid_trees:
        model = sklearn.linear_model.LogisticRegression(max_iter=500)
        idx, y_mask = tree['idx'], tree['y_mask']
        model.fit(embeddings, y_mask)
        cav = model.coef_[0] # vetor com os coeficientes da fronteira de decisão,
                             # é normal ao hiperplano de separação
        if np.linalg.norm(cav) != 0:
            cav = cav / np.linalg.norm(cav)
        cavs.append((idx, cav))
    with open(CAVS_FILE, 'wb') as f:
        dump(cavs, f)
        print(f'Salvando cavs em {CAVS_FILE}')
    return cavs

def get_tcav_scores(cavs: list[tuple[int, np.ndarray]], model: TabPFNClassifier,
                    dist_vec, x: np.ndarray) -> list[tuple[int,float]]:

    gradients = get_model_gradients(model, dist_vec, x)
    print(gradients.shape)
    scores = []

    for (idx, cav) in cavs:
        tcav_score_all = np.dot(gradients, cav)
        tcav_score = 0
        scores.append((idx, np.mean(tcav_score_all > 0)))

    return scores