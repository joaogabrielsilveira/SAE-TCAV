import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from typing_extensions import Any

def get_cavs(valid_trees: list[dict[str, Any]], embeddings: np.ndarray) -> list[tuple[int, np.ndarray]]:
    cavs = []
    for tree in valid_trees:
        model = LogisticRegression(max_iter=500)
        idx, y_mask = tree['idx'], tree['y_mask']
        model.fit(embeddings, y_mask)
        cav = model.coef_[0] # vetor com os coeficientes da fronteira de decisão,
                          # é normal ao hiperplano de separação
        cavs.append((idx, cav))

    return cavs

def get_tcav_scores(cavs: list[tuple[int, np.ndarray]], inputs: np.ndarray, y_pred: np.ndarray)\
        -> list[tuple[int,float]]:
    linear_tabpfn = LogisticRegression(max_iter=1000)

    linear_tabpfn.fit(inputs, y_pred)
    gradient = torch.Tensor(linear_tabpfn.coef_[0])
    scores = []

    for (idx, cav) in cavs:
        tcav_score = np.dot(cav, gradient)
        scores.append((tcav_score, idx))

    return scores