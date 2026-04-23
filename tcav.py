from sklearn.linear_model import LogisticRegression
import torch
from typing_extensions import Any

def get_cavs(valid_trees: list[dict[str, Any]], embeddings: torch.Tensor) -> list[tuple[torch.Tensor, int]]:
    embeddings = embeddings.cpu().detach()
    cavs = []
    for tree in valid_trees:
        model = LogisticRegression(max_iter=500)
        idx, y_mask = tree['idx'], tree['y_mask']
        model.fit(embeddings, y_mask)
        cav = model.coef_[0] # vetor com os coeficientes da fronteira de decisão,
                          # é normal ao hiperplano de separação
        cavs.append((torch.Tensor(cav), idx))

    return cavs

def get_tcav_scores(cavs: list[tuple[int, torch.Tensor]], inputs: torch.Tensor, y_pred: torch.Tensor)\
        -> tuple[int,float]:
    linear_tabpfn = LogisticRegression(max_iter=1000)
    inputs = inputs.cpu().detach()
    y_pred = y_pred.cpu().detach()

    linear_tabpfn.fit(inputs, y_pred)
    gradient = torch.Tensor(linear_tabpfn.coef_[0])
    scores = []

    for (idx, cav) in cavs:
        tcav_score = torch.dot(cav, gradient).item()
        scores.append((tcav_score, idx))

    return scores