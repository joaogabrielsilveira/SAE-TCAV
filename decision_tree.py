import numpy as np
import torch
from sklearn import tree
from sklearn.metrics import accuracy_score, recall_score
from sklearn.tree import DecisionTreeClassifier

def get_binary_targets(train_activations: torch.Tensor) -> list[tuple[int, float]]:
    bin_targets = []
    train_activations = train_activations.cpu().detach()

    for col in range(train_activations.shape[1]):
        # lista com todas as ativações para o embedding atual
        cur_concept = train_activations[:, col]

        # threshold: ativação maior que 50% das positivas (mediana)
        cur_concept_positive = cur_concept[cur_concept > 0]
        if cur_concept_positive.shape[0] > 0:
            threshold = np.median(cur_concept_positive)
            bin_targets.append((col, threshold))

    return bin_targets

def train_binary_trees(train_activations: torch.Tensor, test_activations: torch.Tensor,
                       model_data: dict[str, np.ndarray], feature_names: list[str], max_depth:int=2) -> list[tuple[int, DecisionTreeClassifier]]:
    train_activations = train_activations.cpu().detach()
    test_activations = test_activations.cpu().detach()

    bin_targets = get_binary_targets(train_activations)

    X_train = model_data['X_train']
    X_test = model_data['X_test']

    valid_trees = []
    for (idx, target) in bin_targets:
        cur_train_activations = train_activations[:, idx]
        cur_test_activations = test_activations[:, idx]

        train_target_mask = torch.Tensor([int(act > target) for act in cur_train_activations]) ## y
        test_target_mask = torch.Tensor([int(act > target) for act in cur_test_activations])

        if torch.count_nonzero(train_target_mask) == 0 or torch.count_nonzero(test_target_mask) == 0:
            # print(f'Fator {idx}: conjunto vazio encontrado, pulando')
            continue

        clf = tree.DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced')
        clf.fit(X_train, train_target_mask)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(test_target_mask, y_pred)
        rec = recall_score(test_target_mask, y_pred)

        # print(f'Arvore do fator {idx}: accuracy={acc}, recall={rec}')
        if acc >= 90 / 100 and rec >= 25 / 100:
            # print(f'Arvore {idx} aprovada!!! boa boa boa')
            valid_trees.append((idx, clf))
            tree.export_graphviz(clf, out_file=f'models/tree_factor_{idx}.dot', feature_names=feature_names)

    print(f'Arvores boas encontradas: {len(valid_trees)} {[idx for (idx, clf) in valid_trees]}')
    return valid_trees
