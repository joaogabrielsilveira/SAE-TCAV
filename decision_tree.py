import numpy as np
import torch
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.model_selection import GridSearchCV
from pickle import dump, load
from filepaths import get_env_path
from typing_extensions import Any

MIN_POSITIVE_SAMPLES = 50
TREE_MODEL_PATH = get_env_path('models/trees/params')
TREE_GRAPH_PATH = get_env_path('models/trees/graphs')

def get_binary_targets(train_activations: torch.Tensor) -> list[tuple[int, float]]:
    bin_targets = []
    train_activations = train_activations.cpu().detach()

    for col in range(train_activations.shape[1]):
        # lista com todas as ativações para o embedding atual
        cur_concept = train_activations[:, col]

        # threshold: ativação maior que 50% das positivas (mediana)
        cur_concept_positive = cur_concept[cur_concept > 0]
        if cur_concept_positive.shape[0] >= MIN_POSITIVE_SAMPLES:
            threshold = np.median(cur_concept_positive)
            bin_targets.append((col, threshold))

    return bin_targets

def train_binary_trees(train_activations: torch.Tensor, test_activations: torch.Tensor,
                       model_data: dict[str, np.ndarray], feature_names: list[str], max_depth:int=5)\
        -> list[dict[str, Any]]:
    train_activations = train_activations.cpu().detach()
    test_activations = test_activations.cpu().detach()

    bin_targets = get_binary_targets(train_activations)

    X_train = model_data['X_train']
    X_test = model_data['X_test']

    valid_trees = []
    param_grid = {
        'class_weight': ['balanced', None],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': list(range(5, 10)),
        'splitter': ['best', 'random']
    }
    for (idx, target) in bin_targets:
        cur_train_activations = train_activations[:, idx]
        cur_test_activations = test_activations[:, idx]

        train_target_mask = torch.Tensor([int(act > target) for act in cur_train_activations]) ## y
        test_target_mask = torch.Tensor([int(act > target) for act in cur_test_activations])

        if torch.count_nonzero(train_target_mask) == 0 or torch.count_nonzero(test_target_mask) == 0:
            # print(f'Fator {idx}: conjunto vazio encontrado, pulando')
            continue

        clf = DecisionTreeClassifier(max_depth=max_depth, class_weight='balanced',
                                     min_samples_leaf=50, max_leaf_nodes=8)

        if os.path.exists(f'{TREE_MODEL_PATH}/{idx}.pkl'):
            with open(f'{TREE_MODEL_PATH}/{idx}.pkl', 'rb') as f:
                clf = load(f)
        else:
            clf.fit(X_train, train_target_mask)
            sel = GridSearchCV(clf, param_grid, scoring='f1', n_jobs=8)
            sel.fit(X_train, train_target_mask)
            clf = sel.best_estimator_
            with open(f'{TREE_MODEL_PATH}/{idx}.pkl', 'wb') as f:
                dump(clf, f, protocol=5)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(test_target_mask, y_pred)
        rec = recall_score(test_target_mask, y_pred)
        f1 = f1_score(test_target_mask, y_pred)

        print(f'Arvore do fator {idx}: f1={f1}, accuracy={acc}, recall={rec}')
        if f1 >= 0.5:
            print(f'Arvore {idx} aprovada!!! boa boa boa')
            valid_trees.append({'model': clf, 'idx': idx, 'y_mask': train_target_mask})
            export_graphviz(clf, out_file=f'{TREE_GRAPH_PATH}/{idx}.dot', feature_names=feature_names)
            # text_rules = export_text(
            #     clf,
            #     feature_names=feature_names,
            #     show_weights=True
            # )
            # print(text_rules)

    print(f'Arvores boas encontradas: {len(valid_trees)}')
    return valid_trees
