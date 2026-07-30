import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.model_selection import GridSearchCV
from pickle import dump, load
from filepaths import get_env_path
from typing_extensions import Any
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from progress_utils import progress_iter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
from results import translate_event_name
import re
import pandas as pd
from typing import Sequence
from pathlib import Path

MIN_POSITIVE_SAMPLES = 50
TREE_MODEL_PATH = get_env_path('models/trees/params')
TREE_GRAPH_PATH = get_env_path('models/trees/graphs')

# função gerada pelo google gemini, apenas para verificar regras, não é a função final
def extrair_regras_positivas(modelo_arvore, nomes_das_features, scaler: StandardScaler=None, cid_dict=None):
    """
    Navega pela árvore de decisão e retorna uma lista de strings legíveis
    contendo apenas as regras que levam à previsão da classe positiva (1).
    """
    tree_ = modelo_arvore.tree_

    # Mapeia os IDs das features para os nomes reais passados
    feature_names = [
        nomes_das_features[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]

    regras_positivas = []

    def percorrer_arvore(node, caminho_atual):
        # Se NÃO for uma folha (ou seja, é um nó de divisão)
        if tree_.feature[node] != -2:
            nome_feature = feature_names[node]
            nome_feature += f' ({translate_event_name(feature_names[node], cid_dict)})'
            limiar = tree_.threshold[node]
            feature_idx = tree_.feature[node]

            if scaler is not None:
                mean = scaler.mean_[feature_idx]
                stdev = scaler.scale_[feature_idx]

                limiar = (limiar * stdev) + mean

            # 1. Desce para a esquerda (regra: <= limiar)
            regra_esquerda = f"{nome_feature} <= {limiar:.3f}"
            percorrer_arvore(tree_.children_left[node], caminho_atual + [regra_esquerda])

            # 2. Desce para a direita (regra: > limiar)
            regra_direita = f"{nome_feature} > {limiar:.3f}"
            percorrer_arvore(tree_.children_right[node], caminho_atual + [regra_direita])

        # Se for uma folha final
        else:
            # Pega a distribuição dos PESOS (para calcular a confiança)
            valores_classes = tree_.value[node][0]
            classe_prevista = np.argmax(valores_classes)

            if classe_prevista == 1:
                # Usa os pesos para calcular a porcentagem de confiança
                peso_total = np.sum(valores_classes)
                confianca = valores_classes[1] / peso_total

                # A MÁGICA AQUI: Pega a contagem física real de pacientes!
                qtd_pacientes_reais = tree_.n_node_samples[node]

                regra_completa = " E ".join(caminho_atual)

                # Atualiza a frase para mostrar a quantidade real
                frase = f"SE ( {regra_completa} ) ENTÃO Conceito Ativo [Confiança: {confianca:.0%} | Casos: {qtd_pacientes_reais}]"
                regras_positivas.append(frase)

    # Inicia a busca a partir da raiz (nó 0) com um caminho vazio
    percorrer_arvore(0, [])

    return regras_positivas

# igualmente à função anterior; será revisitada
def extrair_regras_resumidas(modelo_arvore, nomes_das_features, scaler=None, dicionario=None):
    tree_ = modelo_arvore.tree_
    feature_names = [nomes_das_features[i] if i != -2 else "undefined!" for i in tree_.feature]
    regras_positivas = []

    # MÁGICA 1: Função que "olha para o futuro" para ver se pode resumir
    def sub_arvore_pura_positiva(node):
        if tree_.feature[node] == -2: # É uma folha
            valores = tree_.value[node][0]
            return np.argmax(valores) == 1 # Retorna True se prever classe 1

        # Olha para os dois filhos
        esq_positivo = sub_arvore_pura_positiva(tree_.children_left[node])
        dir_positivo = sub_arvore_pura_positiva(tree_.children_right[node])

        # Só resume se TODOS os caminhos abaixo levarem à classe 1
        return esq_positivo and dir_positivo

    # Tradutor embutido
    def traduzir(nome):
        if not dicionario: return nome
        if nome.lower() in dicionario:
            return f"[{dicionario[nome.lower()]}]".capitalize()
        match = re.search(r"DIAGN_([A-Z0-9]+)", nome)

        if match and match.group(1).lower() in dicionario:
            return f"[{dicionario[match.group(1).lower()]}]".capitalize()
        return nome

    def percorrer_arvore(node, limites_atuais):

        # Se este galho inteiro só leva para a Classe 1, podemos resumir agora!
        if sub_arvore_pura_positiva(node):
            valores = tree_.value[node][0]
            confianca = valores[1] / np.sum(valores) # Confiança média ponderada
            casos = tree_.n_node_samples[node]       # Soma total de casos das folhas filhas

            condicoes = []
            for feat, bounds in limites_atuais.items():
                nome_traduzido = traduzir(feat)
                b_min = bounds['min']
                b_max = bounds['max']

                # Monta a regra de forma inteligente (X > min, X <= max, ou min < X <= max)
                if b_min != -np.inf and b_max != np.inf:
                    condicoes.append(f"{b_min:.2f} < {nome_traduzido} ({feat}) <= {b_max:.2f}")
                elif b_min != -np.inf:
                    condicoes.append(f"{nome_traduzido} ({feat}) > {b_min:.2f}")
                elif b_max != np.inf:
                    condicoes.append(f"{nome_traduzido} ({feat}) <= {b_max:.2f}")

            regra_str = " E ".join(condicoes) if condicoes else "TODOS OS CASOS"
            regras_positivas.append(f"SE ( {regra_str}) ENTÃO Conceito Ativo [Confiança Média: {confianca:.0%} | Casos Totais: {casos}]")

            return # Interrompe a descida (Resume a árvore)

        # Se a árvore for "mista" (tem positivos e negativos), continua descendo
        if tree_.feature[node] != -2:
            feature_idx = tree_.feature[node]
            nome_feature = feature_names[node]
            limiar_escalado = tree_.threshold[node]

            if scaler is not None:
                limiar = (limiar_escalado * scaler.scale_[feature_idx]) + scaler.mean_[feature_idx]
            else:
                limiar = limiar_escalado

            # MÁGICA 2: Gravar Mínimos e Máximos em vez de empilhar strings

            # Caminho da Esquerda (<= limiar)
            limites_esq = {k: dict(v) for k, v in limites_atuais.items()}
            if nome_feature not in limites_esq: limites_esq[nome_feature] = {'min': -np.inf, 'max': np.inf}
            limites_esq[nome_feature]['max'] = min(limites_esq[nome_feature]['max'], limiar)
            percorrer_arvore(tree_.children_left[node], limites_esq)

            # Caminho da Direita (> limiar)
            limites_dir = {k: dict(v) for k, v in limites_atuais.items()}
            if nome_feature not in limites_dir: limites_dir[nome_feature] = {'min': -np.inf, 'max': np.inf}
            limites_dir[nome_feature]['min'] = max(limites_dir[nome_feature]['min'], limiar)
            percorrer_arvore(tree_.children_right[node], limites_dir)

    # Começa a busca na raiz (nó 0) com limites vazios
    percorrer_arvore(0, {})
    return regras_positivas

def get_binary_targets(train_activations: np.ndarray, perc=50, model_type:str='ReLU') -> list[tuple[int, float]]:
    bin_targets = []
    # print(f'model_type: {model_type}')

    for col in range(train_activations.shape[1]):
        # lista com todas as ativações para o embedding atual
        cur_concept = train_activations[:, col]

        # threshold: ativação maior que 50% das positivas (mediana)
        cur_concept_positive = cur_concept[cur_concept > 0]
        if model_type == 'ReLU':
            if cur_concept_positive.shape[0] > 0:
                threshold = np.percentile(cur_concept_positive, perc)
                bin_targets.append((col, threshold))
        elif model_type == 'TopK':
            if cur_concept_positive.shape[0] > 0:
                threshold = np.min(cur_concept_positive)
                bin_targets.append((col, threshold))
        else:
            raise ValueError('Invalid model type')



    return bin_targets

def mask_from_rule(rule: str, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    rule_parts = rule.split(' AND ')

    mask = np.ones(X.shape[0], dtype=bool)

    for condition in rule_parts:
        condition = condition.strip()
        if not condition:
            continue

        if '<=' in condition:
            feature, threshold = condition.split(' <= ')
            threshold = float(threshold)
            if feature not in feature_names:
                continue
            feature_idx = feature_names.index(feature)
            mask = mask & (X[:, feature_idx] <= threshold)

        elif '>' in condition:
            feature, threshold = condition.split(' > ')
            threshold = float(threshold)
            if feature not in feature_names:
                continue
            feature_idx = feature_names.index(feature)
            mask = mask & (X[:, feature_idx] > threshold)

    return mask

def get_rules_from_text(text_rules: str) -> pd.DataFrame:
    path = []
    tree_rules = []

    for line in text_rules.split('\n'):
        depth = line.count('|   ')

        # "class:" indica um nó folha (o final de uma regra)
        if 'class' in line:
            cls = line.split('class:')[1].strip()
            tree_rules.append({'Path': ' AND '.join(path), 'Class': cls})

        # nós com filhos indicam a continuação de uma regra
        else:
            cond = line.split('|--- ')[-1].strip()
            # "volta" o conjunto de regras para a altura do nó atual
            path = path[:depth]
            path.append(cond)

    return pd.DataFrame(tree_rules)

def train_binary_trees(train_activations: np.ndarray, X: np.ndarray,
                       feature_names: list[str], model_type: str='ReLU',
                       max_depth:int=15,
                       factor_ids: Sequence[int] | None = None,
                       min_positive_samples: int = MIN_POSITIVE_SAMPLES,
                       show_progress: bool = False,
                       progress_desc: str = "High-precision tree rules")\
        -> list[dict[str, Any]]:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    if min_positive_samples < 1:
        raise ValueError("min_positive_samples must be positive")
    selected_factors = None
    if factor_ids is not None:
        selected_factors = {int(factor_id) for factor_id in factor_ids}
        invalid = [
            factor_id for factor_id in selected_factors
            if factor_id < 0 or factor_id >= train_activations.shape[1]
        ]
        if invalid:
            raise IndexError(f"factor_ids outside activation matrix: {sorted(invalid)}")

    percentiles = [90, 80, 70, 60, 50]
    valid_rules = {p: [] for p in percentiles}

    fit_tasks: list[tuple[int, int, float]] = []
    for perc in percentiles:
        bin_targets = get_binary_targets(train_activations, perc, model_type)
        if selected_factors is not None:
            bin_targets = [
                target for target in bin_targets if target[0] in selected_factors
            ]
        fit_tasks.extend((perc, idx, target) for idx, target in bin_targets)

    for perc, idx, target in progress_iter(
        fit_tasks,
        enabled=show_progress,
        desc=progress_desc,
        total=len(fit_tasks),
        unit="factor-fit",
        leave=False,
    ):
        cur_train_activations = train_activations[:, idx]

        train_target_mask = (cur_train_activations >= target)  # y
        n_high = train_target_mask.sum()

        if n_high < min_positive_samples:
            train_target_mask = (cur_train_activations > 0)
            n_high = train_target_mask.sum()
            if train_target_mask.sum() < min_positive_samples:
                continue

        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=0.01,
            random_state=42
        )

        clf.fit(X, train_target_mask)

        text_rules = export_text(
            clf,
            feature_names=feature_names,
            max_depth=clf.get_depth()
        )

        tree_rules_df = get_rules_from_text(text_rules=text_rules)

        best_rule = None
        best_recall = None
        best_prec = None
        num_true = None

        for _, row in tree_rules_df.iterrows():
            rule = row['Path']
            true_mask = mask_from_rule(rule, X, feature_names)

            if true_mask.sum() == 0:
                continue

            n_true_positive = (true_mask & train_target_mask).sum()
            recall = n_true_positive / max(n_high, 1)
            precision = float(train_target_mask[true_mask].mean())

            if precision < 0.9 or recall < 0.25:
                continue

            if best_recall is None or recall > best_recall:
                best_rule = row
                best_prec = precision
                best_recall = recall
                num_true = true_mask.sum()

        if best_rule is not None:
            valid_rules[perc].append({
                "Factor": idx,
                "Rule": best_rule['Path'],
                "Class": best_rule['Class'],
                "Precision": best_prec,
                "Recall": best_recall,
                "Patients": num_true,
                "Patients_concept": n_high
            })
    return valid_rules

def get_rules_forced(train_activations: np.ndarray, X: np.ndarray, surviving_concepts: np.ndarray, tree_rules_df: pd.DataFrame,  perc: int, feature_names: list[str], model_type: str='ReLU', balanced:bool = False, graph_output_dir: str | Path = TREE_GRAPH_PATH):
    if tree_rules_df is not None:
        surviving_concepts = np.asarray(surviving_concepts, dtype=np.int64)
        existing_rules = tree_rules_df['Factor'].tolist()
        if existing_rules:
            non_existing_mask = ~(np.isin(surviving_concepts, existing_rules))
            surviving_concepts = surviving_concepts[non_existing_mask]

    # print(train_activations.shape)
    bin_targets = get_binary_targets(train_activations, perc, model_type)
    bin_targets = [(idx, target) for (idx, target) in bin_targets if idx in surviving_concepts]

    valid_rules = []
    for concept, target in bin_targets:
        concept_activations = np.asarray(train_activations[:, concept], dtype=np.float32)

        y_high = (concept_activations >= target)
        n_high = y_high.sum()

        if n_high == 0:
            y_high = (concept_activations > 0)
            n_high = y_high.sum()

            if n_high == 0:
                continue

        clf = DecisionTreeClassifier(
            max_depth=15,
            min_samples_leaf=0.01,
            class_weight='balanced' if balanced else None,
            random_state=42
        )
        clf.fit(X, y_high)

        text_rules = export_text(
            clf,
            feature_names=feature_names,
            max_depth=clf.get_depth()
        )

        tree_rules_df = get_rules_from_text(text_rules=text_rules)

        best_f1 = None
        best_rule = None
        best_recall = None
        best_precision = None
        num_true = None

        graph_dir = Path(graph_output_dir)
        graph_dir.mkdir(parents=True, exist_ok=True)
        export_graphviz(
            decision_tree=clf,
            out_file=str(graph_dir / f'{concept}.dot'),
            max_depth=clf.get_depth(),
            feature_names=feature_names,
        )

        for idx, row in tree_rules_df.iterrows():
            # print(f'Concept {concept}, Row {idx}')
            if row['Class'] in [0, 0.0, False, 'False']:
                # print('False rule')
                continue

            rule = row['Path']
            true_mask = mask_from_rule(rule, X, feature_names)

            if true_mask.sum() == 0:
                # print('No true patients for rule')
                continue

            n_true_positive = (true_mask & y_high).sum()
            recall = n_true_positive / max(n_high, 1)
            precision = float(y_high[true_mask].mean())
            f1 = 2 * (precision * recall) / (precision + recall)
            if np.isnan(f1):
                # print('Invalid f1')
                continue

            # print(f'Rule found, prec={precision}, rec={recall}')
            if best_rule is None or f1 > best_f1:
                # print(f'Is new best rule')
                best_f1 = f1
                best_rule = row
                best_recall = recall
                best_precision = precision
                num_true = true_mask.sum()


        if best_rule is not None:
            # print(f'Found rule for concept {concept}')
            valid_rules.append({
                "Factor": concept,
                "Rule": best_rule['Path'],
                "Class": best_rule['Class'],
                "Precision": best_precision,
                "Recall": best_recall,
                "Patients": num_true,
                "Patients_concept": n_high
            })

    return valid_rules
