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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from results import translate_event_name
import re

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

def select_best_tree(cv_results):
    prec = cv_results['mean_test_precision']
    rec = cv_results['mean_test_recall']

    valid_indices = np.where((prec > 0.75) & (rec > 0.20))[0]
    if len(valid_indices) > 0:
        best = valid_indices[np.argmax(prec[valid_indices])]
        return best
    else:
        valid_indices = np.where(rec >= 0.1)[0]
        if len(valid_indices) > 0:
            best = valid_indices[np.argmax(prec[valid_indices])]
            return best
    
    return np.argmax(prec)

def get_binary_targets(train_activations: np.ndarray) -> list[tuple[int, float]]:
    bin_targets = []

    for col in range(train_activations.shape[1]):
        # lista com todas as ativações para o embedding atual
        cur_concept = train_activations[:, col]

        # threshold: ativação maior que 50% das positivas (mediana)
        cur_concept_positive = cur_concept[cur_concept > 0]
        if cur_concept_positive.shape[0] >= MIN_POSITIVE_SAMPLES:
            threshold = np.median(cur_concept_positive)
            bin_targets.append((col, threshold))
            #print(f'Fator {col}: target={threshold}')

    return bin_targets

def train_binary_trees(train_activations: np.ndarray, X: np.ndarray,
                       feature_names: list[str], max_depth:int=5)\
        -> list[dict[str, Any]]:
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    bin_targets = get_binary_targets(train_activations)
    valid_trees = []
    rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
    param_grid = {
        'tree__class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 5}, {0: 1, 1: 10}],
        'tree__criterion': ['gini', 'entropy', 'log_loss'],
        #'tree__max_depth': [5, 7, 10, None],
        'tree__max_depth': [max_depth],
        'tree__splitter': ['best'],
        'tree__min_samples_leaf': [10, 25, 50],
        'tree__max_leaf_nodes': [8, 16, 32, None]
    }
    for (idx, target) in bin_targets:
        cur_train_activations = train_activations[:, idx]

        train_target_mask = cur_train_activations > target # y 

        if np.count_nonzero(train_target_mask) <= MIN_POSITIVE_SAMPLES:
            #print(f'Fator {idx}: conjunto vazio encontrado, pulando')
            continue

        pipeline = Pipeline([('undersampler', rus), ("tree", DecisionTreeClassifier())])

        if os.path.exists(f'{TREE_MODEL_PATH}/{idx}.pkl'):
            with open(f'{TREE_MODEL_PATH}/{idx}.pkl', 'rb') as f:
                pipeline = load(f)
        else:
            pipeline.fit(X, train_target_mask)
            sel = GridSearchCV(pipeline, param_grid, scoring={'precision': make_scorer(precision_score, zero_division=0),
                                                         'recall': make_scorer(recall_score, zero_division=0)},
                                                           n_jobs=-1, refit=select_best_tree)
            sel.fit(X, train_target_mask)
            pipeline = sel.best_estimator_
            with open(f'{TREE_MODEL_PATH}/{idx}.pkl', 'wb') as f:
                dump(pipeline, f, protocol=5)

        y_pred = pipeline.predict(X)
        acc = precision_score(train_target_mask, y_pred)
        rec = recall_score(train_target_mask, y_pred)
        f1 = f1_score(train_target_mask, y_pred)

        metrics = {
            'acc': acc,
            'rec': rec,
            'f1': f1
        }

        # print(f'Arvore do fator {idx}: f1={f1}, precision={acc}, recall={rec}')
        if f1 >= 0.5:
            # print(f'Arvore {idx} aprovada')
            valid_trees.append({'model': pipeline.steps[1][1], 'idx': idx, 'y_mask': train_target_mask, 'metrics': metrics})
            export_graphviz(pipeline['tree'], out_file=f'{TREE_GRAPH_PATH}/{idx}.dot', feature_names=feature_names)
            # text_rules = export_text(
            #     clf,
            #     feature_names=feature_names,
            #     show_weights=True
            # )
            # print(text_rules)

    print(f'Arvores boas encontradas: {len(valid_trees)}')
    return valid_trees
