from tabpfn import TabPFNClassifier
from tabpfn_extensions import TabPFNEmbedding
from sklearn.model_selection import train_test_split
import torch
import os
import numpy as np
from database import get_data, impute_data, normalize_data
import pandas as pd

TRAINING_EMBEDDING_FILE = 'models/tabpfn_emb_train.npy'
TEST_EMBEDDING_FILE = 'models/tabpfn_emb_test.npy'

def get_tabpfn_model(df: pd.DataFrame, get_embeddings=False) -> (TabPFNClassifier |
                                                                 tuple[TabPFNClassifier, np.ndarray, np.ndarray,
                                                                        dict[str, np.ndarray]]):
    """ Cria um modelo tabpfn com os dados fornecidos.
        Se get_embeddings for verdadeiro, retorna uma tupla com o modelo e os embeddings de
        treino e teste, repectivamente. Se possível, os embeddings são extraídos de um arquivo salvo,
        caso contrário são extraídos do próprio modelo. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X, y = get_data(df)

    # Separação dos dados de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Imputação dos dados
    X_train_imputed, y_train_imputed = impute_data(X_train), y_train
    X_test_imputed, y_test_imputed = impute_data(X_test), y_test

    # Normalização dos dados
    X_train_normalized, y_train_normalized = normalize_data(X_train), y_train.astype(np.float64)
    X_test_normalized, y_test_normalized = normalize_data(X_test), y_test.astype(np.float64)

    clf = TabPFNClassifier(device=device, n_estimators=1)
    clf.fit(X_train, y_train)

    if get_embeddings:
        if os.path.exists(TRAINING_EMBEDDING_FILE) and os.path.exists(TEST_EMBEDDING_FILE):
            print("Extraindo embeddings do arquivo")
            train_embeddings = np.load(TRAINING_EMBEDDING_FILE)
            test_embeddings = np.load(TEST_EMBEDDING_FILE)
        else:
            print("Extraindo embeddings do modelo")
            embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=10)
            if not os.path.exists(TRAINING_EMBEDDING_FILE):
                train_embeddings = embedding_extractor.get_embeddings(X_train_normalized, y_train_normalized,
                                                                      X_train_normalized, data_source='train')
                with open(TRAINING_EMBEDDING_FILE, 'wb') as train_emb:
                    np.save(train_emb, train_embeddings)
            else:
                train_embeddings = np.load(TRAINING_EMBEDDING_FILE)

            if not os.path.exists(TEST_EMBEDDING_FILE):
                test_embeddings = embedding_extractor.get_embeddings(X_train_normalized, y_train_normalized,
                                                                     X_test_normalized, data_source='test')
                with open(TEST_EMBEDDING_FILE, 'wb') as test_emb:
                    np.save(test_emb, test_embeddings)
            else:
                test_embeddings = np.load(TEST_EMBEDDING_FILE)

        return clf, train_embeddings, test_embeddings, {'X_train': X_train, 'X_test': X_test,
                                                        'y_train': y_train, 'y_test': y_test,
                                                        'X_train_imputed': X_train_imputed,
                                                        'X_test_imputed': X_test_imputed,
                                                        'y_train_imputed': y_train_imputed,
                                                        'y_test_imputed': y_test_imputed}

    return clf

