from tabpfn import TabPFNClassifier
from tabpfn_extensions import TabPFNEmbedding
import torch
import os
import numpy as np
from database import impute_data, normalize_data
from filepaths import get_env_path

BATCH_SIZE = 512

class TabPFNEvalConfig:
    rng_seed: int = 42
    tabpfn_model_name: str = 'dr_tabpfn'
    batch_size_predict: int = BATCH_SIZE
    run_id: str = "demo_tabpfn_step1"

TRAINING_EMBEDDING_FILE = get_env_path('models/tabpfn/tabpfn_emb_train.npy')
TEST_EMBEDDING_FILE = get_env_path('models/tabpfn/tabpfn_emb_test.npy')
PRED_BIN_FILE = get_env_path('models/tabpfn/y_pred_bin.npy')
PRED_PROB_FILE = get_env_path('models/tabpfn/y_pred_prob.npy')


def get_tabpfn_model(arrays: dict[str, np.ndarray], get_embeddings=False, get_pred=False) -> (TabPFNClassifier |
                                                                                              tuple[
                                                                                                  TabPFNClassifier, np.ndarray, np.ndarray,
                                                                                                  dict[
                                                                                                      str, np.ndarray]
                                                                                              ]):
    """ Cria um modelo tabpfn com os dados fornecidos.
        Se get_embeddings for verdadeiro, retorna uma tupla com o modelo e os embeddings de
        treino e teste, repectivamente. Se possível, os embeddings são extraídos de um arquivo salvo,
        caso contrário são extraídos do próprio modelo. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Separação dos dados de treino e teste
    X_train = arrays['X_train']
    y_train = arrays['y_train']
    X_test = arrays['X_test']
    y_test = arrays['y_test']

    # Imputação dos dados
    X_train_imputed, y_train_imputed = impute_data(X_train), y_train
    X_test_imputed, y_test_imputed = impute_data(X_test), y_test

    # Normalização dos dados
    X_train_normalized, y_train_normalized = normalize_data(X_train_imputed), y_train_imputed.astype(np.float64)
    X_test_normalized, y_test_normalized = normalize_data(X_test_imputed), y_test_imputed.astype(np.float64)

    clf = TabPFNClassifier(device=device, n_estimators=1)
    clf.fit(X_train_normalized, y_train_normalized)
    y_pred_bin = None
    y_pred_prob = None

    if get_pred:
        if os.path.exists(PRED_BIN_FILE):
            y_pred_bin = np.load(PRED_BIN_FILE)
        else:
            y_pred_list = []
            for i in range(0, X_test_normalized.shape[0], BATCH_SIZE):
                print(f'Batch {i // BATCH_SIZE}: {i}-{i + BATCH_SIZE}')
                y_pred_list.append(clf.predict(X_test_normalized[i:i + BATCH_SIZE, :]))

            y_pred_bin = np.concatenate(y_pred_list, axis=0)

            with open(PRED_BIN_FILE, 'wb') as pred:
                np.save(pred, y_pred_bin)

        if os.path.exists(PRED_PROB_FILE):
            y_pred_prob = np.load(PRED_PROB_FILE)
        else:
            y_pred_list = []
            for i in range(0, X_test_normalized.shape[0], BATCH_SIZE):
                print(f'Batch {i // BATCH_SIZE}: {i}-{i + BATCH_SIZE}')
                y_pred_list.append(clf.predict_proba(X_test_normalized[i:i + BATCH_SIZE, :])[:, 1])

            y_pred_prob = np.concatenate(y_pred_list, axis=0)

            with open(PRED_PROB_FILE, 'wb') as pred:
                np.save(pred, y_pred_prob)

    if get_embeddings:
        if os.path.exists(TRAINING_EMBEDDING_FILE) and os.path.exists(TEST_EMBEDDING_FILE):
            print("Extraindo embeddings do arquivo")
            train_embeddings = np.load(TRAINING_EMBEDDING_FILE)
            test_embeddings = np.load(TEST_EMBEDDING_FILE)
        else:
            print("Extraindo embeddings do modelo")
            embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=10)
            if not os.path.exists(TRAINING_EMBEDDING_FILE):
                train_emb_list = []
                for i in range(0, X_train_normalized.shape[0], BATCH_SIZE):
                    print(f'Batch {i // BATCH_SIZE}: {i}-{i+BATCH_SIZE}')
                    train_embeddings = embedding_extractor.get_embeddings(X_train_normalized, y_train_normalized,
                                                                          X_train_normalized[i:i + BATCH_SIZE, :],
                                                                          data_source='train')
                    if train_embeddings.ndim == 3:
                        train_embeddings = np.mean(train_embeddings, axis=0)
                    train_emb_list.append(train_embeddings)
                train_embeddings = np.concatenate(train_emb_list, axis=0)

                with open(TRAINING_EMBEDDING_FILE, 'wb') as train_emb:
                    np.save(train_emb, train_embeddings)
            else:
                train_embeddings = np.load(TRAINING_EMBEDDING_FILE)

            if not os.path.exists(TEST_EMBEDDING_FILE):
                test_emb_list = []
                for i in range(0, X_test_normalized.shape[0], BATCH_SIZE):
                    print(f'Batch {i // BATCH_SIZE}: {i}-{i + BATCH_SIZE}')
                    test_embeddings = embedding_extractor.get_embeddings(X_train_normalized, y_train_normalized,
                                                                         X_test_normalized[i:i + BATCH_SIZE, :],
                                                                         data_source='test')
                    if test_embeddings.ndim == 3:
                        test_embeddings = np.mean(test_embeddings, axis=0)
                    test_emb_list.append(test_embeddings)
                test_embeddings = np.concatenate(test_emb_list, axis=0)

                with open(TEST_EMBEDDING_FILE, 'wb') as test_emb:
                    np.save(test_emb, test_embeddings)
            else:
                test_embeddings = np.load(TEST_EMBEDDING_FILE)

        return clf, train_embeddings, test_embeddings, {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'X_train_imputed': X_train_imputed,
            'X_test_imputed': X_test_imputed,
            'y_train_imputed': y_train_imputed,
            'y_test_imputed': y_test_imputed,
            'X_train_normalized': X_train_normalized,
            'X_test_normalized': X_test_normalized,
            'y_train_normalized': y_train_normalized,
            'y_test_normalized': y_test_normalized,
            'y_pred_bin': y_pred_bin,
            'y_pred_prob': y_pred_prob
        }

    return clf

# def fit_dr_tabpfn(X_train: np.ndarray, y_train: np.ndarray, train_years: np.ndarray, eval_cfg: TabPFNEvalConfig):
#     X_train_np = np.ascontiguousarray(X_train.astype(np.float32, copy=True))
#     y_train_np = np.ascontiguousarray(y_train.astype(np.float32, copy=True))
#
#     # permitir que o tabfpn modifique as entradas, se necessário
#     X_train_np.setflags(write=True)
#     y_train_np.setflags(write=True)
#
#     train_years = np.asarray(train_years).astype(int)
#     # um index (dominio) por ano
#     year_to_domain_train = {y: i for i, y in sorted(set(enumerate(train_years)))}
#     dist_shift_domain_train_np = np.ascontiguousarray(
#         np.array([year_to_domain_train[int(y)] for y in train_years], dtype=np.int64)
#     )

    