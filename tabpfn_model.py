# tabpfn_model.py

from tabpfn import TabPFNClassifier
from tabpfn_extensions import TabPFNEmbedding
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import os
import numpy as np
from database import get_data, impute_data, normalize_data

TRAINING_EMBEDDING_FILE = 'models/tabpfn_emb_train.npy'
TEST_EMBEDDING_FILE = 'models/tabpfn_emb_test.npy'

def get_tabpfn_model(df):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f'Device: {device}')

    X, y = get_data(df)
    X, y = impute_data(X), y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, y_train = normalize_data(X_train), y_train.astype(np.float64)
    X_test, y_test = normalize_data(X_test), y_test.astype(np.float64)

    # model = LogisticRegression()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X=X_test)
    # print(
    #     f"Vanilla Logistic Regression Accuracy: {accuracy_score(y_test, y_pred):.4f}",
    # )

    clf = TabPFNClassifier(device=device, n_estimators=1)

    # print("Métricas do tabpfn: ")
    if not(os.path.exists(TRAINING_EMBEDDING_FILE) and os.path.exists(TEST_EMBEDDING_FILE)):
      clf.fit(X_train, y_train)
      predictions = clf.predict(X_test)
      # print("Accuracy", accuracy_score(y_test, predictions))

    print("Extraindo embeddings do modelo")
    embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=10)

    if not os.path.exists(TRAINING_EMBEDDING_FILE):
      print("Embeddings de Treino")
      train_embeddings = embedding_extractor.get_embeddings(X_train, y_train,
                                                            X_train, data_source='train')
      with open(TRAINING_EMBEDDING_FILE, 'wb') as train_emb:
        np.save(train_emb, train_embeddings)
    else:
      train_embeddings = np.load(TRAINING_EMBEDDING_FILE)

    if not os.path.exists(TEST_EMBEDDING_FILE):
      print("Embeddings de Teste")
      test_embeddings =  embedding_extractor.get_embeddings(X_train, y_train,
                                                          X_test, data_source='test')
      with open(TEST_EMBEDDING_FILE, 'wb') as test_emb:
        np.save(test_emb, test_embeddings)
    else:
      test_embeddings = np.load(TEST_EMBEDDING_FILE)

    # model = LogisticRegression()
    # model.fit(train_embeddings[0], y_train)
    # y_pred = model.predict(test_embeddings[0])
    # print(
    #     f"Logistic Regression with TabPFN (K-Fold CV) Accuracy: {accuracy_score(y_test, y_pred):.4f}",
    # )

    return clf, train_embeddings, test_embeddings
