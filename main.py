from dotenv import load_dotenv

load_dotenv()
import torch
from database import open_parquet, open_feather, create_outcome, get_vars, COVID_DB_PATH, RENAL_DB_PATH, prepare_database, get_tabpfn_arrays
from sae import train_sae_model
from tabpfn_model import get_tabpfn_model
from decision_tree import train_binary_trees
from tcav import get_cavs, get_tcav_scores

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = open_feather(RENAL_DB_PATH)
    prep_out = prepare_database(df=df)

    train_rows = prep_out["train_rows"]
    test_rows = prep_out["test_rows"]
    top_k_events = prep_out["top_k_events"]
    train_years = prep_out["train_years"]
    test_years = prep_out["test_years"]

    # print("train_rows:", train_rows.shape)
    # print("test_rows :", test_rows.shape)
    # print("n top_k_events:", len(top_k_events))
    # print("train_years:", train_years)
    # print("test_years :", test_years)

    tabpfn_arrays = get_tabpfn_arrays(prep_out)

    clf, train_embeddings, test_embeddings, data = get_tabpfn_model(tabpfn_arrays, get_embeddings=True, get_pred=True)
    train_inputs = torch.from_numpy(train_embeddings).squeeze().to(device)
    test_inputs = torch.from_numpy(test_embeddings).squeeze().to(device)
    model = train_sae_model(train_inputs)
    feature_names = get_vars(df)
    feature_names.remove('outcome')
    trees = train_binary_trees(model.encode(train_inputs), model.encode(test_inputs), data, feature_names)
    cavs = get_cavs(trees, train_inputs)
    tcav_scores = get_tcav_scores(cavs, data['X_train_normalized'], data['y_pred_bin'])