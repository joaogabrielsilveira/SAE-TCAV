from dotenv import load_dotenv
load_dotenv()
import torch
from covid_database import open_parquet, create_outcome, get_data, get_vars, DB_PATH
from sae import train_sae_model
from tabpfn_model import get_tabpfn_model
from decision_tree import train_binary_trees
from tcav import get_cavs, get_tcav_scores

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = open_parquet(DB_PATH)
    df = create_outcome(df)
    clf, train_embeddings, test_embeddings, data = get_tabpfn_model(df, get_embeddings=True, get_pred=True)
    train_inputs = torch.from_numpy(train_embeddings).squeeze().to(device)
    test_inputs = torch.from_numpy(test_embeddings).squeeze().to(device)
    model = train_sae_model(train_inputs)
    feature_names = get_vars(df)
    feature_names.remove('outcome')
    trees = train_binary_trees(model.encode(train_inputs), model.encode(test_inputs), data, feature_names)
    cavs = get_cavs(trees, train_inputs)
    tcav_scores = get_tcav_scores(cavs, data['X_train_normalized'], data['y_pred_bin'])