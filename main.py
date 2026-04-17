from dotenv import load_dotenv
load_dotenv()
import torch
from database import open_parquet, create_outcome, DB_PATH
from sae import SAE, train_sae_model
from tabpfn_model import get_tabpfn_model

if __name__ == '__main__':
    df = open_parquet(DB_PATH)
    df = create_outcome(df)
    clf, train_embeddings, test_embeddings = get_tabpfn_model(df)
    inputs = torch.from_numpy(train_embeddings).squeeze()
    model = train_sae_model(inputs)
