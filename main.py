from dotenv import load_dotenv
load_dotenv()
import torch
from database import open_feather, get_vars, RENAL_DB_PATH, prepare_database, get_tabpfn_arrays, scale_df_data
from sae import train_sae_model
from tabpfn_model import fit_dr_tabpfn, walkforward_evaluate_tabpfn, TabPFNEvalConfig, load_or_extract_embeddings, \
    EmbeddingExtractConfig, scale_embeddings, temporal_test_subsplits
from decision_tree import train_binary_trees
from tcav import get_cavs, get_tcav_scores
from filepaths import get_env_path
from pickle import dump, load
import os
import pandas as pd

PREPARED_DB_PATH = get_env_path('data/renal/prep.pkl')

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df = open_feather(RENAL_DB_PATH)
    if not os.path.exists(PREPARED_DB_PATH):
        prep_out = prepare_database(df=df)
        tabpfn_arrays = get_tabpfn_arrays(prep_out)
        with open(PREPARED_DB_PATH, 'wb') as f:
            dump(prep_out, f)
    else:
        with open(PREPARED_DB_PATH, 'rb') as f:
            prep_out = load(f)
            tabpfn_arrays = get_tabpfn_arrays(prep_out)

    train_rows = prep_out['train_rows']
    test_rows = prep_out['test_rows']
    top_k_events = prep_out['top_k_events']

    X_train_np = tabpfn_arrays['X_train']
    y_train_np = tabpfn_arrays['y_train']
    years_train_np = tabpfn_arrays['years_train']

    X_test_np = tabpfn_arrays['X_test']
    y_test_np = tabpfn_arrays['y_test']
    years_test_np = tabpfn_arrays['years_test']

    eval_cfg = TabPFNEvalConfig()
    fit_out = fit_dr_tabpfn(X_train_np, y_train_np, years_train_np, eval_cfg)

    drift_model = fit_out["model"]
    model_add_x_device = fit_out["model_add_x_device"]
    example_add_shape = fit_out["example_add_shape"]

    print(f"Fit time: {fit_out['fit_time_sec']:.2f}s")
    print("additional_x device:", model_add_x_device)
    print("example_add_shape:", example_add_shape)

    wf = walkforward_evaluate_tabpfn(
        drift_model=drift_model,
        test_rows=test_rows,
        top_k_events=top_k_events,
        train_years=years_train_np,
        model_add_x_device=model_add_x_device,
        batch_size_predict=512,
        example_add_shape=example_add_shape,
    )

    results_per_year = wf["results_per_year"]
    year_to_domain_combined = wf["year_to_domain_combined"]
    test_rows_checked = wf["test_rows_checked"]

    results_df = pd.DataFrame(results_per_year).sort_values("year").reset_index(drop=True)

    feature_cols = list(top_k_events)
    X_train_df = train_rows[feature_cols].copy()
    X_test_df = test_rows_checked[feature_cols].copy()

    scaler, X_train_df, X_test_df = scale_df_data(
        X_train_df, X_test_df, feature_cols
    )

    print(results_df)

    emb_cfg = EmbeddingExtractConfig()

    emb_out = load_or_extract_embeddings(
        drift_model,
        X_train_np,
        X_test_np,
        years_train_np,
        years_test_np,
        year_to_domain_combined,
        "",
        emb_cfg,
        model_add_x_device,
        example_add_shape,
    )

    train_emb = emb_out['train_emb_flat']
    test_emb = emb_out['test_emb_flat']

    train_emb_scaled, test_emb_scaled = scale_embeddings(train_emb, test_emb)

    y_test = (test_rows_checked["DEATH"] > 0).astype(int).to_numpy(copy=True)
    split_idx = temporal_test_subsplits(
        y_test,
        42
    )

    idx_test_discover = split_idx["idx_test_discover"]
    idx_test_cav_train = split_idx["idx_test_cav_train"]
    idx_test_tcav_eval = split_idx["idx_test_tcav_eval"]
    idx_test_held_out = split_idx["idx_test_held_out"]

    print("Discovery :", len(idx_test_discover), f"({len(idx_test_discover) / len(y_test):.1%})")
    print("CAV Train :", len(idx_test_cav_train), f"({len(idx_test_cav_train) / len(y_test):.1%})")
    print("TCAV Eval :", len(idx_test_tcav_eval), f"({len(idx_test_tcav_eval) / len(y_test):.1%})")
    print("Held-out  :", len(idx_test_held_out), f"({len(idx_test_held_out) / len(y_test):.1%})")

    y_test_discover = y_test[idx_test_discover]
    y_test_cav_train = y_test[idx_test_cav_train]
    y_test_tcav_eval = y_test[idx_test_tcav_eval]
    y_test_held_out = y_test[idx_test_held_out]

    years_test_discover = years_test_np[idx_test_discover]
    years_test_cav_train = years_test_np[idx_test_cav_train]
    years_test_tcav_eval = years_test_np[idx_test_tcav_eval]
    years_test_held_out = years_test_np[idx_test_held_out]

    embeddings_discovery = test_emb_scaled[idx_test_discover]

    model = train_sae_model(torch.tensor(embeddings_discovery))
    #
    # encoded_train = model.encode(train_inputs).cpu().detach().numpy()
    # encoded_test = model.encode(test_inputs).cpu().detach().numpy()
    #
    # feature_names = get_vars(df)
    # trees = train_binary_trees(encoded_train, encoded_test, data, feature_names)
    #
    # cavs = get_cavs(trees, encoded_train)
    # tcav_scores_bin = get_tcav_scores(cavs, data['X_train_normalized'], data['y_pred_bin'])
    # tcav_scores_prob = get_tcav_scores(cavs, data['X_train_normalized'], data['y_pred_prob'])