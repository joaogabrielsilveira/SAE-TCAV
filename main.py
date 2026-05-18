import numpy as np
from dotenv import load_dotenv
load_dotenv()
import torch
from database import open_feather, get_vars, RENAL_DB_PATH, prepare_database, get_tabpfn_arrays, scale_df_data
from sae import train_sae_model
from tabpfn_model import fit_dr_tabpfn, walkforward_evaluate_tabpfn, TabPFNEvalConfig, load_or_extract_embeddings, \
    EmbeddingExtractConfig, scale_embeddings, scale_embeddings_l2, temporal_test_subsplits, TRAINING_EMBEDDING_FILE, TEST_EMBEDDING_FILE, PRED_PROB_FILE
from decision_tree import train_binary_trees, get_binary_targets, extrair_regras_resumidas
from tcav import get_cavs, get_tcav_scores
from filepaths import get_env_path
from pickle import dump, load
import os
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn import tree as sktree
from decision_tree import extrair_regras_positivas
from results import cid10_dict, translate_event_name, events_dict

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
    cid = cid10_dict()

    X_train_np = tabpfn_arrays['X_train']
    #print(X_train_np.columns)
    #input()
    y_train_np = tabpfn_arrays['y_train']
    years_train_np = tabpfn_arrays['years_train']

    X_test_np = tabpfn_arrays['X_test']
    y_test_np = tabpfn_arrays['y_test']
    years_test_np = tabpfn_arrays['years_test']

    #print("train_rows:", train_rows.shape)
    #print("test_rows :", test_rows.shape)
    #print("n top_k_events:", len(top_k_events))
    #print("train_years:", np.unique(years_train_np))
    #print("test_years :", np.unique(years_test_np))

    #print("X_train_np:", X_train_np.shape)
    #print("y_train_np:", y_train_np.shape, "| pos_ratio:", y_train_np.mean())

    eval_cfg = TabPFNEvalConfig()
    fit_out = fit_dr_tabpfn(X_train_np, y_train_np, years_train_np, eval_cfg)

    drift_model: TabPFNClassifier = fit_out["model"]
    model_add_x_device = fit_out["model_add_x_device"]
    example_add_shape = fit_out["example_add_shape"]

    #print(f"Fit time: {fit_out['fit_time_sec']:.2f}s")
    #print("additional_x device:", model_add_x_device)
    #print("example_add_shape:", example_add_shape)

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
    #print(results_df)

    feature_cols = list(top_k_events)
    X_train_df = train_rows[feature_cols].copy()
    X_test_df = test_rows[feature_cols].copy()

    scaler, X_train_df, X_test_df = scale_df_data(
        X_train_df, X_test_df, feature_cols
    )

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

    train_emb = emb_out['train_emb_flat'].astype(np.float64)
    test_emb = emb_out['test_emb_flat'].astype(np.float64)

    #print(np.mean(train_emb), np.mean(test_emb))

    #train_emb_scaled, test_emb_scaled = scale_embeddings_l2(train_emb, test_emb)
    train_emb_scaled, test_emb_scaled = scale_embeddings(train_emb, test_emb)
    #print(test_emb_scaled.var(), train_emb_scaled.var())

    y_test = (test_rows["DEATH"] > 0).astype(int).to_numpy(copy=True)
    split_idx = temporal_test_subsplits(
        y_test,
        42
    )

    idx_test_discover = split_idx["idx_test_discover"]
    idx_test_cav_train = split_idx["idx_test_cav_train"]
    idx_test_tcav_eval = split_idx["idx_test_tcav_eval"]
    idx_test_held_out = split_idx["idx_test_held_out"]

    #print("Discovery :", len(idx_test_discover), f"({len(idx_test_discover) / len(y_test):.1%})")
    #print("CAV Train :", len(idx_test_cav_train), f"({len(idx_test_cav_train) / len(y_test):.1%})")
    #print("TCAV Eval :", len(idx_test_tcav_eval), f"({len(idx_test_tcav_eval) / len(y_test):.1%})")
    #print("Held-out  :", len(idx_test_held_out), f"({len(idx_test_held_out) / len(y_test):.1%})")

    y_test_discover = y_test[idx_test_discover]
    y_test_cav_train = y_test[idx_test_cav_train]
    y_test_tcav_eval = y_test[idx_test_tcav_eval]
    y_test_held_out = y_test[idx_test_held_out]

    years_test_discover = years_test_np[idx_test_discover]
    years_test_cav_train = years_test_np[idx_test_cav_train]
    years_test_tcav_eval = years_test_np[idx_test_tcav_eval]
    years_test_held_out = years_test_np[idx_test_held_out]

    embeddings_discovery = test_emb_scaled[idx_test_discover]
    # print(embeddings_discovery.var())

    model_train = train_sae_model(torch.tensor(train_emb_scaled), data_source='training')
    # model_disc = train_sae_model(torch.tensor(embeddings_discovery), data_source='discovery')

    rng = np.random.default_rng(42)
    model_device = next(model_train.parameters()).device
    n_cav = len(idx_test_cav_train)
    idx_local = np.arange(n_cav)
    idx_tree_train = rng.choice(idx_local, size=int(n_cav * 0.5), replace=False)
    idx_cav_final_train = np.setdiff1d(idx_local, idx_tree_train)

    # embeddings usados para treinar a árvore de decisão são encodados pelo modelo
    embeddings_dt =  model_train.encode(torch.tensor(train_emb_scaled, dtype=torch.float32).to(model_device)).cpu().detach().numpy()
    #thresh = get_binary_targets(embeddings_dt)

    # entradas relativas aos embeddings usados pra treinar a árvore
    # X_cav_train_df = test_rows_checked.iloc[idx_test_cav_train][feature_cols].reset_index(drop=True)
    # X_feat_tree_train = X_cav_train_df.iloc[idx_tree_train].copy().to_numpy()
    X_feat_tree_train = X_train_df[feature_cols].reset_index(drop=True).copy().to_numpy()
    
    trees = train_binary_trees(embeddings_dt, X_feat_tree_train, feature_cols)
    for tree in trees:
        metrics = tree['metrics']
        # print(f'Tree {tree["idx"]}: {tree["model"]}')
        # print(f'Fator {tree["idx"]}: f1={metrics["f1"]:.2f}, precision={metrics["acc"]:.2f}, recall={metrics["rec"]:.2f}')
    
    y_pred_all = []
    for year in np.unique(years_test_np):
        with open(f'{PRED_PROB_FILE}{year}.pkl', 'rb') as f:
            y_pred_year = load(f)['y_pred_bin']
        y_pred_all.append(y_pred_year)
    y_pred_all = np.concatenate(y_pred_all, axis=0)

    embeddings_cav = train_emb
    cavs = get_cavs(trees, embeddings_cav)

    X_test_tcav_score = X_test_np[idx_test_tcav_eval]
    dist = np.asarray([year_to_domain_combined[int(y)] for y in years_test_np[idx_test_tcav_eval]], dtype=np.int64)
    tcav_scores = get_tcav_scores(cavs=cavs, model=drift_model, x=X_test_tcav_score, dist_vec=dist)
    significant_factors = []
    for idx, score in tcav_scores:
        if 0.4 <= score <= 0.6:
            # regra fraca
            pass
        else:
            significant_factors.append((idx, score))

    full_events_dict = cid | events_dict
    for idx, score in significant_factors:
        print(f'Fator {idx}: TCAV score = {score:.2f} ({"PREVENÇÃO" if score < 0.4 else "RISCO"})')
        for tree in trees:
            if tree['idx'] == idx:
                rule = sktree.export_text(tree['model'], feature_names=feature_cols)
                true_text = 'PREVENÇÃO' if score < 0.4 else 'RISCO'
                false_text = 'RISCO' if score < 0.4 else 'PREVENÇÃO'
                rules = extrair_regras_resumidas(tree['model'], feature_cols, scaler, dicionario=full_events_dict)
                metrics = tree['metrics']
                print(f'Árvore do fator {tree["idx"]}: Precisão={metrics["acc"]:.2f}, Recall={metrics["rec"]:.2f}, F1={metrics["f1"]:.2f}')
                for i, rule in enumerate(rules):                
                    print(f'Regra {i}: {rule}')

                input()
    
    print(f'Total de fatores significativos encontrados: {len(significant_factors)}')