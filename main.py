import pickle

import numpy as np
from dotenv import load_dotenv
load_dotenv()
import torch
from database import open_feather, get_vars, RENAL_DB_PATH, prepare_database, get_tabpfn_arrays, scale_df_data
from sae import train_sae_model
from tabpfn_model import fit_dr_tabpfn, walkforward_evaluate_tabpfn, TabPFNEvalConfig, load_or_extract_embeddings, \
    EmbeddingExtractConfig, scale_embeddings, scale_embeddings_l2, temporal_test_subsplits, TRAINING_EMBEDDING_FILE, TEST_EMBEDDING_FILE, PRED_PROB_FILE
from decision_tree import train_binary_trees, get_binary_targets, extrair_regras_resumidas
# from tcav import get_cavs, get_tcav_scores
from filepaths import get_env_path
from pickle import dump, load
import os
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn import tree as sktree
from decision_tree import extrair_regras_positivas
from results import cid10_dict, translate_event_name, events_dict
# from TCAV.renal_framework.src.tabpfn_pipeline.evaluation import walkforward_evaluate_tabpfn

PREPARED_DB_PATH = get_env_path('data/renal/prep.pkl')

if __name__ == '__main__':
    # my_train_emb = np.load(TRAINING_EMBEDDING_FILE).squeeze().astype(np.float32)
    # my_test_emb = np.load(TEST_EMBEDDING_FILE).squeeze().astype(np.float32)

    # nb_train_emb = np.load('TCAV/renal_framework/results/demo_tabpfn/embeddings/train_emb_flat.npy').astype(np.float32)
    # nb_test_emb = np.load('TCAV/renal_framework/results/demo_tabpfn/embeddings/test_emb_flat.npy').astype(np.float32)

    # print(f'My train emb shape: {my_train_emb.shape}, mean: {my_train_emb.mean()}, var: {my_train_emb.var()}')
    # print(f'My test emb shape: {my_test_emb.shape}, mean: {my_test_emb.mean()}, var: {my_test_emb.var()}')
    # print(f'NB train emb shape: {nb_train_emb.shape}, mean: {nb_train_emb.mean()}, var: {nb_train_emb.var()}')
    # print(f'NB test emb shape: {nb_test_emb.shape}, mean: {nb_test_emb.mean()}, var: {nb_test_emb.var()}')

    # print(np.count_nonzero(my_train_emb != nb_train_emb))
    # print(np.count_nonzero(my_test_emb != nb_test_emb))

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
    y_train_np = tabpfn_arrays['y_train']
    years_train_np = tabpfn_arrays['years_train']

    X_test_np = tabpfn_arrays['X_test']
    y_test_np = tabpfn_arrays['y_test']
    years_test_np = tabpfn_arrays['years_test']

    # print(f'years_train: {years_train_np.shape}')
    # print(f'years_test: {years_test_np.shape}')

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

    # print(f"Fit time: {fit_out['fit_time_sec']:.2f}s")
    # print("additional_x device:", model_add_x_device)
    # print("example_add_shape:", example_add_shape)

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
    # print(results_df)

    feature_cols = list(top_k_events)
    X_train_df = train_rows[feature_cols].copy()
    X_test_df = test_rows[feature_cols].copy()

    # print(f'X_train_df: {X_train_df.shape}')
    # print(f'X_test_df : {X_test_df.shape}')

    scaler, X_train_df, X_test_df = scale_df_data(
        X_train_df, X_test_df, feature_cols
    )

    # print(f'X_train_df_norm: {X_train_df.shape})
    # print(f'X_test_df_norm : {X_test_df.shape}')

    emb_cfg = EmbeddingExtractConfig()
    emb_out = load_or_extract_embeddings(
        model=drift_model,
        X_train_np=X_train_np,
        X_test_np=X_test_np,
        years_train=years_train_np,
        years_test=years_test_np,
        year_to_domain_map=year_to_domain_combined,
        embeddings_dir="",
        cfg=emb_cfg,
        device=model_add_x_device,
        example_add_shape=example_add_shape,
    )

    train_emb = emb_out['train_emb_flat'].astype(np.float32)
    test_emb = emb_out['test_emb_flat'].astype(np.float32)

    #print(np.mean(train_emb), np.mean(test_emb))

    #train_emb_scaled, test_emb_scaled = scale_embeddings_l2(train_emb, test_emb)
    train_emb_scaled, test_emb_scaled = scale_embeddings(train_emb, test_emb)
    # print(train_emb_scaled.dtype, train_emb_scaled.mean(), train_emb_scaled.var())
    # print(test_emb_scaled.mean(), test_emb_scaled.var())

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

    emb_tcav_eval = test_emb_scaled[idx_test_tcav_eval]
    emb_discovery = test_emb_scaled[idx_test_discover]

    sae = train_sae_model(torch.tensor(emb_discovery), data_source='discovery', save_data=False, use_decoder_bias=False)
    sae.eval()
    sae_codes_train = sae.encode(torch.tensor(emb_discovery)).cpu().detach().numpy()
    sae_codes_test = sae.encode(torch.tensor(test_emb_scaled[idx_test_cav_train])).cpu().detach().numpy()

    sparsity_level = (sae_codes_train <= 1e-5).mean()
    active_per_sample = (sae_codes_train > 1e-5).sum(axis=1).mean()

    print(f'SAE codes (discovery): {sae_codes_train.shape}')
    print(f"SAE near-zero activations: {sparsity_level:.2%}")
    print(f"SAE active neurons/sample: {active_per_sample:.2f} / {sae_codes_train.shape[1]}")
    print(f"SAE mean abs activation: {np.abs(sae_codes_train).mean():.4f}")
    print(f"SAE max activation: {sae_codes_train.max():.4f}")
    print(f'SAE dead neurons: {np.all(sae_codes_train <= 1e-5, axis=0).sum().item()}')

    rng = np.random.default_rng(42)
    model_device = next(sae.parameters()).device
    n_cav = len(idx_test_cav_train)
    idx_local = np.arange(n_cav)
    idx_tree_train = rng.choice(idx_local, size=int(n_cav * 0.5), replace=False)
    idx_cav_final_train = np.setdiff1d(idx_local, idx_tree_train)

    
    embeddings_dt = sae_codes_test[idx_tree_train]
    X_cav_train_df = test_rows_checked.iloc[idx_test_cav_train][feature_cols].reset_index(drop=True)
    X_feat_tree_train = X_cav_train_df.iloc[idx_tree_train].copy().to_numpy()

    print("X_feat_dt_train:", X_feat_tree_train.shape)
    print("codes_dt_train:", embeddings_dt.shape)

    with open('X_dt.pkl', 'rb') as f:
        X_dt_nb = load(f)
    with open('codes_dt.pkl', 'rb') as f:
        codes_dt = load(f)

    print(f'Codes diferentes? {np.count_nonzero(np.count_nonzero(codes_dt != embeddings_dt, axis=0))}')
    print(f'X diferentes? {np.count_nonzero(X_feat_tree_train != X_dt_nb)}')
    
    tree_rules = train_binary_trees(embeddings_dt, X_feat_tree_train, feature_cols)
    best_p = max(tree_rules.keys(), key=lambda p: len(tree_rules[p]))
    rules_df = pd.DataFrame(tree_rules[best_p])

    embeddings_cav = test_emb_scaled[idx_test_cav_train]
    embeddings_cav_training = test_emb[idx_cav_final_train]
    y_cav_training = y_test_cav_train[idx_cav_final_train]

    # cavs = get_cavs(trees, embeddings_cav)

    # X_test_tcav_score = X_test_np[idx_test_tcav_eval]
    # dist = np.asarray([year_to_domain_combined[int(y)] for y in years_test_np[idx_test_tcav_eval]], dtype=np.int64)
    # tcav_scores = get_tcav_scores(cavs=cavs, model=drift_model, x=X_test_tcav_score, dist_vec=dist)
    # significant_factors = []
    # for idx, score in tcav_scores:
    #     if 0.4 <= score <= 0.6:
    #         # regra fraca
    #         pass
    #     else:
    #         significant_factors.append((idx, score))

    # full_events_dict = cid | events_dict
    # for idx, score in significant_factors:
    #     print(f'Fator {idx}: TCAV score = {score:.2f} ({"PREVENÇÃO" if score < 0.4 else "RISCO"})')
    #     for tree in trees:
    #         if tree['idx'] == idx:
    #             rule = sktree.export_text(tree['model'], feature_names=feature_cols)
    #             true_text = 'PREVENÇÃO' if score < 0.4 else 'RISCO'
    #             false_text = 'RISCO' if score < 0.4 else 'PREVENÇÃO'
    #             rules = extrair_regras_resumidas(tree['model'], feature_cols, scaler, dicionario=full_events_dict)
    #             metrics = tree['metrics']
    #             print(f'Árvore do fator {tree["idx"]}: Precisão={metrics["acc"]:.2f}, Recall={metrics["rec"]:.2f}, F1={metrics["f1"]:.2f}')
    #             for i, rule in enumerate(rules):                
    #                 print(f'Regra {i}: {rule}')

    #             input()
    
    # print(f'Total de fatores significativos encontrados: {len(significant_factors)}')