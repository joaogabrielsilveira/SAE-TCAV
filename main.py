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
from tcav import train_cavs_from_rules, get_model_gradients, get_tcav_scores, robust_tcav_significance_test, run_feature_association_dual_split, run_sparse_readout_dual_split
from filepaths import get_env_path
from pickle import dump, load
import os
import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn import tree as sktree
from decision_tree import extrair_regras_positivas
from results import cid10_dict, translate_event_name, events_dict, tcav_result_df_from_concepts, translate_event_names
# from TCAV.renal_framework.src.tabpfn_pipeline.evaluation import walkforward_evaluate_tabpfn

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

    from sae_compare import run_sae_random_comparison, plot_run_results
    import matplotlib.pyplot as plt
    inputs = emb_discovery
    pair_criteria = 'cos_sim'
    model_n = range(3, 15)
    alphas = [1e-1]
    scaling_factors = [1.5]
    res, model_res = run_sae_random_comparison(
        model_nums=model_n,
        alphas=alphas,
        embs=inputs, 
        scaling_factors=scaling_factors, 
        model_type='ReLU',
        pair_criteria=pair_criteria    
    )

    for alpha in alphas:
        plot_run_results(full_results=res, model_results=model_res, alpha=alpha,
                         scaling_factors=scaling_factors, model_n=model_n, pair_criteria=pair_criteria)

    exit()
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
    X_cav_final_df = X_cav_train_df.iloc[idx_cav_final_train].reset_index(drop=True)

    # print("X_feat_dt_train:", X_feat_tree_train.shape)
    # print("codes_dt_train:", embeddings_dt.shape)
    
    tree_rules = train_binary_trees(embeddings_dt, X_feat_tree_train, feature_cols)
    best_p = max(tree_rules.keys(), key=lambda p: len(tree_rules[p]))
    rules_df = pd.DataFrame(tree_rules[best_p])

    embeddings_cav = test_emb_scaled[idx_test_cav_train]
    embeddings_cav_training = embeddings_cav[idx_cav_final_train]
    embeddings_cav_training_encoded = sae_codes_test[idx_cav_final_train]
    y_cav_training = y_test_cav_train[idx_cav_final_train]

    high_quantile = 1.0 - (best_p / 100.0)

    X_eval = X_test_np[idx_test_tcav_eval]
    dist_eval = np.array([year_to_domain_combined[y] for y in years_test_tcav_eval], dtype=np.int64)
    grads = get_model_gradients(model=drift_model, X=X_eval, dist_vec=dist_eval)
    cavs = train_cavs_from_rules(rules_per_percentile=tree_rules[best_p], X_cav_train_df=X_cav_final_df, cav_train_emb=embeddings_cav_training,
                                 cav_train_emb_encoded=embeddings_cav_training_encoded, y_cav_train=y_cav_training,
                                 feature_cols=feature_cols, emb_scaler=scaler, high_quantile=high_quantile, min_pos_samples=50, random_state=42)
    tcav_scores = get_tcav_scores(cavs=cavs.values(), grads=grads)
    for cav in cavs.values():
        idx = cav['Factor']
        cav['TCAV_score'] = tcav_scores[idx]
    
    for rule in tree_rules[best_p]:
        idx = rule['Factor']
        if idx in cavs:
            prec, rec = rule['Precision'], rule['Recall']
            cavs[idx]['Precision'] = prec
            cavs[idx]['Recall'] = rec
    
    robust_tcav_results = {}
    for idx, info in cavs.items():
        robust_result = robust_tcav_significance_test(
            concept_idx=idx, embs=embeddings_cav_training,
            idx_pos=info['positive_idx'],
            idx_neg=info['negative_idx'],
            model_grads=grads, scaler_emb=scaler,
            sample_fraction=1.0, rng_seed=42
        )
        robust_tcav_results[idx] = robust_result
        if robust_result['is_significant']:
            # print(f'Factor {idx}: p={robust_result["p_value"]}, t={robust_result["t_stat"]}')
            pass
    
    significant_concepts = {}
    for idx in cavs:
        if robust_tcav_results[idx]['is_significant'] and abs(cavs[idx]['TCAV_score'] - 0.5) > 0.1:
            significant_concepts[idx] = cavs[idx]
            significant_concepts[idx]['p_value'] = robust_tcav_results[idx]['p_value']
            significant_concepts[idx]['t_stat'] = robust_tcav_results[idx]['t_stat']
            significant_concepts[idx]['Precision'] = cavs[idx]['Precision']
            significant_concepts[idx]['Recall'] = cavs[idx]['Recall']

    for idx, info in significant_concepts.items():
        # print(f'Factor {idx} (TCAV = {info["TCAV_score"]}): {info["Rule"]}')
        # print(f'p={robust_tcav_results[idx]["p_value"]}, t={robust_tcav_results[idx]["t_stat"]}')
        continue
    
    print(f'Total de fatores significativos encontrados: {len(significant_concepts)}')
    for idx, info in significant_concepts.items():
        print(f'Factor {info["Factor"]}: Precision={info["Precision"]:.4f}, Recall={info["Recall"]:.4f}')
        print(f'TCAV_score={info["TCAV_score"]:.2f} ({"PREVENÇÃO" if info["TCAV_score"] < 0.4 else "RISCO"}), p_value={info["p_value"]:.4f}, t_stat={info["t_stat"]:.4f}')
        print(f'Rule: {translate_event_names(full_text=info["Rule"], cid_dict=cid)}')
        print()
    # tcav_df = tcav_result_df_from_concepts(significant_concepts)
    # print(tcav_df)

    tcav_eval_t = torch.tensor(test_emb_scaled[idx_test_tcav_eval])
    tcav_eval_concept_activations = sae.encode(tcav_eval_t).detach().cpu().numpy()

    held_out_t = torch.tensor(test_emb_scaled[idx_test_held_out])
    held_out_concept_activations = sae.encode(held_out_t).detach().cpu().numpy()

    cav_train_t = torch.tensor(test_emb_scaled[idx_test_cav_train])
    cav_train_concept_activations = sae.encode(cav_train_t).detach().cpu().numpy()

    X_test_df = test_rows_checked[feature_cols].copy()
    X_tcav_eval = X_test_df.to_numpy()[idx_test_tcav_eval]
    X_held_out = X_test_df.to_numpy()[idx_test_held_out]
    X_cav_train = X_test_df.to_numpy()[idx_test_cav_train]

    # print("tcav_eval_concept_activations:", tcav_eval_concept_activations.shape)
    # print("held_out_concept_activations :", held_out_concept_activations.shape)
    # print("cav_train_concept_activations:", cav_train_concept_activations.shape)

    fa_results_eval, fa_results_held_out, consistency_df = run_feature_association_dual_split(
        significant_factors=significant_concepts,
        tcav_eval_concept_activations=tcav_eval_concept_activations,
        held_out_concept_activations=held_out_concept_activations,
        X_tcav_eval=X_tcav_eval,
        X_held_out=X_held_out,
        feature_cols=feature_cols,
        quantile=0.1
    )

    print(consistency_df)

    _, _, sre_df = run_sparse_readout_dual_split(
        significant_factors=significant_concepts, cav_train_concept_activations=cav_train_concept_activations,
        tcav_eval_concept_activations=tcav_eval_concept_activations, held_out_concept_activations=held_out_concept_activations,
        X_cav_train=X_cav_train, X_tcav_eval=X_tcav_eval, X_held_out=X_held_out, feature_cols=feature_cols, cv=5,
        overfit_drop_warn_threshold=0.2
    )

    pd.set_option('display.max_columns', None)
    print(sre_df)
