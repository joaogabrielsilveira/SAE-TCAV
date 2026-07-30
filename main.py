import pickle

import numpy as np
from dotenv import load_dotenv
load_dotenv()
import torch
from database import open_feather, get_vars, RENAL_DB_PATH, prepare_database, get_tabpfn_arrays, scale_df_data, compare_distributions
from sae import train_sae_model
from tabpfn_model import fit_dr_tabpfn, walkforward_evaluate_tabpfn, TabPFNEvalConfig, load_or_extract_embeddings, \
    EmbeddingExtractConfig, scale_embeddings, scale_embeddings_l2, temporal_test_subsplits, TRAINING_EMBEDDING_FILE, TEST_EMBEDDING_FILE, PRED_PROB_FILE
from decision_tree import train_binary_trees, get_binary_targets, extrair_regras_resumidas, get_rules_forced
from tcav import train_cavs_from_rules, get_model_gradients, get_tcav_scores, robust_tcav_significance_test, run_feature_association_dual_split, run_sparse_readout_dual_split, get_significant_concepts
from filepaths import get_env_path
from pickle import dump, load
import os
import pandas as pd
from pathlib import Path
from tabpfn import TabPFNClassifier
from results import cid10_dict, translate_event_name, events_dict, tcav_result_df_from_concepts, translate_event_names
import matplotlib.pyplot as plt

PREPARED_DB_PATH = get_env_path('data/renal/prep.pkl')
CONCEPT_STATS_DIR = Path('stats/concepts')

if __name__ == '__main__':
    CONCEPT_STATS_DIR.mkdir(parents=True, exist_ok=True)
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

    cut_features = False
    
    feature_cols = list(np.asarray(top_k_events))
    X_train_df = train_rows[feature_cols].copy()
    X_test_df = test_rows[feature_cols].copy()

    data_shift_df, features_to_keep, features_to_keep_name = compare_distributions(x_i=X_train_np, x_j=X_test_np, top_k_events=top_k_events)
    
    X_train_start = train_rows[train_rows['year'] == 2000][feature_cols].to_numpy()
    X_train_end = train_rows[train_rows['year'] == max(years_train_np)][feature_cols].to_numpy()

    data_shift_train, _, _ = compare_distributions(X_train_start, X_train_end, top_k_events)

    X_test_start = test_rows[test_rows['year'] == min(years_test_np)][feature_cols].to_numpy()
    X_test_end = test_rows[test_rows['year'] == max(years_test_np)][feature_cols].to_numpy()

    data_shift_test, _, _ = compare_distributions(X_test_start, X_test_end, top_k_events)

    data_shift_df.to_csv('stats/data_shift.csv')
    data_shift_train.to_csv('stats/data_shift_train.csv')
    data_shift_test.to_csv('stats/data_shift_test.csv')

    if not cut_features:
        features_to_keep = np.ones(len(top_k_events), dtype=bool)
        features_to_keep_name = top_k_events
    
    X_train_np = X_train_np[:, features_to_keep]
    X_test_np = X_test_np[:, features_to_keep]

    train_rows = train_rows[features_to_keep_name + ['patient_id', 'year', 'DEATH']]
    test_rows = test_rows[features_to_keep_name + ['patient_id', 'year', 'DEATH']]

    # print(f'years_train: {years_train_np.shape}')
    # print(f'years_test: {years_test_np.shape}')
    # print("train_rows:", train_rows.shape)
    # print("test_rows :", test_rows.shape)
    # print("n top_k_events:", len(top_k_events))
    # print("train_years:", np.unique(years_train_np))
    # print("test_years :", np.unique(years_test_np))
    # print("X_train_np:", X_train_np.shape)
    # print("y_train_np:", y_train_np.shape, "| pos_ratio:", y_train_np.mean())

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
        use_cache=True
    )

    results_per_year = wf["results_per_year"]
    year_to_domain_combined = wf["year_to_domain_combined"]
    test_rows_checked = wf["test_rows_checked"]
    test_rows_checked = test_rows

    # results_df = pd.DataFrame(results_per_year).sort_values("year").reset_index(drop=True)
    # print(results_df)

    scaler, X_train_df, X_test_df = scale_df_data(
        X_train_df, X_test_df, feature_cols
    )

    # print(f'X_train_df_norm: {X_train_df.shape})
    # print(f'X_test_df_norm : {X_test_df.shape}')

    emb_cfg = EmbeddingExtractConfig()
    emb_cfg.use_cache = True
    emb_out = load_or_extract_embeddings(
        model=drift_model,
        X_train_np=X_train_np,
        X_test_np=X_test_np,
        years_train=years_train_np,
        years_test=years_test_np,
        year_to_domain_map=year_to_domain_combined,
        embeddings_dir=None,
        cfg=emb_cfg,
        device=model_add_x_device,
        example_add_shape=example_add_shape,
    )

    embedding_info = {
        'model': drift_model,
        'X_train_np': X_train_np,
        'ytd_map': year_to_domain_combined,
        'years_train': years_train_np,
        'device': model_add_x_device,
        'example_add_shape': example_add_shape, 
        'cfg': emb_cfg
    }

    train_emb = emb_out['train_emb_flat'].astype(np.float32)
    test_emb = emb_out['test_emb_flat'].astype(np.float32)
    train_emb_scaled, test_emb_scaled, scaler_emb = scale_embeddings(train_emb, test_emb, fit_test=False)

    from instance_selection import infer_with_selected_instances
    test_indices = range(0, len(test_rows))
    np.random.seed(42)
    infer_indices = np.random.choice(test_indices, size=50, replace=False)
    X_infer_np = X_test_np[infer_indices, :]
    y_infer_np = y_test_np[infer_indices]
    years_infer_np = years_test_np[infer_indices]
    data = infer_with_selected_instances(
        base_model=drift_model,
        base_sae=train_sae_model(inputs=torch.tensor(train_emb_scaled, dtype=torch.float32), save_data=False, use_cache=False, alpha=1.0),
        X_train_np=X_train_np,
        years_train_np=years_train_np,
        y_train_np=y_train_np,
        train_embs=train_emb_scaled,
        X_infer_np=X_infer_np,
        y_infer_np=y_infer_np,
        years_infer_np=years_infer_np,
        emb_scaler=scaler_emb,
        add_x_device=model_add_x_device,
        add_x_shape=example_add_shape
    )
    for key, val in data.items():
        print(f'{key}: {val}')

    mean_train = np.mean(train_emb, axis=0)
    std_train = np.std(train_emb, axis=0)

    mean_test = np.mean(test_emb, axis=0)
    std_test = np.std(test_emb, axis=0)

    rows = []
    for idx in range(train_emb.shape[1]):
        rows.append({
            'Factor index': idx,
            'Mean diff': abs(mean_train[idx] - mean_test[idx]),
            'Std diff': abs(std_train[idx] - std_test[idx]),
            'Training mean': mean_train[idx], 'Training std': std_test[idx],
            'Test mean': mean_test[idx], 'Test std': std_test[idx]
        })
    data_shift_df = pd.DataFrame(rows)
    with open('stats/embeddings_shift.csv', 'wb') as f:
        data_shift_df.to_csv(f)

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

    from sae_compare import run_sae_random_comparison, plot_run_results, run_random_comparison_year_differences
    import matplotlib.pyplot as plt
    inputs = emb_discovery
    pair_criteria = 'cos_sim'
    model_type = 'ReLU'
    model_n = range(3, 15)
    alphas = [0.1, 0.5, 1.0]
    ks = [8]
    scaling_factors = [1.5]

    res, model_res, full_pair_df, sae_list = run_sae_random_comparison(
        model_nums=model_n,
        hyper_params=alphas if model_type == 'ReLU' else ks,
        embs=inputs, 
        scaling_factors=scaling_factors, 
        model_type=model_type,
        pair_criteria=pair_criteria,
    )
    pd.set_option('display.max_rows', 20)
    full_pair_df['is_relevant_pair'] = full_pair_df[pair_criteria] >= 0.7
    original_sae_df = full_pair_df[full_pair_df['sae_i_idx'] == 0].copy()

    stats_by_concept = original_sae_df.groupby('original_concept').agg(
        survival_rate=('is_relevant_pair', 'mean'),
        mean_cos_sim=('cos_sim', 'mean'),
        mean_overlap=('overlap', 'mean')
    ).reset_index().sort_values('survival_rate', ascending=False)

    surviving_concepts = stats_by_concept[stats_by_concept['survival_rate'] > 0.0]

    pair_sae_idx = 1
    original_sae_first_pairing_df = original_sae_df[original_sae_df['sae_j_idx'] == pair_sae_idx].copy()

    first_pairing_survivors = original_sae_first_pairing_df[original_sae_first_pairing_df['is_relevant_pair'] == True]
    original_survivors = first_pairing_survivors['original_concept'].tolist()
    pairs = first_pairing_survivors['best_pair'].tolist()

    with open(CONCEPT_STATS_DIR / f'surviving_concepts_{model_type}_sae.pkl', 'wb') as out:
        dump(surviving_concepts, out)
    if model_type == 'ReLU':
        for alpha in alphas:
            plot_run_results(full_results=res, model_results=model_res, hyper_param=alpha,
                            scaling_factors=scaling_factors, model_n=model_n, pair_criteria=pair_criteria, model_type=model_type)
    if model_type == 'TopK':
        for k in ks:
            plot_run_results(full_results=res, model_results=model_res, hyper_param=k,
                            scaling_factors=scaling_factors, model_n=model_n, pair_criteria=pair_criteria, model_type=model_type)
            
    sae = sae_list[0]['model']
    codes_train_tensor = []
    codes_test_tensor = []
    for s in sae_list:
        model = s['model']
        if model_type == 'TopK':
            sae_codes_train = model.encode(x=torch.tensor(inputs))[1].cpu().detach().numpy()
            sae_codes_test = model.encode(torch.tensor(test_emb_scaled[idx_test_cav_train]))[1].cpu().detach().numpy()
        else:
            sae_codes_train = model.encode(x=torch.tensor(inputs)).cpu().detach().numpy()
            sae_codes_test = model.encode(torch.tensor(test_emb_scaled[idx_test_cav_train])).cpu().detach().numpy()
        
        codes_train_tensor.append(sae_codes_train)
        codes_test_tensor.append(sae_codes_test)
    
    codes_train_tensor = np.asarray(codes_train_tensor, dtype=np.float32)
    codes_test_tensor = np.asarray(codes_test_tensor, dtype=np.float32)

    sae_codes_train = codes_train_tensor[0]
    sae_codes_test = codes_test_tensor[0]

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

    tree_rules = train_binary_trees(embeddings_dt, X_feat_tree_train, feature_cols, model_type=model_type)
    best_p = max(tree_rules.keys(), key=lambda p: len(tree_rules[p]))
    rules_df = pd.DataFrame(tree_rules[best_p])

    forced_rules = get_rules_forced(train_activations=embeddings_dt,
                                    X=X_feat_tree_train,
                                    surviving_concepts=original_survivors, 
                                    tree_rules_df=rules_df, 
                                    perc=best_p, feature_names=feature_cols, model_type=model_type
                                )

    original_concept_rules = pd.DataFrame(forced_rules)[['Factor', 'Rule', 'Precision', 'Recall']]
    print(original_concept_rules)
    original_concept_rules = original_concept_rules.rename(columns={'Factor': 'original_concept', 'Rule': 'original_rule', 'Precision': 'original_prec', 'Recall': 'original_rec'})
    original_concept_rules = pd.merge(
        left=original_concept_rules,
        right=original_sae_first_pairing_df[['original_concept', 'cos_sim', 'overlap']],
        how='inner',
        on='original_concept'
    )

    pair_concept_rules = pd.DataFrame(get_rules_forced(train_activations=(codes_test_tensor[pair_sae_idx])[idx_tree_train], X=X_feat_tree_train, surviving_concepts=pairs, tree_rules_df=None, perc=best_p, feature_names=feature_cols, model_type=model_type))
    pair_concept_rules = pair_concept_rules[['Factor', 'Rule', 'Precision', 'Recall']]
    pair_concept_rules = pair_concept_rules.rename(columns={'Factor': 'pair_concept', 'Rule': 'pair_rule', 'Precision': 'pair_prec', 'Recall': 'pair_rec'})

    rules_paired_df = pd.concat([original_concept_rules, pair_concept_rules], axis=1).dropna()
    rules_paired_df['original_rule'] = [translate_event_names(full_text=rule, cid_dict=cid) for rule in rules_paired_df['original_rule']]
    rules_paired_df['pair_rule'] = [translate_event_names(full_text=rule, cid_dict=cid) for rule in rules_paired_df['pair_rule']]

    with open(CONCEPT_STATS_DIR / f'paired_rules_{model_type}_sae.pkl', 'wb') as out:
        dump(rules_paired_df, out)
    print(rules_paired_df)

    forced_rules_df = pd.DataFrame(forced_rules)

    with open(CONCEPT_STATS_DIR / f'forced_rules_{model_type}_sae.pkl', 'wb') as out:
        dump(forced_rules_df, out)

    all_rules_df = pd.concat([rules_df, forced_rules_df])
    for info in forced_rules:
        tree_rules[best_p].append(info)

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

    significant_concepts, significant_df = get_significant_concepts(
        cavs=cavs, tcav_scores=tcav_scores, best_rules=tree_rules[best_p],
        grads=grads, embs=embeddings_cav_training, scaler=scaler_emb
    )

    significant_concepts = pd.merge(
        significant_df,
        stats_by_concept,
        left_on='Factor',
        right_on='original_concept',
        how='inner'
    )
    significant_concepts['translated_rule'] = [translate_event_names(full_text=rule, cid_dict=cid) for rule in significant_concepts['Rule']]

    print(f'Total de fatores significativos encontrados: {len(significant_concepts)}')
    for idx, row in significant_concepts.iterrows():
        print(f'Factor {row["Factor"]}: Precision={row["Precision"]:.4f}, Recall={row["Recall"]:.4f}')
        print(f'TCAV_score={row["TCAV_score"]:.2f} ({"PREVENÇÃO" if row["TCAV_score"] < 0.4 else "RISCO"}), p_value={row["p_value"]:.4f}, t_stat={row["t_stat"]:.4f}')
        print(f'Survival rate: {(row["survival_rate"] * 100):.2f}%')
        print(f'Mean cosine similarity: {row["mean_cos_sim"]:.4f}, Mean overlap: {row["mean_overlap"]:.4f}')
        print(f'Rule: {row["translated_rule"]}')
        print()
    
    graph_columns = ['Factor', 'Precision', 'Recall', 'TCAV_score', 'survival_rate', 'mean_cos_sim', 'mean_overlap', 'translated_rule']
    with open(CONCEPT_STATS_DIR / f'significant_factors_{model_type}_sae.pkl', 'wb') as out:
        dump(significant_concepts[graph_columns], out)
    
    num_latents = int(192 * max(scaling_factors))
    non_zero_concept_mask = ~(np.all(sae_codes_train <= 1e-5, axis=0))
    remaining_concepts = [i for i in range(num_latents) if (i not in significant_concepts['Factor']) and (i not in surviving_concepts) and (non_zero_concept_mask[i] == True)]
    np.random.seed(42)
    random_concepts = np.random.choice(remaining_concepts, size=len(original_survivors), replace=False)
    forced_random_rules = get_rules_forced(train_activations=embeddings_dt, X=X_feat_tree_train, surviving_concepts=random_concepts,
                                           tree_rules_df=None, perc=best_p, feature_names=feature_cols, model_type=model_type)

    print(pd.DataFrame(forced_random_rules))
    print(f'Regras geradas por conceitos aleatorios: {len(forced_random_rules)}/{len(random_concepts)}')
    print(f'Regras geradas por conceitos pareados  : {len(original_concept_rules)}/{len(original_survivors)}')
    exit()

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
