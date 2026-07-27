# Cross-run SAE semantic comparison

## Scope and compatibility

This feature adds an opt-in semantic layer beside existing SAE geometry and CAV/TCAV analysis. It does not replace factor matching, the legacy high-precision decision-tree rule, or existing artifact names.

Existing behavior remains authoritative for CAV construction:

- `decision_tree.train_binary_trees` produces the precision-constrained primary single rule; `decision_tree.get_rules_forced` preserves the existing fallback path.
- `tcav.train_cavs_from_rules` uses those rules to choose clean positive examples, then learns CAVs in TabPFN embedding space.
- `sae_compare.get_concepts_matching` continues one-to-one Hungarian matching by decoder-direction cosine or high-activation cohort overlap.
- Existing `main.py` execution and legacy files remain unchanged. Semantic experiments write to a separate namespace.

Semantic rule sets serve cross-run description and transfer only. They are recall-oriented OR-of-ANDs models and must not feed `train_cavs_from_rules`. Keeping both paths separate avoids contaminating CAV positives with broader, lower-precision semantic coverage.

## Implemented modules

### `semantic_rules.py`

Owns backend-independent semantic types and evaluation:

- `Condition`: numeric `<=` or `>` condition with feature index/name, raw threshold, optional fitting-set ECDF threshold, and configured clinical groups.
- `Rule`: conjunction of conditions.
- `RuleSet`: union of rules. Its mask is the OR of constituent rule masks.
- `ActivationTargetSpec` and `FittedActivationTarget`: positive-only activation quantiles whose cutoff is learned once on fitting data and then frozen.
- `BinaryMetrics`: prevalence, coverage, precision, recall, F2, lift, WRAcc, and target/prediction Jaccard.
- `select_rule_set`: deterministic exhaustive or beam search over rule subsets. Precision, lift, objective, and marginal-recall constraints are evaluated on the complete union mask, never averages of per-rule scores.

Default targets are top 10%, 25%, and 50% of strictly positive factor activations. A target becomes explicitly invalid when its fitting partition contains too few positive activations.

### `stable_rule_backend.py`

Implements candidate discovery with randomized shallow `sklearn.tree.DecisionTreeClassifier` models:

1. Bootstrap fitting rows or complete patient groups.
2. Train multiple seeded randomized trees per outer bootstrap.
3. Extract paths to positive leaves as structured conjunctions.
4. Collapse repeated conditions on one feature to strongest lower/upper bounds.
5. Record raw thresholds plus ECDF-normalized thresholds computed only from semantic-fitting data.
6. Retain bootstrap/tree seeds, fit and out-of-bag counts, precision, recall, and warnings.

Outer bootstraps define recurrence. Multiple trees within one bootstrap improve candidate coverage but never count as independent recurrence events.

### `semantic_config.py`

Provides strict dataclass configuration for activation targets, constrained selection, discovery, runtime, artifacts, and clinical-group mapping. Unknown top-level keys fail early. `load_clinical_groups` accepts one-to-many feature mappings from JSON; unmapped features remain distinguishable singleton groups during discovery.

### `semantic_artifacts.py`

Provides deterministic JSON hashing, derived seeds, environment version capture, and a content-addressed artifact store supporting JSON, JSONL, and compressed NumPy arrays.

### `semantic_compare.py`

Owns stability families and held-out transfer:

- `rule_similarity` combines cohort Jaccard (0.50), exact feature Jaccard (0.20), clinical-group Jaccard (0.20), and direction/normalized-threshold compatibility (0.10). `RuleSimilarityConfig` exposes weights, cluster threshold, ECDF-threshold tolerance, and minimum recurrence.
- `cluster_rule_families` performs deterministic complete-link clustering and returns serializable `RuleFamilyClusteringResult`. Families cannot merge unless every cross-family rule pair meets configured similarity threshold.
- `RuleFamily` records distinct-bootstrap recurrence, feature/group recurrence, raw and normalized threshold distributions, cohort-overlap distribution, representative medoid, member rules, occurrence references, and retention status.
- `retained_representatives` exposes one representative from each retained family; `recurrent_representatives` remains compatibility helper for explicit recurrence filtering.
- `compare_semantic_pair` evaluates both directions on one aligned final matrix and returns serializable `SemanticPairComparison`: directional and self metrics, per-metric symmetric mean/min, cohort Jaccards, and detailed exact-feature/clinical-group set agreement. `compare_rule_sets_symmetric` preserves concise call style used by orchestration.

### `semantic_splits.py`

Owns patient-grouped split logic without importing TabPFN. `tabpfn_model.semantic_test_subsplits` is a compatibility wrapper, allowing semantic tests and orchestration to use the lightweight module while existing callers can stay in `tabpfn_model`.

### Existing integration seams

- `sae_compare.encode_sae` performs inference through one already-trained SAE; `encode_sae_runs` encodes identical records through every configured run. `train_all_saes` now retains model, run ID, seed, model type, and explicit `decoder_directions` while preserving legacy keys.
- `semantic_splits.semantic_test_subsplits` adds deterministic patient-grouped partitions; `tabpfn_model` re-exports a wrapper and preserves `temporal_test_subsplits` unchanged.
- `tcav.get_model_gradients` accepts an optional cache path and rejects cache entries with a mismatched record count. `compare_tcav_pair` reports CAV cosine and, when gradients exist, raw TCAV scores, absolute difference, effect signs, and sign agreement.

## Data separation and flow

`semantic_test_subsplits` assigns complete patient histories to four disjoint partitions. Default row targets approximate existing proportions:

1. `semantic_fit` (33%): fit activation cutoffs; bootstrap trees; discover candidates; compute normalized thresholds and out-of-bag discovery diagnostics.
2. `semantic_select` (33.5%): compute rule-cohort similarity, cluster families, choose representatives/objectives, and select the constrained OR-of-rules subset.
3. `tcav_eval` (16.75%): existing TCAV evaluation only. This partition is not semantic model-selection data.
4. `semantic_final` (16.75%): untouched cross-run semantic evaluation.

Splitting happens once by patient ID. Every SAE run receives identical record indices. For each configured SAE run and split, existing TabPFN embeddings are encoded without retraining:

```text
shared records -> TabPFN embeddings -> trained SAE A -> factor activations A
                                \----> trained SAE B -> factor activations B
shared raw tabular features -------------------------> semantic rules
```

Raw tabular features are rule inputs. SAE activations define binary targets. Each run/factor/target fits its own cutoff using only `semantic_fit`; that frozen cutoff labels `semantic_select` and `semantic_final`. Thus matched factors can have different activation distributions while transfer still uses exactly the same held-out records.

Per-factor processing:

1. Fit top-positive activation targets on `semantic_fit` activations.
2. Discover structured rules on `semantic_fit` raw features and target labels.
3. Cluster equivalent occurrences on `semantic_select` using direction, exact feature overlap, configured clinical-group overlap, compatible fitting-set-normalized thresholds, and selected-cohort overlap.
4. Count each recurrent family once per outer bootstrap. Retain only families meeting recurrence threshold.
5. Select family representatives and the final OR set on `semantic_select`, optimizing F2 or recall subject to aggregate precision, lift, size, length, and marginal-recall constraints.
6. Freeze cutoffs, family representatives, and selected rule sets before touching `semantic_final`.

V1 treats discovery settings and constraint values as predeclared experiment configuration. To tune them, run candidate configurations through `semantic_fit`/`semantic_select`, choose by the configured aggregate objective on `semantic_select`, then lock one configuration before a single `semantic_final` evaluation. Never choose settings from pair-transfer results. Built-in orchestration intentionally does not automate a held-out grid search.

## Cross-run comparison

Geometry still chooses one-to-one pairs. Semantic results annotate those pairs; they do not alter Hungarian assignments.

For factor `i` in run A and matched factor `j` in run B, at each activation target:

- Apply `R_i` to shared final raw features and score against frozen target `y_j`.
- Apply `R_j` to the same rows and score against frozen target `y_i`.
- Preserve both directional precision, recall, F2, lift, WRAcc, and Jaccard.
- Report arithmetic mean and minimum for each symmetric summary, especially `F2_mean` and `F2_min`.
- Report target-cohort Jaccard and selected-rule-cohort Jaccard.
- Report exact input-feature and configured clinical-group set equality/Jaccard.
- Join decoder cosine and high-activation overlap from existing matching.
- Keep CAV/TCAV compatibility alongside semantic results. `compare_tcav_pair` computes it from legacy CAVs/gradients. Library callers may pass nested or keyed `functional_by_factor` entries containing `CAV`/`cav` and `TCAV_score`/`tcav_score`; pair results then include CAV cosine, TCAV scores/difference, and effect-sign agreement. CLI does not rebuild CAVs and retains precomputed pair metadata supplied as extra match fields.

Directional values remain first-class. High `i_to_j` with low `j_to_i` can reveal semantic containment: run A's rule describes a subset of run B, while run B's broader rule does not isolate run A.

Pair artifacts retain structured comparison plus flat compatibility keys. Representative transfer fragment:

```json
{
  "left_factor_id": "0:17",
  "right_factor_id": "1:42",
  "left_to_right": {
    "source_factor_id": "0:17",
    "target_factor_id": "1:42",
    "metrics": {"precision": 0.71, "recall": 0.83, "f2": 0.80}
  },
  "right_to_left": {
    "source_factor_id": "1:42",
    "target_factor_id": "0:17",
    "metrics": {"precision": 0.66, "recall": 0.74, "f2": 0.72}
  },
  "symmetric_metrics": {
    "f2": {
      "left_to_right": 0.80,
      "right_to_left": 0.72,
      "mean": 0.76,
      "minimum": 0.72
    }
  },
  "t_mean": 0.76,
  "t_min": 0.72,
  "mean": {"f2": 0.76, "recall": 0.785},
  "min": {"f2": 0.72, "recall": 0.74},
  "target_cohort_jaccard": 0.68,
  "selected_cohort_jaccard": 0.64,
  "exact_feature_jaccard": 0.75,
  "exact_feature_equal": false,
  "clinical_group_jaccard": 1.0,
  "clinical_group_equal": true
}
```

Artifact also contains `i_to_j`/`j_to_i` metric aliases, self metrics, complete feature/group set-agreement objects, selected/activation cohort Jaccards, sample counts, prevalence, and coverage. Mean/min cover precision, recall, F2, lift, WRAcc, and Jaccard. Values above are illustrative.

## Rule discovery backend choice

Implementation uses a custom stable-rule layer over scikit-learn randomized trees.

This is SIRUS-style recurrence and stability, not an exact reimplementation of SIRUS fitting or prediction aggregation.

Reasons:

- Scikit-learn is already required by repository; no new compiled or cross-language runtime.
- Structured conditions, patient-group bootstraps, out-of-bag diagnostics, deterministic provenance, clinical metadata, and OR-union selection are direct requirements.
- Candidate generation remains replaceable behind a narrow backend boundary.
- Exact rule strings are never stability identity; downstream families use semantic/cohort similarity.

WoodTapper is a useful Python SIRUS implementation, but adds version/platform constraints and its native prediction aggregation does not directly implement required constrained OR-of-rules selection. It remains a possible optional candidate backend after compatibility and serialization contract tests.

`imodels` SkopeRules can also generate candidates, but its precision-oriented rule screening is poorly aligned with recall-oriented union selection. Depending on it would still require custom bootstrap recurrence, equivalence clustering, clinical grouping, and aggregate constrained search. It is therefore not the default.

## Configuration and CLI

`semantic_experiment.example.json` shows schema `1.0`:

- `activation_targets.positive_fractions`, `min_positive_samples`
- `objective.objective`: `f2` or `recall`
- `objective.min_precision`, `min_lift`, `max_rules`, `max_rule_length`, `min_marginal_recall`
- `objective.exhaustive_candidate_limit`, `beam_width`
- `discovery.n_bootstraps`, `trees_per_bootstrap`, `max_depth`, `min_samples_leaf`
- `discovery.backend` (currently only `randomized_tree`), `max_features`, `splitter`
- `discovery.positive_leaf_probability`, `min_positive_leaf_samples`
- `discovery.max_candidates_per_bootstrap`, `min_family_recurrence`, `family_similarity_threshold`
- `runtime.seed`, `n_jobs`, `cache`, `artifact_dir`
- `clinical_groups_path`

Clinical group mappings live in external JSON, as shown by `clinical_groups.example.json`. One feature may belong to several groups. Unmapped features become singleton `feature:<name>` groups; manifest reports mapping count, coverage, and unmapped names. Production experiments should version this taxonomy with experiment configuration and record its content hash.

The semantic pipeline is opt-in. `semantic_experiment.py` accepts already-computed SAE activations:

```bash
python semantic_experiment.py \
  --config semantic_experiment.example.json \
  --bundle semantic_inputs.npz \
  --matches matched_factors.json
```

`--force` bypasses a matching result cache. Command prints content-addressed output directory.

Bundle must contain `X`, `outcome`, `patient_ids`, `feature_names`, and at least one `activations_run_<id>` array. Optional `record_keys` improves split auditing. Activations must already be aligned with `X`; use `sae_compare.encode_sae_runs` upstream when starting from trained SAE objects. Matches JSON must be a list. Rows may use existing names (`sae_i_idx`, `sae_j_idx`, `original_concept`, `best_pair`) or semantic names (`run_i`, `run_j`, `factor_i`, `factor_j`); extra matching fields such as `cos_sim` and `overlap` are retained as geometry.

Library callers can invoke `learn_factor_semantics` for one run/factor/threshold or `run_semantic_comparison` for the complete aligned experiment. Existing no-argument/notebook-style usage remains unaffected.

`requirements-semantic.txt` lists additive NumPy, scikit-learn, and pytest bounds for this layer. It does not attempt to replace the existing TabPFN/PyTorch environment.

## Artifacts

Semantic outputs belong under `<artifact_dir>/<experiment_hash>/`, separate from legacy `stats/`, `models/tcav/cavs.pkl`, and `models/tcav/grads.pkl` names.

Written files:

- `manifest.json`: schema, resolved configuration, experiment hash, tabular/activation/split/clinical-group fingerprints, environment versions, and record/pair counts.
- `splits.npz`: partition indices plus hashed record keys needed to prove cross-run alignment; avoid raw patient identifiers.
- `semantic_rules.jsonl`: one run/factor/target record containing fitted cutoff, family recurrence/stability, representative rules, bootstrap seeds/diagnostics, and selection metrics/status.
- `pair_results.jsonl`: complete directional pair records.
- `pair_metrics.csv`: flattened analysis table.
- `result.json`: complete cached return payload, including manifest, semantic models, and pair results.

Experiment hash includes resolved configuration, semantic source fingerprint, tabular data, stratification outcome, patient groups, optional record keys, aligned activation matrices, split indices, clinical-group mapping, match rows, and feature names. Activation bytes therefore identify effective SAE outputs even though SAE weight files are not CLI inputs. `--force` recomputes. Never reuse a cache solely because record count matches; optional legacy gradient-cache shape check is only a minimum guard.

## Runtime profiles

Use explicit profile values in checked experiment JSON rather than hidden behavior:

- Smoke: about 3 outer bootstraps and 5-10 trees each; exercises full flow, not scientific output.
- Standard: 30 outer bootstraps and 50-100 trees each; default development/research iteration.
- Publication: 100 outer bootstraps and about 1,000 shallow trees each after runtime validation.

Main cost scales with matched SAE runs, referenced factors, activation targets, bootstraps, trees, and candidate subsets. Orchestrator learns only factors referenced by match rows. It skips invalid/dead factors early, caps candidates per bootstrap using fitting/out-of-bag diagnostics, filters recurrence before subset search, uses exhaustive search only below configured candidate limit, switches to deterministic beam search above it, and caches complete matching experiments. Current backend runs sequentially and config rejects `n_jobs` values other than `1`; field reserves future deterministic parallel execution.

## Reproducibility contract

- Base seed is configuration, not process state.
- Derive stage seeds from stable hashes of semantic identifiers such as run, factor, target, bootstrap, and tree.
- Use `numpy.random.SeedSequence` for bootstrap/tree descendants.
- Sort rules, families, candidates, pairs, and serialized mappings before tie-breaking or writing.
- Persist per-bootstrap seeds/diagnostics plus family bootstrap IDs in `semantic_rules.jsonl`.
- Record Python, NumPy, scikit-learn, SciPy, pandas, and Torch versions where available.
- Record split indices/hashes and verify every cross-run comparison uses same ordered final record keys.
- Freeze activation cutoffs after semantic fitting. Never recompute quantiles on selection, TCAV, or final data.
- Keep final semantic records inaccessible to discovery, family construction, objective choice, constraints, and rule-set selection.

## Failure behavior

Dead factors, insufficient positive fitting samples, single-class bootstraps, empty candidate pools, no recurrent families, and no feasible constrained subset produce explicit invalid reasons/warnings plus empty rule sets. They do not silently relax constraints or fall back to final held-out data.

## Deferred extension points

Initial implementation intentionally keeps one-to-one geometric matching and per-factor semantics. Later work can consume frozen rule sets and aligned cohort masks for:

- ablation/destruction-effect transfer;
- one-to-many or many-to-many factor matching;
- grouping split or absorbed SAE features;
- optional WoodTapper or SkopeRules candidate backends.

These additions should not change activation-target fitting, strict held-out separation, directional transfer schemas, or legacy CAV isolation.
