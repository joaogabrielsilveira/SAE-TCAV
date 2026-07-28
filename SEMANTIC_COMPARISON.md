# Cross-run SAE semantic comparison

## Scope and compatibility

This feature adds an opt-in semantic layer beside existing SAE geometry and CAV/TCAV analysis. It does not replace factor matching, the legacy high-precision decision-tree rule, or existing artifact names.

Existing behavior remains authoritative for CAV construction:

- `decision_tree.train_binary_trees` produces the precision-constrained primary single rule; `decision_tree.get_rules_forced` preserves the existing fallback path.
- `tcav.train_cavs_from_rules` uses those rules to choose clean positive examples, then learns CAVs in TabPFN embedding space.
- `sae_compare.get_concepts_matching` continues one-to-one Hungarian matching by decoder-direction cosine or high-activation cohort overlap.
- Existing `main.py` execution and legacy files remain unchanged. Semantic experiments write to a separate namespace.

Semantic rule sets serve cross-run description and transfer only. They are recall-oriented OR-of-ANDs models and must not feed `train_cavs_from_rules`. Keeping both paths separate avoids contaminating CAV positives with broader, lower-precision semantic coverage.

## Installation and required inputs

Use Python 3.11 and Git. The repository depends on a pinned Drift-Resilient
TabPFN fork because the current code uses its domain-shift, embedding, and
gradient APIs; PyPI `tabpfn` is not a compatible substitute.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-semantic.txt
python -m pytest -q
```

`requirements-semantic.txt` contains the complete direct Python dependency set
for semantic comparison and the upstream SAE, tree, CAV/TCAV, and dataset code.
The TabPFN Git dependency includes the checkpoint selected by
`TabPFNEvalConfig`.

Dependencies cannot supply experiment data. There are now two supported entry
points:

- `main-comparison.py` starts from the renal event Feather file and runs data
  preparation, TabPFN, the configured SAE runs, geometric matching, the legacy
  high-precision rule/CAV/TCAV path, and the complete semantic comparison.
- `semantic_experiment.py` starts from already-computed, aligned SAE
  activations. It is useful for rerunning only the semantic layer.

The lower-level semantic CLI expects:

- an aligned NPZ bundle containing raw tabular features, outcomes, patient IDs,
  feature names, and activations from each already-trained SAE run;
- a JSON list of one-to-one factor matches;
- an optional external clinical-group mapping.

The exact schemas and commands are documented below. `main.py` remains an
unchanged, outdated research script. Its early `exit()` calls do not affect
`main-comparison.py`, and they should not be commented out for the new
workflow. A non-renal dataset must first be adapted to the runner's prepared
data seam or directly to the semantic bundle contract.

## Complete renal runner

`main-comparison.py` is a thin command-line wrapper around the importable
`comparison_runner.run_comparison` interface. The default configuration is
`comparison_runner.example.json`.

For a Feather file in the repository root:

```bash
python main-comparison.py \
  --config comparison_runner.example.json \
  --device cuda \
  --data tidy_event_data.feather
```

`--device cuda` fails immediately if CUDA is unavailable, preventing an
expensive run from silently falling back to CPU. Use `--device auto` or the
configuration default when the same configuration must also work on
CPU-only machines.

The Feather file must use the current renal event schema: `patient_id`,
`date`, and `event`, including an exact `DEATH` event. The preparation code
builds patient-year count rows and derives the binary `DEATH` outcome.

The default complete run:

1. Prepares the renal train/test rows and fits Drift-Resilient TabPFN.
2. Evaluates TabPFN by test year and extracts aligned train/test embeddings.
3. Makes one patient-grouped four-way split of test records.
4. Trains one SAE per configured seed on `semantic_fit` (the example uses
   seeds `42` and `135`).
5. Re-encodes every test record through every frozen SAE.
6. Performs one-to-one Hungarian matching and keeps pairs with cosine at least
   `0.7`.
7. Learns the existing precision-first single rule for each selected factor,
   constructs CAVs on `semantic_select`, and evaluates TCAV on `tcav_eval`.
8. Learns recurrent recall-oriented semantic rule sets and evaluates pooled
   and class-separated symmetric transfer on `semantic_final`.

### GPU execution controls

The complete runner uses the selected accelerator for TabPFN, SAE training,
batched SAE activation encoding, and TCAV gradients. The example configuration
starts with conservative settings suitable for a large GPU:

```json
{
  "accelerator": {"device": "auto"},
  "tabpfn": {"batch_size": 1024},
  "sae": {"encoding_batch_size": 8192},
  "functional": {"gradient_batch_size": 512}
}
```

For the first GPU run, pass `--device cuda`. If CUDA runs out of memory,
reduce the batch size for the failing stage; completed and failed stage names
are persisted as described below. TCAV gradients remain
FP32 because transfer scores depend on gradient/CAV dot-product signs.
Mixed-precision and TF32 modes are intentionally not enabled in this initial
optimization.

`stage_metrics.json` is updated after every stage and records wall-clock
seconds, resolved device, and—on CUDA—peak allocated and reserved VRAM. The
same information, total timed seconds, GPU name, compute capability, CUDA
version, and total GPU memory are copied into `runner_manifest.json`. This
makes the first run a useful profiling run without requiring an external
profiler.

Stable randomized-tree discovery, CAV logistic regression, rule clustering,
and tabular preparation remain CPU operations. A GPU therefore accelerates
the neural stages but does not remove the need for later deterministic CPU
parallelism when running millions of shallow tree fits.

### Progress reporting

Progress bars are enabled by default with `"show_progress": true` in the
complete runner and `"runtime": {"show_progress": true}` in the standalone
semantic configuration. They track TabPFN years and embedding batches, SAE
runs and epochs, geometric run pairs, high-precision tree fits, TCAV gradient
batches and factors, semantic factor/threshold representations, rule
bootstraps, and final pair evaluation. Nested bars are transient so completed
runs leave a readable terminal history.

Disable all progress bars for redirected logs or automated jobs:

```bash
python main-comparison.py --no-progress --data tidy_event_data.feather
python semantic_experiment.py --no-progress --config ... --bundle ... --matches ...
```

Progress settings are presentation-only and are deliberately excluded from
content-addressed experiment hashes. Turning bars on or off therefore reuses
the same scientific artifacts.

No stage imports or executes `main.py`. Use configuration fields rather than
editing that file.

Useful command-line overrides:

```bash
# Recompute even when a complete cached result exists.
python main-comparison.py --force --data tidy_event_data.feather

# Recompute only one cache stage group. Repeat the option when needed.
python main-comparison.py --force-stage sae --data tidy_event_data.feather
python main-comparison.py --force-stage functional --force-stage semantic \
  --data tidy_event_data.feather

# Include every Hungarian assignment, regardless of cosine. This can be very
# expensive with the standard bootstrap configuration.
python main-comparison.py --all-pairs --data tidy_event_data.feather

# Run geometry and stable semantic rules without rebuilding CAV/TCAV.
python main-comparison.py --skip-functional --data tidy_event_data.feather
```

For a smaller first run, set `matching.max_pairs_per_run_pair` in
`comparison_runner.example.json` and reduce `discovery.n_bootstraps` and
`discovery.trees_per_bootstrap` in the semantic configuration. The runner
prints the selected pair/factor count and estimated randomized-tree fits before
semantic rule discovery.

The runner creates a content-addressed directory below
`<artifact_dir>/<runner_hash>/`. Its main outputs are:

- `summary.json` and `runner_manifest.json`;
- `cache_refs.json`, identifying every shared stage artifact used by the run;
- `prepared.pkl`, `embeddings.npz`, `activations.npz`, and `sae/run_<id>.pt`;
- `splits.npz` and `semantic_inputs.npz`, preserving aligned records for audit
  or lower-level reruns;
- `matches_all.{json,csv}` and `matched_factors.{json,csv}`;
- `high_precision_rules.jsonl`, `functional.json`,
  `functional_cavs.npz`, and scoped decision-tree/gradient artifacts;
- `semantic/<experiment_hash>/`, containing all semantic artifacts documented
  later in this file, including `pair_metrics_by_class.csv`.

The runner hash includes dataset bytes, scientific runner/semantic
configuration, external clinical-group mapping bytes, and relevant source
files. Execution-only cache paths, verification policy, progress, device
request, and inference batch sizes are excluded. A matching `summary.json` is
the complete-result cache.

### Stage-aware shared cache

Expensive intermediate results live below
`<artifact_dir>/_cache/v2/<stage>/<sha256>/`, outside runner-hash directories.
Changing one configuration field therefore invalidates only stages whose
scientific dependencies changed.

Important examples:

- adding an SAE seed reuses prepared data, splits, TabPFN, embeddings, existing
  SAE models/activations, and existing SAE-pair matchings;
- changing matching score limits reuses raw Hungarian matchings;
- changing semantic objectives reuses randomized-tree bootstraps and rule
  families, then reruns constrained selection;
- increasing semantic bootstraps computes only new bootstrap IDs;
- changing clinical groups reuses tree discovery and reclusters rule families;
- changing class analysis reruns only final held-out evaluation;
- changing DOT export, documentation, progress, or CSV formatting never
  invalidates embeddings or SAE models.

Every entry has a versioned manifest, dependency/source/environment identity,
payload sizes and SHA-256 checksums, and a completion marker. Writes use a
per-key POSIX lock and atomic directory publication. Missing, partial, corrupt,
misaligned, or checksum-mismatched entries are quarantined and recomputed.
Run-local legacy filenames remain compatibility/audit exports; lookup uses only
the shared cache.

Configuration remains backward compatible:

```json
{
  "use_cache": true,
  "cache_dir": null,
  "cache_verification": "checksum"
}
```

`cache_dir: null` means `<artifact_dir>/_cache/v2`.
`cache_verification` accepts `checksum` (default) or `manifest`.
`use_cache: false` bypasses shared reads and writes while retaining ordinary
run artifacts.

`--force` recomputes every stage. Repeatable `--force-stage` accepts
`prepared`, `splits`, `tabpfn`, `embeddings`, `sae`, `activations`, `matching`,
`functional`, or `semantic`. Forced computation never overwrites a valid
canonical entry. If fresh output differs, `cache_refs.json` records the
nondeterminism.

Cache retention is explicit; nothing is automatically deleted:

```bash
python -m comparison_cache inspect --root stats/comparison/_cache/v2
python -m comparison_cache prune --root stats/comparison/_cache/v2 \
  --unreferenced
python -m comparison_cache prune --root stats/comparison/_cache/v2 \
  --older-than-days 90
# Apply only after reviewing dry-run output.
python -m comparison_cache prune --root stats/comparison/_cache/v2 \
  --unreferenced --apply
```

Cache payloads include pickle/Torch files and are trusted-local artifacts.
Never load a cache copied from an untrusted source.

## Implemented modules

### `comparison_cache.py`

Provides shared stage resolution through one deep interface. It owns
content-derived keys, POSIX locks, atomic publication, checksums, quarantine,
hit/miss/force reporting, `cache_refs.json`, inspection, and explicit pruning.

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

Provides strict dataclass configuration for activation targets, constrained selection, discovery, runtime, additive class-separated evaluation, artifacts, and clinical-group mapping. `class_analysis.enabled` defaults to `true`; setting it to `false` omits class-specific evaluation while leaving pooled results unchanged. Unknown top-level keys and non-boolean class-analysis values fail early. `load_clinical_groups` accepts one-to-many feature mappings from JSON; unmapped features remain distinguishable singleton groups during discovery.

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
- Keep CAV/TCAV compatibility alongside semantic results. `compare_tcav_pair` computes it from legacy CAVs/gradients. Library callers may pass nested or keyed `functional_by_factor` entries containing `CAV`/`cav` and `TCAV_score`/`tcav_score`; pair results then include CAV cosine, TCAV scores/difference, and effect-sign agreement. The lower-level semantic CLI retains precomputed pair metadata; the complete renal runner rebuilds the high-precision rules and functional artifacts before invoking it.

Directional values remain first-class. High `i_to_j` with low `j_to_i` can reveal semantic containment: run A's rule describes a subset of run B, while run B's broader rule does not isolate run A.

### Additive evaluation by observed class

When `class_analysis.enabled` is `true`, the pipeline also partitions `semantic_final` by the observed outcome supplied as `outcome_for_stratification`. It evaluates the same frozen activation cutoffs and rule sets on each class subset. It does not train more SAEs, refit rules, fit class-specific cutoffs, or expose final rows to model selection.

The existing pooled `transfer` object remains unchanged. Every target-level pair result gains a sibling `class_analysis` list, ordered deterministically by class value. Each item records:

- `class_value`, final-subset sample count, and positive-target counts for both matched factors;
- `valid` plus explicit reasons when either factor has no positive activation targets in that class;
- the same directional and symmetric transfer structure used by pooled analysis.

Prevalence, precision, recall, F2, lift, WRAcc, coverage, target/prediction Jaccard, selected-cohort Jaccard, and activation-cohort Jaccard are recomputed from that class subset. Both directions use the same ordered held-out rows. Exact feature and clinical-group agreement remain pair-level properties because class analysis reuses the pooled frozen rule sets.

Pair-level `class_threshold_stability` maps each serialized class value to its F2 minimum, maximum, and range across configured activation targets. This complements the existing pooled threshold-stability summary; it does not replace it. Geometry and CAV/TCAV fields also remain unchanged and pair-level.

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
- `class_analysis.enabled`: additive final evaluation by observed outcome class; defaults to `true`
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

`requirements-semantic.txt` installs both the semantic layer and its upstream
TabPFN/PyTorch/data-science environment. Python 3.11 is the supported target.

## Artifacts

Semantic outputs belong under `<artifact_dir>/<experiment_hash>/`, separate from legacy `stats/`, `models/tcav/cavs.pkl`, and `models/tcav/grads.pkl` names.

Written files:

- `manifest.json`: schema, resolved configuration, experiment hash, tabular/activation/split/clinical-group fingerprints, environment versions, record/pair counts, and `class_analysis` metadata (`enabled`, deterministically ordered final class values, and final class support).
- `splits.npz`: partition indices plus hashed record keys needed to prove cross-run alignment; avoid raw patient identifiers.
- `semantic_rules.jsonl`: one run/factor/target record containing fitted cutoff, family recurrence/stability, representative rules, bootstrap seeds/diagnostics, and selection metrics/status.
- `pair_results.jsonl`: complete pooled directional pair records plus additive target-level class analyses and pair-level class threshold stability when enabled.
- `pair_metrics.csv`: existing pooled flattened analysis table; columns and one-row-per-pair/target behavior remain unchanged.
- `pair_metrics_by_class.csv`: additive one-row-per-pair/target/class analysis table, written only when class analysis is enabled.
- `result.json`: complete cached return payload, including manifest, semantic models, and pair results.

Experiment hash includes resolved configuration, semantic source fingerprint, tabular data, stratification outcome, patient groups, optional record keys, aligned activation matrices, split indices, clinical-group mapping, match rows, and feature names. Activation bytes therefore identify effective SAE outputs even though SAE weight files are not CLI inputs. `--force` recomputes. Internal semantic caches are finer: one entry per bootstrap, family clustering, and constrained selection. Gradient reuse checks model identity, ordered record keys, feature matrix, domain vector, shape, and checksum rather than record count alone.

## Runtime profiles

Use explicit profile values in checked experiment JSON rather than hidden behavior:

- Smoke: about 3 outer bootstraps and 5-10 trees each; exercises full flow, not scientific output.
- Standard: 30 outer bootstraps and 50-100 trees each; default development/research iteration.
- Publication: 100 outer bootstraps and about 1,000 shallow trees each after runtime validation.

Main cost scales with matched SAE runs, referenced factors, activation targets, bootstraps, trees, and candidate subsets. Orchestrator learns only factors referenced by match rows. It skips invalid/dead factors early, caps candidates per bootstrap using fitting/out-of-bag diagnostics, filters recurrence before subset search, uses exhaustive search only below configured candidate limit, switches to deterministic beam search above it, and caches both complete experiments and granular semantic stages. Current backend runs sequentially and config rejects `n_jobs` values other than `1`; field reserves future deterministic parallel execution.

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

Dead factors, insufficient positive fitting samples, single-class bootstraps, empty candidate pools, no recurrent families, and no feasible constrained subset produce explicit invalid reasons/warnings plus empty rule sets. A final class subset with no positive target for either matched factor retains its class-result shape but is marked invalid with explicit reasons. These cases do not silently relax constraints or fall back to final held-out data.

## Deferred extension points

Initial implementation intentionally keeps one-to-one geometric matching and per-factor semantics. Later work can consume frozen rule sets and aligned cohort masks for:

- ablation/destruction-effect transfer;
- one-to-many or many-to-many factor matching;
- grouping split or absorbed SAE features;
- optional WoodTapper or SkopeRules candidate backends.

These additions should not change activation-target fitting, strict held-out separation, directional transfer schemas, or legacy CAV isolation.
