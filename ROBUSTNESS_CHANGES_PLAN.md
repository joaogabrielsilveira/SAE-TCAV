# Multi-percentile robustness and matching diagnostics

## Summary

- Fit embedding `StandardScaler` only on `test_raw[idx_semantic_fit]`; transform all test embeddings with that frozen scaler. Transform train embeddings only for compatibility, never fitting on them.
- Compute activation profiles on `semantic_select` at percentiles 70, 80, and 90.
- Add unfiltered cosine-Hungarian, overlap-Hungarian, and directed nearest-neighbor analyses for every run pair under `scope="all"`.
- Use the analysis thresholds for the configured selector: cosine uses `cosine_analysis_threshold`; overlap evaluates every analysis percentile, then keeps the percentile with the most threshold-qualified factors and that percentile's assignment. New robustness analyses remain separate from the selected pairs passed downstream.
- Use `cosine >= 0.60`, `overlap >= 0.70`, and strict recurrence highlighting `recurrence > 0.50`.

## Pipeline and interfaces

- Change `DefaultComparisonAdapter.embeddings(...)` to accept semantic splits.
  - Fit scaler on `test_raw[idx_semantic_fit]`.
  - Produce `test_scaled = scaler.transform(test_raw)`, including semantic-fit itself because SAE training consumes that subset.
  - Preserve `train_scaled = scaler.transform(train_raw)` for compatibility only.
  - Persist scaler plus fit provenance: split name, row count, fit-index fingerprint, mean fingerprint, and scale fingerprint.
  - Make `embeddings_scaled` cache depend on raw embeddings and semantic-fit indices; changing split seed must invalidate scaling and downstream SAE stages.

- Extend `MatchingRunnerConfig` while preserving `scope`, `criterion`, and `max_pairs_per_run_pair`; remove `minimum_score`:
  - `analysis_percentiles=(70, 80, 90)`
  - `cosine_analysis_threshold=0.60`
  - `overlap_analysis_threshold=0.70`
  - `nearest_neighbor_top_k=3`
  - `alternative_score_deltas=(0.05, 0.10)`
  - For overlap-selection ties, choose the highest percentile deterministically.
  - Normalize JSON lists to tuples and validate ordered unique percentiles, score ranges, positive top-k, valid deltas, and selected overlap percentile membership.

- Refactor activation thresholding in `sae_compare.py`:
  - Add percentile-aware computation returning masks and per-factor thresholds for all requested percentiles.
  - Preserve positive-activation-only percentile fitting, strict activation mask comparison `activation > percentile_threshold`, and all-false masks for dead factors.
  - Keep the existing singular `high_activation_matrix(..., perc=90)` as a compatibility adapter for legacy callers.
  - Runner stores profiles from `activation_matrix[idx_semantic_select]`, never final-held-out data.

- Add a pure deep module, `robustness_matching.py`, with one main interface:
  - `analyze_run_pair(left_directions, right_directions, left_profiles, right_profiles, top_k) -> PairMatchingAnalysis`.
  - Result contains cosine matrix, overlap matrix per percentile, cosine-Hungarian assignment, overlap-Hungarian assignment per percentile, directed top-three nearest neighbors, reciprocal/collision diagnostics, and nearest-neighbor-versus-Hungarian gaps.
  - Deterministic ranking uses descending score then ascending target-factor ID.
  - Rectangular matrices remain valid; unassigned Hungarian factors are recorded as missing and count as recurrence failures later.

## Matching and artifact contract

- For every unordered run pair from `scope="all"`:
  - Compute cosine matrix once.
  - Compute overlap/Jaccard matrices independently at percentiles 70, 80, and 90.
  - Compute primary cosine-Hungarian assignment and attach `cos_sim`, `overlap_p70`, `overlap_p80`, and `overlap_p90`.
  - Compute secondary overlap-Hungarian assignment separately for each percentile.
  - Compute directed nearest neighbors in both directions for cosine and every overlap percentile.

- Preserve current downstream artifacts:
  - `matches_all.{json,csv}` continues representing the configured selector's complete Hungarian assignment.
  - `matched_factors.{json,csv}` applies the configured criterion's analysis threshold and per-pair cap.
  - When overlap is selected, compute an assignment at every analysis percentile, count threshold-qualified factors, and retain the winning percentile plus its factors; ties choose the highest percentile.
  - Functional and semantic stages receive only `matched_factors`, never unions or nearest-neighbor matches.

- Add run-local artifacts:
  - `high_activation_profiles.npz`: masks and threshold vectors keyed by run and percentile.
  - `matching/matrices/run_<i>__run_<j>.npz`: `cosine`, `overlap_p70`, `overlap_p80`, and `overlap_p90`.
  - `matching/manifest.json`: schema version, thresholds, percentiles, run-pair coverage, shapes, filenames, fingerprints, and row counts.
  - `cosine_hungarian_matches.{json,csv}`: primary fixed assignment with threshold-pass flags and two-of-three overlap-consistency flag.
  - `overlap_hungarian_matches.{json,csv}`: secondary assignments, one row set per percentile.
  - `nearest_neighbor_candidates.{json,csv}`: up to three ranked candidates per directed source factor and metric.
  - `matching_diagnostics.{json,csv}`: best/second/third scores, score gaps, threshold validity, valid-alternative flags for 0.05 and 0.10, reciprocal status, target collision counts, Hungarian score, and `nearest_minus_hungarian`.

- Nearest-neighbor rules:
  - Candidate passes minimum using `>= 0.60` for cosine or `>= 0.70` for overlap.
  - Rank-two/rank-three candidate is valid at delta `d` when it passes threshold and `best_score - candidate_score <= d`.
  - Store raw and threshold-qualified reciprocal/collision fields.
  - Compare cosine nearest neighbors with cosine-Hungarian scores; compare each overlap nearest neighbor with same-percentile overlap-Hungarian scores.

- Cache each pair as matrices plus threshold-independent raw assignments and rankings. Apply score thresholds and delta flags after cache loading, allowing threshold changes to reuse matrices and assignments. Validate matrix shapes, finite values, score ranges, assignment uniqueness, rank coverage, and factor bounds.

## `robustness_analysis.ipynb`

- Extend strict artifact loading:
  - Require matching manifest, cosine/overlap Hungarian tables, nearest-neighbor diagnostics, and matrix index.
  - Require `scope="all"` and exactly one artifact bundle for every `R choose 2` run pair.
  - Verify matrix fingerprints and confirm every stored score equals its indexed matrix value.
  - Build recurrence from unfiltered analysis artifacts, not `matched_factors.csv`.

- Compute directed evidence for every factor `(r,k)` against every other run:
  - Reverse unordered pair assignments when `r` is stored as right run.
  - Enumerate all SAE-manifest factors; missing assignments count as failures.
  - Cosine recurrence uses fixed cosine-Hungarian matches and `cos_sim >= 0.60`.
  - Primary overlap recurrence keeps those same cosine-Hungarian matches fixed and evaluates `overlap_p >= 0.70` separately at 70, 80, and 90.
  - Cross-percentile consistency passes for a target run when at least two of three overlaps meet 0.70; compute its recurrence over `R-1`.
  - Secondary overlap recurrence uses each percentile's own overlap-Hungarian assignment.
  - Mark recurrent factors only when recurrence is strictly greater than 0.50.

- Export:
  - `factor_recurrence_primary.csv`
  - `factor_recurrence_secondary_overlap.csv`
  - `recurrence_highlights.csv`
  - `nearest_neighbor_summary.csv`
  - `nearest_neighbor_collisions.csv`
  - `nearest_neighbor_ambiguities.csv`
  - Plots comparing recurrence distributions, highlighted-factor counts, cross-percentile consistency, reciprocal rates, collisions, ambiguity rates, and nearest/Hungarian score gaps.

- Preserve existing canonical temporal and semantic-transfer analyses. Keep current semantic `robust` subset distinct from new geometric recurrence labels to prevent metric conflation.
- Extend `analysis_manifest.json` with matching-manifest fingerprint, thresholds, percentiles, recurrence counts, collision/ambiguity counts, output filenames, and `source_artifacts_modified=false`.
- Clear stale notebook outputs after schema changes; only commit refreshed outputs after executing against artifacts produced by updated runner.

## Tests and acceptance

- Scaling tests prove scaler statistics come from semantic-fit test embeddings, not train embeddings, and every split uses same frozen transform.
- Activation tests cover all three percentiles, positive-only thresholds, dead factors, tied thresholds, deterministic ordering, and compatibility wrapper.
- Synthetic matching tests cover:
  - Different cosine and overlap Hungarian assignments.
  - Percentile-dependent overlap assignments.
  - Many-to-one nearest-neighbor collisions.
  - Reciprocal and non-reciprocal matches.
  - Rank-two/rank-three alternatives at both deltas.
  - Exact-threshold equality passing.
  - Nearest/Hungarian gaps, ties, and rectangular matrices.
- Recurrence tests cover denominator `R-1`, reverse pair orientation, missing assignments, strict `>0.50` highlighting, fixed-cosine overlap evaluation, two-of-three consistency, and secondary overlap matching.
- Runner tests verify `R choose 2` coverage, cache reuse, artifact round trips, manifest fingerprints, metric-specific selector thresholds, percentile winner selection, and unchanged downstream isolation.
- Update `comparison_runner.example.json` and `SEMANTIC_COMPARISON.md` with new fields, formulas, artifact schemas, selector separation, and quadratic matrix-storage cost.
- Verification target: supported Python 3.11 environment from `requirements-semantic.txt`. Current planning shell uses Python 3.14 without `torch` or `pandas`, so baseline tests could not collect there.
