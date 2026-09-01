# Clean, Full-Data Temporal Synthesis Redesign

## Summary

Refactor the patched synthesis code into a maintainable downstream analysis pipeline, then rerun it from the same frozen enrichment and CRI manifests. The default run will:

- Use every eligible cached reference year and complete member/factor vector.
- Train the outcome-free core DAE with `k=2`.
- Exhaustively evaluate all 216 DAE configurations with nested all-year LOYO.
- Automatically select the fastest deterministic CPU/CUDA execution strategy.
- Resume safely after interruption.
- Treat sparse, constant TCAV results as an audited sensitivity rather than a valid latent-dimensionality input.
- Never import or execute TabPFN, SAE, matching, rule, CAV, TCAV, enrichment, or CRI builders.

Stop the current run now because it is early in the second of eight outer folds and cannot persist completed search work. Preserve its log as the pre-optimization timing audit.

## Implementation Changes

### 1. Clean architecture and interfaces

- Keep `temporal_metric_synthesis.py` as a thin CLI and compatibility facade exposing:
  - `MetricSynthesisConfig`
  - `MetricSynthesisRuntimeConfig`
  - `build_metric_synthesis(...)`
  - `load_metric_synthesis(...)`
- Move alignment, supervised analysis, dimensionality, DAE search, and artifact publication into focused `temporal_synthesis` modules. Keep `temporal_robustness_autoencoder.py` limited to preprocessing, model definition, corruption, fitting, and inference.
- Delete the obsolete two-year DAE implementation and consolidate duplicated LOYO, reconstruction, aggregation, and device-selection logic.
- Replace compressed one-line statements with typed dataclasses, named result objects, docstrings describing scientific invariants, and one declarative metric-profile registry.
- Separate scientific configuration from runtime configuration:
  - Scientific configuration affects artifact identity.
  - Device, worker count, progress display, logging, and resume policy are runtime settings.
- Fingerprint every synthesis/autoencoder source module, numerical-library versions, input manifests, and resolved numerical backend. Worker count does not alter identity.
- Load only the required frozen tables while validating their descriptors and enrichment-CRI linkage.

### 2. Full-data and high-throughput defaults

The no-extra-flags default will use:

- Profiles: `core` plus audited `p50_tcav_extended`.
- DAE: enabled, `profile="core"`, `latent_dimensions=2`.
- All reference years dynamically present in the complete core member vectors; for the pinned artifacts this is 2007-2014.
- All 1,222 complete non-distance-zero core member/factor vectors.
- All 216 prespecified DAE candidates.
- Eight outer LOYO tests, seven inner reference folds per outer test, and validation seeds 42, 43, and 44.
- `device="auto"`, `executor="auto"`, `workers="auto"`, and resume enabled.

Implement a common search-job abstraction with stable seeds derived from outer year, inner year, candidate, and validation seed.

- CUDA execution uses a thread pool with thread-local CUDA streams, fixed train/validation tensors on-device, and per-job CUDA generators.
- CPU execution uses persistent worker processes with one Torch thread per worker to avoid oversubscription.
- Corruption is generated directly on the selected device.
- Search mode records validation history and selected epoch without computing or synchronizing unused clean-training MSE each epoch.
- Validation synchronization remains because early stopping depends on it.
- Model initialization and output ordering remain deterministic regardless of job scheduling.

Before the real search, `auto` benchmarks a fixed 20-epoch set of 28 representative jobs:

- CPU concurrency: 1, 7, and up to 28 physical-core workers.
- CUDA concurrency: 1, 7, 14, and up to 28 streams, bounded by available CPU threads and device memory.
- Select the valid executor with the highest jobs/second after a warm-up.
- Record every benchmark, selected executor, throughput, stage duration, and peak CUDA memory in runtime telemetry.
- Explicit `--device`, `--executor`, or `--workers` overrides bypass selection and are validated strictly.

Checkpoint completed candidate-fold groups atomically. Also checkpoint selected outer configurations and outer OOF evaluation results. Resume only when input hashes, scientific configuration, source fingerprints, and numerical backend match exactly. An interruption may lose only currently in-flight jobs.

Fully vectorized candidate networks are deferred from the initial rewrite because they require a custom batched optimizer and complicate maintainability. The adaptive CPU/CUDA executor is the default production optimization; telemetry will provide evidence for a later vectorized implementation.

### 3. Statistical corrections

Add a metric-quality artifact for every profile containing finite count, missing fraction, unique-value count, variance, effective rank, and eligibility reason.

For dimensionality analysis:

- Detect zero-variance metrics before standardization.
- Do not silently remove a prespecified metric and continue under the same profile name.
- Mark `p50_tcav_extended` dimensionality as `not_estimable_zero_variance` because observed `u_tcav` is always 1.0.
- Emit TCAV coverage and raw/reference-value audits, but no six-metric PCA/FA/parallel-analysis claims.
- Keep the full-data five-metric core analysis as the primary dimensionality result.
- Fit FA only for `k < effective_rank` and `k < metric_count`.
- Capture warnings per fit and publish convergence state, iterations, finite likelihood, reconstruction error, communalities, and exclusion reason.
- Apply deterministic varimax rotation, factor ordering, and sign orientation only to valid converged fits.
- Run parallel analysis using the actual estimable metric count.
- Add reference/split-cluster bootstrap stability for PCA subspaces and FA loadings using the configured 1,000 repetitions.

For supervised analysis:

- Keep original Death-F1 as headline and balanced-context as a separate sensitivity.
- Preserve common-complete and maximal-history populations, LOYO and forward chaining, and fold-local scaling/selection.
- Keep p50 TCAV prediction output only as an explicitly underpowered sensitivity.
- Add fields stating that TCAV and its temporal delta have zero predictive variation and prohibit attributing any improvement to TCAV.
- Implement the originally planned reference-cluster bootstrap intervals and paired sign-flip comparison for concepts versus performance history.
- Continue excluding Death-F1 from all PCA, FA, DAE, reconstruction, and DAE-selection operations.

## Artifacts, Logging, and Notebook

- Publish schema-versioned `metric_synthesis_<hash>` artifacts only after complete checksum validation.
- Include scientific configuration, runtime configuration, source fingerprints, numerical environment, data-coverage audit, executor benchmark, stage telemetry, resume history, and all derived tables in the manifest.
- Maintain resumable state under a clearly named work directory; remove it only after the final manifest validates.
- Write progress to console and `temporal-metric-synthesis.log` by default, including completed/total jobs, outer year, throughput, ETA, selected backend, early stopping, checkpoint publication, and resume events.
- Atomically update a canonical-manifest pointer only after successful validation. The notebook loads this pointer rather than a stale hard-coded synthesis hash.
- Update the notebook with:
  - Core dimensionality and loadings.
  - Explicit TCAV non-estimability/coverage panel.
  - Supervised headline and sensitivity labels.
  - DAE/PCA/FA OOF reconstruction comparisons.
  - Latent trajectories, Procrustes stability, and runtime summary.
- After the corrected artifact passes validation, rename the existing log as the pre-optimization audit and remove only incomplete work products. Leave prior completed content-addressed artifacts intact but noncanonical.

The default CLI remains explicit about frozen inputs:

```bash
PYTHONPATH=. .venv/bin/python temporal_metric_synthesis.py \
  --enrichment-manifest <completed-enrichment-manifest> \
  --cri-manifest <completed-cri-manifest>
```

Optional controls include `--device`, `--executor`, `--workers`, `--no-resume`, `--skip-dae`, and `--log-file`.

## Test and Acceptance Plan

- Unit-test exact alignment, all-year discovery, complete member selection, metric-quality audits, zero variance, rank deficiency, FA factor limits, convergence reporting, varimax orientation, and TCAV sensitivity labels.
- Prove preprocessing, corruption, early stopping, PCA/FA, Ridge tuning, and degradation thresholds see training references only.
- Test deterministic device-native corruption and model initialization across serial and concurrent schedules.
- Test CUDA streams when CUDA is available; otherwise skip CUDA-specific tests while fully exercising the CPU executor.
- Simulate interruption after candidate and outer-fold checkpoints, then verify resumed output equals uninterrupted output.
- Test adaptive executor selection with controlled benchmark timings and resource limits.
- Verify changing any scientific source module invalidates reuse, while changing only worker count does not.
- Statically assert that synthesis modules do not import upstream builders.
- Run the ordinary unit suite plus a reduced-grid all-reference integration test before the canonical exhaustive run.
- Validate the final pinned-data artifact has:
  - 1,222 complete core DAE input vectors.
  - All eight available outer reference years.
  - `8 x 7 x 216 = 12,096` nested search-fold records.
  - Three DAE OOF seeds for every outer year.
  - No uncaptured numerical or convergence warnings.
  - TCAV dimensionality marked non-estimable with its zero variance and missingness documented.
  - A complete checksummed manifest and canonical pointer.
- Compare the adaptive executor against serial execution on the benchmark workload and record the measured speedup; scientific acceptance remains based on identical grouping and reconstruction criteria, not GPU utilization alone.

## Assumptions and Defaults

- The current running process will be stopped before implementation.
- Frozen enrichment and CRI artifacts remain immutable and are the only scientific inputs.
- Missing utilities are not imputed; "maximum data" means every eligible cached complete vector, not fabricated targets.
- Core `k=2` is the human-approved DAE configuration.
- Exhaustive nested validation remains scientifically authoritative; adaptive execution changes scheduling only.
- The DAE remains exploratory unless it consistently beats matched PCA out of sample across reference years and seeds.
