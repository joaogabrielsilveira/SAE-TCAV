"""Forecast next-year deterioration from completed matching-system artifacts.

This artifact-only builder never fits mortality models or extracts concepts.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np

from artifact_storage import (atomic_write_json, atomic_write_jsonl_gzip,
                              canonical_json, file_sha256, read_artifact,
                              validate_descriptor)
from temporal_synthesis.profiles import profile_metrics

LOG = logging.getLogger(__name__)
PARENT = "5fd57eb7b61700cda81e"
ENRICHMENT = "a3034b25b327c7484446"
CRI = "cri_d6d075a79151863ac633"
WINDOWS = "windows_6d74dea4c3ee4c6403df"
OUTCOMES = {"frozen_f1": ("death_f1_at_frozen_threshold", -1),
            "average_precision": ("death_average_precision", -1),
            "brier": ("brier_score", 1)}
STRATA = ("system", "outcome", "cohort_view", "activation_target", "profile")
ROW_KEY = ("reference_year", "patient_split_seed", "forecast_origin_year")
CONCEPT_KEY = ("reference_year", "patient_split_seed", "cohort_view", "activation_target", "temporal_distance")
MODELS = ("history", "history_raw", "history_cri", "history_pca", "history_fa", "history_dae", "raw", "cri")


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def key(row, fields):
    return tuple(row[n] for n in fields)


def finite(value):
    return value is not None and bool(np.isfinite(value))


def unique_index(rows, fields):
    result = {}
    for row in rows:
        identity = key(row, fields)
        if identity in result:
            raise ValueError(f"Duplicate identity {identity}")
        result[identity] = row
    return result


def checked_manifest(path, artifact_names=None):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("complete") is not True:
        raise ValueError(f"Incomplete manifest: {path}")
    artifacts = manifest.get("artifacts", manifest.get("aggregate_artifacts", {}))
    for name in artifacts if artifact_names is None else artifact_names:
        validate_descriptor(path.parent, artifacts[name])
    return manifest


def table(path, manifest, name):
    return read_artifact(Path(path).parent, manifest["artifacts"][name])


def write_table(root, name, rows):
    descriptor = atomic_write_jsonl_gzip(Path(root) / f"{name}.jsonl.gz", rows)
    descriptor["path"] = f"{name}.jsonl.gz"
    return descriptor


def configure_logging(root):
    Path(root).mkdir(parents=True, exist_ok=True)
    log_path = (Path(root) / "progress.log").resolve()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in logger.handlers):
        handler = logging.FileHandler(log_path)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return log_path


@dataclass(frozen=True)
class ForecastConfig:
    activation_targets: tuple = (.1, .3, .5)
    outcomes: tuple = tuple(OUTCOMES)
    profiles: tuple = ("core", "p50_tcav_extended")
    designs: tuple = ("held_reference", "forward")
    models: tuple = MODELS
    ridge_alphas: tuple = (1e-4, 1e-3, 1e-2, .1, 1., 10., 100.)
    bootstrap_replicates: int = 1000
    seed: int = 42
    dae_seeds: tuple = (42, 43, 44)
    dae_epochs: int = 250
    dae_device: str = "cpu"

    def __post_init__(self):
        for values, allowed in ((self.outcomes, OUTCOMES), (self.models, MODELS),
                                (self.designs, ("held_reference", "forward")),
                                (self.profiles, ("core", "p50_tcav_extended"))):
            if not values or not set(values).issubset(allowed):
                raise ValueError(f"Unsupported configuration: {values}")
        if self.bootstrap_replicates < 1 or self.dae_epochs < 1 or not self.dae_seeds:
            raise ValueError("Positive bootstrap/epoch counts and DAE seeds are required")


def original_inputs(repo):
    """Validate provenance and rebuild performance on the original concept cohorts."""
    repo = Path(repo)
    pp = repo / "stats/temporal_robustness" / PARENT / "parent_manifest.json"
    ep = pp.parent / "derived" / ENRICHMENT / "manifest.json"
    cp = pp.parent / "derived" / CRI / "manifest.json"
    wp = repo / "stats/temporal_performance_windows" / WINDOWS / "manifest.json"
    parent = checked_manifest(pp)
    enrichment = checked_manifest(ep, ("headline_factor_metrics", "tcav_significance"))
    cri = checked_manifest(cp, ("cri_family_universe",))
    windows = checked_manifest(wp, ("record_probabilities", "thresholds", "yearly_metrics", "legacy_parent_metric_parity"))
    if parent["runner_hash"] != PARENT or enrichment["parent_manifest_sha256"] != file_sha256(pp):
        raise ValueError("Enrichment/parent identity mismatch")
    if cri["enrichment_manifest_sha256"] != file_sha256(ep):
        raise ValueError("CRI/enrichment identity mismatch")
    if windows["dependency_sha256"]["parent_manifest"] != file_sha256(pp):
        raise ValueError("Legacy window probabilities belong to another parent")
    if windows["population_fingerprints"] != parent["population_fingerprints"]:
        raise ValueError("Population identity mismatch")
    import pickle
    from temporal_production import ProductionTemporalAdapter
    from temporal_robustness import _fingerprint_population
    population = None
    candidates = [pp.parent.parent / "_population/prepared.pkl"]
    candidates += sorted((pp.parent.parent / "_population_cache/prepared").glob("*/prepared.pkl"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        with candidate.open("rb") as handle:
            prepared = pickle.load(handle)
        proposed = ProductionTemporalAdapter()._population_from_prepared(prepared)
        if _fingerprint_population(proposed) == parent["population_fingerprints"]:
            population = proposed
            LOG.info("Exact retained population verified: %s", candidate)
            break
    if population is None:
        raise ValueError("No retained population matches the immutable parent")
    exposed, t0, split_hashes = {}, {}, {}
    for success in parent["successful_experiments"]:
        old = Path(success["manifest"])
        sp = pp.parent / old.parent.parent.name / old.parent.name / old.name
        if file_sha256(sp) != success["manifest_fingerprint"]:
            raise ValueError(f"Changed split manifest {sp}")
        sm = checked_manifest(sp, ("reference_roles",))
        rows = table(sp, sm, "reference_roles")
        k = (sm["reference_year"], sm["patient_split_seed"])
        exposed[k] = {str(r["patient_id"]) for r in rows if r["role"] != "t0_evaluation"}
        t0[k] = {int(r["row_index"]) for r in rows if r["role"] == "t0_evaluation"}
        if exposed[k] & {str(r["patient_id"]) for r in rows if r["role"] == "t0_evaluation"}:
            raise ValueError("Reference evaluation patients overlap concept fitting")
        split_hashes[str(k)] = file_sha256(sp)
    thresholds = unique_index([r for r in table(wp, windows, "thresholds") if r["window"] == "legacy_reference_only"],
                              ("reference_year", "patient_split_seed"))
    groups = defaultdict(list)
    descriptor = windows["artifacts"]["record_probabilities"]
    LOG.info("Reading validated legacy probabilities (streaming all window records)")
    with gzip.open(wp.parent / descriptor["path"], "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if row["window"] == "legacy_reference_only":
                groups[key(row, ("reference_year", "patient_split_seed", "test_year"))].append(row)
    from temporal_performance_windows import metric_bundle
    performance, audits = [], []
    for (ref, seed, year), records in sorted(groups.items()):
        unique_index(records, ("row_index",))
        expected_rows = t0[(ref, seed)] if year == ref else set(np.flatnonzero(population.years == year).tolist())
        if {r["row_index"] for r in records} != expected_rows:
            raise ValueError("Saved probabilities do not cover original evaluation records")
        for r in records:
            i = r["row_index"]
            if (str(population.patient_ids[i]) != str(r["patient_id"]) or int(population.outcomes[i]) != r["outcome"]
                    or str(population.record_keys[i]) != r["record_key"] or int(population.years[i]) != year):
                raise ValueError("Probability row differs from retained population")
        threshold = {r["frozen_threshold"] for r in records}
        if len(threshold) != 1:
            raise ValueError("Cutoff changed within trajectory")
        threshold = threshold.pop()
        selected = thresholds[(ref, seed)]
        expected = selected.get("frozen_threshold", selected.get("threshold"))
        if threshold != expected:
            raise ValueError(f"Cutoff differs from reference validation: {selected}")
        if year == ref and {r["row_index"] for r in records} != t0[(ref, seed)]:
            raise ValueError("Reference support differs from concept evaluation")
        for cohort in ("all_comer", "pipeline_unseen"):
            rows = records if cohort == "all_comer" else [r for r in records if str(r["patient_id"]) not in exposed[(ref, seed)]]
            support = digest(sorted((r["row_index"], r["record_key"], str(r["patient_id"]), r["outcome"]) for r in rows))
            identity = {"system": "original", "concept_protocol": "original_reference_roles",
                        "reference_year": ref, "patient_split_seed": seed, "test_year": year,
                        "temporal_distance": year-ref, "cohort_view": cohort,
                        "cohort_definition": "original_all_fitting_roles_v1", "support_fingerprint": support,
                        "frozen_threshold": threshold}
            metrics = metric_bundle([r["outcome"] for r in rows], [r["death_probability"] for r in rows], threshold,
                                    minimum_deaths=10, minimum_survivors=30) if rows else {"valid": False, "invalid_reason": "empty_cohort"}
            performance.append({**identity, **metrics})
            audits.append({**identity, "record_count": len(rows), "d0_alias_verified": year == ref,
                           "window_unseen_count": sum(r["exposure_cohort"] == "pipeline_unseen" for r in records)})
    factors = table(ep, enrichment, "headline_factor_metrics")
    perf = unique_index(performance, ("reference_year", "patient_split_seed", "test_year", "cohort_view"))
    for r in factors:
        match = perf.get(key(r, ("reference_year", "patient_split_seed", "test_year", "cohort_view")))
        if match and r.get("future_denominator") is not None and r["future_denominator"] != match["record_count"]:
            raise ValueError("Concept and probability record denominators differ")
    return {"performance": performance, "factors": factors,
            "tcav": table(ep, enrichment, "tcav_significance"),
            "universe": table(cp, cri, "cri_family_universe"), "membership_audit": audits,
            "sources": {"parent": file_sha256(pp), "enrichment": file_sha256(ep),
                        "cri": file_sha256(cp), "windows": file_sha256(wp), "splits": split_hashes},
            "system": "original"}


def build_forecast_rows(performance, config):
    """Construct targets without requiring future concept observations."""
    groups = defaultdict(list)
    for r in performance:
        groups[key(r, ("system", "reference_year", "patient_split_seed", "cohort_view"))].append(r)
    rows, excluded = [], []
    for group, observations in sorted(groups.items()):
        index = unique_index(observations, ("test_year",))
        if len({r["frozen_threshold"] for r in observations}) != 1:
            raise ValueError("Cutoff must remain frozen across years")
        for (year,), current in sorted(index.items()):
            previous, future = index.get((year-1,)), index.get((year+1,))
            for outcome in config.outcomes:
                column, sign = OUTCOMES[outcome]
                identity = {**dict(zip(("system", "reference_year", "patient_split_seed", "cohort_view"), group)),
                            "outcome": outcome, "forecast_origin_year": year, "outcome_year": year+1,
                            "temporal_distance": year-group[1], "cohort_definition": current["cohort_definition"]}
                reason = None
                if previous is None or future is None:
                    reason = "missing_consecutive_performance"
                elif not all(r.get("valid") and finite(r.get(column)) for r in (previous, current, future)):
                    reason = "insufficient_performance_support"
                if reason:
                    excluded.append({**identity, "reason": reason})
                    continue
                for profile in config.profiles:
                    for activation in config.activation_targets:
                        if profile == "p50_tcav_extended" and activation != .5:
                            continue
                        rows.append({**identity, "activation_target": activation, "profile": profile,
                                     "current": current[column], "previous_deterioration": sign*(current[column]-previous[column]),
                                     "deterioration": sign*(future[column]-current[column]),
                                     "previous_support": previous["support_fingerprint"],
                                     "current_support": current["support_fingerprint"], "future_support": future["support_fingerprint"]})
    unique_index(rows, STRATA + ROW_KEY)
    return rows, excluded


def training_rows(rows, held_reference, origin, design):
    return [r for r in rows if r["reference_year"] != held_reference and
            (design != "forward" or (r["reference_year"] < held_reference and r["outcome_year"] <= origin))]


def weights(rows):
    """Equal references, equal forecast years within reference, equal splits within year."""
    counts, years = defaultdict(int), defaultdict(set)
    for r in rows:
        counts[(r["reference_year"], r["forecast_origin_year"])] += 1
        years[r["reference_year"]].add(r["forecast_origin_year"])
    w = np.array([1/(len(years[r["reference_year"]])*counts[(r["reference_year"], r["forecast_origin_year"])]) for r in rows])
    return w / np.mean(w)


class FoldFeatures:
    """Training-only normalization and representation cache, shared across outcomes."""
    def __init__(self, bundle, root, config):
        self.bundle, self.root, self.config = bundle, Path(root), config
        self.cache = {}

    def get(self, train, requested, representation, seed=42):
        from temporal_cri import CRIAnalysisConfig, calibrate_taus, compute_member_utilities
        from temporal_metric_synthesis import build_metric_vectors, system_concept_features
        from temporal_unified_analysis import UnifiedAnalysisConfig
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA, FactorAnalysis

        profile, activation, cohort = (train[0][n] for n in ("profile", "activation_target", "cohort_view"))
        points = sorted({(r["reference_year"], r["patient_split_seed"], r["temporal_distance"]-offset)
                         for r in train for offset in (0, 1)})
        identity = digest([profile, activation, cohort, points, representation, seed])[:24]
        path = self.root / "transforms" / f"{identity}.json"
        if identity not in self.cache:
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["payload_sha256"] != digest(saved["payload"]):
                    raise ValueError(f"Corrupt transform checkpoint {path}")
                self.cache[identity] = saved["payload"]
            else:
                LOG.info("Fitting %s transform %s (%s, activation=%s, seed=%s)", representation, identity, cohort, activation, seed)
                refs = {r["reference_year"] for r in train}
                cc = CRIAnalysisConfig()
                factors = [r for r in self.bundle["factors"] if r.get("matching_view") == "intersection"
                           and r.get("rule_source") == "semantic" and r.get("target_role") == "primary"
                           and r["activation_target"] == activation]
                taus = calibrate_taus([r for r in factors if r["reference_year"] in refs], cc)
                if not taus:
                    taus = [{"activation_target": activation, "metric": m, "tau": cc.tau_fallback,
                             "used_fallback": True, "reference_member_count": 0} for m in ("prevalence", "activation")]
                members = compute_member_utilities(factors, [r for r in self.bundle["universe"] if r["activation_target"] == activation],
                                                   taus, UnifiedAnalysisConfig(), cc)
                vectors, _ = build_metric_vectors({"headline_factor_metrics": factors, "tcav_significance": self.bundle["tcav"]},
                                                  {"cri_member_utilities": members})
                if cohort == "pipeline_unseen" and self.bundle["system"] == "original":
                    vectors += [{**r, "cohort_view": cohort} for r in vectors if r["cohort_view"] == "all_comer" and r["temporal_distance"] == 0]
                vectors = [r for r in vectors if r["cohort_view"] == cohort]
                metrics = profile_metrics(profile)
                summaries = system_concept_features(vectors, profile)
                pointkeys = {(r, s, cohort, activation, d) for r, s, d in points}
                available = [r for r in summaries if all(finite(r.get(m)) for m in metrics)]
                fit_summaries = [r for r in available if key(r, CONCEPT_KEY) in pointkeys]
                transformed, state = {}, {}
                families = defaultdict(list)
                for v in vectors:
                    families[key(v, CONCEPT_KEY) + (v["factor_family_uid"],)].append(v)
                complete_families = {k: vs for k, vs in families.items() if all(all(finite(v.get(m)) for m in metrics) for v in vs)}
                def assign(row, value):
                    transformed[canonical_json(key(row, CONCEPT_KEY))] = {"values": np.asarray(value).tolist(), "coverage": row["concept_coverage"]}
                if representation in ("raw", "cri"):
                    for r in available:
                        values = [r[m] for m in metrics]
                        if representation == "cri":
                            complete = [vs for k, vs in complete_families.items() if k[:-1] == key(r, CONCEPT_KEY)]
                            values = [np.median([np.mean([v["cri_arithmetic"] for v in vs]) for vs in complete])] if complete else [np.nan]
                        if np.isfinite(values).all():
                            assign(r, values)
                elif len(fit_summaries) >= 2:
                    if representation in ("pca", "fa"):
                        fitx = np.array([[r[m] for m in metrics] for r in fit_summaries])
                        scaler = StandardScaler().fit(fitx)
                        z = scaler.transform(fitx)
                        if np.max(np.std(z, axis=0)) < 1e-12:
                            for r in available:
                                assign(r, [0., 0.])
                            state = {"constant": True, "location": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
                        else:
                            reducer = (PCA(n_components=2) if representation == "pca" else FactorAnalysis(n_components=2, random_state=42)).fit(z)
                            for r in available:
                                assign(r, reducer.transform(scaler.transform([[r[m] for m in metrics]]))[0])
                            state = {"location": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                                     "components": reducer.components_.tolist(), "mean": reducer.mean_.tolist(),
                                     "noise_variance": np.asarray(getattr(reducer, "noise_variance_", [])).tolist()}
                    elif representation == "dae":
                        import torch
                        from temporal_robustness_autoencoder import (DenoisingAutoencoderConfig, fit_denoising_autoencoder, fit_utility_preprocessor)
                        valid = [v for vs in complete_families.values() for v in vs]
                        fitting = [v for v in valid if key(v, CONCEPT_KEY) in pointkeys and v["temporal_distance"] > 0]
                        if len(fitting) >= 2:
                            fitx = np.array([[v[m] for m in metrics] for v in fitting])
                            pre = fit_utility_preprocessor(fitx, "standard_linear")
                            model_config = DenoisingAutoencoderConfig(latent_dimensions=2, epochs=self.config.dae_epochs,
                                                                    seed=seed, device=self.config.dae_device)
                            result = fit_denoising_autoencoder(pre.transform(fitx), model_config, progress=False, restore_best=False)
                            with torch.no_grad():
                                latent = result["model"].encode(torch.as_tensor(pre.transform([[v[m] for m in metrics] for v in valid]), device=result["device"])).cpu().numpy()
                            latent_families = defaultdict(list)
                            for v, value in zip(valid, latent):
                                latent_families[key(v, CONCEPT_KEY) + (v["factor_family_uid"],)].append(value)
                            for r in available:
                                vals = [np.median(vs, axis=0) for k, vs in latent_families.items() if k[:-1] == key(r, CONCEPT_KEY)]
                                if vals:
                                    assign(r, np.median(vals, axis=0))
                            state = {"location": pre.location.tolist(), "scale": pre.scale.tolist(), "config": asdict(model_config),
                                     "weights": {k: v.cpu().numpy().tolist() for k, v in result["model"].state_dict().items()},
                                     "training_member_count": len(fitting), "epochs": self.config.dae_epochs}
                payload = {"identity": identity, "training_points": points, "training_references": sorted(refs),
                           "taus": taus, "representation": representation, "seed": seed, "state": state, "features": transformed}
                atomic_write_json(path, {"payload_sha256": digest(payload), "payload": payload})
                self.cache[identity] = payload
        payload = self.cache[identity]
        output = []
        for r in requested:
            k = key(r, CONCEPT_KEY)
            now = payload["features"].get(canonical_json(k))
            prior = payload["features"].get(canonical_json(k[:-1] + (k[-1]-1,)))
            output.append(None if now is None or prior is None else
                          now["values"] + (np.array(now["values"])-prior["values"]).tolist() + [now["coverage"]])
        return output, identity


def fit_predict_model(train, test, name, features, config, alpha, seed=42):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    history = lambda rows: [[r["current"], r["previous_deterioration"], r["temporal_distance"]] for r in rows]
    transform_id = None
    if name.startswith("history"):
        x, z = history(train), history(test)
    else:
        x, z = [[] for _ in train], [[] for _ in test]
    if name != "history":
        representation = name.removeprefix("history_")
        values, transform_id = features.get(train, train+test, representation, seed)
        a, b = values[:len(train)], values[len(train):]
        if any(v is None for v in a+b):
            return None, {"reason": "missing_concept_representation", "transform": transform_id}
        x, z = [h+c for h, c in zip(x, a)], [h+c for h, c in zip(z, b)]
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), [r["deterioration"] for r in train], sample_weight=weights(train))
    prediction = model.predict(scaler.transform(z))
    return prediction, {"transform": transform_id, "scaler_location": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
                        "coefficients": model.coef_.tolist(), "intercept": float(model.intercept_), "alpha": alpha}


def eligible(rows):
    return len(rows) >= 3 and len({r["reference_year"] for r in rows}) >= 2


def nested_prediction(train, test, name, design, features, config):
    seeds = config.dae_seeds if name == "history_dae" else (config.seed,)
    losses = {a: [] for a in config.ridge_alphas}
    for held in sorted({r["reference_year"] for r in train}):
        origins = sorted({r["forecast_origin_year"] for r in train if r["reference_year"] == held}) if design == "forward" else [None]
        for origin in origins:
            inner = training_rows(train, held, origin, design)
            validation = [r for r in train if r["reference_year"] == held and (origin is None or r["forecast_origin_year"] == origin)]
            if not eligible(inner):
                continue
            for alpha in config.ridge_alphas:
                predictions = [fit_predict_model(inner, validation, name, features, config, alpha, seed)[0] for seed in seeds]
                if any(p is None for p in predictions):
                    continue
                pred = np.mean(predictions, axis=0)
                losses[alpha].extend((r["reference_year"], r["forecast_origin_year"], abs(p-r["deterioration"])) for r, p in zip(validation, pred))
    def loss(values):
        cells, refs = defaultdict(list), defaultdict(list)
        for ref, year, v in values:
            cells[(ref, year)].append(v)
        for (ref, _), vs in cells.items():
            refs[ref].append(np.mean(vs))
        return float(np.mean([np.mean(vs) for vs in refs.values()]))
    scores = {a: loss(v) for a, v in losses.items() if v}
    alpha = min(scores, key=lambda a: (scores[a], a)) if scores else 1.
    results = [fit_predict_model(train, test, name, features, config, alpha, seed) for seed in seeds]
    if any(p is None for p, _ in results):
        return None, {"reason": "missing_concept_representation", "fits": [s for _, s in results]}
    return np.mean([p for p, _ in results], axis=0), {"alpha": alpha, "inner_scores": scores, "fallback_alpha": not scores,
                                                   "seeds": seeds, "fits": [s for _, s in results]}


def summarize_predictions(predictions, config):
    summaries, comparisons = [], []
    groups = defaultdict(list)
    for r in predictions:
        groups[key(r, STRATA + ("design",))].append(r)
    def reference_losses(rows, power):
        cells, refs = defaultdict(list), defaultdict(list)
        for r in rows:
            cells[(r["reference_year"], r["forecast_origin_year"])].append(abs(r["prediction"]-r["deterioration"])**power)
        for (ref, _), values in cells.items():
            refs[ref].append(np.mean(values))
        return {ref: float(np.mean(values)) for ref, values in refs.items()}
    for group, rows in groups.items():
        identity = dict(zip(STRATA + ("design",), group))
        by_model = defaultdict(list)
        for r in rows:
            by_model[r["model"]].append(r)
        for name, values in list(by_model.items()):
            mae, mse = reference_losses(values, 1), reference_losses(values, 2)
            summaries.append({**identity, "model": name, "mae": float(np.mean(list(mae.values()))),
                              "rmse": float(np.sqrt(np.mean(list(mse.values())))), "rows": len(values), "references": len(mae)})
            if name in ("no_change", "training_mean", "history_maximal"):
                continue
            for baseline in ("history", "no_change"):
                if baseline == name:
                    continue
                base = unique_index(by_model[baseline], ROW_KEY)
                paired = [r for r in values if key(r, ROW_KEY) in base]
                matched_base = [base[key(r, ROW_KEY)] for r in paired]
                if not paired:
                    continue
                for metric, power in (("mae", 1), ("rmse", 2)):
                    ml, bl = reference_losses(paired, power), reference_losses(matched_base, power)
                    refs = sorted(ml)
                    m, b = np.array([ml[r] for r in refs]), np.array([bl[r] for r in refs])
                    reduce = np.sqrt if power == 2 else lambda a: a
                    gain = float(reduce(b.mean())-reduce(m.mean()))
                    lower = upper = None
                    if len(refs) >= 2:
                        rng = np.random.default_rng(config.seed)
                        ix = rng.integers(0, len(refs), size=(config.bootstrap_replicates, len(refs)))
                        draws = reduce(b[ix].mean(axis=1))-reduce(m[ix].mean(axis=1))
                        lower, upper = np.quantile(draws, [.025, .975]).tolist()
                    comparisons.append({**identity, "model": name, "baseline": baseline, "metric": metric,
                                        "improvement": gain, "lower_95": lower, "upper_95": upper,
                                        "paired_rows": len(paired), "references": len(refs),
                                        "support_fingerprint": digest(sorted(key(r, ROW_KEY) for r in paired)),
                                        "primary": name == "history_raw" and baseline == "history" and metric == "mae" and
                                        group[1:5] == ("frozen_f1", "all_comer", .5, "core") and group[-1] == "forward"})
    return summaries, comparisons


def build_forecasting(bundle, output_root, config=None):
    config = config or ForecastConfig()
    import sklearn
    import torch
    sources = {name: file_sha256(Path(__file__).with_name(name)) for name in
               ("temporal_concept_forecasting.py", "temporal_cri.py", "temporal_metric_synthesis.py", "temporal_robustness_autoencoder.py")}
    identity = digest({"config": asdict(config), "sources": sources, "inputs": bundle["sources"],
                       "system": bundle["system"], "numpy": np.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__})[:20]
    root = Path(output_root) / f"forecast_{identity}"
    configure_logging(root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        load_forecasting(manifest_path)
        LOG.info("Completed run verified: %s", manifest_path)
        return manifest_path
    LOG.info("Starting %s forecast run %s; checkpoints and log: %s", bundle["system"], identity, root)
    rows, exclusions = build_forecast_rows(bundle["performance"], config)
    features = FoldFeatures(bundle, root, config)
    groups = defaultdict(list)
    for r in rows:
        groups[key(r, STRATA)].append(r)
    predictions, folds = [], []
    started = time.monotonic()
    for group_number, (group, values) in enumerate(sorted(groups.items()), 1):
        LOG.info("Stratum %d/%d: %s; %d forecast rows", group_number, len(groups), group, len(values))
        for design in config.designs:
            for held in sorted({r["reference_year"] for r in values}):
                origins = sorted({r["forecast_origin_year"] for r in values if r["reference_year"] == held}) if design == "forward" else [None]
                for origin in origins:
                    fold_id = digest([group, design, held, origin])[:24]
                    checkpoint = root / "folds" / f"{fold_id}.json"
                    if checkpoint.exists():
                        saved = json.loads(checkpoint.read_text())
                        if saved["sha256"] != digest(saved["payload"]):
                            raise ValueError(f"Corrupt fold checkpoint: {checkpoint}")
                        payload = saved["payload"]
                        predictions.extend(payload["predictions"]); folds.extend(payload["folds"])
                        continue
                    train = training_rows(values, held, origin, design)
                    test = [r for r in values if r["reference_year"] == held and (origin is None or r["forecast_origin_year"] == origin)]
                    fold = {**dict(zip(STRATA, group)), "design": design, "held_reference": held, "origin": origin,
                            "train_rows": len(train), "train_references": sorted({r["reference_year"] for r in train}),
                            "maximum_training_outcome_year": max((r["outcome_year"] for r in train), default=None), "fold_id": fold_id}
                    fp, ff = [], []
                    if not eligible(train):
                        ff.append({**fold, "status": "non_estimable", "reason": "insufficient_training_references"})
                    else:
                        raw, _ = features.get(train, train+test, "raw")
                        common_train = [r for r, f in zip(train, raw[:len(train)]) if f is not None]
                        common_test = [r for r, f in zip(test, raw[len(train):]) if f is not None]
                        jobs = [(m, common_train, common_test) for m in config.models] + [("history_maximal", train, test)]
                        ff.extend({**fold, "status": "excluded", "reason": "missing_current_or_previous_concepts", **{n: r[n] for n in ROW_KEY}}
                                  for r, f in zip(test, raw[len(train):]) if f is None)
                        for name, tr, te in jobs:
                            if not eligible(tr) or not te:
                                ff.append({**fold, "model": name, "status": "non_estimable", "reason": "insufficient_concept_support"})
                                continue
                            LOG.info("Forecast model=%s design=%s held=%s origin=%s train=%d test=%d", name, design, held, origin, len(tr), len(te))
                            pred, state = nested_prediction(tr, te, "history" if name == "history_maximal" else name, design, features, config)
                            ff.append({**fold, "model": name, "status": "complete" if pred is not None else "non_estimable", **state})
                            if pred is not None:
                                fp.extend({**r, "design": design, "model": name, "prediction": float(p), "fold_id": fold_id} for r, p in zip(te, pred))
                        if eligible(common_train):
                            mean = float(np.average([r["deterioration"] for r in common_train], weights=weights(common_train)))
                            for name, value in (("no_change", 0.), ("training_mean", mean)):
                                fp.extend({**r, "design": design, "model": name, "prediction": value, "fold_id": fold_id} for r in common_test)
                    payload = {"predictions": fp, "folds": ff}
                    atomic_write_json(checkpoint, {"payload": payload, "sha256": digest(payload)})
                    predictions.extend(fp); folds.extend(ff)
                    LOG.info("Fold complete: %s ref=%s origin=%s train=%d test=%d predictions=%d elapsed=%.1fs",
                             design, held, origin, len(train), len(test), len(fp), time.monotonic()-started)
    summaries, comparisons = summarize_predictions(predictions, config)
    artifacts = {name: write_table(root, name, data) for name, data in
                 (("performance", bundle["performance"]), ("membership_audit", bundle["membership_audit"]),
                  ("measured_concepts", bundle["factors"]), ("forecast_rows", rows), ("exclusions", exclusions),
                  ("folds", folds), ("predictions", predictions), ("summaries", summaries), ("comparisons", comparisons))}
    transforms = [{"path": str(p.relative_to(root)), "sha256": file_sha256(p)} for p in sorted((root / "transforms").glob("*.json"))]
    atomic_write_json(manifest_path, {"schema_version": "concept_forecasting_v1", "complete": True,
                                    "system": bundle["system"], "config": asdict(config), "sources": bundle["sources"],
                                    "source_code": sources, "artifacts": artifacts, "transforms": transforms,
                                    "persistent_log": "progress.log", "elapsed_seconds": time.monotonic()-started})
    LOG.info("Forecasting complete: %s", manifest_path)
    return manifest_path


def load_forecasting(path):
    manifest = checked_manifest(path)
    for d in manifest.get("transforms", []):
        if file_sha256(Path(path).parent / d["path"]) != d["sha256"]:
            raise ValueError("Transformation checksum mismatch")
    return {"manifest": manifest, **{n: table(path, manifest, n) for n in manifest["artifacts"]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("stats/temporal_concept_forecasting"))
    parser.add_argument("--window-concepts", type=Path, help="Completed window-concept manifest; otherwise use pinned original artifacts")
    parser.add_argument("--config", type=Path, help="Optional ForecastConfig JSON (changes run identity)")
    args = parser.parse_args(argv)
    configure_logging(args.output)
    config = ForecastConfig(**json.loads(args.config.read_text())) if args.config else ForecastConfig()
    try:
        if args.window_concepts:
            manifest = checked_manifest(args.window_concepts)
            for system in manifest["systems"]:
                bundle = {n: table(args.window_concepts, manifest, f"{system}_{n}") for n in ("performance", "factors", "tcav", "universe", "membership_audit")}
                bundle.update(system=system, sources={"window_concepts": file_sha256(args.window_concepts)})
                build_forecasting(bundle, args.output, config)
        else:
            build_forecasting(original_inputs(args.repo), args.output, config)
    except Exception:
        LOG.exception("Forecasting failed; completed checkpoints remain reusable")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
