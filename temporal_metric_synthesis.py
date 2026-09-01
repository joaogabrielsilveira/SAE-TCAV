"""Leakage-safe downstream temporal robustness analysis.

This module reads two *completed* manifests.  It intentionally never imports
the enrichment or CRI builders, so running it cannot rerun scientific work.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from temporal_synthesis.config import MetricSynthesisConfig, MetricSynthesisRuntimeConfig
from temporal_synthesis.dimensionality import bootstrap_stability, dimensionality as _analyze_dimensionality, metric_quality
from temporal_synthesis.profiles import CORE_METRICS, EXTENDED_METRICS, profile_metrics

from artifact_storage import (ARTIFACT_SCHEMA_VERSION, atomic_write_json,
                              atomic_write_jsonl_gzip, canonical_json,
                              descriptor_for_file, file_sha256, read_artifact,
                              validate_descriptor)

MEMBER_KEYS = ("reference_year", "patient_split_seed", "factor_family_uid", "member_sae_seed", "member_factor_id", "activation_target")
FAMILY_KEYS = ("reference_year", "patient_split_seed", "factor_family_uid", "activation_target")
SYSTEM_KEYS = ("reference_year", "patient_split_seed", "cohort_view", "activation_target", "temporal_distance")
LOGGER = logging.getLogger(__name__)


def _progress(iterable, *, total: int, description: str):
    """Use tqdm only for an interactive/info-level command-line run."""
    if not LOGGER.isEnabledFor(logging.INFO):
        return iterable
    from tqdm.auto import tqdm
    return tqdm(iterable, total=total, desc=description, unit="step", leave=False)


def _read_manifest(path: str | Path, *, label: str,
                   artifact_names: Sequence[str] | None = None) -> tuple[Path, dict[str, Any]]:
    path = Path(path)
    LOGGER.info("Loading and validating %s manifest: %s", label, path)
    if not path.is_file(): raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True: raise RuntimeError(f"{label} manifest is incomplete")
    names = tuple(manifest.get("artifacts", {})) if artifact_names is None else tuple(artifact_names)
    for name in names:
        if name not in manifest.get("artifacts", {}):
            raise RuntimeError(f"{label} lacks required completed table: {name}")
        _validate_descriptor(path.parent, manifest["artifacts"][name])
    return path, manifest


def _validate_descriptor(root: Path, descriptor: Mapping[str, Any]) -> None:
    """Validate tables through storage and binary checkpoints by checksum."""
    if descriptor.get("format") != "npz":
        validate_descriptor(root, descriptor)
        return
    path = root / str(descriptor["path"])
    if not path.is_file() or file_sha256(path) != descriptor.get("sha256"):
        raise RuntimeError(f"NPZ artifact changed or is missing: {path}")
    if path.stat().st_size != descriptor.get("size_bytes"):
        raise RuntimeError(f"NPZ artifact size changed: {path}")
    with np.load(path, allow_pickle=False): pass


def validate_inputs(enrichment_manifest: str | Path, cri_manifest: str | Path) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    needed_e = {"headline_factor_metrics", "tcav_significance", "primary_performance", "performance_variants"}
    needed_c = {"cri_member_utilities", "cri_family_universe"}
    ep, enrichment = _read_manifest(enrichment_manifest, label="enrichment", artifact_names=sorted(needed_e))
    cp, cri = _read_manifest(cri_manifest, label="CRI", artifact_names=sorted(needed_c))
    if cri.get("enrichment_manifest_sha256") != file_sha256(ep):
        raise RuntimeError("CRI does not belong to the supplied enrichment manifest")
    if not needed_e.issubset(enrichment.get("artifacts", {})): raise RuntimeError("enrichment lacks required completed tables")
    if not needed_c.issubset(cri.get("artifacts", {})): raise RuntimeError("CRI lacks required completed tables")
    LOGGER.info("Validated immutable input linkage; no upstream builders will be invoked")
    return ep, enrichment, cp, cri


def _source_fingerprints() -> dict[str, str]:
    """Fingerprint every downstream scientific source module."""
    sources = [Path(__file__), Path(__file__).with_name("temporal_robustness_autoencoder.py")]
    sources.extend(sorted(Path(__file__).with_name("temporal_synthesis").glob("*.py")))
    return {str(path.relative_to(Path(__file__).parent)): file_sha256(path) for path in sources}


def _numerical_environment(resolved_device: str | None = None) -> dict[str, Any]:
    import scipy
    import sklearn
    import torch
    return {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "torch": torch.__version__, "resolved_device": resolved_device,
            "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version()}


def synthesis_hash(enrichment_manifest: Path, cri_manifest: Path, config: MetricSynthesisConfig,
                   resolved_backend: str | None = None) -> str:
    value = canonical_json({"enrichment_sha256": file_sha256(enrichment_manifest), "cri_sha256": file_sha256(cri_manifest),
                            "scientific_config": config.to_dict(), "source_fingerprints": _source_fingerprints(),
                            "numerical_environment": _numerical_environment(resolved_backend)})
    return hashlib.sha256(value.encode()).hexdigest()[:20]


def _load(path: Path, manifest: Mapping[str, Any], names: Sequence[str] | None = None) -> dict[str, list[dict[str, Any]]]:
    selected = set(manifest["artifacts"]) if names is None else set(names)
    LOGGER.info("Loading %d required completed tables from %s", len(selected), path.parent)
    return {name: read_artifact(path.parent, manifest["artifacts"][name]) for name in selected
            if manifest["artifacts"][name].get("format") != "npz"}


def _key(row: Mapping[str, Any], names: Sequence[str]) -> tuple[Any, ...]: return tuple(row.get(name) for name in names)
def _finite(x: Any) -> bool: return x is not None and bool(np.isfinite(x))


def _exact_index(rows: Sequence[Mapping[str, Any]], names: Sequence[str], label: str) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result = {}
    for row in rows:
        key = _key(row, names)
        if key in result and dict(result[key]) != dict(row): raise ValueError(f"duplicate nonidentical {label} mapping: {key}")
        result[key] = row
    return result


def build_metric_vectors(enrichment: Mapping[str, Sequence[Mapping[str, Any]]], cri: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Exact member join and reference-oriented utilities, with a TCAV audit."""
    members = cri["cri_member_utilities"]
    LOGGER.info("Constructing reference-oriented metric vectors from %d CRI member rows", len(members))
    # Match the immutable CRI estimand, not the notebook's additional views.
    selected_factor = [r for r in enrichment["headline_factor_metrics"] if r.get("matching_view") == "intersection" and r.get("rule_source") == "semantic" and r.get("target_role") == "primary"]
    selected_tcav = [r for r in enrichment["tcav_significance"] if r.get("matching_view") == "intersection" and r.get("rule_source") == "semantic" and r.get("target_role") == "primary"]
    factor = _exact_index(selected_factor, MEMBER_KEYS + ("cohort_view", "temporal_distance"), "factor")
    tcav = _exact_index(selected_tcav, MEMBER_KEYS + ("cohort_view", "temporal_distance"), "TCAV")
    reference_tcav: dict[tuple[Any, ...], float] = {}
    for key, row in tcav.items():
        if key[-2:] == ("all_comer", 0) and _finite(row.get("tcav")): reference_tcav[key[:len(MEMBER_KEYS)]] = float(row["tcav"])
    vectors, audit = [], []
    for member in members:
        key = _key(member, MEMBER_KEYS + ("cohort_view", "temporal_distance"))
        row = factor.get(key)
        if row is None: raise ValueError(f"missing exact factor mapping for CRI member: {key}")
        cosine = row.get("feature_association_cosine")
        association = None if not _finite(cosine) else float(np.clip((float(cosine) + 1.) / 2., 0., 1.))
        trow, ref = tcav.get(key), reference_tcav.get(key[:len(MEMBER_KEYS)])
        raw = None if trow is None else trow.get("tcav")
        utility = None if not (_finite(raw) and ref is not None) else float(np.clip(1 - abs(float(raw) - ref), 0., 1.))
        vectors.append({**dict(member), "u_feature_association": association, "u_tcav": utility,
                        "feature_association_valid": row.get("feature_association_valid"),
                        "tcav_raw": raw, "tcav_reference": ref,
                        "tcav_valid": None if trow is None else trow.get("tcav_valid"),
                        "tcav_p_value": None if trow is None else trow.get("p_value"),
                        "tcav_q_value": None if trow is None else trow.get("q_value")})
        audit.append({**{n: member.get(n) for n in MEMBER_KEYS + ("cohort_view", "temporal_distance")},
                      "tcav_present": trow is not None, "tcav_finite": _finite(raw), "tcav_reference_present": ref is not None,
                      "tcav_valid": None if trow is None else trow.get("tcav_valid")})
    return vectors, audit


def _profile_metrics(profile: str) -> tuple[str, ...]: return profile_metrics(profile)


def system_concept_features(vectors: Sequence[Mapping[str, Any]], profile: str) -> list[dict[str, Any]]:
    metrics = _profile_metrics(profile); output = []
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in vectors:
        if profile == "p50_tcav_extended" and float(row["activation_target"]) != .5: continue
        groups.setdefault(_key(row, SYSTEM_KEYS), []).append(row)
    for key, rows in sorted(groups.items(), key=lambda x: str(x[0])):
        family: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
        for row in rows: family.setdefault(_key(row, FAMILY_KEYS), []).append(row)
        complete = [items for items in family.values() if all(all(_finite(item.get(m)) for m in metrics) for item in items)]
        values = {m: (float(np.median([np.median([float(item[m]) for item in items]) for items in complete])) if complete else None) for m in metrics}
        output.append({**{name: value for name, value in zip(SYSTEM_KEYS, key)}, "profile": profile,
                       "family_denominator": len(family), "complete_family_count": len(complete),
                       "concept_coverage": len(complete) / len(family) if family else 0., **values})
    return output


def build_early_warning_rows(performance: Sequence[Mapping[str, Any]], features: Sequence[Mapping[str, Any]], profile: str) -> list[dict[str, Any]]:
    fi = _exact_index(features, SYSTEM_KEYS, "concept system")
    # performance identity intentionally includes variant: original and balanced are separate systems.
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in performance:
        variant = row.get("variant", "original")
        groups.setdefault((variant,) + _key(row, ("reference_year", "patient_split_seed", "cohort_view")), []).append(row)
    rows = []
    for (variant, ref, seed, cohort), values in groups.items():
        by_d = {int(x["temporal_distance"]): x for x in values if _finite(x.get("death_f1"))}
        for d, current in by_d.items():
            previous, future = by_d.get(d - 1), by_d.get(d + 1)
            if previous is None or future is None: continue
            for activation in sorted({x["activation_target"] for x in features if x["reference_year"] == ref and x["patient_split_seed"] == seed and x["cohort_view"] == cohort and x["temporal_distance"] == d}):
                now, prior = fi.get((ref, seed, cohort, activation, d)), fi.get((ref, seed, cohort, activation, d - 1))
                if now is None or prior is None: continue
                record = {"variant": variant, "profile": profile, "reference_year": ref, "patient_split_seed": seed, "cohort_view": cohort, "activation_target": activation, "temporal_distance": d, "target_year": current.get("test_year"), "death_f1_previous": previous["death_f1"], "death_f1_current": current["death_f1"], "death_f1_next": future["death_f1"], "death_f1_degradation": float(current["death_f1"] - future["death_f1"]), "previous_degradation": float(previous["death_f1"] - current["death_f1"]), "concept_coverage": now["concept_coverage"]}
                for metric in _profile_metrics(profile):
                    record[metric] = now.get(metric); record[f"delta_{metric}"] = None if not (_finite(now.get(metric)) and _finite(prior.get(metric))) else float(now[metric] - prior[metric])
                cri_metrics = ("u_f2", "u_jaccard", "u_prevalence", "u_activation")
                record["current_cri"] = None if not all(_finite(now.get(x)) for x in cri_metrics) else float(np.mean([now[x] for x in cri_metrics]))
                record["delta_current_cri"] = None if not all(_finite(now.get(x)) and _finite(prior.get(x)) for x in cri_metrics) else float(record["current_cri"] - np.mean([prior[x] for x in cri_metrics]))
                rows.append(record)
    return rows


def _ridge_oof(rows: Sequence[Mapping[str, Any]], profile: str, *, forward: bool = False,
               maximal_history: bool = False,
               ridge_alphas: Sequence[float] = (1e-4, 1e-3, 1e-2, .1, 1., 10., 100.)) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    variants = sorted({str(row.get("variant", "original")) for row in rows})
    if len(variants) > 1:
        predictions, folds, summary = [], [], []
        for variant_name in variants:
            result = _ridge_oof([row for row in rows if str(row.get("variant", "original")) == variant_name], profile,
                                forward=forward, maximal_history=maximal_history, ridge_alphas=ridge_alphas)
            predictions.extend(result[0]); folds.extend(result[1]); summary.extend(result[2])
        return predictions, folds, summary
    variant = variants[0] if variants else "original"
    metrics = _profile_metrics(profile); history = ("death_f1_current", "previous_degradation", "temporal_distance")
    concepts = metrics + tuple("delta_" + x for x in metrics) + ("concept_coverage",)
    specs = ({"performance_history_maximal_coverage": history} if maximal_history else
             {"performance_history": history, "concept_robustness": concepts, "history_plus_concepts": history + concepts, "current_cri": ("current_cri", "delta_current_cri")})
    def reference_weights(values: Sequence[Mapping[str, Any]]) -> np.ndarray:
        counts: dict[int, int] = {}
        for value in values: counts[int(value["reference_year"])] = counts.get(int(value["reference_year"]), 0) + 1
        return np.asarray([1. / counts[int(value["reference_year"])] for value in values])
    required = history if maximal_history else history + concepts + ("current_cri", "delta_current_cri")
    complete = [dict(r) for r in rows if all(_finite(r.get(x)) for x in required)]
    predictions: list[dict[str, Any]]=[]; folds=[]
    refs = sorted({int(r["reference_year"]) for r in complete})
    evaluation = "forward" if forward else "LOYO"
    population = "maximal_history" if maximal_history else "common_complete_case"
    LOGGER.info("Ridge %s evaluation for %s/%s (%s): %d rows, %d reference groups", evaluation, profile, variant, population, len(complete), len(refs))
    for held in _progress(refs, total=len(refs), description=f"Ridge {profile} {evaluation}"):
        LOGGER.info("Ridge %s %s: holding out reference year %s", profile, evaluation, held)
        test = [r for r in complete if int(r["reference_year"]) == held]
        for model, names in specs.items():
            train = [r for r in complete if int(r["reference_year"]) != held]
            cached_fits: dict[int | None, Any] = {}
            if forward:
                # A row-specific forward train set is recorded and fit separately below.
                train = train
            for testrow in test:
                selected_train = [r for r in train if not forward or (int(r["reference_year"]) < held and int(r["target_year"]) < int(testrow["target_year"]))]
                if len(selected_train) < 3 or len({r["reference_year"] for r in selected_train}) < 2:
                    predictions.append({**testrow, "evaluation": "forward" if forward else "loyo", "model": model, "held_out_reference_year": held, "prediction": None, "noninformative_reason": "insufficient_training_groups"}); continue
                cache_key = int(testrow["target_year"]) if forward else None
                if cache_key in cached_fits:
                    scaler, reg, alpha = cached_fits[cache_key]
                    prediction=float(reg.predict(scaler.transform([[testrow[x] for x in names]]))[0])
                    predictions.append({**testrow, "evaluation":"forward" if forward else "loyo", "model":model,"held_out_reference_year":held,"prediction":prediction,"absolute_error":abs(testrow["death_f1_degradation"]-prediction),"squared_error":(testrow["death_f1_degradation"]-prediction)**2,"selected_alpha":alpha})
                    continue
                # Inner LOYO alpha selection and scaling happen strictly within selected outer training rows.
                candidates=[]
                inner_refs=sorted({int(r["reference_year"]) for r in selected_train})
                for alpha in ridge_alphas:
                    errors=[]
                    for inner in inner_refs:
                        a=[r for r in selected_train if int(r["reference_year"]) != inner]; b=[r for r in selected_train if int(r["reference_year"]) == inner]
                        if not a or not b: continue
                        scaler=StandardScaler().fit([[r[x] for x in names] for r in a]); reg=Ridge(alpha=alpha).fit(scaler.transform([[r[x] for x in names] for r in a]), [r["death_f1_degradation"] for r in a], sample_weight=reference_weights(a))
                        errors.append(float(np.mean(np.abs(reg.predict(scaler.transform([[r[x] for x in names] for r in b])) - np.asarray([r["death_f1_degradation"] for r in b])))))
                    candidates.append((float(np.mean(errors)) if errors else np.inf, -alpha, alpha))
                alpha=min(candidates)[2]
                scaler=StandardScaler().fit([[r[x] for x in names] for r in selected_train]); reg=Ridge(alpha=alpha).fit(scaler.transform([[r[x] for x in names] for r in selected_train]), [r["death_f1_degradation"] for r in selected_train], sample_weight=reference_weights(selected_train))
                prediction=float(reg.predict(scaler.transform([[testrow[x] for x in names]]))[0])
                cached_fits[cache_key] = (scaler, reg, alpha)
                predictions.append({**testrow, "evaluation":"forward" if forward else "loyo", "model":model,"held_out_reference_year":held,"prediction":prediction,"absolute_error":abs(testrow["death_f1_degradation"]-prediction),"squared_error":(testrow["death_f1_degradation"]-prediction)**2,"selected_alpha":alpha})
            folds.append({"evaluation":"forward" if forward else "loyo", "model":model,"variant":variant,"analysis_population":population,"held_out_reference_year":held,"entire_reference_held_out":all(int(r["reference_year"]) != held for r in train),"feature_names":list(names)})
        # baselines share exactly the complete set.
        for row in test:
            baseline_train = [r for r in complete if r["reference_year"] != held and (not forward or (int(r["reference_year"]) < held and int(r["target_year"]) < int(row["target_year"])))]
            training_mean = None if not baseline_train else float(np.mean([np.mean([x["death_f1_degradation"] for x in baseline_train if x["reference_year"] == ref]) for ref in sorted({x["reference_year"] for x in baseline_train})]))
            baseline_names = (("zero_degradation_maximal_coverage",0.),("training_mean_maximal_coverage",training_mean)) if maximal_history else (("zero_degradation",0.),("training_mean",training_mean))
            for name, pred in baseline_names:
                predictions.append({**row,"evaluation":"forward" if forward else "loyo","model":name,"held_out_reference_year":held,"prediction":pred,
                                    "absolute_error":None if pred is None else abs(row["death_f1_degradation"]-pred),
                                    "squared_error":None if pred is None else (row["death_f1_degradation"]-pred)**2,
                                    "noninformative_reason":None if pred is not None else "insufficient_forward_training_groups"})
    grouped={}
    for row in predictions:
        if row.get("prediction") is not None: grouped.setdefault((row["evaluation"],row["model"]),[]).append(row)
    summary=[]
    for (evaluation,model), values in sorted(grouped.items()):
        per=[np.mean([x["absolute_error"] for x in values if x["reference_year"]==ref]) for ref in sorted({x["reference_year"] for x in values})]
        summary.append({"profile":profile,"variant":variant,"analysis_population":population,"evaluation":evaluation,"model":model,"oof_row_count":len(values),"oof_reference_count":len(per),"mae_macro_reference":float(np.mean(per)),"mae_micro":float(np.mean([x["absolute_error"] for x in values])),"rmse_micro":float(np.sqrt(np.mean([x["squared_error"] for x in values]))),"zero_target_fraction":float(np.mean([x["death_f1_degradation"]==0 for x in values])),"target_variance":float(np.var([x["death_f1_degradation"] for x in values]))})
    return predictions, folds, summary


def _reference_weights(values: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Give every reference year equal aggregate fitting weight."""
    counts: dict[int, int] = {}
    for value in values: counts[int(value["reference_year"])] = counts.get(int(value["reference_year"]), 0) + 1
    return np.asarray([1. / counts[int(value["reference_year"])] for value in values])


def _latent_design(train_rows: Sequence[Mapping[str, Any]], evaluation_rows: Sequence[Mapping[str, Any]],
                   profile: str, method: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit PCA/FA only on training concept states and produce current/delta scores.

    The transform sees no Death-F1 values.  Current and prior metric states are
    both included in the training fit so a delta in latent space is a fair
    representation of the corresponding raw-utility delta.
    """
    from sklearn.decomposition import FactorAnalysis, PCA
    from sklearn.preprocessing import StandardScaler
    names = _profile_metrics(profile)
    if method not in {"pca2", "fa2"}: raise ValueError(f"unknown latent representation: {method}")
    def current(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray([[float(row[name]) for name in names] for row in rows], dtype=float)
    def prior(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray([[float(row[name]) - float(row[f"delta_{name}"]) for name in names] for row in rows], dtype=float)
    train_current, train_prior = current(train_rows), prior(train_rows)
    utility_scaler = StandardScaler().fit(np.vstack((train_current, train_prior)))
    reducer = (PCA(n_components=2, random_state=0) if method == "pca2" else FactorAnalysis(n_components=2, random_state=0))
    reducer.fit(utility_scaler.transform(np.vstack((train_current, train_prior))))
    def design(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        now = reducer.transform(utility_scaler.transform(current(rows)))
        before = reducer.transform(utility_scaler.transform(prior(rows)))
        history = np.asarray([[float(row["death_f1_current"]), float(row["previous_degradation"]), float(row["temporal_distance"]), float(row["concept_coverage"])] for row in rows])
        return np.hstack((history, now, now - before))
    label = "PC" if method == "pca2" else "FA"
    return design(train_rows), design(evaluation_rows), ["death_f1_current", "previous_degradation", "temporal_distance", "concept_coverage", f"{label}1_current", f"{label}2_current", f"delta_{label}1", f"delta_{label}2"]


def _latent_ridge_oof(rows: Sequence[Mapping[str, Any]], profile: str, config: MetricSynthesisConfig, *,
                      method: str, forward: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Nested, grouped Ridge evaluation for a fold-fitted two-dimensional representation."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    variants = sorted({str(row.get("variant", "original")) for row in rows})
    if len(variants) > 1:
        predictions, folds, summary = [], [], []
        for variant_name in variants:
            result = _latent_ridge_oof([row for row in rows if str(row.get("variant", "original")) == variant_name],
                                       profile, config, method=method, forward=forward)
            predictions.extend(result[0]); folds.extend(result[1]); summary.extend(result[2])
        return predictions, folds, summary
    variant = variants[0] if variants else "original"
    metrics = _profile_metrics(profile)
    required = ("death_f1_current", "previous_degradation", "temporal_distance", "concept_coverage") + metrics + tuple(f"delta_{metric}" for metric in metrics)
    complete = [dict(row) for row in rows if all(_finite(row.get(name)) for name in required)]
    refs = sorted({int(row["reference_year"]) for row in complete})
    evaluation = "forward" if forward else "loyo"
    model = f"history_plus_{method}"
    label = "PC" if method == "pca2" else "FA"
    feature_names = ["death_f1_current", "previous_degradation", "temporal_distance", "concept_coverage", f"{label}1_current", f"{label}2_current", f"delta_{label}1", f"delta_{label}2"]
    LOGGER.info("Fold-fitted %s Ridge %s evaluation for %s/%s: %d complete rows, %d reference groups", method, evaluation, profile, variant, len(complete), len(refs))
    predictions, folds = [], []
    for held in _progress(refs, total=len(refs), description=f"Ridge {model} {evaluation}"):
        test = [row for row in complete if int(row["reference_year"]) == held]
        outer_train = [row for row in complete if int(row["reference_year"]) != held]
        test_blocks: dict[int | None, list[Mapping[str, Any]]] = {}
        for row in test: test_blocks.setdefault(int(row["target_year"]) if forward else None, []).append(row)
        for target_year, test_block in test_blocks.items():
            train = [row for row in outer_train if not forward or (int(row["reference_year"]) < held and int(row["target_year"]) < int(target_year))]
            if len(train) < 3 or len({int(row["reference_year"]) for row in train}) < 2:
                predictions.extend({**test_row, "evaluation": evaluation, "model": model, "held_out_reference_year": held,
                                    "prediction": None, "noninformative_reason": "insufficient_training_groups"} for test_row in test_block)
                continue
            candidates = []
            for alpha in config.ridge_alphas:
                errors = []
                for inner in sorted({int(row["reference_year"]) for row in train}):
                    inner_train = [row for row in train if int(row["reference_year"]) != inner]
                    inner_test = [row for row in train if int(row["reference_year"]) == inner]
                    if not inner_train or not inner_test: continue
                    x_train, x_test, _ = _latent_design(inner_train, inner_test, profile, method)
                    scaler = StandardScaler().fit(x_train)
                    regression = Ridge(alpha=alpha).fit(scaler.transform(x_train), [row["death_f1_degradation"] for row in inner_train], sample_weight=_reference_weights(inner_train))
                    errors.append(float(np.mean(np.abs(regression.predict(scaler.transform(x_test)) - np.asarray([row["death_f1_degradation"] for row in inner_test])))))
                candidates.append((float(np.mean(errors)) if errors else np.inf, -float(alpha), float(alpha)))
            alpha = min(candidates)[2]
            x_train, x_test, _ = _latent_design(train, test_block, profile, method)
            scaler = StandardScaler().fit(x_train)
            regression = Ridge(alpha=alpha).fit(scaler.transform(x_train), [row["death_f1_degradation"] for row in train], sample_weight=_reference_weights(train))
            for test_row, prediction in zip(test_block, regression.predict(scaler.transform(x_test))):
                prediction = float(prediction)
                predictions.append({**test_row, "evaluation": evaluation, "model": model, "held_out_reference_year": held,
                                    "prediction": prediction, "absolute_error": abs(test_row["death_f1_degradation"] - prediction),
                                    "squared_error": (test_row["death_f1_degradation"] - prediction) ** 2, "selected_alpha": alpha,
                                    "representation_fit_training_references_only": True})
        folds.append({"evaluation": evaluation, "model": model, "variant": variant, "analysis_population": "common_complete_case", "held_out_reference_year": held,
                      "entire_reference_held_out": all(int(row["reference_year"]) != held for row in outer_train),
                      "representation_fit_training_references_only": True, "latent_dimensions": 2,
                      "feature_names": feature_names})
    values = [row for row in predictions if row.get("prediction") is not None]
    per_reference = [np.mean([row["absolute_error"] for row in values if int(row["reference_year"]) == ref]) for ref in sorted({int(row["reference_year"]) for row in values})]
    summary = [] if not values else [{"profile": profile, "variant": variant, "analysis_population": "common_complete_case", "evaluation": evaluation, "model": model, "oof_row_count": len(values),
                                      "oof_reference_count": len(per_reference), "mae_macro_reference": float(np.mean(per_reference)),
                                      "mae_micro": float(np.mean([row["absolute_error"] for row in values])),
                                      "rmse_micro": float(np.sqrt(np.mean([row["squared_error"] for row in values]))),
                                      "zero_target_fraction": float(np.mean([row["death_f1_degradation"] == 0 for row in values])),
                                      "target_variance": float(np.var([row["death_f1_degradation"] for row in values]))}]
    return predictions, folds, summary


def _split_noise_threshold(rows: Sequence[Mapping[str, Any]], quantile: float) -> tuple[float, int, int]:
    """Estimate a minimum detectable F1 change from split-seed replication.

    Concept rows duplicate a performance measurement over activation targets, so
    the noise estimator first reduces them to one degradation per split seed.
    Residuals around a fixed reference/cohort/distance median represent ordinary
    refit/split variation, not temporal change itself.
    """
    by_seed: dict[tuple[Any, ...], float] = {}
    for row in rows:
        key = _key(row, ("variant", "reference_year", "patient_split_seed", "cohort_view", "temporal_distance", "target_year"))
        value = float(row["death_f1_degradation"])
        if key in by_seed and not np.isclose(by_seed[key], value): raise ValueError(f"nonidentical degradation duplicated across concept rows: {key}")
        by_seed[key] = value
    groups: dict[tuple[Any, ...], list[float]] = {}
    for key, value in by_seed.items():
        groups.setdefault((key[0], key[1], key[3], key[4], key[5]), []).append(value)
    residuals = []
    replicate_groups = 0
    for values in groups.values():
        if len(values) < 2: continue
        replicate_groups += 1
        centre = float(np.median(values))
        residuals.extend(abs(value - centre) for value in values)
    if not residuals: raise RuntimeError("cannot estimate split-noise threshold: no performance cells have at least two split seeds")
    return float(np.quantile(residuals, quantile)), replicate_groups, len(residuals)


def _event_design(train_rows: Sequence[Mapping[str, Any]], evaluation_rows: Sequence[Mapping[str, Any]],
                  profile: str, model: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build outcome-free event-model features; PCA/FA is fitted within train."""
    history = ("death_f1_current", "previous_degradation", "temporal_distance")
    if model == "performance_history":
        return (np.asarray([[float(row[name]) for name in history] for row in train_rows]),
                np.asarray([[float(row[name]) for name in history] for row in evaluation_rows]), list(history))
    if model == "history_plus_concepts":
        names = _profile_metrics(profile) + tuple(f"delta_{name}" for name in _profile_metrics(profile)) + ("concept_coverage",)
        all_names = history + names
        return (np.asarray([[float(row[name]) for name in all_names] for row in train_rows]),
                np.asarray([[float(row[name]) for name in all_names] for row in evaluation_rows]), list(all_names))
    if model == "history_plus_pca2": return _latent_design(train_rows, evaluation_rows, profile, "pca2")
    if model == "history_plus_fa2": return _latent_design(train_rows, evaluation_rows, profile, "fa2")
    raise ValueError(f"unknown material-degradation event model: {model}")


def _event_summary(predictions: Sequence[Mapping[str, Any]], profile: str) -> list[dict[str, Any]]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    output = []
    identity = ("reference_year", "patient_split_seed", "cohort_view", "activation_target", "temporal_distance", "target_year")
    pairs = sorted({(str(row.get("variant", "original")), str(row["evaluation"])) for row in predictions})
    for variant, evaluation in pairs:
        subset = [row for row in predictions if str(row.get("variant", "original")) == variant and row["evaluation"] == evaluation]
        models = sorted({str(row["model"]) for row in subset})
        key_sets = [{_key(row, identity) for row in subset if row["model"] == model and row.get("probability") is not None} for model in models]
        common_keys = set.intersection(*key_sets) if key_sets else set()
        for model in models:
            values = [row for row in subset if row["model"] == model and row.get("probability") is not None and _key(row, identity) in common_keys]
            if not values: continue
            per_reference = [np.mean([row["brier_score"] for row in values if int(row["reference_year"]) == ref]) for ref in sorted({int(row["reference_year"]) for row in values})]
            target, probability = np.asarray([row["material_degradation"] for row in values]), np.asarray([row["probability"] for row in values])
            output.append({"profile": profile, "variant": variant, "analysis_population": "common_complete_case_and_estimable_models", "evaluation": evaluation, "model": model, "oof_row_count": len(values), "oof_reference_count": len(per_reference),
                           "brier_macro_reference": float(np.mean(per_reference)), "brier_micro": float(np.mean([row["brier_score"] for row in values])),
                           "log_loss_micro": float(np.mean([row["log_loss"] for row in values])), "event_fraction": float(target.mean()),
                           "auroc_micro": None if len(np.unique(target)) < 2 else float(roc_auc_score(target, probability)),
                           "average_precision_micro": None if len(np.unique(target)) < 2 else float(average_precision_score(target, probability))})
    return output


def _material_degradation_oof(rows: Sequence[Mapping[str, Any]], profile: str, config: MetricSynthesisConfig, *,
                              forward: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Nested grouped prediction of decline beyond train-derived split noise."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    variants = sorted({str(row.get("variant", "original")) for row in rows})
    if len(variants) > 1:
        predictions, folds, thresholds, summary = [], [], [], []
        for variant_name in variants:
            result = _material_degradation_oof([row for row in rows if str(row.get("variant", "original")) == variant_name],
                                               profile, config, forward=forward)
            predictions.extend(result[0]); folds.extend(result[1]); thresholds.extend(result[2]); summary.extend(result[3])
        return predictions, folds, thresholds, summary
    variant = variants[0] if variants else "original"
    metrics = _profile_metrics(profile)
    required = ("death_f1_current", "previous_degradation", "temporal_distance", "concept_coverage") + metrics + tuple(f"delta_{metric}" for metric in metrics)
    complete = [dict(row) for row in rows if all(_finite(row.get(name)) for name in required)]
    refs = sorted({int(row["reference_year"]) for row in complete})
    models = ("performance_history", "history_plus_concepts", "history_plus_pca2", "history_plus_fa2")
    cs = (.01, .1, 1., 10., 100.)
    evaluation = "forward" if forward else "loyo"
    LOGGER.info("Material-degradation %s event evaluation for %s/%s: %d complete rows, %d reference groups, noise q=%.2f", evaluation, profile, variant, len(complete), len(refs), config.material_degradation_noise_quantile)
    predictions, folds, thresholds = [], [], []
    for held in _progress(refs, total=len(refs), description=f"Material degradation {profile} {evaluation}"):
        test = [row for row in complete if int(row["reference_year"]) == held]
        outer_train = [row for row in complete if int(row["reference_year"]) != held]
        test_blocks: dict[int | None, list[Mapping[str, Any]]] = {}
        for row in test: test_blocks.setdefault(int(row["target_year"]) if forward else None, []).append(row)
        for target_year, test_block in test_blocks.items():
            train = [row for row in outer_train if not forward or (int(row["reference_year"]) < held and int(row["target_year"]) < int(target_year))]
            if len(train) < 3 or len({int(row["reference_year"]) for row in train}) < 2:
                for model in ("no_material_degradation",) + models:
                    predictions.extend({**test_row, "evaluation": evaluation, "model": model, "held_out_reference_year": held, "probability": None,
                                        "noninformative_reason": "insufficient_training_groups"} for test_row in test_block)
                continue
            try:
                threshold, group_count, residual_count = _split_noise_threshold(train, config.material_degradation_noise_quantile)
            except RuntimeError as error:
                thresholds.append({"profile": profile, "variant": variant, "evaluation": evaluation, "held_out_reference_year": held, "target_year": target_year,
                                   "noise_quantile": config.material_degradation_noise_quantile, "minimum_detectable_degradation": None,
                                   "training_replicate_group_count": 0, "training_split_residual_count": 0,
                                   "training_reference_years": sorted({int(row["reference_year"]) for row in train}), "status": "not_estimable", "reason": str(error)})
                for model in ("no_material_degradation",) + models:
                    predictions.extend({**test_row, "evaluation": evaluation, "model": model, "held_out_reference_year": held, "probability": None,
                                        "noninformative_reason": "split_noise_threshold_not_estimable"} for test_row in test_block)
                continue
            thresholds.append({"profile": profile, "variant": variant, "evaluation": evaluation, "held_out_reference_year": held, "target_year": target_year,
                               "noise_quantile": config.material_degradation_noise_quantile, "minimum_detectable_degradation": threshold,
                               "training_replicate_group_count": group_count, "training_split_residual_count": residual_count,
                               "training_reference_years": sorted({int(row["reference_year"]) for row in train})})
            y_train = np.asarray([float(row["death_f1_degradation"]) > threshold for row in train], dtype=int)
            y_test = np.asarray([float(row["death_f1_degradation"]) > threshold for row in test_block], dtype=int)
            for model in ("no_material_degradation",) + models:
                if model == "no_material_degradation":
                    probability = np.zeros(len(test_block))
                    selected_c = None
                elif len(np.unique(y_train)) < 2:
                    probability = None; selected_c = None
                else:
                    candidates = []
                    for c in cs:
                        errors = []
                        for inner in sorted({int(row["reference_year"]) for row in train}):
                            inner_train = [row for row in train if int(row["reference_year"]) != inner]
                            inner_test = [row for row in train if int(row["reference_year"]) == inner]
                            if not inner_train or not inner_test: continue
                            try:
                                inner_threshold, _, _ = _split_noise_threshold(inner_train, config.material_degradation_noise_quantile)
                            except RuntimeError:
                                continue
                            inner_y_train = np.asarray([float(row["death_f1_degradation"]) > inner_threshold for row in inner_train], dtype=int)
                            if len(np.unique(inner_y_train)) < 2: continue
                            inner_y_test = np.asarray([float(row["death_f1_degradation"]) > inner_threshold for row in inner_test], dtype=int)
                            x_train, x_test, _ = _event_design(inner_train, inner_test, profile, model)
                            scaler = StandardScaler().fit(x_train)
                            classifier = LogisticRegression(C=c, max_iter=1000, solver="lbfgs", random_state=config.seed).fit(scaler.transform(x_train), inner_y_train, sample_weight=_reference_weights(inner_train))
                            probability_inner = classifier.predict_proba(scaler.transform(x_test))[:, 1]
                            errors.append(float(np.mean((probability_inner - inner_y_test) ** 2)))
                        candidates.append((float(np.mean(errors)) if errors else np.inf, -c, c))
                    selected_c = min(candidates)[2]
                    x_train, x_test, feature_names = _event_design(train, test_block, profile, model)
                    scaler = StandardScaler().fit(x_train)
                    classifier = LogisticRegression(C=selected_c, max_iter=1000, solver="lbfgs", random_state=config.seed).fit(scaler.transform(x_train), y_train, sample_weight=_reference_weights(train))
                    probability = classifier.predict_proba(scaler.transform(x_test))[:, 1]
                if probability is None:
                    predictions.extend({**test_row, "evaluation": evaluation, "model": model, "held_out_reference_year": held,
                                        "minimum_detectable_degradation": threshold, "probability": None, "noninformative_reason": "single_class_training_target"} for test_row in test_block)
                    continue
                for test_row, event, probability_value in zip(test_block, y_test, probability):
                    probability_value = float(probability_value)
                    log_loss = float(-(event * np.log(max(probability_value, 1e-15)) + (1 - event) * np.log(max(1 - probability_value, 1e-15))))
                    predictions.append({**test_row, "evaluation": evaluation, "model": model, "held_out_reference_year": held,
                                        "minimum_detectable_degradation": threshold, "material_degradation": int(event), "probability": probability_value,
                                        "brier_score": float((probability_value - event) ** 2), "log_loss": log_loss,
                                        "selected_c": selected_c, "representation_fit_training_references_only": model in {"history_plus_pca2", "history_plus_fa2"}})
            folds.append({"profile": profile, "variant": variant, "analysis_population": "common_complete_case", "evaluation": evaluation, "held_out_reference_year": held, "target_year": target_year,
                          "entire_reference_held_out": all(int(row["reference_year"]) != held for row in outer_train),
                          "noise_threshold_fit_training_references_only": True, "representation_fit_training_references_only": True,
                          "models": list(models)})
    return predictions, folds, thresholds, _event_summary(predictions, profile)


def dimensionality(features: Sequence[Mapping[str, Any]], profile: str, config: MetricSynthesisConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return _analyze_dimensionality(features, profile, config)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Atomically serialize validated DAE weights without pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(handle); temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as output: np.savez_compressed(output, **arrays)
        with np.load(temporary, allow_pickle=False) as bundle:
            if set(bundle.files) != set(arrays): raise RuntimeError("DAE checkpoint validation changed arrays")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return descriptor_for_file(path, relative_to=path.parent)


def dae_member_vectors(vectors: Sequence[Mapping[str, Any]], profile: str) -> list[dict[str, Any]]:
    """Complete member/factor vectors, excluding the trivial reference distance."""
    names = _profile_metrics(profile)
    return [dict(row) for row in vectors
            if int(row["temporal_distance"]) != 0
            and (profile != "p50_tcav_extended" or float(row["activation_target"]) == .5)
            and all(_finite(row.get(name)) for name in names)]


def _dae_candidates() -> list[dict[str, Any]]:
    """Small, prespecified grid; decoder/scale combinations are coherent."""
    result = []
    for dropout in (0., .02, .05, .10):
        for noise in (0., .01, .02):
            for hidden in (4, 8, 16):
                for learning_rate in (3e-4, 1e-3):
                    for transform, activation in (("raw_sigmoid", "sigmoid"), ("standard_linear", "linear"), ("logit_linear", "linear")):
                        result.append({"dropout_probability": dropout, "noise_std": noise, "hidden_dimensions": hidden,
                                       "learning_rate": learning_rate, "transform": transform, "output_activation": activation})
    return result


def _weighted_loss_weights(matrix: np.ndarray) -> tuple[float, ...]:
    """Equalize raw metric variance without observing any performance outcome."""
    variance = np.maximum(np.var(matrix, axis=0), 1e-6)
    weights = 1. / variance
    return tuple((weights / weights.mean()).astype(float))


def _run_dae_split(train_raw: np.ndarray, validation_raw: np.ndarray | None, candidate: Mapping[str, Any], *,
                   dimensions: int, epochs: int, patience: int, seed: int, device: str, progress: bool) -> tuple[dict[str, Any], Any, Any]:
    """Fit preprocessing and DAE solely on a split's training references."""
    from temporal_robustness_autoencoder import (DenoisingAutoencoderConfig, fit_denoising_autoencoder,
                                                  fit_utility_preprocessor)
    preprocessor = fit_utility_preprocessor(train_raw, str(candidate["transform"]))
    train = preprocessor.transform(train_raw)
    validation = None if validation_raw is None else preprocessor.transform(validation_raw)
    config = DenoisingAutoencoderConfig(latent_dimensions=dimensions, hidden_dimensions=int(candidate["hidden_dimensions"]),
                                        dropout_probability=float(candidate["dropout_probability"]), noise_std=float(candidate["noise_std"]),
                                        epochs=epochs, early_stopping_patience=patience, learning_rate=float(candidate["learning_rate"]),
                                        output_activation=str(candidate["output_activation"]), metric_loss_weights=_weighted_loss_weights(train),
                                        seed=seed, device=device)
    fitted = fit_denoising_autoencoder(train, config, validation, progress=progress,
                                       search_mode=validation is not None)
    return fitted, preprocessor, config


def _run_dae_search_job(job: Any, *, matrix: np.ndarray, reference_values: np.ndarray,
                        outer_training_references: tuple[int, ...], candidate: Mapping[str, Any],
                        dimensions: int, epochs: int, patience: int, device: str,
                        profile: str) -> dict[str, Any]:
    """Pickle-safe execution unit for CPU processes and CUDA threads."""
    train_mask = np.isin(reference_values, outer_training_references) & (reference_values != job.inner_year)
    validation_mask = reference_values == job.inner_year
    fitted, preprocessor, _ = _run_dae_split(
        matrix[train_mask], matrix[validation_mask], candidate, dimensions=dimensions,
        epochs=epochs, patience=patience, seed=job.seed, device=device, progress=False,
    )
    reconstructed = preprocessor.inverse_transform(fitted["validation_reconstruction"])
    error = matrix[validation_mask] - reconstructed
    return {"profile": profile, "outer_held_out_reference_year": job.outer_year,
            "inner_held_out_reference_year": job.inner_year, "candidate_index": job.candidate_index,
            "validation_seed": job.validation_seed, "stable_job_seed": job.seed,
            "row_count": int(validation_mask.sum()), "validation_mae": float(np.mean(abs(error))),
            "validation_mse": float(np.mean(error ** 2)), "best_epoch": fitted["best_epoch"], **candidate}


def _model_outputs(fitted: Mapping[str, Any], transformed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode and reconstruct a transformed matrix without refitting anything."""
    import torch
    model = fitted["model"]
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        latent, reconstruction = model(torch.as_tensor(transformed, dtype=torch.float32, device=device))
    return latent.cpu().numpy(), reconstruction.cpu().numpy()


def _reconstruction_rows(*, profile: str, method: str, seed: int, held_reference_year: int,
                         names: Sequence[str], truth: np.ndarray, reconstructed: np.ndarray,
                         dimensions: int, best_epoch: int | None = None) -> list[dict[str, Any]]:
    error = truth - reconstructed
    base = {"profile": profile, "method": method, "seed": seed, "held_out_reference_year": held_reference_year,
            "latent_dimensions": dimensions, "row_count": len(truth), "best_epoch": best_epoch}
    rows = [{**base, "metric": "overall", "mae": float(np.mean(abs(error))), "mse": float(np.mean(error ** 2))}]
    rows.extend({**base, "metric": name, "mae": float(np.mean(abs(error[:, index]))), "mse": float(np.mean(error[:, index] ** 2))}
                for index, name in enumerate(names))
    return rows


def _aggregate_latent(rows: Sequence[Mapping[str, Any]], dimensions: int, keys: Sequence[str], label: str, profile: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows: groups.setdefault(_key(row, keys), []).append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        record = {"profile": profile, "aggregation": label, **dict(zip(keys, key)), "row_count": len(group)}
        for index in range(dimensions):
            values = [float(row[f"latent_{index + 1}"]) for row in group]
            record.update({f"latent_{index + 1}_median": float(np.median(values)), f"latent_{index + 1}_q25": float(np.quantile(values, .25)), f"latent_{index + 1}_q75": float(np.quantile(values, .75))})
        result.append(record)
    return result


def _procrustes_stability(embeddings: Sequence[Mapping[str, Any]], dimensions: int, profile: str) -> list[dict[str, Any]]:
    """Align all model embeddings before comparing 2-D geometry."""
    if len(embeddings) < 2: return []
    from scipy.linalg import orthogonal_procrustes
    result = []
    for left_index, left in enumerate(embeddings):
        for right in embeddings[left_index + 1:]:
            a, b = np.asarray(left["latent"], dtype=float), np.asarray(right["latent"], dtype=float)
            a = a - a.mean(axis=0); b = b - b.mean(axis=0)
            a /= max(np.linalg.norm(a), 1e-12); b /= max(np.linalg.norm(b), 1e-12)
            rotation, _ = orthogonal_procrustes(b, a)
            aligned = b @ rotation
            result.append({"profile": profile, "latent_dimensions": dimensions,
                           "left_seed": left["seed"], "left_held_out_reference_year": left["held_reference_year"],
                           "right_seed": right["seed"], "right_held_out_reference_year": right["held_reference_year"],
                           "procrustes_rmse": float(np.sqrt(np.mean((a - aligned) ** 2))),
                           "aligned_coordinate_correlation": float(np.mean([np.corrcoef(a[:, i], aligned[:, i])[0, 1] for i in range(dimensions)]))})
    return result


def nested_reference_year_splits(reference_years: Sequence[int]) -> list[tuple[int, tuple[int, ...]]]:
    """Return every reference once as outer test with every other year inside."""
    references = tuple(sorted({int(reference) for reference in reference_years}))
    if len(references) < 4: raise ValueError("nested reference validation needs at least four distinct years")
    return [(outer, tuple(reference for reference in references if reference != outer)) for outer in references]


def train_dae_representation(vectors: Sequence[Mapping[str, Any]], profile: str, dimensions: int,
                             config: MetricSynthesisConfig,
                             runtime: MetricSynthesisRuntimeConfig | None = None,
                             resume_store: Any | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    """Nested-LOYO DAE selection and evaluation using every cached reference year.

    Every outer reference year is evaluated.  Its DAE configuration and epoch
    count are selected by inner LOYO using only the other reference years.
    Death-F1 and performance artifacts are absent from this API.
    """
    from sklearn.decomposition import FactorAnalysis, PCA
    from sklearn.preprocessing import StandardScaler
    runtime = runtime or MetricSynthesisRuntimeConfig()
    from temporal_synthesis.execution import SearchJob, resolve_executor, run_search_jobs
    execution = resolve_executor(runtime)
    names = _profile_metrics(profile)
    rows = dae_member_vectors(vectors, profile)
    if len(rows) < max(3, dimensions + 1): raise RuntimeError("insufficient complete member vectors for selected DAE dimension")
    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float32)
    reference_values = np.asarray([int(row["reference_year"]) for row in rows], dtype=np.int64)
    references = sorted({int(row["reference_year"]) for row in rows})
    if len(references) < 4: raise RuntimeError("nested DAE validation requires at least four reference years")
    candidates = _dae_candidates()
    LOGGER.info("Nested DAE %s: %d complete member vectors, all %d reference years, %d configurations", profile, len(rows), len(references), len(candidates))
    search_folds: list[dict[str, Any]] = []
    selected_by_outer: list[dict[str, Any]] = []
    oof_metrics: list[dict[str, Any]] = []
    embeddings: list[dict[str, Any]] = []

    outer_splits = nested_reference_year_splits(references)
    for outer_held, outer_training_tuple in _progress(outer_splits, total=len(outer_splits), description=f"Nested DAE outer LOYO {profile}"):
        outer_training_references = list(outer_training_tuple)
        outer_candidate_summaries = []
        for candidate_index, candidate in enumerate(candidates, start=1):
            group_name = f"search.outer-{outer_held}.candidate-{candidate_index}"
            candidate_folds = None if resume_store is None else resume_store.load(group_name)
            if candidate_folds is None:
                jobs = [SearchJob(outer_held, inner_held, candidate_index, config.seed)
                        for inner_held in outer_training_references]
                runner = partial(_run_dae_search_job, matrix=matrix, reference_values=reference_values,
                                 outer_training_references=tuple(outer_training_references), candidate=candidate,
                                 dimensions=dimensions, epochs=config.dae_epochs,
                                 patience=config.dae_early_stopping_patience, device=execution.device,
                                 profile=profile)
                group_started = time.perf_counter()
                candidate_folds = run_search_jobs(jobs, runner, execution)
                duration = max(time.perf_counter() - group_started, 1e-9)
                LOGGER.info("DAE search outer=%s candidate=%d: %d jobs in %.2fs (%.2f jobs/s)",
                            outer_held, candidate_index, len(jobs), duration, len(jobs) / duration)
                if resume_store is not None:
                    resume_store.save(group_name, candidate_folds)
                    LOGGER.info("Published resumable candidate-fold checkpoint: %s", group_name)
            search_folds.extend(candidate_folds)
            outer_candidate_summaries.append({"candidate_index": candidate_index,
                                              "mean_validation_mse_macro_reference": float(np.mean([row["validation_mse"] for row in candidate_folds])),
                                              "mean_validation_mae_macro_reference": float(np.mean([row["validation_mae"] for row in candidate_folds])),
                                              "median_best_epoch": int(round(np.median([row["best_epoch"] for row in candidate_folds]))), **candidate})
            if candidate_index % 24 == 0:
                LOGGER.info("Nested DAE %s outer %s: evaluated %d/%d configurations", profile, outer_held, candidate_index, len(candidates))
        selected = min(outer_candidate_summaries, key=lambda row: (row["mean_validation_mse_macro_reference"], row["mean_validation_mae_macro_reference"], row["candidate_index"]))
        selected_by_outer.append({"profile": profile, "outer_held_out_reference_year": outer_held,
                                  "outer_training_reference_years": outer_training_references, **selected})
        outer_group = f"oof.outer-{outer_held}"
        completed_outer = None if resume_store is None else resume_store.load(outer_group)
        if completed_outer is not None:
            oof_metrics.extend(completed_outer["metrics"])
            embeddings.extend({**row, "latent": np.asarray(row["latent"], dtype=np.float32)}
                              for row in completed_outer["embeddings"])
            LOGGER.info("Resumed completed outer OOF checkpoint: %s", outer_group)
            continue
        selected_candidate = {name: selected[name] for name in ("dropout_probability", "noise_std", "hidden_dimensions", "learning_rate", "transform", "output_activation")}
        train_mask = np.asarray([int(row["reference_year"]) != outer_held for row in rows]); test_mask = ~train_mask
        outer_metrics_start = len(oof_metrics)
        outer_embeddings_start = len(embeddings)
        for seed in config.dae_validation_seeds:
            fitted, preprocessor, _ = _run_dae_split(matrix[train_mask], None, selected_candidate,
                                                      dimensions=dimensions, epochs=int(selected["median_best_epoch"]),
                                                      patience=config.dae_early_stopping_patience, seed=seed,
                                                      device=execution.device, progress=False)
            test_transformed = preprocessor.transform(matrix[test_mask])
            _, dae_reconstruction_transformed = _model_outputs(fitted, test_transformed)
            dae_reconstruction = preprocessor.inverse_transform(dae_reconstruction_transformed)
            oof_metrics.extend(_reconstruction_rows(profile=profile, method="dae", seed=seed, held_reference_year=outer_held,
                                                     names=names, truth=matrix[test_mask], reconstructed=dae_reconstruction,
                                                     dimensions=dimensions, best_epoch=fitted["best_epoch"]))
            # PCA and FA use their own train-only standardization, independent
            # of whichever DAE transform won the inner search.
            baseline_scaler = StandardScaler().fit(matrix[train_mask])
            baseline_train = baseline_scaler.transform(matrix[train_mask]); baseline_test = baseline_scaler.transform(matrix[test_mask])
            pca = PCA(n_components=dimensions, random_state=seed).fit(baseline_train)
            pca_reconstruction = np.clip(baseline_scaler.inverse_transform(pca.inverse_transform(pca.transform(baseline_test))), 0., 1.)
            oof_metrics.extend(_reconstruction_rows(profile=profile, method="pca", seed=seed, held_reference_year=outer_held,
                                                     names=names, truth=matrix[test_mask], reconstructed=pca_reconstruction, dimensions=dimensions))
            fa = FactorAnalysis(n_components=dimensions, random_state=seed).fit(baseline_train)
            fa_reconstruction = np.clip(baseline_scaler.inverse_transform(fa.transform(baseline_test) @ fa.components_ + fa.mean_), 0., 1.)
            oof_metrics.extend(_reconstruction_rows(profile=profile, method="fa", seed=seed, held_reference_year=outer_held,
                                                     names=names, truth=matrix[test_mask], reconstructed=fa_reconstruction, dimensions=dimensions))
            all_latent, _ = _model_outputs(fitted, preprocessor.transform(matrix))
            embeddings.append({"seed": seed, "held_reference_year": outer_held, "latent": all_latent})
        if resume_store is not None:
            resume_store.save(outer_group, {
                "metrics": oof_metrics[outer_metrics_start:],
                "embeddings": [{**row, "latent": np.asarray(row["latent"]).tolist()}
                               for row in embeddings[outer_embeddings_start:]],
            })
            LOGGER.info("Published resumable outer OOF checkpoint: %s", outer_group)

    # Aggregate every nested inner fold to choose the descriptive full-data fit.
    candidate_summary = []
    for candidate_index, candidate in enumerate(candidates, start=1):
        folds = [row for row in search_folds if row["candidate_index"] == candidate_index]
        candidate_summary.append({"profile": profile, "candidate_index": candidate_index,
                                  "outer_reference_group_count": len(references), "nested_inner_fold_count": len(folds),
                                  "mean_validation_mse_macro_nested_fold": float(np.mean([row["validation_mse"] for row in folds])),
                                  "mean_validation_mae_macro_nested_fold": float(np.mean([row["validation_mae"] for row in folds])),
                                  "median_best_epoch": int(round(np.median([row["best_epoch"] for row in folds]))), **candidate})
    selected_full = min(candidate_summary, key=lambda row: (row["mean_validation_mse_macro_nested_fold"], row["mean_validation_mae_macro_nested_fold"], row["candidate_index"]))
    full_candidate = {name: selected_full[name] for name in ("dropout_probability", "noise_std", "hidden_dimensions", "learning_rate", "transform", "output_activation")}

    oof_summary = []
    for method in ("dae", "pca", "fa"):
        for metric in ("overall",) + tuple(names):
            selected_rows = [row for row in oof_metrics if row["method"] == method and row["metric"] == metric]
            oof_summary.append({"profile": profile, "method": method, "metric": metric, "latent_dimensions": dimensions,
                                "fold_count": len(selected_rows), "reference_group_count": len(references),
                                "mae_macro_reference_seed": float(np.mean([row["mae"] for row in selected_rows])),
                                "mse_macro_reference_seed": float(np.mean([row["mse"] for row in selected_rows]))})
    pca_by_fold = {(row["seed"], row["held_out_reference_year"]): row for row in oof_metrics if row["method"] == "pca" and row["metric"] == "overall"}
    dae_pca_comparison = [{"profile": profile, "seed": row["seed"], "held_out_reference_year": row["held_out_reference_year"],
                           "dae_mse": row["mse"], "pca_mse": pca_by_fold[(row["seed"], row["held_out_reference_year"])]["mse"],
                           "dae_minus_pca_mse": row["mse"] - pca_by_fold[(row["seed"], row["held_out_reference_year"])]["mse"],
                           "dae_beats_pca": bool(row["mse"] < pca_by_fold[(row["seed"], row["held_out_reference_year"])]["mse"])}
                          for row in oof_metrics if row["method"] == "dae" and row["metric"] == "overall"]
    consistently_beats_pca = bool(dae_pca_comparison) and all(row["dae_beats_pca"] for row in dae_pca_comparison)
    stability = _procrustes_stability(embeddings, dimensions, profile)

    fitted, preprocessor, fitted_config = _run_dae_split(matrix, None, full_candidate, dimensions=dimensions,
                                                          epochs=int(selected_full["median_best_epoch"]), patience=config.dae_early_stopping_patience,
                                                          seed=config.seed, device=execution.device, progress=runtime.progress)
    latent, reconstruction = _model_outputs(fitted, preprocessor.transform(matrix))
    full_metrics = _reconstruction_rows(profile=profile, method="dae_full_fit", seed=config.seed, held_reference_year=-1,
                                        names=names, truth=matrix, reconstructed=preprocessor.inverse_transform(reconstruction),
                                        dimensions=dimensions, best_epoch=fitted["best_epoch"])
    latent_rows = [{**{key: row[key] for key in MEMBER_KEYS + ("cohort_view", "temporal_distance")}, "profile": profile,
                    **{f"latent_{index + 1}": float(latent[row_index, index]) for index in range(dimensions)}} for row_index, row in enumerate(rows)]
    family_trajectories = _aggregate_latent(latent_rows, dimensions, FAMILY_KEYS + ("cohort_view", "temporal_distance"), "family", profile)
    system_trajectories = _aggregate_latent(latent_rows, dimensions, SYSTEM_KEYS, "system", profile)
    checkpoint = {name: value.detach().cpu().numpy() for name, value in fitted["model"].state_dict().items()}
    checkpoint.update(preprocessor.checkpoint_arrays()); checkpoint["metric_names"] = np.asarray(names, dtype="U"); checkpoint["latent_dimensions"] = np.asarray([dimensions], dtype=np.int64)
    details = {"profile": profile, "latent_dimensions": dimensions, "input_level": "complete_member_factor_vectors",
               "input_row_count": len(rows), "all_available_reference_years": references, "reference_group_count": len(references),
               "validation_design": "nested_LOYO_all_reference_years", "outer_reference_years": references,
               "nested_inner_folds_per_outer": len(references) - 1, "candidate_count": len(candidates),
               "device": fitted["device"], "torch_version": fitted["torch_version"], "seed": config.seed,
               "executor": execution.executor, "workers": execution.workers,
               "validation_seeds": list(config.dae_validation_seeds), "outcome_used": False, "monotone_score": dimensions == 1,
               "selected_full_candidate_index": selected_full["candidate_index"], "selected_full_candidate": full_candidate,
               "full_fit_best_epoch": fitted["best_epoch"], "full_fit_config": fitted_config.to_dict(),
               "oof_beats_pca_consistently": consistently_beats_pca,
               "recommendation": "retain_exploratory_dae" if consistently_beats_pca else "use_pca_or_fa_working_representation"}
    return {"dae_nested_search_folds": search_folds, "dae_nested_selected_candidates": selected_by_outer,
            "dae_hyperparameter_search_summary": candidate_summary, "dae_oof_reconstruction_metrics": oof_metrics,
            "dae_oof_reconstruction_summary": oof_summary, "dae_oof_vs_pca": dae_pca_comparison,
            "dae_reconstruction_metrics": full_metrics, "dae_training_history": fitted["history"], "dae_latent_vectors": latent_rows,
            "dae_family_latent_trajectories": family_trajectories, "dae_latent_trajectories": system_trajectories,
            "dae_stability": stability, "dae_training_details": [details]}, checkpoint


def build_metric_synthesis(enrichment_manifest_path: str | Path, cri_manifest_path: str | Path,
                           config: MetricSynthesisConfig | None = None,
                           runtime_config: MetricSynthesisRuntimeConfig | None = None) -> Path:
    config = config or MetricSynthesisConfig()
    runtime_config = runtime_config or MetricSynthesisRuntimeConfig()
    synthesis_started = time.perf_counter()
    from temporal_synthesis.execution import resolve_executor
    execution = resolve_executor(runtime_config)
    LOGGER.info("Starting temporal metric synthesis (profiles=%s, parallel repetitions=%d)", ", ".join(config.profiles), config.parallel_repetitions)
    ep, em, cp, cm = validate_inputs(enrichment_manifest_path, cri_manifest_path)
    identifier = synthesis_hash(ep, cp, config, execution.device)
    root = cp.parent.parent / f"metric_synthesis_{identifier}"
    out = root / "manifest.json"
    if out.is_file():
        existing=json.loads(out.read_text());
        if existing.get("complete") is True:
            for d in existing.get("artifacts",{}).values(): _validate_descriptor(root,d)
            LOGGER.info("Reusing validated derived artifact: %s", out)
            return out
    e = _load(ep, em, ("headline_factor_metrics", "tcav_significance", "performance_variants", "primary_performance"))
    c = _load(cp, cm, ("cri_member_utilities", "cri_family_universe"))
    vectors, audit = build_metric_vectors(e, c)
    products = {"metric_vectors": vectors, "tcav_coverage_audit": audit,
                "runtime_telemetry": [{"stage": "executor_selection", "device": execution.device,
                                       "executor": execution.executor, "workers": execution.workers,
                                       "selection_reason": execution.selection_reason}]}
    sample = []
    prediction_flow = []
    for profile in config.profiles:
        LOGGER.info("Starting profile: %s", profile)
        features=system_concept_features(vectors,profile); performance=e["performance_variants"]
        derived=build_early_warning_rows(performance,features,profile); loyo,folds,metrics=_ridge_oof(derived,profile,ridge_alphas=config.ridge_alphas); forward,ffolds,fmetrics=_ridge_oof(derived,profile,forward=True,ridge_alphas=config.ridge_alphas)
        maximal_loyo, maximal_folds, maximal_metrics = _ridge_oof(derived, profile, maximal_history=True, ridge_alphas=config.ridge_alphas)
        maximal_forward, maximal_forward_folds, maximal_forward_metrics = _ridge_oof(derived, profile, forward=True, maximal_history=True, ridge_alphas=config.ridge_alphas)
        latent_loyo, latent_folds, latent_metrics = [], [], []
        latent_forward, latent_forward_folds, latent_forward_metrics = [], [], []
        for representation in ("pca2", "fa2"):
            p, f, m = _latent_ridge_oof(derived, profile, config, method=representation)
            latent_loyo.extend(p); latent_folds.extend(f); latent_metrics.extend(m)
            p, f, m = _latent_ridge_oof(derived, profile, config, method=representation, forward=True)
            latent_forward.extend(p); latent_forward_folds.extend(f); latent_forward_metrics.extend(m)
        event_loyo, event_folds, event_thresholds, event_metrics = _material_degradation_oof(derived, profile, config)
        event_forward, event_forward_folds, event_forward_thresholds, event_forward_metrics = _material_degradation_oof(derived, profile, config, forward=True)
        quality = metric_quality(features, profile)
        spectrum,loadings,parallel,fa=dimensionality(features,profile,config)
        bootstrap = bootstrap_stability(features, profile, config)
        fa_loadings = [{"profile": profile, "dimensions": row["dimensions"], "metric": metric, **{f"FA{i+1}": value for i, value in enumerate(values)}} for row in fa if row.get("status") == "valid" for metric, values in zip(_profile_metrics(profile), row["loadings"])]
        not_estimable = next((row["status"] for row in fa if str(row.get("status", "")).startswith("not_estimable")), None)
        dimension_summary = [{"profile": profile, "status": not_estimable or "estimable", "parallel_retained_dimensions": int(sum(row["retained"] for row in parallel)), "kaiser_retained_dimensions": int(sum(row["kaiser_retained"] for row in spectrum)), "distance_zero_excluded": True, "row_count": len([r for r in features if int(r["temporal_distance"]) != 0 and all(_finite(r.get(x)) for x in _profile_metrics(profile))])}]
        all_metrics = metrics + fmetrics + latent_metrics + latent_forward_metrics + maximal_metrics + maximal_forward_metrics
        comparison = [{"profile": profile, "evaluation": row["evaluation"], "model": row["model"], "mae_macro_reference": row["mae_macro_reference"],
                       "variant": row["variant"], "analysis_population": row["analysis_population"],
                       "improvement_vs_performance_history": next((base["mae_macro_reference"] - row["mae_macro_reference"] for base in all_metrics if base["evaluation"] == row["evaluation"] and base["variant"] == row["variant"] and base["model"] == ("performance_history_maximal_coverage" if row["analysis_population"] == "maximal_history" else "performance_history")), None),
                       "improvement_vs_zero_degradation": next((base["mae_macro_reference"] - row["mae_macro_reference"] for base in all_metrics if base["evaluation"] == row["evaluation"] and base["variant"] == row["variant"] and base["model"] == ("zero_degradation_maximal_coverage" if row["analysis_population"] == "maximal_history" else "zero_degradation")), None)} for row in all_metrics]
        tcav_zero_variation = profile == "p50_tcav_extended" and any(
            row["metric"] == "u_tcav" and row["eligibility_reason"] == "zero_variance" for row in quality
        )
        for row in comparison:
            row["tcav_predictive_variation"] = "zero" if tcav_zero_variation else "not_applicable"
            row["delta_tcav_predictive_variation"] = "zero" if tcav_zero_variation else "not_applicable"
            row["improvement_attributable_to_tcav"] = False
            row["sensitivity_label"] = "underpowered_p50_tcav_sensitivity" if profile == "p50_tcav_extended" else "primary_core_analysis"
        from temporal_synthesis.supervised import compare_oof_models
        uncertainty = []
        combined_predictions = loyo + forward
        for variant_name in sorted({str(row.get("variant", "original")) for row in combined_predictions}):
            for evaluation_name in ("loyo", "forward"):
                subset = [row for row in combined_predictions if str(row.get("variant", "original")) == variant_name and row.get("evaluation") == evaluation_name]
                for candidate_name in ("concept_robustness", "history_plus_concepts"):
                    result = compare_oof_models(subset, candidate=candidate_name,
                                                repetitions=config.bootstrap_repetitions, seed=config.seed)
                    uncertainty.append({"profile": profile, "variant": variant_name,
                                        "evaluation": evaluation_name, **result})
        products[f"system_concept_features_{profile}"]=features; products[f"early_warning_rows_{profile}"]=derived; products[f"metric_quality_{profile}"]=quality; products[f"dimensionality_bootstrap_stability_{profile}"]=bootstrap; products[f"ridge_oof_predictions_{profile}"]=loyo+forward; products[f"ridge_fold_audit_{profile}"]=folds+ffolds; products[f"ridge_metrics_{profile}"]=metrics+fmetrics; products[f"ridge_maximal_history_oof_predictions_{profile}"]=maximal_loyo+maximal_forward; products[f"ridge_maximal_history_fold_audit_{profile}"]=maximal_folds+maximal_forward_folds; products[f"ridge_maximal_history_metrics_{profile}"]=maximal_metrics+maximal_forward_metrics; products[f"ridge_model_comparisons_{profile}"]=comparison; products[f"ridge_model_comparison_uncertainty_{profile}"]=uncertainty; products[f"ridge_latent_oof_predictions_{profile}"]=latent_loyo+latent_forward; products[f"ridge_latent_fold_audit_{profile}"]=latent_folds+latent_forward_folds; products[f"ridge_latent_metrics_{profile}"]=latent_metrics+latent_forward_metrics; products[f"material_degradation_oof_predictions_{profile}"]=event_loyo+event_forward; products[f"material_degradation_fold_audit_{profile}"]=event_folds+event_forward_folds; products[f"material_degradation_noise_thresholds_{profile}"]=event_thresholds+event_forward_thresholds; products[f"material_degradation_metrics_{profile}"]=event_metrics+event_forward_metrics; products[f"pca_spectrum_{profile}"]=spectrum; products[f"pca_loadings_{profile}"]=loadings; products[f"parallel_analysis_{profile}"]=parallel; products[f"fa_diagnostics_{profile}"]=fa; products[f"fa_loadings_{profile}"]=fa_loadings; products[f"dimension_summary_{profile}"]=dimension_summary
        sample.append({"profile":profile,"metric_vectors":len(vectors),"system_features":len(features),"early_warning_rows":len(derived)})
        required_common = ("death_f1_current", "previous_degradation", "temporal_distance") + _profile_metrics(profile) + tuple(f"delta_{name}" for name in _profile_metrics(profile)) + ("concept_coverage", "current_cri", "delta_current_cri")
        for variant_name in sorted({str(row["variant"]) for row in derived}):
            variant_rows = [row for row in derived if str(row["variant"]) == variant_name]
            prediction_flow.append({"profile": profile, "variant": variant_name, "early_warning_rows_all_available": len(variant_rows),
                                    "history_complete_rows": sum(all(_finite(row.get(name)) for name in ("death_f1_current", "previous_degradation", "temporal_distance")) for row in variant_rows),
                                    "common_complete_case_rows": sum(all(_finite(row.get(name)) for name in required_common) for row in variant_rows),
                                    "reference_year_count": len({int(row["reference_year"]) for row in variant_rows})})
        LOGGER.info("Completed profile %s: %d system features, %d early-warning rows", profile, len(features), len(derived))
    products["sample_flow"] = sample
    products["prediction_sample_flow"] = prediction_flow
    dae_checkpoint: dict[str, np.ndarray] | None = None
    if config.dae_profile is None:
        products["dae_status"] = [{"status":"awaiting_human_dimension_selection", "dae_profile":None, "dae_latent_dimensions":None, "outcome_used":False}]
    else:
        from temporal_synthesis.execution import ResumeStore
        resume_store = ResumeStore(
            root.parent / f"metric_synthesis_{identifier}_work",
            {"metric_synthesis_hash": identifier, "source_fingerprints": _source_fingerprints(),
             "numerical_environment": _numerical_environment(execution.device)},
            enabled=runtime_config.resume,
        )
        dae_products, dae_checkpoint = train_dae_representation(
            vectors, config.dae_profile, config.dae_latent_dimensions, config,
            runtime_config, resume_store,
        )
        from temporal_synthesis.execution import shutdown_search_executors
        shutdown_search_executors()
        products.update(dae_products)
        details = dae_products["dae_training_details"][0]
        products["dae_status"] = [{"status": "oof_better_than_pca" if details["oof_beats_pca_consistently"] else "exploratory_dae_did_not_consistently_beat_pca",
                                   "dae_profile": config.dae_profile, "dae_latent_dimensions": config.dae_latent_dimensions,
                                   "outcome_used": False, "recommendation": details["recommendation"]}]
    peak_cuda_memory = None
    if execution.device.startswith("cuda"):
        import torch
        peak_cuda_memory = int(torch.cuda.max_memory_allocated(execution.device))
    products["runtime_telemetry"].append({"stage": "analysis_complete", "duration_seconds": time.perf_counter() - synthesis_started,
                                          "peak_cuda_memory_bytes": peak_cuda_memory})
    descriptors={}
    for name,rows in products.items():
        LOGGER.info("Writing derived table %s (%d rows)", name, len(rows))
        d=atomic_write_jsonl_gzip(root/f"{name}.jsonl.gz",rows); d["path"]=f"{name}.jsonl.gz"; descriptors[name]=d
    if dae_checkpoint is not None:
        LOGGER.info("Writing checksummed DAE checkpoint")
        descriptors["dae_weights"] = _write_npz(root / "dae_weights.npz", dae_checkpoint)
    for descriptor in descriptors.values():
        _validate_descriptor(root, descriptor)
    manifest={"schema_version":"2.0","artifact_schema_version":ARTIFACT_SCHEMA_VERSION,"complete":True,"metric_synthesis_hash":identifier,"enrichment_manifest":str(ep),"enrichment_manifest_sha256":file_sha256(ep),"cri_manifest":str(cp),"cri_manifest_sha256":file_sha256(cp),"source_fingerprints":_source_fingerprints(),"scientific_config":config.to_dict(),"runtime_config":runtime_config.to_dict(),"numerical_environment":_numerical_environment(execution.device),"executor_selection":{"device":execution.device,"executor":execution.executor,"workers":execution.workers,"selection_reason":execution.selection_reason},"artifacts":descriptors}
    atomic_write_json(out,manifest,compact=False)
    pointer = root.parent / "metric_synthesis_canonical.json"
    atomic_write_json(pointer, {"manifest": str(out), "metric_synthesis_hash": identifier}, compact=False)
    LOGGER.info("Completed temporal metric synthesis: %s", out)
    return out


def load_metric_synthesis(manifest_path: str | Path) -> dict[str,list[dict[str,Any]]]:
    path = Path(manifest_path)
    if path.is_file():
        candidate = json.loads(path.read_text(encoding="utf-8"))
        if "manifest" in candidate and candidate.get("complete") is not True:
            resolved = Path(str(candidate["manifest"]))
            path = resolved if resolved.is_absolute() else path.parent / resolved
    path,manifest=_read_manifest(path,label="metric synthesis")
    return _load(path,manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrichment-manifest", required=True)
    parser.add_argument("--cri-manifest", required=True)
    parser.add_argument("--skip-dae", action="store_true")
    parser.add_argument("--dae-epochs", type=int, default=250)
    parser.add_argument("--dae-patience", type=int, default=30)
    parser.add_argument("--dae-seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--executor", default="auto", choices=("auto", "serial", "thread", "process"))
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log-file", default="temporal-metric-synthesis.log")
    parser.add_argument("--material-degradation-noise-quantile", type=float, default=.95)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    args = parser.parse_args()
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)
    workers: int | str = args.workers if args.workers == "auto" else int(args.workers)
    scientific = MetricSynthesisConfig(
        dae_profile=None if args.skip_dae else "core",
        dae_latent_dimensions=None if args.skip_dae else 2,
        dae_epochs=args.dae_epochs,
        dae_early_stopping_patience=args.dae_patience,
        dae_validation_seeds=tuple(args.dae_seeds),
        material_degradation_noise_quantile=args.material_degradation_noise_quantile,
    )
    runtime = MetricSynthesisRuntimeConfig(device=args.device, executor=args.executor, workers=workers,
                                           resume=not args.no_resume, log_file=args.log_file)
    print(build_metric_synthesis(args.enrichment_manifest, args.cri_manifest, scientific, runtime))
