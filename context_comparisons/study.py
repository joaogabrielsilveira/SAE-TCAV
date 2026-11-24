"""One-change-at-a-time additions without modifying the original running study."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from concept_study import measures
from context_comparisons import choices, methods, paired, provenance
from context_comparisons.cache import MissingPredictions, PredictionStore
from context_comparisons.geometry import retrieval_diagnostics
from context_comparisons.native import implementation_identity
from context_comparisons.representations import native_concepts
from context_intervention import data, experiment, predictor, representations, study as original_study
from context_intervention.analysis import primary_builds


ORIGINAL = Path("stats/context_intervention")
OUTPUT = Path("stats/context_comparisons")
CORRECTIONS = {
    "sampling": "common random numbers: one key per context row and (seed, draw); hard and graded draws keep "
                "the uniform draw's eligible rows (Efraimidis-Spirakis keys for graded weights)",
    "removal_order": "priority, then cosine similarity to the representative patients (was population row order)",
    "development_split": "original stable 60/40 patient hash split; no future outcomes or entrants used",
    "baseline": "cumulative arms recomputed with these corrections; original study arms are not mixed in",
}


def atomic_json(path, value):
    def finite(item):
        if isinstance(item, dict):
            return {key: finite(v) for key, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(v) for v in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        return item

    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(finite(value), indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def original_details(reference, update_year, original=ORIGINAL):
    name = f"{reference['build']}_{reference['reference']}_{update_year}"
    path = original / f"development_{name}.json"
    details = json.loads(path.read_text())
    provenance.validate(reference, details["provenance"])
    return details


def signature(reference, details):
    current = provenance.validate(reference, details["provenance"])
    digest = hashlib.sha256(json.dumps(current, sort_keys=True).encode())
    digest.update(json.dumps(implementation_identity(), sort_keys=True).encode())
    for path in sorted(Path("context_comparisons").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def frame(reference, year, probabilities, method, mortality, threshold, metadata=None):
    queries = data.evaluation_rows(reference, year)
    population = reference["population"]
    rows = []
    for seed, values in zip(experiment.SEEDS, probabilities):
        if len(values) != len(queries):
            raise ValueError("Evaluation predictions are not aligned")
        for query, probability in zip(queries, values):
            rows.append({"build": reference["build"], "reference": reference["reference"], "year": year,
                         "update_year": year - 1, "patient_id": str(population["patient_id"][query]),
                         "sampling_seed": seed, "label": int(population["y"][query]), "method": method,
                         "probability": float(probability), "threshold": threshold, "tuned_threshold": np.nan,
                         "reference_mortality": mortality, **(metadata or {})})
    return pd.DataFrame(rows)


def candidate_records(labels, form, predict, threshold, mortality):
    """Score every option of one form on validation; ``predict(form, strength)`` supplies probabilities."""
    if len(np.unique(labels)) < 2:
        return []
    return [choices.score_option(labels, predict(option, strength), threshold, mortality, option, strength)
            for option, strength in choices.options(form)]


def metric_comparison(reference, update_year, output=OUTPUT, original=ORIGINAL):
    """Compare validation decisions with no additional model inference.

    All candidate priorities stay Death-F1-based. Only the validation objective
    changes. If a newly selected option has no test cache, retain the decision
    and mark test evaluation unavailable instead of launching expensive fits.

    This pass re-reads the original study's decisions, so it deliberately keeps
    the original split, sampler and removal order (``methods``). Its predictions
    are saved separately and are not mixed with the corrected comparison arms.
    """
    output.mkdir(parents=True, exist_ok=True)
    details = original_details(reference, update_year, original)
    record, _ = original_study.reference_info(reference["build"], reference["reference"])
    table = measures.load_build(record)
    threshold, mortality = float(record["death_threshold"]["last_3"]), float(table["reference_mortality"])
    validation = data.development_rows(reference, update_year, "validation")
    candidates = data.development_rows(reference, update_year, "development")
    test = data.evaluation_rows(reference, update_year + 1)
    size, draws = details["provenance"]["sample_size"], details["provenance"]["draws"]
    store = PredictionStore(reference, original / "predictions", output / "predictions", allow_inference=False)
    decision_rows, test_frames, records_by_form = [], [], {}
    for representation in ("features", "concepts"):
        path = original / f"priorities_{reference['build']}_{reference['reference']}_{update_year}_{representation}.npy"
        priorities = np.load(path, allow_pickle=False)
        def predict(option, strength):
            return methods.predictions(reference, priorities, validation, option, strength, store, size, draws)

        for form_name in choices.FORMS:
            records = (candidate_records(reference["population"]["y"][validation], form_name, predict, threshold,
                                         mortality) if len(np.unique(reference["population"]["y"][candidates])) == 2 else [])
            records_by_form[f"{representation}_{form_name}"] = records
            winners = {objective: choices.choose(records, objective) for objective in choices.OBJECTIVES}
            frozen = details[representation]["interventions"][form_name]["choice"]
            if any(winners["death_f1"][key] != frozen[key] for key in ("form", "strength")):
                raise ValueError("Recomputed Death-F1 selection differs from the original runner")
            for objective, winner in winners.items():
                result = {"build": reference["build"], "reference": reference["reference"],
                          "update_year": update_year, "year": update_year + 1, "representation": representation,
                          "intervention": form_name, "objective": objective, "choice": winner,
                          "different_from_death_f1": any(winner[k] != winners["death_f1"][k] for k in ("form", "strength"))}
                metadata = methods.policy_metadata(reference, priorities, winner["form"], winner["strength"], size,
                                                   form_name, winner.get("reason", ""))
                result["effective_policy"] = metadata
                result["random_direct_control"] = ("unavailable: selected policy test predictions missing"
                    if metadata["effective_form"] == "direct" else "not applicable: no direct removal")
                try:
                    probabilities = methods.predictions(reference, priorities, test, winner["form"], winner["strength"],
                                                         store, size, draws)
                except MissingPredictions:
                    result["evaluation"] = "new test inference required; not run by the cheap comparison"
                else:
                    result["evaluation"] = "cached"
                    test_frames.append(frame(reference, update_year + 1, probabilities,
                                              f"{objective}_{representation}_{form_name}", mortality, threshold, metadata))
                    if metadata["effective_form"] == "direct":
                        try:
                            control = methods.random_removal_predictions(reference, priorities, test, winner["strength"],
                                                                          store, size)
                        except MissingPredictions:
                            result["random_direct_control"] = "unavailable: matching control needs new inference; not run"
                        else:
                            result["random_direct_control"] = "cached, exactly matched by stratum and removed count"
                            control_metadata = {**metadata, "effective_form": "random_direct", "guided_adjustment": False}
                            test_frames.append(frame(reference, update_year + 1, control,
                                f"{objective}_{representation}_random_direct", mortality, threshold, control_metadata))
                decision_rows.append(result)
    name = f"{reference['build']}_{reference['reference']}_{update_year}"
    result = {"source_identity": signature(reference, details), "decisions": decision_rows,
              "candidate_validation_metrics": records_by_form, "cache_hits": store.hits,
              "new_model_inferences": store.new_predictions,
              "candidate_priority_metric": "death_f1", "scope": "cumulative development"}
    atomic_json(output / f"metric_choices_{name}.json", result)
    if test_frames:
        pd.concat(test_frames, ignore_index=True).to_csv(output / f"metric_predictions_{name}.csv.gz",
                                                       index=False, compression="gzip")
    return result


def prediction_store(reference, update_year, output=OUTPUT, original=ORIGINAL):
    """Inference store shared by all corrected arms of one update; query supersets are reused."""
    supersets = (paired.development_rows(reference, update_year, "development"),
                 paired.development_rows(reference, update_year, "validation"),
                 data.evaluation_rows(reference, update_year + 1))
    return PredictionStore(reference, original / "predictions", output / "predictions", True, supersets=supersets)


def evaluate_variant(reference, update_year, representation, vectors, scope, variant, details,
                     output=OUTPUT, original=ORIGINAL, store=None):
    """One corrected arm: paired sampling, similarity-ordered removal, stable patient hash split."""
    record, _ = original_study.reference_info(reference["build"], reference["reference"])
    mortality = float(measures.load_build(record)["reference_mortality"])
    threshold = float(record["death_threshold"]["last_3"])
    full_candidates = paired.development_rows(reference, update_year, "development")
    full_validation = paired.development_rows(reference, update_year, "validation")
    population = reference["population"]
    candidate_mask = population["year"][full_candidates] == update_year if scope == "latest_year" else np.ones(len(full_candidates), bool)
    validation_mask = population["year"][full_validation] == update_year if scope == "latest_year" else np.ones(len(full_validation), bool)
    candidates, validation = full_candidates[candidate_mask], full_validation[validation_mask]
    test = data.evaluation_rows(reference, update_year + 1)
    context_vectors, candidate_vectors = vectors
    context = reference["context_rows"]
    size, draws = details["provenance"]["sample_size"], details["provenance"]["draws"]
    store = store or prediction_store(reference, update_year, output, original)
    representation_varies = len(context_vectors) > 1 and bool(np.any(np.ptp(context_vectors, axis=0) > 0))
    probabilities, _ = data.cached_predictor_rows(reference, candidates) if len(candidates) else (np.array([]), None)
    geometry = retrieval_diagnostics(context_vectors, candidate_vectors[candidate_mask],
        population["year"][context], population["y"][context], population["y"][candidates], probabilities, threshold)
    reasons = []
    if len(np.unique(population["y"][candidates])) < 2 or len(np.unique(population["y"][validation])) < 2:
        reasons.append("development lacks both outcomes")
    geometry["fallback_enforced"] = variant == "native_context"
    if variant == "native_context" and not geometry["any_discriminating_stratum"]:
        reasons.append("no within-stratum cosine discrimination")
    elif variant != "native_context" and not representation_varies:
        reasons.append("constant context representation")
    insufficient = bool(reasons)
    closeness = np.zeros(len(context))
    if insufficient:
        priorities, effects, neighborhoods, neighborhood_details = np.zeros(len(context)), [], [], []
    else:
        neighborhoods, neighborhood_details = experiment.make_neighborhoods(
            context_vectors, candidate_vectors[candidate_mask], population["year"][context], population["y"][context],
            population["y"][candidates], probabilities, threshold)
        representatives = [row for detail in neighborhood_details for row in detail.get("representative_rows", [])]
        closeness = paired.similarity(context_vectors, candidate_vectors[candidate_mask][representatives])
        priorities, effects = paired.neighborhood_priorities(reference, neighborhoods, candidates, threshold,
                                                              mortality, store, size, draws)
    result = {"scope": scope, "context_representation": variant, "representation": representation,
              "context_representation_varies": representation_varies,
              "retrieval_geometry": geometry, "fallback_reason": "; ".join(reasons),
              "counts": {"candidate_rows": len(candidates), "candidate_deaths": int(population["y"][candidates].sum()),
                         "validation_rows": len(validation), "validation_deaths": int(population["y"][validation].sum())},
              "neighborhoods": neighborhood_details, "effects": effects, "choices": {}}

    def predict(queries, option, strength):
        return paired.predictions(reference, priorities, closeness, queries, option, strength, store, size, draws)

    frames = []
    for form_name in choices.FORMS:
        records = [] if insufficient else candidate_records(population["y"][validation], form_name,
            lambda option, strength: predict(validation, option, strength), threshold, mortality)
        winner = choices.choose(records, "death_f1")
        if insufficient:
            winner["reason"] = "; ".join(reasons)
        metadata = methods.policy_metadata(reference, priorities, winner["form"], winner["strength"], size,
                                           form_name, winner.get("reason", ""))
        result["choices"][form_name] = {"winner": winner, "validation": records, "effective_policy": metadata}
        method_name = f"{variant}_{representation}_{form_name}"
        frames.append(frame(reference, update_year + 1, predict(test, winner["form"], winner["strength"]), method_name,
                            mortality, threshold, metadata))
        if metadata["effective_form"] == "direct":
            control = paired.random_removal_predictions(reference, priorities, closeness, test, winner["strength"],
                                                        store, size)
            frames.append(frame(reference, update_year + 1, control, f"{variant}_{representation}_random_direct", mortality,
                                threshold, {**metadata, "effective_form": "random_direct", "guided_adjustment": False}))
    result["cache_hits"], result["new_model_inferences"] = store.hits, store.new_predictions
    return frames, result, priorities


def common_controls(reference, update_year, details, store):
    """Original context and paired uniform sampling, the shared controls of every corrected arm."""
    record, _ = original_study.reference_info(reference["build"], reference["reference"])
    mortality = float(measures.load_build(record)["reference_mortality"])
    threshold = float(record["death_threshold"]["last_3"])
    test = data.evaluation_rows(reference, update_year + 1)
    size, draws = details["provenance"]["sample_size"], details["provenance"]["draws"]
    zeros = np.zeros(len(reference["context_rows"]))
    frames = []
    for method, form in (("original", "original"), ("uniform_sampling", "uniform")):
        probabilities = paired.predictions(reference, zeros, zeros, test, form, 0., store, size, draws)
        metadata = methods.policy_metadata(reference, zeros, form, 0., size, form)
        frames.append(frame(reference, update_year + 1, probabilities, method, mortality, threshold, metadata))
    return frames


def run_update(reference, update_year, output=OUTPUT, original=ORIGINAL, native_vectors=None):
    """Corrected cumulative, latest-year and native-context arms, each against corrected controls.

    The cumulative arm is recomputed here with the original split and corrected sampler and removal order,
    so that latest-year and native-context results change exactly one thing relative to it.
    The original study's own arms are not copied in.
    """
    output.mkdir(parents=True, exist_ok=True)
    details = original_details(reference, update_year, original)
    identity = signature(reference, details)
    name = f"{reference['build']}_{reference['reference']}_{update_year}"
    target, complete = output / f"predictions_{name}.csv.gz", output / f"comparisons_{name}.json"
    if complete.exists():
        if json.loads(complete.read_text())["source_identity"] != identity:
            raise ValueError("Existing comparisons use different source identities")
        if target.exists():
            return
    metric_comparison(reference, update_year, output, original)
    store = prediction_store(reference, update_year, output, original)
    frames, results = common_controls(reference, update_year, details, store), {}
    candidates = paired.development_rows(reference, update_year, "development")
    validation = paired.development_rows(reference, update_year, "validation")
    all_rows = np.concatenate((candidates, validation))
    _, records = original_study.reference_info(reference["build"], reference["reference"])
    vectors = representations.representations(reference, all_rows, records, original / "embeddings")
    for representation in ("features", "concepts"):
        print(f"  Cumulative {representation}, reference {reference['reference']}, update {update_year}", flush=True)
        selected_vectors = (vectors[representation][0], vectors[representation][1][:len(candidates)])
        new_frames, result, priorities = evaluate_variant(reference, update_year, representation, selected_vectors,
            "cumulative", "cumulative", details, output, original, store)
        frames.extend(new_frames)
        results[f"cumulative_{representation}"] = result
        np.save(output / f"cumulative_priorities_{name}_{representation}.npy", priorities)
        if update_year == reference["reference"]:
            for item in new_frames:
                latest = item.copy()
                latest["method"] = latest["method"].str.replace("cumulative_", "latest_year_", n=1, regex=False)
                frames.append(latest)
            results[f"latest_year_{representation}"] = {"identical_to_reference_year_cumulative": True}
        else:
            print(f"  Latest-year {representation}, reference {reference['reference']}, update {update_year}", flush=True)
            new_frames, result, priorities = evaluate_variant(reference, update_year, representation, selected_vectors,
                "latest_year", "latest_year", details, output, original, store)
            frames.extend(new_frames)
            results[f"latest_year_{representation}"] = result
            np.save(output / f"latest_priorities_{name}_{representation}.npy", priorities)
    if native_vectors is None:
        native_vectors = native_concepts(reference, records, output, original)
    print(f"  Native context concepts, reference {reference['reference']}, update {update_year}", flush=True)
    new_frames, result, priorities = evaluate_variant(reference, update_year, "concepts",
        (native_vectors, vectors["concepts"][1][:len(candidates)]), "cumulative", "native_context", details, output,
        original, store)
    frames.extend(new_frames)
    results["native_context_concepts"] = result
    np.save(output / f"native_priorities_{name}.npy", priorities)
    pd.concat(frames, ignore_index=True).to_csv(target, index=False, compression="gzip")
    atomic_json(complete, {"source_identity": identity, "comparisons": results, "corrections": CORRECTIONS,
                           "design": "one change at a time against a corrected cumulative arm; not a full factorial experiment"})


def run_metrics_only(output=OUTPUT, original=ORIGINAL):
    """Cheap, CPU-only decisions for reference/year cells already completed by the original run."""
    for record in primary_builds(measures.load_index()):
        files = list(original.glob(f"development_{record['build']}_{record['reference']}_*.json"))
        if not files:
            continue
        reference = predictor.open_reference(record["build"], record["reference"], "cpu")
        for path in sorted(files):
            update = int(path.stem.rsplit("_", 1)[1])
            print(f"Cached metric comparison: reference {record['reference']}, update {update}", flush=True)
            metric_comparison(reference, update, output, original)


def run_all(output=OUTPUT, original=ORIGINAL):
    import torch
    torch.set_num_threads(8)
    for record in primary_builds(measures.load_index()):
        reference = predictor.open_reference(record["build"], record["reference"], "cpu")
        _, records = original_study.reference_info(record["build"], record["reference"])
        states = native_concepts(reference, records, output, original)
        for update_year in range(record["reference"], 2014):
            run_update(reference, update_year, output, original, states)
