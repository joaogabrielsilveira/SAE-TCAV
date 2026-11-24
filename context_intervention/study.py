"""Chronological development of context choices and next-year predictions."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from concept_study import measures, paths
from context_intervention import data, experiment, predictor, representations
from context_intervention import concepts as concept_analysis
from context_intervention import sampling
from context_intervention.analysis import primary_builds
from context_intervention.evaluation import choose_threshold


def reference_info(build, year):
    index = measures.load_index()
    record = next(item for item in primary_builds(index) if item["build"] == build and item["reference"] == year)
    concepts = [item for item in index["concepts"] if item["build"] == build and item["reference"] == year]
    return record, concepts


def run_update(reference, update_year, output, sample_size=500, draws=4):
    """Develop on outcomes through update_year, then score held-out year + 1."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    build, ref = reference["build"], reference["reference"]
    record, concepts = reference_info(build, ref)
    threshold = float(record["death_threshold"]["last_3"])
    table = measures.load_build(record)
    reference_mortality = float(table["reference_mortality"])
    candidates = data.development_rows(reference, update_year, "development")
    validation = data.development_rows(reference, update_year, "validation")
    test = data.evaluation_rows(reference, update_year + 1)
    population = reference["population"]
    if not len(candidates) or not len(validation) or not len(test):
        raise ValueError("Development, validation and evaluation rows must be present")
    labels = population["y"]
    context = reference["context_rows"]
    source_files = sorted(Path("context_intervention").glob("*.py"))
    source_files += [Path("concept_subgroup_drift_pilot.py"), Path("tabpfn_model.py")]
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    input_files = [paths.PATIENT_TABLE / record["file"]]
    input_files += [paths.PATIENT_TABLE / "concepts" / item["file"] for item in concepts]
    input_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in input_files}
    dictionaries = {}
    for seed in reference["cache"].config["sae_seeds"]:
        unit = reference["cache"].load(f"S/{seed}/{ref}")
        dictionaries[str(seed)] = {key: unit[key] for key in ("sae_identity", "scaler_identity")}
    predictions_hash = hashlib.sha256()
    for name in ("row_index", "probabilities", "embeddings"):
        predictions_hash.update(np.ascontiguousarray(reference["predictor"][name]).tobytes())
    provenance = {"context_identity": reference["predictor"]["context_identity"],
                  "checkpoint_sha256": reference["predictor"]["checkpoint_sha256"],
                  "population_fingerprint": str(population["fingerprint"]),
                  "patient_table_index": measures.index_hash(), "sample_size": sample_size,
                  "draws": draws, "sources": source_hashes, "patient_tables": input_hashes,
                  "dictionaries": dictionaries, "cached_predictions": predictions_hash.hexdigest()}
    target = output / f"predictions_{build}_{ref}_{update_year}.csv.gz"
    concept_target = output / f"concepts_{build}_{ref}_{update_year}.csv.gz"
    details_path = output / f"development_{build}_{ref}_{update_year}.json"
    priorities_files = [output / f"priorities_{build}_{ref}_{update_year}_{name}.npy" for name in ("features", "concepts")]
    if details_path.exists():
        if json.loads(details_path.read_text()).get("provenance") != provenance:
            raise ValueError("Existing update was built from different sources or settings")
        if target.exists() and concept_target.exists() and all(path.exists() for path in priorities_files):
            return target
    development = np.concatenate([candidates, validation])
    vectors = representations.representations(reference, development, concepts, output / "embeddings")
    candidate_probability, _ = data.cached_predictor_rows(reference, candidates)
    context_patient_ids = np.unique(population["patient_id"][context])
    policy_details = {"provenance": provenance,
                      "counts": {"candidate_rows": len(candidates), "candidate_deaths": int(labels[candidates].sum()),
                                 "validation_rows": len(validation), "validation_deaths": int(labels[validation].sum()),
                                 "evaluation_rows": len(test),
                                 "original_context_rows": len(context),
                                 "evaluation_with_patient_history_in_context": int(np.isin(
                                     population["patient_id"][test], context_patient_ids).sum())}}
    context_keys, context_counts = np.unique(np.column_stack((population["year"][context], labels[context])),
                                              axis=0, return_counts=True)
    policy_details["context_composition"] = [{"year": int(key[0]), "outcome": int(key[1]), "rows": int(count)}
                                             for key, count in zip(context_keys, context_counts)]
    output_rows = []
    cache = output / "predictions"
    concept_frames = []
    timing_file = paths.OUTPUT / "extensions/error_analysis/rows.csv"
    timing_lookup = {}
    if timing_file.exists():
        timing = pd.read_csv(timing_file, usecols=["build", "reference", "row", "active_months", "death_month"])
        timing = timing[(timing["build"] == build) & (timing["reference"] == ref)]
        timing_lookup = {int(row.row): (row.active_months, row.death_month) for row in timing.itertuples()}

    def append_concept_comparison(name, priorities, form, strength):
        frame = concept_analysis.comparison(reference, test, priorities, form, strength, threshold, concepts,
                                            sample_size, draws)
        frame["build"], frame["reference"], frame["year"], frame["method"] = build, ref, update_year + 1, name
        concept_frames.append(frame)

    def append_predictions(name, probabilities, threshold_tuned):
        for seed, probability in zip(experiment.SEEDS, probabilities):
            for row, value in zip(test, probability):
                output_rows.append({"build": build, "reference": ref, "update_year": update_year,
                                    "year": update_year + 1, "patient_id": str(population["patient_id"][row]),
                                    "label": int(labels[row]), "method": name, "sampling_seed": seed,
                                    "probability": float(value), "threshold": threshold,
                                    "tuned_threshold": threshold_tuned, "reference_mortality": reference_mortality})

    zeros = np.zeros(len(context))
    original = experiment.intervention_predictions(reference, zeros, test, "original", 0, cache, sample_size, draws)
    original_validation = experiment.intervention_predictions(reference, zeros, validation, "original", 0, cache,
                                                               sample_size, draws)
    tuned = choose_threshold(labels[validation], original_validation.mean(axis=0), threshold, reference_mortality)
    append_predictions("original", original, tuned)
    append_concept_comparison("original", zeros, "original", 0.)
    uniform = experiment.intervention_predictions(reference, zeros, test, "uniform", 0, cache, sample_size, draws)
    uniform_validation = experiment.intervention_predictions(reference, zeros, validation, "uniform", 0, cache,
                                                              sample_size, draws)
    tuned = choose_threshold(labels[validation], uniform_validation.mean(axis=0), threshold, reference_mortality)
    append_predictions("uniform_sampling", uniform, tuned)

    for name in ("features", "concepts"):
        print(f"  Developing {name} neighborhoods", flush=True)
        context_vectors, query_vectors = vectors[name]
        neighborhoods, details = experiment.make_neighborhoods(
            context_vectors, query_vectors[:len(candidates)], population["year"][context], labels[context],
            labels[candidates], candidate_probability, threshold)
        for detail in details:
            if "representative_rows" in detail:
                selected = candidates[detail.pop("representative_rows")]
                representatives = []
                for row in selected:
                    active_months, death_month = timing_lookup.get(int(row), (np.nan, np.nan))
                    representatives.append({"population_row": int(row), "patient_id": str(population["patient_id"][row]),
                                            "year": int(population["year"][row]), "label": int(labels[row]),
                                            "model_feature_events": float(population["X"][row].sum()),
                                            "active_months": float(active_months) if np.isfinite(active_months) else None,
                                            "death_month": float(death_month) if np.isfinite(death_month) else None})
                detail["representatives"] = representatives
        priorities, effects = experiment.neighborhood_effects(reference, neighborhoods, candidates, threshold,
                                                               reference_mortality, cache, sample_size, draws)
        policy_details[name] = {"neighborhoods": details, "effects": effects, "interventions": {}}
        for form in ("direct", "hard_sampling", "graded_sampling"):
            print(f"  Validating {name}: {form}", flush=True)
            if len(np.unique(labels[candidates])) < 2 or len(np.unique(labels[validation])) < 2:
                choice = {"form": "original", "strength": 0., "reason": "development lacks both outcomes"}
                comparisons = []
            else:
                choice, comparisons = experiment.choose_intervention(reference, priorities, validation, form,
                                                                       threshold, reference_mortality, cache,
                                                                       sample_size, draws)
            if choice["form"] in ("direct", "hard_sampling"):
                retained = sampling.select_removals(population["year"][context], labels[context], priorities,
                                                    choice["strength"], sample_size)
                choice["removed_rows"] = int((~retained).sum())
            else:
                choice["removed_rows"] = 0
            choice["supplied_context_rows"] = (len(context) - choice["removed_rows"] if choice["form"] == "direct"
                                               else len(context) if choice["form"] == "original" else sample_size)
            policy_details[name]["interventions"][form] = {"choice": choice, "validation": comparisons}
            probabilities = experiment.intervention_predictions(reference, priorities, test, choice["form"],
                                                                 choice["strength"], cache, sample_size, draws)
            validation_predictions = experiment.intervention_predictions(reference, priorities, validation,
                                       choice["form"], choice["strength"], cache, sample_size, draws)
            tuned = choose_threshold(labels[validation], validation_predictions.mean(axis=0), threshold,
                                     reference_mortality)
            append_predictions(f"{name}_{form}", probabilities, tuned)
            append_concept_comparison(f"{name}_{form}", priorities, choice["form"], choice["strength"])
            if form == "direct" and choice["form"] == "direct":
                random = experiment.direct_random_control(reference, priorities, choice["strength"], test, cache,
                                                          sample_size)
                random_validation = experiment.direct_random_control(reference, priorities, choice["strength"],
                                                                      validation, cache, sample_size)
                tuned_random = choose_threshold(labels[validation], random_validation.mean(axis=0), threshold,
                                                 reference_mortality)
                append_predictions(f"{name}_random_direct", random, tuned_random)
        np.save(output / f"priorities_{build}_{ref}_{update_year}_{name}.npy", priorities)
        if update_year > ref:
            previous_path = output / f"development_{build}_{ref}_{ref}.json"
            previous_priorities = output / f"priorities_{build}_{ref}_{ref}_{name}.npy"
            if not previous_path.exists() or not previous_priorities.exists():
                raise ValueError("Run the reference-year update before later updates")
            prior_details = json.loads(previous_path.read_text())
            if prior_details.get("provenance") != provenance:
                raise ValueError("Reference-year policy source or settings changed; recompute it first")
            first = prior_details[name]["interventions"]
            fixed_priorities = np.load(previous_priorities, allow_pickle=False)
            for form in ("direct", "hard_sampling", "graded_sampling"):
                choice = first[form]["choice"]
                probabilities = experiment.intervention_predictions(reference, fixed_priorities, test, choice["form"],
                                                                     choice["strength"], cache, sample_size, draws)
                validation_predictions = experiment.intervention_predictions(
                    reference, fixed_priorities, validation, choice["form"], choice["strength"], cache,
                    sample_size, draws)
                tuned = choose_threshold(labels[validation], validation_predictions.mean(axis=0), threshold,
                                         reference_mortality)
                append_predictions(f"{name}_{form}_reference_fixed", probabilities, tuned)
                if form == "direct" and choice["form"] == "direct":
                    random = experiment.direct_random_control(reference, fixed_priorities, choice["strength"], test,
                                                              cache, sample_size)
                    random_validation = experiment.direct_random_control(reference, fixed_priorities,
                                                                          choice["strength"], validation, cache,
                                                                          sample_size)
                    tuned_random = choose_threshold(labels[validation], random_validation.mean(axis=0), threshold,
                                                    reference_mortality)
                    append_predictions(f"{name}_random_direct_reference_fixed", random, tuned_random)
    pd.DataFrame(output_rows).to_csv(target, index=False, compression="gzip")
    pd.concat(concept_frames, ignore_index=True).to_csv(concept_target, index=False, compression="gzip")
    temporary = details_path.with_name(details_path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(policy_details, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, details_path)
    return target


def run_all(output="stats/context_intervention", sample_size=500, draws=4):
    """Run updates in reference-year order; reuse only matching completed results."""
    import torch

    torch.set_num_threads(8)
    for record in primary_builds(measures.load_index()):
        build, year = record["build"], record["reference"]
        print(f"Opening {build} reference {year}", flush=True)
        reference = predictor.open_reference(build, year, "cpu")
        for update_year in range(year, 2014):
            print(f"Developing through {update_year}; evaluating {update_year + 1}", flush=True)
            target = run_update(reference, update_year, output, sample_size, draws)
            print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    run_all()
    from context_intervention.report import summarize
    print(json.dumps(summarize("stats/context_intervention"), indent=2), flush=True)
