"""Priority runner contracts using synthetic mocks only; execute on Fiji."""

import itertools
import json
from pathlib import Path

import pytest

import run_temporal_concept_forecasting as runner
from temporal_handoff import EmbeddingsReady


@pytest.fixture
def runner_case(monkeypatch, tmp_path):
    calls, stage_a = [], []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "original_inputs", lambda *args: {})

    def forecast(*args):
        stage_a.append(args)
        return tmp_path / "forecast.json"

    monkeypatch.setattr(runner, "build_forecasting", forecast)
    monkeypatch.setattr(runner, "build_report", lambda *args, **kwargs: None)

    def stop_at_first_fit(*args, **kwargs):
        calls.append(kwargs)
        raise EmbeddingsReady(tmp_path / "mock-handoff.json")

    monkeypatch.setattr(runner, "run_window_concepts", stop_at_first_fit)
    return dict(calls=calls, stage_a=stage_a, tmp=tmp_path, argv=[
        "--repo", str(tmp_path), "--output", str(tmp_path / "out"),
        "--window-output", str(tmp_path / "windows"), "--device", "cpu",
    ])


def test_runner_defaults_to_priority_grid_without_excluded_pilot(runner_case):
    case = runner_case
    assert runner.main(case["argv"]) == 0
    assert len(case["calls"]) == 1
    call = case["calls"][0]
    assert not call.get("pilot", False), "Priority execution must not launch the excluded 2015 pilot"
    expected = list(itertools.product(range(2007, 2015), (42,), ("reference_only_common", "last_3")))
    assert call.get("jobs") == expected, "Default Stage B must be exactly the requested 16 logical jobs"


def test_runner_canonicalizes_explicit_selection(runner_case):
    case = runner_case
    argv = case["argv"] + [
        "--reference-years", "2014", "2007", "2014",
        "--windows", "last_3", "reference_only_common", "last_3",
        "--patient-split-seeds", "42", "42",
    ]
    assert runner.main(argv) == 0
    assert case["calls"][0]["jobs"] == [
        (2007, 42, "reference_only_common"), (2007, 42, "last_3"),
        (2014, 42, "reference_only_common"), (2014, 42, "last_3"),
    ]


def test_runner_can_explicitly_request_original_wider_grid(runner_case):
    case = runner_case
    argv = case["argv"] + [
        "--reference-years", *map(str, range(2007, 2016)),
        "--windows", "reference_only_common", "last_3", "all_history",
        "--patient-split-seeds", "42", "43", "44",
    ]
    assert runner.main(argv) == 0
    call = case["calls"][0]
    assert not call.get("pilot", False)
    assert len(call["jobs"]) == 81
    assert (2015, 44, "all_history") in call["jobs"]


def test_runner_forwards_read_only_completed_reuse_source(runner_case):
    case = runner_case
    source = case["tmp"] / "old-window-output"
    source.mkdir()
    assert runner.main(case["argv"] + ["--reuse-completed-from", str(source)]) == 0
    assert case["calls"][0].get("reuse_completed_from") == source
    assert list(source.iterdir()) == []


@pytest.mark.parametrize("flags", [
    ["--reference-years", "2006"], ["--reference-years", "2016"],
    ["--reference-years"], ["--windows", "unknown"], ["--windows"],
    ["--patient-split-seeds", "0"], ["--patient-split-seeds", "45"],
    ["--patient-split-seeds"],
])
def test_runner_rejects_invalid_scope_before_scientific_work(runner_case, flags):
    case = runner_case
    with pytest.raises(SystemExit) as stopped:
        runner.main(case["argv"] + flags)
    assert stopped.value.code == 2
    assert case["calls"] == [] and case["stage_a"] == []
    assert not (case["tmp"] / "out" / "execution_status.json").exists()


def test_empty_jobs_rejected_before_reading_artifacts(monkeypatch, tmp_path):
    import temporal_window_concepts as module
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid scope reached artifact loading")
    monkeypatch.setattr(module, "checked_manifest", forbidden)
    with pytest.raises(ValueError, match="nonempty"):
        module.run_window_concepts(tmp_path, tmp_path / "out", jobs=[], device="cpu")


@pytest.mark.parametrize("jobs", [
    [(2006, 42, "last_3")], [(2016, 42, "last_3")],
    [(2007, True, "last_3")], [(2007, 45, "last_3")],
    [(2007.0, 42, "last_3")], [(2007, 42, "unknown")],
    [(2007, 42)], [None],
])
def test_invalid_jobs_reject_before_scientific_work(monkeypatch, tmp_path, jobs):
    import temporal_window_concepts as module
    monkeypatch.setattr(module, "checked_manifest", lambda *args: pytest.fail("Scientific work started"))
    with pytest.raises(ValueError):
        module.run_window_concepts(tmp_path, tmp_path / "out", jobs=jobs, device="cpu")


@pytest.fixture
def window_case(monkeypatch, tmp_path):
    import json
    import numpy as np
    from types import SimpleNamespace
    import temporal_window_concepts as module
    import temporal_config
    import temporal_production
    import temporal_performance_windows as windows
    import comparison_runner
    import temporal_metric_synthesis
    from artifact_storage import atomic_write_json, file_sha256
    from temporal_concept_forecasting import checked_manifest, write_table

    population = SimpleNamespace(
        years=np.repeat(np.arange(2007, 2016), 3),
        outcomes=np.zeros(27, dtype=int),
        patient_ids=np.array([f"patient-{i}" for i in range(27)]),
        record_keys=np.array([f"record-{i}" for i in range(27)]),
        X=np.zeros((27, 2)), feature_names=("a", "b"), validate=lambda: None,
    )
    parent = {"config": dict(comparison_config_path="compare.yaml",
        semantic_config_path="semantic.yaml", dataset_path="data.csv"),
        "dependent_config_fingerprints": {
            name: {"sha256": value} for name, value in (
                ("comparison_config_path", "compare.yaml"), ("semantic_config_path", "semantic.yaml"))},
        "population_fingerprints": {}}
    wm = {"config": {}, "dependency_sha256": {
        "parent_manifest": "parent_manifest.json", "tabpfn_checkpoint": {"sha256": "checkpoint"}}}
    def manifest(path, *args):
        if Path(path).name == "parent_manifest.json":
            return parent
        if Path(path).parent.name == module.WINDOWS:
            return wm
        return checked_manifest(path, *args)
    monkeypatch.setattr(module, "checked_manifest", manifest)
    monkeypatch.setattr(module, "file_sha256",
        lambda path: file_sha256(path) if Path(path).is_file() else Path(path).name)
    monkeypatch.setattr(module, "configure_logging", lambda *args: None)
    monkeypatch.setattr(temporal_config.TemporalRobustnessConfig, "from_dict",
        lambda cfg: SimpleNamespace(**cfg))
    monkeypatch.setattr(windows.WindowExperimentConfig, "from_dict",
        lambda cfg: SimpleNamespace(**cfg))
    loader = SimpleNamespace(_base_config=SimpleNamespace(tabpfn=SimpleNamespace(model_name="mock")),
        load_retained_population=lambda *args: population)
    monkeypatch.setattr(temporal_production, "ProductionTemporalAdapter", lambda: loader)
    monkeypatch.setattr(comparison_runner, "_tabpfn_checkpoint_fingerprint",
        lambda *args: {"sha256": "checkpoint"})
    monkeypatch.setattr(temporal_metric_synthesis, "_numerical_environment", lambda *args: {"mock": "cpu"})
    monkeypatch.setattr(windows, "post_death_exclusion_mask",
        lambda *args: (np.ones(27, dtype=bool), []))
    def roles(ref):
        start = (ref - 2007) * 3
        return {name: np.array([start + offset]) for offset, name in enumerate(module.CONCEPT_ROLES)}
    monkeypatch.setattr(module, "_reference_roles", lambda pp, parent, ref, seed: roles(ref))
    monkeypatch.setattr(module, "historical_roles",
        lambda population, ref, seed, reference, years, keep: (reference, []))
    monkeypatch.setattr(windows, "build_training_indices",
        lambda **kw: np.flatnonzero(np.isin(population.years,
            windows.effective_window_years(kw["logical_window"], kw["reference_year"]))))
    monkeypatch.setattr(module, "cohort_masks",
        lambda population, ref, evaluation, *args: {(ref, "all_comer"): np.ones(len(evaluation), dtype=bool)})
    completed_calls = []
    def reuse(output, inputs, job, workspace, identity, *args):
        completed_calls.append((inputs, job, identity))
        artifacts = {"performance": write_table(workspace, "performance", [
            {"reference_year": job["ref"], "patient_split_seed": job["seed"], "test_year": job["ref"]}])}
        atomic_write_json(workspace / "completed.json",
            {"complete": True, "identity": identity, "artifacts": artifacts})
        return True
    monkeypatch.setattr(module, "reuse_reviewed_completed_job", reuse)
    monkeypatch.setattr(windows.ProductionWindowAdapter, "fit_predict",
        lambda *args, **kwargs: pytest.fail("Synthetic completed fit was recomputed"))
    def run(jobs=None, pilot=False, directory="out"):
        path = module.run_window_concepts(tmp_path, tmp_path / directory,
            jobs=jobs, pilot=pilot, device="cpu")
        return path, json.loads(path.read_text())
    return SimpleNamespace(run=run, completed=completed_calls, module=module)


def test_selected_manifest_binds_scope_and_preserves_2007_alias(window_case):
    case = window_case
    jobs = case.module.select_window_jobs(range(2007, 2015), ("reference_only_common", "last_3"), (42,))
    path, manifest = case.run(jobs)
    assert manifest.get("selection") == [list(job) for job in jobs], "Manifest must persist exact canonical scope"
    assert manifest["logical_jobs"] == 16 and manifest["distinct_fits"] == 15
    assert manifest["systems"] == ["last_3", "reference_only_common"]
    assert len(case.completed) == 15
    import json
    inputs = json.loads((path.parent / "run_identity.json").read_text())["inputs"]
    assert inputs["selection"] == manifest["selection"]


def test_scope_changes_aggregate_identity_not_equivalent_fit_identity(window_case):
    case = window_case
    path_a, _ = case.run([(2007, 42, "reference_only_common")])
    identity_a = case.completed[-1][2]
    path_b, manifest_b = case.run([(2007, 42, "last_3")])
    assert path_a.parent != path_b.parent, "Different logical scope must not return an existing aggregate"
    assert case.completed[-1][2] == identity_a
    assert manifest_b["systems"] == ["last_3"]


def test_lower_level_default_and_explicit_pilot_remain_supported(window_case):
    case = window_case
    _, full = case.run()
    _, pilot = case.run(pilot=True, directory="pilot")
    assert full["logical_jobs"] == 81
    assert pilot["logical_jobs"] == 2 and pilot["pilot"] is True


def test_explicit_jobs_are_deduplicated_without_cartesian_expansion(window_case):
    jobs = [(2014, 42, "last_3"), (2007, 42, "reference_only_common"), (2014, 42, "last_3")]
    _, manifest = window_case.run(jobs)
    assert manifest["logical_jobs"] == 2
    assert manifest.get("selection") == [[2007, 42, "reference_only_common"], [2014, 42, "last_3"]]



def _snapshot(root):
    from artifact_storage import file_sha256
    return {str(p.relative_to(root)): (file_sha256(p), p.stat().st_mtime_ns)
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def _write_prior_fits(root, captured, sources):
    """Rebuild captured per-fit inputs as a finished prior run recorded under other sources."""
    from artifact_storage import atomic_write_json
    from temporal_concept_forecasting import digest, write_table
    written = []
    for inputs, job, _ in captured:
        prior_inputs = {**inputs, "sources": sources}
        identity = digest(prior_inputs)[:20]
        run = root / f"window_concepts_{identity}"
        source = run / "fits" / digest({"run": identity, **job})[:24] / f"reference_{job['ref']}" / f"split_{job['seed']}"
        source.mkdir(parents=True)
        atomic_write_json(run / "run_identity.json", {"identity": identity, "inputs": prior_inputs})
        rows = [{"reference_year": job["ref"], "patient_split_seed": job["seed"], "test_year": job["ref"]}]
        atomic_write_json(source / "completed.json", {"complete": True, "actual_device": "cpu",
            "identity": digest({"run": identity, **job})[:24], "effective_years": list(job["years"]),
            "artifacts": {"performance": write_table(source, "performance", rows)}})
        written.append(source)
    return written


REUSE_JOBS = [(2008, 42, "reference_only_common"), (2009, 42, "reference_only_common")]


@pytest.fixture
def prior_case(window_case, monkeypatch, tmp_path):
    case = window_case
    case.run(REUSE_JOBS, directory="learn")
    case.prior = tmp_path / "prior"
    case.captured = list(case.completed)
    monkeypatch.setattr(case.module, "reuse_reviewed_completed_job", lambda *args, **kwargs: False)
    case.fits = []
    def refuse(operation, *args, **kwargs):
        case.fits.append(args)
        raise RuntimeError("recompute started")
    monkeypatch.setattr(case.module, "run_with_cpu_fallback", refuse)
    case.reuse = lambda: case.module.run_window_concepts(tmp_path, tmp_path / "reused", jobs=REUSE_JOBS,
        device="cpu", reuse_completed_from=case.prior)
    return case


def test_reviewed_prior_root_supplies_completed_fits_without_fitting_or_source_writes(prior_case):
    case = prior_case
    _write_prior_fits(case.prior, case.captured, case.module._REVIEWED_COMPLETED_SOURCES)
    before = _snapshot(case.prior)
    import json
    path = case.reuse()
    manifest = json.loads(path.read_text())
    assert manifest["complete"] and manifest["logical_jobs"] == 2 and manifest["distinct_fits"] == 2
    assert case.fits == [], "Reused fits must not start model fitting, inference or concept extraction"
    assert _snapshot(case.prior) == before
    reused = [json.loads(p.read_text()) for p in path.parent.glob("fits/*/reference_*/split_*/completed.json")]
    assert len(reused) == 2 and all(r["reuse_basis"].startswith("reviewed_source_bundle") for r in reused)


def test_unreviewed_prior_sources_are_recomputed_not_reused(prior_case):
    case = prior_case
    sources = {**case.module._REVIEWED_COMPLETED_SOURCES, "tcav.py": "unreviewed"}
    _write_prior_fits(case.prior, case.captured, sources)
    with pytest.raises(RuntimeError, match="window extraction jobs failed"):
        case.reuse()
    assert len(case.fits) == 2


def test_damaged_matching_prior_artifact_aborts_before_any_later_job(prior_case):
    case = prior_case
    written = _write_prior_fits(case.prior, case.captured, case.module._REVIEWED_COMPLETED_SOURCES)
    (written[0] / "performance.jsonl.gz").write_bytes(b"corrupted")
    with pytest.raises(case.module.ReuseIntegrityError):
        case.reuse()
    assert case.fits == []
    assert not list((case.prior.parent / "reused").glob("window_concepts_*/fits/*/reference_2009"))


# ---- selected-scope report -------------------------------------------------

SELECTED_WINDOWS = ("reference_only_common", "last_3")


def _performance(system, reference, seed=42):
    shift = {"reference_only_common": 0., "last_3": .01, "all_history": .02}.get(system, 0.)
    return [dict(system=system, reference_year=reference, patient_split_seed=seed, test_year=year,
                 temporal_distance=year - reference, cohort_view="all_comer", cohort_definition="synthetic",
                 frozen_threshold=.2, valid=True, support_fingerprint=f"{reference}/{seed}/{year}",
                 death_f1_at_frozen_threshold=.6 - .03 * (year - reference) + shift,
                 death_average_precision=.4 - .02 * (year - reference) + shift,
                 brier_score=.1 + .01 * (year - reference) - shift)
            for year in range(reference, 2016)]


def _write_run(root, system, references, window_sha=None, drop_exclusions=(), extra_predictions=()):
    from artifact_storage import atomic_write_json
    from temporal_concept_forecasting import ForecastConfig, build_forecast_rows, write_table
    config = ForecastConfig(activation_targets=(.5,), profiles=("core",))
    performance = [row for reference in references for row in _performance(system, reference)]
    rows, exclusions = build_forecast_rows(performance, config)
    exclusions = [e for e in exclusions if e["reference_year"] not in drop_exclusions]
    predictions = [{**r, "design": "forward", "model": "no_change", "prediction": 0., "fold_id": "f"} for r in rows]
    predictions += list(extra_predictions)
    concepts = [dict(system=system, reference_year=r["reference_year"], patient_split_seed=r["patient_split_seed"],
                     test_year=r["test_year"], cohort_view=r["cohort_view"]) for r in performance]
    run = Path(root) / f"forecast_{system}"
    run.mkdir(parents=True)
    tables = dict(performance=performance, membership_audit=[], measured_concepts=concepts, forecast_rows=rows,
                  exclusions=exclusions, folds=[], predictions=predictions, summaries=[], comparisons=[])
    atomic_write_json(run / "manifest.json", {
        "schema_version": "concept_forecasting_v1", "complete": True, "system": system,
        "config": {"outcomes": list(config.outcomes)},
        "sources": {"window_concepts": window_sha} if window_sha else {"test": "synthetic"},
        "artifacts": {name: write_table(run, name, values) for name, values in tables.items()},
        "transforms": []})
    return run / "manifest.json"


def _window_manifest(root, references=range(2007, 2015), windows=SELECTED_WINDOWS, seeds=(42,), missing=(), legacy=False):
    from artifact_storage import atomic_write_json
    from temporal_concept_forecasting import write_table
    jobs = [(r, s, w) for r in references for s in seeds for w in windows]
    aliases = [dict(reference_year=r, patient_split_seed=s, window=w, fit_identity=f"fit-{r}-{s}-{w}",
                    completed_manifest="completed.json", completed_manifest_sha256="0", effective_years=[])
               for r, s, w in jobs if (r, s, w) not in missing]
    root = Path(root)
    root.mkdir(parents=True)
    manifest = {"schema_version": "window_concepts_test", "complete": True, "pilot": False,
                "systems": sorted(set(windows)), "failures": [],
                "artifacts": {"aliases": write_table(root, "aliases", aliases)}}
    if not legacy:
        manifest.update(selection=[list(job) for job in jobs], logical_jobs=len(jobs),
                        distinct_fits=len({a["fit_identity"] for a in aliases}))
    atomic_write_json(root / "manifest.json", manifest)
    return root / "manifest.json"


def _report_case(tmp_path, tag="a", references=range(2007, 2015), windows=SELECTED_WINDOWS, seeds=(42,),
                 missing_jobs=(), omit_systems=(), run_references=None, **run_options):
    window = _window_manifest(tmp_path / f"windows_{tag}", references, windows, seeds, missing_jobs)
    from artifact_storage import file_sha256
    sha = file_sha256(window)
    runs = tmp_path / f"runs_{tag}"
    paths = [_write_run(runs, "original", references)]
    paths += [_write_run(runs, system, run_references or references, sha, **run_options)
              for system in windows if system not in omit_systems]
    return dict(paths=paths, window=window, output=tmp_path / f"report_{tag}")


def _report_table(manifest_path, name):
    import json
    from temporal_concept_forecasting import table
    manifest = json.loads(Path(manifest_path).read_text())
    return table(manifest_path, manifest, name)


def test_selected_report_completes_without_all_history(tmp_path):
    import json
    from temporal_forecasting_report import build_report
    case = _report_case(tmp_path)
    path = build_report(case["paths"], case["output"], stage_b_complete=True, window_manifest=case["window"])
    report = json.loads(Path(path).read_text())
    assert report["complete"] is True and report["stage_b_complete"] is True
    scope = report["scope"]
    assert scope["systems"] == ["last_3", "reference_only_common"]
    assert scope["reference_years"] == list(range(2007, 2015)) and scope["patient_split_seeds"] == [42]
    assert scope["logical_jobs"] == 16 and len(scope["selection"]) == 16
    assert "all_history" not in {m["path"] for m in report["forecast_manifests"]}
    assert report["window_manifest"]["sha256"] == scope["window_manifest_sha256"]


def test_selected_report_fails_when_a_selected_system_is_missing(tmp_path):
    from temporal_forecasting_report import build_report
    case = _report_case(tmp_path, omit_systems=("last_3",))
    with pytest.raises(ValueError, match="last_3"):
        build_report(case["paths"], case["output"], stage_b_complete=True, window_manifest=case["window"])


def test_selected_report_fails_when_a_selected_job_is_missing(tmp_path):
    from temporal_forecasting_report import build_report
    case = _report_case(tmp_path, missing_jobs=[(2010, 42, "last_3")])
    with pytest.raises(ValueError, match="2010"):
        build_report(case["paths"], case["output"], stage_b_complete=True, window_manifest=case["window"])
    case = _report_case(tmp_path, tag="b", run_references=range(2007, 2010))
    with pytest.raises(ValueError, match="2010"):
        build_report(case["paths"], case["output"], stage_b_complete=True, window_manifest=case["window"])


def test_selected_report_never_adds_an_excluded_system(tmp_path):
    from temporal_forecasting_report import build_report
    case = _report_case(tmp_path)
    case["paths"].append(_write_run(tmp_path / "runs_extra", "all_history", range(2007, 2015)))
    with pytest.raises(ValueError, match="all_history"):
        build_report(case["paths"], case["output"], stage_b_complete=True, window_manifest=case["window"])


def test_selected_report_rejects_forecasts_from_another_window_manifest(tmp_path):
    from temporal_forecasting_report import build_report
    case = _report_case(tmp_path)
    other = _report_case(tmp_path, tag="other", references=range(2007, 2014))
    with pytest.raises(ValueError, match="window"):
        build_report(case["paths"][:2] + other["paths"][2:], case["output"], stage_b_complete=True,
                     window_manifest=case["window"])


def test_report_identity_differs_between_scopes(tmp_path):
    from temporal_forecasting_report import build_report
    wide = _report_case(tmp_path, tag="wide")
    narrow = _report_case(tmp_path, tag="narrow", references=range(2007, 2014))
    ids = [build_report(c["paths"], c["output"], stage_b_complete=True, window_manifest=c["window"]).parent.name
           for c in (wide, narrow)]
    assert ids[0] != ids[1], "A different scope must not reuse a report"


def test_report_identity_inputs_include_canonical_scope_and_window_checksum(tmp_path, monkeypatch):
    import json
    import temporal_forecasting_report as module
    case = _report_case(tmp_path)
    baseline = module.build_report(case["paths"], case["output"], stage_b_complete=True, window_manifest=case["window"])
    window = json.loads(case["window"].read_text())
    from artifact_storage import file_sha256
    seen = []
    real = module.digest
    monkeypatch.setattr(module, "digest", lambda value: seen.append(value) or real(value))
    module.build_report(case["paths"], tmp_path / "again", stage_b_complete=True, window_manifest=case["window"])
    identity = [value for value in seen if isinstance(value, list) and value and isinstance(value[-1], dict)]
    assert identity, "Report identity must carry a scope record"
    scope = identity[-1][-1]
    assert scope["window_manifest_sha256"] == file_sha256(case["window"])
    assert scope["selection"] == window["selection"]
    assert baseline.parent.name.startswith("report_")


def test_reference_2014_keeps_descriptive_records_and_publishes_explicit_exclusions(tmp_path):
    import json
    from temporal_forecasting_report import build_report
    case = _report_case(tmp_path)
    path = build_report(case["paths"], case["output"], stage_b_complete=True, window_manifest=case["window"])
    report = json.loads(Path(path).read_text())
    exclusions = [r for r in _report_table(path, "forecast_exclusions") if r["reference_year"] == 2014]
    for system in SELECTED_WINDOWS:
        rows = sorted((r["forecast_origin_year"], r["outcome"]) for r in exclusions if r["system"] == system)
        assert rows == sorted((year, outcome) for year in (2014, 2015) for outcome in ("frozen_f1", "average_precision", "brier"))
    by_year = {r["forecast_origin_year"]: r for r in exclusions if r["outcome"] == "frozen_f1" and r["system"] == "last_3"}
    assert by_year[2014]["reason"] == "missing_consecutive_performance" and by_year[2014]["missing_performance_years"] == [2013]
    assert by_year[2015]["missing_performance_years"] == [2016]
    assert all(r["explanation"] and "prediction" not in r for r in exclusions)
    eligibility = {(r["system"], r["reference_year"]): r for r in _report_table(path, "forecast_eligibility")}
    for system in SELECTED_WINDOWS:
        descriptive = eligibility[(system, 2014)]
        assert descriptive["status"] == "descriptive_only" and descriptive["forecast_rows"] == 0
        assert descriptive["performance_years"] == [2014, 2015] and descriptive["performance_rows"] == 2
        assert descriptive["measured_concept_rows"] == 2 and descriptive["exclusion_rows"] == 6
        assert eligibility[(system, 2013)]["status"] == "forecast_eligible"
    assert eligibility[("original", 2014)]["status"] == "descriptive_only"
    comparison = [r for r in _report_table(path, "window_comparisons")
                  if r["question"] == "performance_level" and r["outcome"] == "frozen_f1" and r["system"] == "last_3"]
    assert comparison and comparison[0]["paired_rows"] == sum(2016 - ref for ref in range(2007, 2015)), \
        "2014 performance must remain in descriptive window comparisons"
    assert report["forecast_eligibility"]["descriptive_only_references"] == [2014]
    assert report["limitations"] and any("single patient split" in text for text in report["limitations"])


def test_ineligible_forecast_is_never_published_as_a_prediction(tmp_path):
    from temporal_forecasting_report import build_report
    fake = {"system": "last_3", "reference_year": 2014, "patient_split_seed": 42, "cohort_view": "all_comer",
            "outcome": "frozen_f1", "forecast_origin_year": 2014, "model": "no_change", "prediction": 0.}
    case = _report_case(tmp_path)
    case["paths"][2] = _write_run(tmp_path / "bad", "last_3", range(2007, 2015),
                                  __import__("artifact_storage").file_sha256(case["window"]), extra_predictions=[fake])
    with pytest.raises(ValueError, match="ineligible"):
        build_report(case["paths"], case["output"], stage_b_complete=True, window_manifest=case["window"])


def test_ineligible_forecast_cannot_be_silently_dropped(tmp_path):
    from temporal_forecasting_report import build_report
    case = _report_case(tmp_path, drop_exclusions=(2014,))
    with pytest.raises(ValueError, match="silently"):
        build_report(case["paths"], case["output"], stage_b_complete=True, window_manifest=case["window"])


def test_legacy_full_grid_report_still_requires_and_publishes_four_systems(tmp_path):
    import json
    from temporal_concept_forecasting import digest
    from temporal_forecasting_report import build_report
    from artifact_storage import file_sha256
    refs = range(2007, 2010)
    runs = tmp_path / "legacy"
    names = ("original", "reference_only_common", "last_3", "all_history")
    paths = [_write_run(runs, name, refs) for name in names]
    with pytest.raises(ValueError, match="all four systems"):
        build_report(paths[:3], tmp_path / "short", stage_b_complete=True)
    path = build_report(paths, tmp_path / "full", stage_b_complete=True)
    expected = digest([file_sha256(p) for p in paths] + [True])[:20]
    assert path.parent.name == f"report_{expected}"
    report = json.loads(path.read_text())
    assert report["stage_b_complete"] is True and report.get("scope") is None
    systems = {r["system"] for r in _report_table(path, "window_comparisons")}
    assert systems == {"last_3", "all_history"}
    # A window manifest recorded before selections were persisted keeps the four-system rule.
    window = _window_manifest(tmp_path / "legacy_windows", refs, ("reference_only_common", "last_3", "all_history"), legacy=True)
    with pytest.raises(ValueError, match="all four systems"):
        build_report(paths[:3], tmp_path / "short2", stage_b_complete=True, window_manifest=window)
    assert build_report(paths, tmp_path / "full2", stage_b_complete=True, window_manifest=window)


# ---- artifact-only notebook ------------------------------------------------

NOTEBOOK = Path(__file__).resolve().parents[1] / "temporal_concept_forecasting_analysis.ipynb"
TRAINING_MODULES = ("torch", "comparison_runner", "temporal_window_concepts", "temporal_production",
                    "run_temporal_concept_forecasting", "temporal_handoff", "temporal_robustness_autoencoder",
                    "tabpfn_model", "tcav")
REUSE_BASIS = "reviewed_source_bundle_exact_scientific_inputs_and_membership"
REUSED_FITS = ("fit-2007-42", "fit-2008-42-reference_only_common", "fit-2008-42-last_3",
               "fit-2009-42-reference_only_common", "fit-2009-42-last_3",
               "fit-2010-42-reference_only_common", "fit-2010-42-last_3")


def _notebook_windows(root, reused=REUSED_FITS):
    """Window manifest whose 2007 jobs alias one fit and whose completed fits are real files."""
    from artifact_storage import atomic_write_json, file_sha256
    from temporal_concept_forecasting import write_table
    root = Path(root)
    jobs = [(r, 42, w) for r in range(2007, 2015) for w in SELECTED_WINDOWS]
    aliases = []
    for reference, seed, window in jobs:
        fit = f"fit-{reference}-{seed}" if reference == 2007 else f"fit-{reference}-{seed}-{window}"
        completed = root / "fits" / fit / "completed.json"
        if not completed.exists():
            record = {"complete": True, "identity": fit}
            if fit in reused:
                record.update(reuse_basis=REUSE_BASIS, reused_from=f"/synthetic/prior/{fit}/completed.json",
                              source_run_identity="window_concepts_synthetic", reused_manifest_sha256="a" * 64)
            atomic_write_json(completed, record)
        aliases.append(dict(reference_year=reference, patient_split_seed=seed, window=window, fit_identity=fit,
                            completed_manifest=str(completed.relative_to(root)),
                            completed_manifest_sha256=file_sha256(completed), effective_years=[]))
    atomic_write_json(root / "manifest.json", {
        "schema_version": "window_concepts_test", "complete": True, "pilot": False, "failures": [],
        "systems": sorted(SELECTED_WINDOWS), "selection": [list(job) for job in jobs], "logical_jobs": len(jobs),
        "distinct_fits": len({a["fit_identity"] for a in aliases}),
        "artifacts": {"aliases": write_table(root, "aliases", aliases)}})
    return root / "manifest.json"


def _selected_notebook_report(tmp_path):
    from artifact_storage import file_sha256
    from temporal_forecasting_report import build_report
    window = _notebook_windows(tmp_path / "windows")
    sha = file_sha256(window)
    paths = [_write_run(tmp_path / "runs", "original", range(2007, 2015))]
    paths += [_write_run(tmp_path / "runs", system, range(2007, 2015), sha) for system in SELECTED_WINDOWS]
    return build_report(paths, tmp_path / "report", stage_b_complete=True, window_manifest=window)


def _legacy_notebook_report(tmp_path):
    """Four-system report as written before scopes, eligibility tables and limitations existed."""
    from artifact_storage import atomic_write_json, file_sha256
    from temporal_forecasting_report import build_report
    names = ("original", "reference_only_common", "last_3", "all_history")
    paths = [_write_run(tmp_path / "runs", name, range(2007, 2010)) for name in names]
    path = build_report(paths, tmp_path / "report", stage_b_complete=True)
    manifest = json.loads(path.read_text())
    for name in ("scope", "contextual_systems", "forecast_eligibility", "limitations", "window_manifest"):
        manifest.pop(name)
    for name in ("forecast_eligibility", "forecast_exclusions"):
        manifest["artifacts"].pop(name)
    atomic_write_json(path, manifest)
    atomic_write_json(path.parent.parent / "report_manifest.json", dict(path=str(path), sha256=file_sha256(path)))
    return path


def _outputs_text(notebook):
    chunks = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                chunks.append(output.text)
            elif output.output_type == "error":
                chunks.append("\n".join(output.traceback))
            else:
                chunks.extend(output.data[mime] for mime in ("text/plain", "text/markdown", "text/html")
                              if mime in output.data)
    return "\n".join(chunks)


def _execute_notebook(report, workdir):
    """Run the notebook the way the runner does: parameter cell, CPU only, nothing outside workdir."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(f"REPORT_MANIFEST = {str(report)!r}"))
    notebook.cells.append(nbformat.v4.new_code_cell(
        "import sys, json\nprint('LOADED_TRAINING_MODULES=' + json.dumps(sorted(m for m in "
        f"{list(TRAINING_MODULES)!r} if m in sys.modules)))"))
    with pytest.MonkeyPatch.context() as env:
        for name, value in dict(CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1",
                                MPLCONFIGDIR=str(workdir / "mpl"), IPYTHONDIR=str(workdir / "ipython"),
                                JUPYTER_RUNTIME_DIR=str(workdir / "jupyter")).items():
            env.setenv(name, value)
        nbclient.NotebookClient(notebook, timeout=300, kernel_name="python3",
                                resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()
    text = _outputs_text(notebook)
    loaded = json.loads(text.split("LOADED_TRAINING_MODULES=", 1)[1].splitlines()[0])
    return text, loaded


@pytest.fixture(scope="module")
def selected_notebook(tmp_path_factory):
    root = tmp_path_factory.mktemp("selected_notebook")
    return _execute_notebook(_selected_notebook_report(root), root)


@pytest.fixture(scope="module")
def legacy_notebook(tmp_path_factory):
    root = tmp_path_factory.mktemp("legacy_notebook")
    return _execute_notebook(_legacy_notebook_report(root), root)


def test_notebook_shows_requested_and_completed_scope(selected_notebook):
    text, _ = selected_notebook
    assert "Reference years: 2007-2014" in text
    assert "Windows: last_3, reference_only_common" in text
    assert "Patient split seeds: 42" in text
    assert "Logical jobs: 16" in text and "Distinct fits: 15" in text
    assert "Completed jobs: 16 of 16 requested" in text
    assert "Completed distinct fits: 15" in text


def test_notebook_labels_original_stage_a_as_contextual(selected_notebook):
    text, _ = selected_notebook
    assert "Contextual Stage A systems: original" in text
    assert "not a new fit" in text


def test_notebook_shows_reuse_provenance_when_recorded(selected_notebook):
    text, _ = selected_notebook
    assert "Reused completed fits: 7 distinct fits across 8 of 16 logical jobs" in text
    assert REUSE_BASIS in text and "window_concepts_synthetic" in text
    for fit in REUSED_FITS:
        assert f"/synthetic/prior/{fit}/completed.json" in text
    assert "fit-2011-42-last_3/completed.json" not in text


def test_notebook_shows_2014_as_descriptive_only_with_explicit_exclusions(selected_notebook):
    text, _ = selected_notebook
    assert "Descriptive-only reference years: [2014]" in text
    assert "excluded_not_forecast" in text and "missing_consecutive_performance" in text
    assert "not forecasts and are not zero deterioration" in text
    assert "no_origin_year_with_previous_current_and_future_performance" in text


def test_notebook_states_single_split_and_sae_seed_limitations(selected_notebook):
    text, _ = selected_notebook
    assert "single patient split (42)" in text
    assert "SAE initialization seeds (42/43/44 in this protocol)" in text
    assert "do not replace patient-split replication" in text
    assert "previous, current and future performance" in text


def test_notebook_executes_without_loading_any_fitting_module(selected_notebook, legacy_notebook):
    assert selected_notebook[1] == [] and legacy_notebook[1] == []


def test_notebook_source_never_imports_or_calls_fitting_code():
    import ast
    import re
    source = json.loads(NOTEBOOK.read_text())
    code = "\n".join("".join(cell["source"]) for cell in source["cells"] if cell["cell_type"] == "code")
    imported = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert not imported & set(TRAINING_MODULES), imported
    assert not re.search(r"\b(build_report|run_window_concepts|build_forecasting|subprocess|\.fit\(|\.train\()", code)


def test_notebook_is_committed_without_outputs():
    source = json.loads(NOTEBOOK.read_text())
    code = [cell for cell in source["cells"] if cell["cell_type"] == "code"]
    assert code and all(cell["outputs"] == [] and cell["execution_count"] is None for cell in code)


def test_legacy_report_without_scope_still_renders(legacy_notebook):
    text, _ = legacy_notebook
    assert "Stage B complete: True" in text
    assert "Legacy report: no selected scope was recorded" in text
    assert "Logical jobs" not in text and "Descriptive-only reference years" not in text
    assert "No forecast-eligibility tables were recorded in this report" in text
    assert "Contextual Stage A systems: original" in text
