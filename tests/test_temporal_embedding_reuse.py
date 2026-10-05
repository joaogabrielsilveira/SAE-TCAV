"""Temporal retained-row reuse contracts; execute on Fiji, never the laptop."""

from dataclasses import replace
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from comparison_runner import ComparisonRunnerConfig, DefaultComparisonAdapter


@pytest.fixture
def extraction_case(monkeypatch):
    import tabpfn_model

    retained = np.array(
        [[2., 3.], [5., 7.], [11., 13.], [17., 19.], [23., 29.], [31., 37.]]
    )
    years = np.array([2007, 2007, 2008, 2008, 2009, 2010])
    mapping = np.array([4, 0, 2], dtype=np.int64)
    prepared = SimpleNamespace(
        X_train=retained[mapping].copy(),
        y_train=np.array([0, 1, 0]),
        years_train=years[mapping].copy(),
        X_test=retained.copy(),
        years_test=years.copy(),
        domain_reference_year=2009,
    )
    model = object()
    fit = dict(
        model=model, model_add_x_device="cpu", example_add_shape=None,
        fit_time_sec=0.,
    )
    base = ComparisonRunnerConfig()
    config = replace(
        base, use_cache=False, show_progress=False,
        accelerator=replace(base.accelerator, device="cpu"),
        tabpfn=replace(base.tabpfn, batch_size=2, run_walkforward=False),
    )
    domains = {2007: 0, 2008: 1, 2009: 2, 2010: 3}
    calls = []

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("An injected temporal fit must never be refitted")

    def extract(**kwargs):
        assert kwargs["model"] is model
        calls.append(dict(
            X=np.asarray(kwargs["X"]).copy(),
            years=np.asarray(kwargs["years"]).copy(),
            domains=dict(kwargs["year_to_domain_map"]),
            batch_size=kwargs["cfg"].batch_size,
            strict_domains=kwargs["cfg"].strict_domains,
            is_train=kwargs["is_train"],
            ctx_idx=kwargs["ctx_idx"],
            device=kwargs["device"],
            example_add_shape=kwargs["example_add_shape"],
        ))
        X = np.asarray(kwargs["X"])
        return np.column_stack((X, X[:, 0] ** 2 + X[:, 1]))

    monkeypatch.setattr(tabpfn_model, "fit_dr_tabpfn", forbidden_fit)
    monkeypatch.setattr(tabpfn_model, "extract_embeddings_robust", extract)
    monkeypatch.setattr(tabpfn_model, "flatten_embeddings", lambda value: value)
    return SimpleNamespace(
        prepared=prepared, fit=fit, config=config, domains=domains,
        mapping=mapping, calls=calls, discovery=np.array([1, 3]),
    )


def _require_mapping_contract():
    # Fail explicitly for the missing contract, rather than a keyword TypeError.
    assert "train_to_retained_indices" in inspect.signature(
        DefaultComparisonAdapter.embeddings
    ).parameters, "Temporal embeddings must accept an explicit retained-row mapping"


def _embeddings(case, workspace, **options):
    workspace.mkdir()
    return DefaultComparisonAdapter(None).embeddings(
        case.prepared, {"idx_semantic_fit": case.discovery},
        case.config, workspace, force=False, fitted_state=case.fit,
        explicit_domain_map=case.domains, fitted_identity="immutable-test-fit",
        **options,
    )


def test_unmapped_injected_fit_keeps_independent_extraction(extraction_case, tmp_path):
    case = extraction_case
    result = _embeddings(case, tmp_path / "independent")
    assert len(case.calls) == 2
    np.testing.assert_array_equal(case.calls[0]["X"], case.prepared.X_train)
    np.testing.assert_array_equal(case.calls[1]["X"], case.prepared.X_test)
    assert result.require_model() is case.fit["model"]


def test_mapped_training_export_skips_queries_and_preserves_retained_scaling(
    extraction_case, tmp_path,
):
    _require_mapping_contract()
    case = extraction_case
    baseline = _embeddings(case, tmp_path / "independent")
    case.calls.clear()
    result = _embeddings(
        case, tmp_path / "derived", train_to_retained_indices=case.mapping,
    )

    assert len(case.calls) == 1, "Only the unchanged retained query may be extracted"
    call = case.calls[0]
    np.testing.assert_array_equal(call["X"], case.prepared.X_test)
    np.testing.assert_array_equal(call["years"], case.prepared.years_test)
    assert call["domains"] == case.domains
    assert call["batch_size"] == 2 and call["strict_domains"] is True
    assert call["is_train"] is True and call["ctx_idx"] is None
    assert call["device"] == "cpu" and call["example_add_shape"] is None
    assert result.require_model() is case.fit["model"]
    np.testing.assert_array_equal(result.test_raw, baseline.test_raw)
    np.testing.assert_array_equal(result.test_scaled, baseline.test_scaled)
    np.testing.assert_array_equal(result.train_raw, result.test_raw[case.mapping])
    np.testing.assert_array_equal(result.train_scaled, baseline.train_scaled)
    np.testing.assert_array_equal(result.scaler.mean_, baseline.scaler.mean_)
    np.testing.assert_array_equal(result.scaler.scale_, baseline.scaler.scale_)
    np.testing.assert_array_equal(
        result.scaler.mean_, result.test_raw[case.discovery].mean(axis=0),
    )
    assert not np.shares_memory(result.train_raw, result.test_raw)


@pytest.mark.parametrize("mapping", [
    np.array([4, 0]),
    np.array([[4, 0, 2]]),
    np.array([4., 0., 2.]),
    np.array([True, False, True]),
    np.array([4, 0, 4]),
    np.array([-1, 0, 2]),
    np.array([6, 0, 2]),
    np.array(["4", "0", "2"]),
], ids=["length", "rank", "float", "bool", "duplicate", "negative", "bounds", "string"])
def test_invalid_mapping_rejected_before_inference(extraction_case, tmp_path, mapping):
    _require_mapping_contract()
    case = extraction_case
    with pytest.raises(ValueError, match="(?i)(train|mapping|retained|indices)"):
        _embeddings(case, tmp_path / "invalid", train_to_retained_indices=mapping)
    assert case.calls == []


@pytest.mark.parametrize("mismatch", ["features", "years"])
def test_mapped_row_disagreement_rejected_before_inference(
    extraction_case, tmp_path, mismatch,
):
    _require_mapping_contract()
    case = extraction_case
    if mismatch == "features":
        case.prepared.X_train[0, 0] += 1.
    else:
        case.prepared.years_train[0] = 2008
    with pytest.raises(ValueError, match="(?i)(train|mapping|retained|feature|year)"):
        _embeddings(case, tmp_path / "mismatch", train_to_retained_indices=case.mapping)
    assert case.calls == []


def test_mapping_without_injected_fit_rejected_before_inference(extraction_case, tmp_path):
    _require_mapping_contract()
    case = extraction_case
    with pytest.raises(ValueError, match="(?i)(inject|fit|mapping|retained)"):
        DefaultComparisonAdapter(None).embeddings(
            case.prepared, {"idx_semantic_fit": case.discovery}, case.config,
            tmp_path, force=False, train_to_retained_indices=case.mapping,
        )
    assert case.calls == []


def test_imported_arrays_win_over_valid_mapping_without_rewriting_them(
    extraction_case, tmp_path,
):
    _require_mapping_contract()
    case = extraction_case
    retained = np.arange(18, dtype=float).reshape(6, 3)
    # Deliberately unlike mapped retained rows: legacy values must remain intact.
    train = np.full((3, 3), 123.5)
    train_before, retained_before = train.copy(), retained.copy()
    result = _embeddings(
        case, tmp_path / "imported", train_to_retained_indices=case.mapping,
        imported_raw=(train, retained),
    )
    assert case.calls == []
    np.testing.assert_array_equal(result.train_raw, train_before)
    np.testing.assert_array_equal(result.test_raw, retained_before)
    np.testing.assert_array_equal(train, train_before)
    np.testing.assert_array_equal(retained, retained_before)


class _MappingCaptured(RuntimeError):
    pass


@pytest.fixture
def production_case(monkeypatch, tmp_path):
    import pandas as pd
    from temporal_config import TemporalRobustnessConfig
    from temporal_production import ProductionTemporalAdapter

    adapter = ProductionTemporalAdapter()
    adapter._base_config = ComparisonRunnerConfig()
    adapter._source_prepared = SimpleNamespace(test_rows=pd.DataFrame({"row": range(6)}))
    population = SimpleNamespace(
        X=np.arange(12, dtype=float).reshape(6, 2),
        outcomes=np.array([0, 1, 0, 1, 0, 1]),
        years=np.array([2007, 2007, 2008, 2008, 2009, 2010]),
        feature_names=("a", "b"), patient_ids=np.array(list("abcdef")),
        record_keys=np.array([f"record-{i}" for i in range(6)]),
    )
    roles = {role: np.array([i]) for i, role in enumerate((
        "tabpfn_context", "sae_discovery", "rule_discovery",
        "rule_selection_cav", "t0_evaluation",
    ))}
    options = dict(
        population=population, reference_year=2009,
        split=SimpleNamespace(effective_seed=42), global_roles=roles,
        evaluation_indices=np.array([5, 2, 4, 1, 0, 3]),
        training_indices=np.array([4, 0, 2]),
        domain_map={2007: 0, 2008: 1, 2009: 2, 2010: 3},
        config=TemporalRobustnessConfig(sae_seeds=(42, 43, 44)), workspace=tmp_path,
        fitted_state=dict(model=object()), fitted_identity="immutable-test-fit",
    )
    calls = []

    def capture(self, prepared, splits, config, workspace, **kwargs):
        calls.append((prepared, splits, config, kwargs))
        raise _MappingCaptured()

    monkeypatch.setattr(DefaultComparisonAdapter, "embeddings", capture)
    return SimpleNamespace(adapter=adapter, options=options, calls=calls)


@pytest.mark.parametrize("use_explicit_training", [True, False])
def test_temporal_caller_passes_exact_global_identity_mapping(production_case, use_explicit_training):
    case = production_case
    if not use_explicit_training:
        case.options["training_indices"] = None
    with pytest.raises(_MappingCaptured):
        case.adapter.run_reference_experiment(**case.options)
    prepared, splits, config, kwargs = case.calls[0]
    expected = np.array([2, 4, 1]) if use_explicit_training else np.array([4])
    assert "train_to_retained_indices" in kwargs, "Injected temporal fits must opt into identity reuse"
    np.testing.assert_array_equal(kwargs["train_to_retained_indices"], expected)
    np.testing.assert_array_equal(prepared.X_train, prepared.X_test[expected])
    np.testing.assert_array_equal(prepared.years_train, prepared.years_test[expected])
    assert kwargs["explicit_domain_map"] == case.options["domain_map"]
    assert tuple(config.sae.seeds) == (42, 43, 44)
    np.testing.assert_array_equal(splits["idx_semantic_fit"], [3])


@pytest.mark.parametrize("field,indices", [
    ("training_indices", np.array([4, 0, 4])),
    ("training_indices", np.array([4., 0., 2.])),
    ("training_indices", np.array([-1, 0, 2])),
    ("training_indices", np.array([6, 0, 2])),
    ("evaluation_indices", np.array([5, 2, 4, 1, 0, 0, 3])),
    ("evaluation_indices", np.array([5., 2., 4., 1., 0., 3.])),
], ids=["duplicate-train", "float-train", "negative", "bounds", "duplicate-retained", "float-retained"])
def test_temporal_caller_rejects_invalid_global_indices_before_embeddings(production_case, field, indices):
    case = production_case
    case.options[field] = indices
    with pytest.raises(ValueError, match="(?i)(identity|indices|membership|mapping|retained|train)"):
        case.adapter.run_reference_experiment(**case.options)
    assert case.calls == []


@pytest.mark.parametrize("problem", ["missing-train", "duplicate-record-key", "missing-domain"])
def test_temporal_caller_rejects_incomplete_identity_and_domains(production_case, problem):
    case = production_case
    if problem == "missing-train":
        case.options["evaluation_indices"] = np.array([0, 1, 2, 3, 4])
        case.options["training_indices"] = np.array([0, 5])
    elif problem == "duplicate-record-key":
        case.options["population"].record_keys[3] = case.options["population"].record_keys[4]
    else:
        case.options["domain_map"].pop(2010)
    with pytest.raises(ValueError, match="(?i)(identity|indices|membership|contain|domain|record)"):
        case.adapter.run_reference_experiment(**case.options)
    assert case.calls == []


def test_non_injected_temporal_caller_does_not_opt_in(production_case):
    case = production_case
    case.options["fitted_state"] = None
    case.options["fitted_identity"] = None
    with pytest.raises(_MappingCaptured):
        case.adapter.run_reference_experiment(**case.options)
    assert "train_to_retained_indices" not in case.calls[0][3]


@pytest.mark.parametrize("mode,origin", [
    ("independent", "independent_queries_v1"),
    ("derived", "retained_rows_v1"),
    ("imported", "imported_preserved_v1"),
])
def test_embedding_export_records_actual_training_origin(extraction_case, tmp_path, mode, origin):
    import json
    from semantic_artifacts import array_fingerprint
    case = extraction_case
    options = {}
    if mode != "independent":
        options["train_to_retained_indices"] = case.mapping
    if mode == "imported":
        options["imported_raw"] = (np.full((3, 3), 123.5), np.arange(18.).reshape(6, 3))
    workspace = tmp_path / mode
    result = _embeddings(case, workspace, **options)
    provenance = getattr(result, "training_provenance", None)
    assert provenance is not None, "Embedding exports must identify the training-array origin"
    assert provenance["origin"] == origin
    if mode == "derived":
        assert provenance["mapping"]["indices"] == case.mapping.tolist()
        assert provenance["mapping"]["sha256"] == array_fingerprint(case.mapping)
    else:
        assert provenance["mapping"] is None
    persisted = json.loads((workspace / "embedding_scaler_provenance.json").read_text())
    assert persisted["training_embeddings"] == provenance


def _cached_embeddings(case, cache, workspace, **options):
    workspace.mkdir()
    result = DefaultComparisonAdapter(cache).embeddings(
        case.prepared, {"idx_semantic_fit": case.discovery}, case.config,
        workspace, force=False, fitted_state=case.fit,
        explicit_domain_map=case.domains, fitted_identity="immutable-test-fit", **options,
    )
    return result, {event.stage: event for event in cache.events[-2:]}


def test_independent_and_derived_caches_are_separate_and_derived_resumes(extraction_case, tmp_path):
    from comparison_cache import ComparisonCache
    case = extraction_case
    cache = ComparisonCache(tmp_path / "cache")
    _, independent = _cached_embeddings(case, cache, tmp_path / "independent")
    case.calls.clear()
    _, derived = _cached_embeddings(
        case, cache, tmp_path / "derived", train_to_retained_indices=case.mapping,
    )
    assert len(case.calls) == 1, "Derived policy must not reuse an independent-query cache"
    for stage in ("embeddings_raw", "embeddings_scaled"):
        assert independent[stage].key != derived[stage].key
    case.calls.clear()
    _, resumed = _cached_embeddings(
        case, cache, tmp_path / "resumed", train_to_retained_indices=case.mapping,
    )
    assert case.calls == []
    assert all(event.status == "hit" for event in resumed.values())


def test_mapping_identity_separates_caches_even_when_features_and_outputs_match(extraction_case, tmp_path):
    from comparison_cache import ComparisonCache
    case = extraction_case
    case.prepared.X_test[0] = case.prepared.X_test[4]
    case.prepared.years_test[0] = case.prepared.years_test[4]
    case.prepared.X_train = case.prepared.X_test[case.mapping].copy()
    case.prepared.years_train = case.prepared.years_test[case.mapping].copy()
    cache = ComparisonCache(tmp_path / "cache")
    first, events1 = _cached_embeddings(
        case, cache, tmp_path / "first", train_to_retained_indices=case.mapping,
    )
    mapping2 = np.array([0, 4, 2], dtype=np.int64)
    case.calls.clear()
    second, events2 = _cached_embeddings(
        case, cache, tmp_path / "second", train_to_retained_indices=mapping2,
    )
    np.testing.assert_array_equal(first.train_raw, second.train_raw)
    np.testing.assert_array_equal(first.test_raw, second.test_raw)
    assert len(case.calls) == 1, "Record mapping must bind cache identity, not feature similarity"
    for stage in ("embeddings_raw", "embeddings_scaled"):
        assert events1[stage].key != events2[stage].key


def test_mapping_validation_helper_change_invalidates_raw_cache(extraction_case, tmp_path, monkeypatch):
    import comparison_runner
    from comparison_cache import ComparisonCache
    case = extraction_case
    cache = ComparisonCache(tmp_path / "cache")
    _, before = _cached_embeddings(
        case, cache, tmp_path / "before", train_to_retained_indices=case.mapping,
    )
    original = comparison_runner._validate_train_to_retained_indices

    def reviewed_mapping_check(prepared, indices):
        return original(prepared, indices)

    monkeypatch.setattr(comparison_runner, "_validate_train_to_retained_indices", reviewed_mapping_check)
    case.calls.clear()
    _, after = _cached_embeddings(
        case, cache, tmp_path / "after", train_to_retained_indices=case.mapping,
    )
    assert len(case.calls) == 1, "Mapping-validation source must bind the extraction cache"
    assert before["embeddings_raw"].key != after["embeddings_raw"].key


@pytest.mark.parametrize("origin", ["independent_queries_v1", "retained_rows_v1"])
def test_adapter_preserves_validated_import_origin_and_arrays(extraction_case, tmp_path, origin):
    from semantic_artifacts import array_fingerprint
    from temporal_handoff import ImportedRaw
    case = extraction_case
    retained = np.arange(18.).reshape(6, 3)
    mapping = None if origin == "independent_queries_v1" else {
        "indices": case.mapping.tolist(), "sha256": array_fingerprint(case.mapping),
    }
    source_provenance = {"origin": origin, "mapping": mapping}
    train = np.full((3, 3), 123.5) if mapping is None else retained[case.mapping].copy()
    raw = ImportedRaw(train, retained, source_provenance)
    result = _embeddings(
        case, tmp_path / "preserved", imported_raw=raw,
        train_to_retained_indices=case.mapping,
    )
    assert case.calls == []
    assert result.training_provenance == source_provenance
    np.testing.assert_array_equal(result.train_raw, train)
    np.testing.assert_array_equal(result.test_raw, retained)
    result.training_provenance["origin"] = "changed-result-only"
    assert raw.training_provenance == source_provenance


def test_adapter_rejects_imported_derived_mapping_for_a_different_destination(extraction_case, tmp_path):
    from semantic_artifacts import array_fingerprint
    from temporal_handoff import ImportedRaw
    case = extraction_case
    retained = np.arange(18.).reshape(6, 3)
    other = np.array([0, 4, 2], dtype=np.int64)
    provenance = {"origin": "retained_rows_v1", "mapping": {
        "indices": other.tolist(), "sha256": array_fingerprint(other),
    }}
    raw = ImportedRaw(retained[other].copy(), retained, provenance)
    with pytest.raises(ValueError, match="(?i)(provenance|mapping)"):
        _embeddings(case, tmp_path / "mismatch", imported_raw=raw, train_to_retained_indices=case.mapping)
    assert case.calls == []
