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
