from pathlib import Path

import numpy as np
import pytest

from temporal_metric_synthesis import (
    MEMBER_KEYS, MetricSynthesisConfig, MetricSynthesisRuntimeConfig, build_early_warning_rows,
    _event_summary, _latent_design, _ridge_oof, _split_noise_threshold, build_metric_vectors, dae_member_vectors, dimensionality, nested_reference_year_splits, system_concept_features,
)
from temporal_synthesis.dimensionality import metric_quality, varimax
from temporal_synthesis.execution import ResumeStore, SearchJob, select_fastest_benchmark


def _member(distance=0, cohort="all_comer", **changes):
    row = {"reference_year": 2007, "patient_split_seed": 42, "factor_family_uid": "f",
           "member_sae_seed": 42, "member_factor_id": 1, "activation_target": .5,
           "cohort_view": cohort, "temporal_distance": distance, "test_year": 2007 + distance,
           "u_f2": .9, "u_jaccard": .8, "u_prevalence": .7, "u_activation": .6}
    row.update(changes); return row


def _factor(member, cosine=.5):
    return {**member, "matching_view":"intersection", "rule_source":"semantic", "target_role":"primary", "feature_association_cosine": cosine, "feature_association_valid": True}


def _tcav(member, value=.6):
    return {**member, "matching_view":"intersection", "rule_source":"semantic", "target_role":"primary", "tcav": value, "tcav_valid": False, "p_value": .9, "q_value": .9}


def test_exact_member_join_orients_association_and_keeps_raw_invalid_tcav():
    zero, future = _member(), _member(1)
    vectors, audit = build_metric_vectors(
        {"headline_factor_metrics": [_factor(zero), _factor(future, -1)] , "tcav_significance": [_tcav(zero, .4), _tcav(future, .9)]},
        {"cri_member_utilities": [zero, future]},
    )
    result = vectors[1]
    assert result["u_feature_association"] == 0
    assert result["u_tcav"] == pytest.approx(.5)
    assert result["tcav_valid"] is False
    assert audit[1]["tcav_finite"] is True


def test_missing_exact_factor_mapping_fails_instead_of_row_order_fallback():
    with pytest.raises(ValueError, match="missing exact factor"):
        build_metric_vectors({"headline_factor_metrics": [], "tcav_significance": []}, {"cri_member_utilities": [_member()]})


def test_family_is_one_vote_and_extended_profile_is_p50_only():
    values=[]
    for family, utility in (("a", .2), ("b", .8)):
        row=_member(1, factor_family_uid=family, u_f2=utility, u_jaccard=utility, u_prevalence=utility, u_activation=utility, u_feature_association=utility, u_tcav=utility)
        values.extend([row, {**row, "member_sae_seed": 43, "member_factor_id": 2}])
    core=system_concept_features(values,"core")[0]
    assert core["u_f2"] == pytest.approx(.5)
    assert core["complete_family_count"] == 2
    assert system_concept_features([{**values[0],"activation_target":.7}],"p50_tcav_extended") == []


def test_early_warning_target_uses_current_minus_next_and_requires_three_times():
    features=[]; performance=[]
    for d, f1 in ((0,.9),(1,.8),(2,.5)):
        feature=_member(d, u_feature_association=.5, u_tcav=.5, concept_coverage=1.); features.append(feature)
        performance.append({"reference_year":2007,"patient_split_seed":42,"cohort_view":"all_comer","temporal_distance":d,"test_year":2007+d,"death_f1":f1,"variant":"original"})
    rows=build_early_warning_rows(performance,features,"core")
    assert len(rows)==1
    assert rows[0]["death_f1_degradation"] == pytest.approx(.3)
    assert rows[0]["previous_degradation"] == pytest.approx(.1)


def test_distance_zero_is_excluded_from_dimensionality_fit():
    rows=[]
    for i in range(8):
        rows.append({"reference_year":2007+i,"patient_split_seed":42,"cohort_view":"all_comer","activation_target":.5,"temporal_distance":0, **{x:1. for x in ("u_f2","u_jaccard","u_prevalence","u_activation","u_feature_association")}})
        rows.append({"reference_year":2007+i,"patient_split_seed":42,"cohort_view":"all_comer","activation_target":.5,"temporal_distance":1, **{x:(i+1)/10 for x in ("u_f2","u_jaccard","u_prevalence","u_activation","u_feature_association")}})
    spectrum, _, _, _ = dimensionality(rows,"core",MetricSynthesisConfig(parallel_repetitions=10))
    assert spectrum and spectrum[0]["eigenvalue"] >= 0


def test_dae_uses_complete_member_factor_vectors_not_system_medians():
    rows=[_member(0, u_feature_association=.5), _member(1, u_feature_association=.5),
          _member(1, member_sae_seed=43, member_factor_id=2, u_feature_association=.4),
          _member(2, u_feature_association=None)]
    selected=dae_member_vectors(rows,"core")
    assert len(selected) == 2
    assert {row["member_sae_seed"] for row in selected} == {42,43}


def test_pca_prediction_representation_is_fit_only_on_training_references():
    names=("u_f2", "u_jaccard", "u_prevalence", "u_activation", "u_feature_association")
    def row(reference, offset):
        return {"reference_year":reference, "death_f1_current":.7, "previous_degradation":.1,
                "temporal_distance":1, "concept_coverage":1., **{name:offset + index / 100 for index,name in enumerate(names)},
                **{f"delta_{name}":.01 for name in names}}
    training=[row(2007,.1),row(2007,.2),row(2008,.3),row(2008,.4)]
    held=[row(2009,.5)]
    first_train, first_held, feature_names=_latent_design(training,held,"core","pca2")
    changed_held=[row(2009,99.)]
    second_train, second_held, _=_latent_design(training,changed_held,"core","pca2")
    assert feature_names[-4:] == ["PC1_current","PC2_current","delta_PC1","delta_PC2"]
    assert np.allclose(first_train,second_train)
    assert not np.allclose(first_held,second_held)


def test_split_noise_threshold_deduplicates_activation_targets_and_uses_only_supplied_rows():
    def row(reference, seed, degradation, activation):
        return {"variant":"original", "reference_year":reference, "patient_split_seed":seed, "cohort_view":"all_comer",
                "temporal_distance":1, "target_year":reference + 1, "activation_target":activation,
                "death_f1_degradation":degradation}
    train=[row(2007,1,.00,.5),row(2007,1,.00,.7),row(2007,2,.10,.5),row(2008,1,.00,.5),row(2008,2,.20,.5)]
    threshold, groups, residuals=_split_noise_threshold(train,.95)
    assert groups == 2 and residuals == 4
    assert threshold == pytest.approx(.1)
    held=row(2009,1,.9,.5)
    assert _split_noise_threshold(train,.95)[0] == threshold  # a held-out row cannot alter a train-only threshold
    with pytest.raises(RuntimeError, match="at least two split seeds"):
        _split_noise_threshold([held],.95)


def test_supervised_variants_are_fit_separately_and_history_has_maximal_coverage_view():
    metrics=("u_f2", "u_jaccard", "u_prevalence", "u_activation", "u_feature_association")
    rows=[]
    for variant in ("original", "balanced_context"):
        for reference in range(2007,2011):
            rows.append({"variant":variant,"reference_year":reference,"patient_split_seed":1,"cohort_view":"all_comer",
                         "activation_target":.5,"temporal_distance":1,"target_year":reference+1,
                         "death_f1_current":.7,"previous_degradation":.01,"death_f1_degradation":.02,
                         "concept_coverage":1.,"current_cri":.8,"delta_current_cri":.01,
                         **{name:.8 for name in metrics},**{f"delta_{name}":.01 for name in metrics}})
        rows.append({**rows[-1],"reference_year":2010,"patient_split_seed":2,"u_feature_association":None})
    _,_,common=_ridge_oof(rows,"core")
    history_common=[row for row in common if row["model"]=="performance_history"]
    assert {row["variant"] for row in history_common} == {"original","balanced_context"}
    assert all(row["oof_row_count"] == 4 for row in history_common)
    _,_,maximal=_ridge_oof(rows,"core",maximal_history=True)
    history_max=[row for row in maximal if row["model"]=="performance_history_maximal_coverage"]
    assert all(row["oof_row_count"] == 5 for row in history_max)
    assert all(row["analysis_population"] == "maximal_history" for row in history_max)


def test_nested_dae_outer_validation_uses_every_available_reference_year():
    references=list(range(2007,2015))
    splits=nested_reference_year_splits(references)
    assert [outer for outer,_ in splits] == references
    assert all(len(inner) == 7 and outer not in inner for outer,inner in splits)
    assert all(set(inner) | {outer} == set(references) for outer,inner in splits)


def test_event_summary_compares_models_on_common_estimable_rows():
    rows=[]
    for model in ("baseline","candidate"):
        for seed,event in ((1,0),(2,1)):
            if model == "candidate" and seed == 2: continue
            probability=.1 if event == 0 else .8
            rows.append({"profile":"core","variant":"original","evaluation":"forward","model":model,
                         "reference_year":2008,"patient_split_seed":seed,"cohort_view":"all_comer","activation_target":.5,
                         "temporal_distance":1,"target_year":2009,"material_degradation":event,"probability":probability,
                         "brier_score":(probability-event)**2,"log_loss":.1})
    summary=_event_summary(rows,"core")
    assert len(summary) == 2
    assert all(row["oof_row_count"] == 1 for row in summary)
    assert all(row["analysis_population"] == "common_complete_case_and_estimable_models" for row in summary)


def test_defaults_enable_full_core_dae_and_runtime_is_separate():
    scientific = MetricSynthesisConfig()
    assert scientific.dae_profile == "core" and scientific.dae_latent_dimensions == 2
    assert "device" not in scientific.to_dict() and "workers" not in scientific.to_dict()
    assert MetricSynthesisRuntimeConfig(workers=7).workers == 7


def test_zero_variance_prespecified_tcav_makes_extended_profile_not_estimable():
    rows = []
    for index in range(10):
        rows.append({"reference_year": 2007 + index % 4, "temporal_distance": 1,
                     "u_f2": .1 + index / 100, "u_jaccard": .2 + index / 90,
                     "u_prevalence": .3 + index / 80, "u_activation": .4 + index / 70,
                     "u_feature_association": .5 + index / 60, "u_tcav": 1.})
    quality = metric_quality(rows, "p50_tcav_extended")
    tcav = next(row for row in quality if row["metric"] == "u_tcav")
    assert tcav["eligibility_reason"] == "zero_variance"
    spectrum, loadings, parallel, diagnostics = dimensionality(
        rows, "p50_tcav_extended", MetricSynthesisConfig(parallel_repetitions=10, bootstrap_repetitions=0)
    )
    assert not spectrum and not loadings and not parallel
    assert diagnostics[0]["status"] == "not_estimable_zero_variance"


def test_varimax_has_stable_order_and_positive_largest_loading():
    rotated = varimax(np.asarray([[-.9, .1], [-.2, .8], [-.1, .7]]))
    for column in range(rotated.shape[1]):
        anchor = np.argmax(np.abs(rotated[:, column]))
        assert rotated[anchor, column] >= 0


def test_search_seed_schedule_checkpoint_and_benchmark_selection(tmp_path):
    job = SearchJob(2007, 2008, 9, 42)
    assert job.seed == SearchJob(2007, 2008, 9, 42).seed
    assert job.seed != SearchJob(2007, 2008, 10, 42).seed
    store = ResumeStore(tmp_path / "work", {"identity": "x"})
    store.save("candidate", [{"done": True}])
    assert store.load("candidate") == [{"done": True}]
    selected = select_fastest_benchmark([
        {"valid": True, "jobs_per_second": 2., "workers": 1, "executor": "serial"},
        {"valid": True, "jobs_per_second": 5., "workers": 7, "executor": "process"},
    ])
    assert selected["executor"] == "process"
