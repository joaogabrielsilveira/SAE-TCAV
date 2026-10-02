from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from temporal_concept_forecasting import (ForecastConfig, FoldFeatures, build_forecast_rows,
    training_rows, weights, summarize_predictions, unique_index, build_forecasting,
    load_forecasting, digest, canonical_json)
from temporal_cri import build_family_universe, CRIAnalysisConfig
from temporal_unified_analysis import UnifiedAnalysisConfig
from temporal_window_concepts import historical_roles, cohort_masks, save_binary_checkpoint, load_binary_checkpoint


def fixture_bundle():
    factors, performance = [], []
    for ref in (2007, 2008, 2009, 2010):
        for seed in (42, 43):
            for d in range(4):
                performance.append(dict(system='original', reference_year=ref, patient_split_seed=seed,
                    test_year=ref+d, temporal_distance=d, cohort_view='all_comer', cohort_definition='test',
                    frozen_threshold=.2, valid=True, support_fingerprint=f'{ref}/{seed}/{d}',
                    death_f1_at_frozen_threshold=.5-.03*d, death_average_precision=.3-.01*d,
                    brier_score=.1+.01*d, death_f1_oracle=999))
                for member in (42, 43, 44):
                    factors.append(dict(reference_year=ref, patient_split_seed=seed, factor_family_uid=f'{ref}/{seed}/family',
                        member_sae_seed=member, member_factor_id=3, activation_target=.5, cohort_view='all_comer',
                        temporal_distance=d, test_year=ref+d, matching_view='intersection', rule_source='semantic',
                        target_role='primary', cosine_threshold=.6, overlap_percentile=70, overlap_threshold=.7,
                        geometric_factor_recurrence=1., f2=.8-.02*d, jaccard=.6-.01*d, prevalence_ratio=1.-.03*d,
                        activation_magnitude=2.+.01*member-.1*d, t0_prevalence=.1+member/1000,
                        feature_association_cosine=1.-.05*d, status='stable'))
    return dict(system='original', factors=factors, performance=performance, tcav=[], membership_audit=[],
                universe=build_family_universe(factors, UnifiedAnalysisConfig(), CRIAnalysisConfig()), sources={'test': 'synthetic'})


def config(**kwargs):
    return ForecastConfig(activation_targets=(.5,), profiles=('core',), **kwargs)


def test_target_signs_and_future_concepts_not_required():
    b = fixture_bundle()
    rows, excluded = build_forecast_rows(b['performance'], config())
    assert {r['outcome'] for r in rows} == {'frozen_f1', 'average_precision', 'brier'}
    assert all(r['deterioration'] > 0 and r['outcome_year'] == r['forecast_origin_year']+1 for r in rows)
    assert all('oracle' not in k for r in rows for k in r)
    assert len(excluded) == 48
    b['performance'][1]['frozen_threshold'] = .3
    with pytest.raises(ValueError, match='frozen'):
        build_forecast_rows(b['performance'], config())


def test_forward_restrictions_hold_for_outer_and_inner():
    rows, _ = build_forecast_rows(fixture_bundle()['performance'], config())
    tr = training_rows(rows, 2009, 2010, 'forward')
    assert tr and all(r['reference_year'] < 2009 and r['outcome_year'] <= 2010 for r in tr)
    inner = training_rows(tr, 2008, 2009, 'forward')
    assert all(r['reference_year'] < 2008 and r['outcome_year'] <= 2009 for r in inner)
    assert all(r['reference_year'] != 2009 for r in training_rows(rows, 2009, None, 'held_reference'))


@pytest.mark.parametrize('representation', ['raw', 'cri', 'pca', 'fa', 'dae'])
def test_held_reference_and_future_changes_cannot_change_fitted_transform(tmp_path, representation):
    b = fixture_bundle()
    c = config(outcomes=('frozen_f1',), dae_epochs=2, dae_seeds=(42,))
    rows, _ = build_forecast_rows(b['performance'], c)
    tr = training_rows(rows, 2009, 2010, 'forward')
    te = [r for r in rows if r['reference_year'] == 2009 and r['forecast_origin_year'] == 2010]
    engine = FoldFeatures(b, tmp_path/'a', c)
    original, identity = engine.get(tr, tr+te, representation)
    altered = deepcopy(b)
    for r in altered['factors']:
        if r['reference_year'] >= 2009 or r['test_year'] >= 2010:
            r.update(activation_magnitude=999., t0_prevalence=.999, feature_association_cosine=-.99)
    second = FoldFeatures(altered, tmp_path/'b', c)
    changed, identity2 = second.get(tr, tr+te, representation)
    assert identity == identity2
    assert original[:len(tr)] == changed[:len(tr)]
    assert engine.cache[identity]['state'] == second.cache[identity2]['state']
    assert engine.cache[identity]['taus'] == second.cache[identity2]['taus']
    if representation == 'dae':
        assert engine.cache[identity]['state']['epochs'] == 2
        assert engine.cache[identity]['state']['training_member_count'] > 0


def test_balanced_weights_ignore_number_of_splits_and_years():
    rows = [dict(reference_year=2007, forecast_origin_year=2008)]*3 + [dict(reference_year=2007, forecast_origin_year=2009)] + [dict(reference_year=2008, forecast_origin_year=2009)]*2
    w = weights(rows)
    assert w[:4].sum() == pytest.approx(w[4:].sum())
    assert w[:3].sum() == pytest.approx(w[3])


def test_comparison_pairs_support_and_bootstrap_is_reproducible():
    rows, _ = build_forecast_rows(fixture_bundle()['performance'], config(outcomes=('frozen_f1',)))
    predictions = []
    for i, r in enumerate(rows):
        predictions.extend([{**r, 'design':'forward', 'model':'history', 'prediction':0.1},
                            {**r, 'design':'forward', 'model':'no_change', 'prediction':0.}])
        if i % 3:
            predictions.append({**r, 'design':'forward', 'model':'history_raw', 'prediction':r['deterioration']})
    first = summarize_predictions(predictions, config())[1]
    assert first == summarize_predictions(predictions, config())[1]
    primary = next(r for r in first if r['primary'])
    assert primary['paired_rows'] == sum(r['model']=='history_raw' for r in predictions)
    assert primary['improvement'] > 0 and primary['lower_95'] > 0


def test_duplicates_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        unique_index([{'id':1}, {'id':1}], ('id',))


def test_resume_and_corruption_detection(tmp_path):
    b = fixture_bundle()
    c = config(outcomes=('frozen_f1',), models=('history','history_raw'), designs=('forward',), ridge_alphas=(1.,))
    path = build_forecasting(b, tmp_path, c)
    result = load_forecasting(path)
    assert result['predictions'] and result['folds']
    path.unlink()
    assert build_forecasting(b, tmp_path, c) == path
    assert load_forecasting(path)['predictions'] == result['predictions']
    artifact = path.parent/result['manifest']['artifacts']['predictions']['path']
    artifact.write_bytes(b'corrupted')
    with pytest.raises((ValueError, RuntimeError)):
        load_forecasting(path)


def test_historical_roles_nested_equal_and_patient_disjoint():
    patients = np.array(['context','sae','rule','select','eval'] + [f'h{i}' for i in range(9)] + ['eval','sae'])
    years = np.array([2010]*5 + [2007,2008,2009]*3 + [2008,2008])
    pop = SimpleNamespace(patient_ids=patients, years=years)
    roles = {r: np.array([i]) for i, r in enumerate(('tabpfn_context','sae_discovery','rule_discovery','rule_selection_cav','t0_evaluation'))}
    keep = np.ones(len(years),bool)
    long, audit = historical_roles(pop,2010,42,roles,(2007,2008,2009,2010),keep)
    short, _ = historical_roles(pop,2010,42,roles,(2009,2010),keep)
    for role in ('sae_discovery','rule_discovery','rule_selection_cav'):
        assert set(short[role]).issubset(long[role])
        assert 'eval' not in patients[long[role]]
    historical = [r for r in audit if r['historical_only_patient']]
    assert {sum(r['assigned_role']==role for r in historical) for role in ('sae_discovery','rule_discovery','rule_selection_cav')} == {3}
    assert 15 in long['sae_discovery']
    keep[15] = False
    filtered,_ = historical_roles(pop,2010,42,roles,(2007,2008,2009,2010),keep)
    assert 15 not in filtered['sae_discovery']


def test_window_unseen_d0_is_not_automatically_aliased():
    pop = SimpleNamespace(patient_ids=np.array(['a','a','b','c']), years=np.array([2008,2009,2009,2009]))
    roles = {'t0_evaluation':np.array([1,2]), 'sae_discovery':np.array([3]), 'rule_discovery':np.array([],int), 'rule_selection_cav':np.array([],int)}
    masks = cohort_masks(pop,2009,np.arange(4),roles,np.array([0]),np.array([],int))
    assert masks[(2009,'all_comer')].sum() == 2
    assert masks[(2009,'pipeline_unseen')].sum() == 1


def test_fitted_checkpoint_checksum(tmp_path):
    path = tmp_path/'fit.pkl'
    save_binary_checkpoint(path, {'model':'test'}, 'identity')
    assert load_binary_checkpoint(path,'identity') == {'model':'test'}
    with pytest.raises(ValueError):
        load_binary_checkpoint(path,'another')
    path.write_bytes(b'bad')
    with pytest.raises(ValueError):
        load_binary_checkpoint(path,'identity')


def test_injected_model_is_never_refitted_and_domains_are_explicit(monkeypatch, tmp_path):
    import tabpfn_model
    from comparison_runner import DefaultComparisonAdapter, ComparisonRunnerConfig
    model = object()
    fit = dict(model=model, model_add_x_device='cpu', example_add_shape=None, fit_time_sec=0.)
    prepared = SimpleNamespace(X_train=np.array([[1.,2.],[2.,4.]]), y_train=np.array([0,1]),
        years_train=np.array([2007,2008]), X_test=np.array([[3.,4.],[4.,5.],[5.,6.]]),
        years_test=np.array([2007,2009,2010]), domain_reference_year=2009)
    seen = []
    def extract(**kwargs):
        assert kwargs['model'] is model
        seen.append(kwargs['year_to_domain_map'])
        return kwargs['X']
    def forbidden(*args, **kwargs):
        raise AssertionError('Injected model was refitted')
    monkeypatch.setattr(tabpfn_model,'fit_dr_tabpfn',forbidden)
    monkeypatch.setattr(tabpfn_model,'extract_embeddings_robust',extract)
    monkeypatch.setattr(tabpfn_model,'flatten_embeddings',lambda x:x)
    base = ComparisonRunnerConfig()
    cfg = replace(base, use_cache=False, show_progress=False, tabpfn=replace(base.tabpfn,run_walkforward=False))
    domains = {2007:0,2008:1,2009:2,2010:3}
    result = DefaultComparisonAdapter(None).embeddings(prepared, {'idx_semantic_fit':np.array([0,1])}, cfg,tmp_path,
        force=False,fitted_state=fit,explicit_domain_map=domains,fitted_identity='same-fit')
    assert result.require_model() is model
    assert seen == [domains,domains]
    assert result.year_to_domain[2007] == 0  # reported distance would be -2
    with pytest.raises(ValueError,match='explicit domain'):
        DefaultComparisonAdapter(None).embeddings(prepared, {'idx_semantic_fit':np.array([0,1])},cfg,tmp_path,
                                                  force=False,fitted_state=fit)


def test_matching_window_comparisons_reject_different_record_support():
    from temporal_forecasting_report import matched_window_comparisons
    b = fixture_bundle()
    rows, _ = build_forecast_rows(b['performance'],config(outcomes=('frozen_f1',)))
    baseline = dict(performance=b['performance'], forecast_rows=rows,predictions=[])
    changed = deepcopy(baseline)
    for r in changed['performance']:
        r['support_fingerprint'] += '-different'
    for r in changed['forecast_rows']:
        r['current_support'] += '-different'
    assert matched_window_comparisons({'reference_only_common':baseline,'all_history':changed}) == []


def test_strict_embeddings_never_drop_domain_after_error():
    from tabpfn_model import batch_get_embeddings
    class BrokenModel:
        def __init__(self):
            self.calls = []
        def get_embeddings(self, x, **kwargs):
            self.calls.append(kwargs)
            if kwargs:
                raise RuntimeError('domain failure')
            return x
    model = BrokenModel()
    with pytest.raises(RuntimeError,match='domain failure'):
        batch_get_embeddings(model,np.ones((2,2)),np.array([0,1]),strict_domains=True)
    assert len(model.calls) == 1 and 'additional_x' in model.calls[0]
