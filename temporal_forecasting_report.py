"""Artifact-only report assembly and matched comparisons across fitted systems."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from artifact_storage import atomic_write_json, file_sha256
from temporal_concept_forecasting import (OUTCOMES, ROW_KEY, STRATA, checked_manifest, digest, key,
    load_forecasting, table, unique_index, write_table)

LEGACY_SYSTEMS = {'original', 'reference_only_common', 'last_3', 'all_history'}
EXCLUSION_EXPLANATIONS = {
    'missing_consecutive_performance': 'A forecast needs previous, current and future annual performance from the same frozen system; '
                                       'the listed years are not measured, so no forecast was attempted.',
    'insufficient_performance_support': 'Previous, current or future performance is present but invalid or non-finite for this outcome; '
                                        'no forecast was attempted.'}


def clustered_difference(rows, value, replicates=1000, seed=42):
    cells, references = defaultdict(list), defaultdict(list)
    for row in rows:
        cells[(row['reference_year'], row['year'])].append(row[value])
    for (ref, _), values in cells.items():
        references[ref].append(np.mean(values))
    values = np.array([np.mean(v) for _, v in sorted(references.items())])
    if not len(values):
        return dict(estimate=None, lower_95=None, upper_95=None, references=0, paired_rows=0)
    lower = upper = None
    if len(values) > 1:
        rng = np.random.default_rng(seed)
        draws = values[rng.integers(0, len(values), (replicates, len(values)))].mean(axis=1)
        lower, upper = np.quantile(draws, [.025, .975]).tolist()
    return dict(estimate=float(values.mean()), lower_95=lower, upper_95=upper,
                references=len(values), paired_rows=len(rows))


def matched_window_comparisons(runs):
    """Only compare identical record membership; all-comers are the primary cohort."""
    baseline = runs.get('reference_only_common')
    if baseline is None:
        return []
    output = []
    perf_keys = ('reference_year', 'patient_split_seed', 'test_year', 'cohort_view', 'support_fingerprint')
    bi = unique_index(baseline['performance'], perf_keys)
    forecast_keys = tuple(n for n in STRATA if n != 'system') + ROW_KEY + ('previous_support','current_support','future_support')
    base_rows = unique_index(baseline['forecast_rows'], forecast_keys)
    pred_keys = tuple(n for n in STRATA if n != 'system') + ROW_KEY + ('design','previous_support','current_support','future_support')
    base_predictions = {m: unique_index([r for r in baseline['predictions'] if r['model']==m], pred_keys)
                        for m in ('history', 'history_raw')}
    for system, run in runs.items():
        if system not in ('last_3','all_history'):
            continue
        groups = defaultdict(list)
        for row in run['performance']:
            other = bi.get(key(row, perf_keys))
            if other is None or not (other.get('valid') and row.get('valid')):
                continue
            for outcome, (column, _) in OUTCOMES.items():
                groups[('performance_level',outcome,row['cohort_view'],None,None,None)].append(
                    dict(reference_year=row['reference_year'], year=row['test_year'], difference=row[column]-other[column]))
        for row in run['forecast_rows']:
            other = base_rows.get(key(row, forecast_keys))
            if other is None:
                continue
            groups[('deterioration',row['outcome'],row['cohort_view'],row['activation_target'],row['profile'],None)].append(
                dict(reference_year=row['reference_year'], year=row['forecast_origin_year'], difference=row['deterioration']-other['deterioration']))
        pi = {m: unique_index([r for r in run['predictions'] if r['model']==m], pred_keys) for m in ('history','history_raw')}
        common = set(pi['history']) & set(pi['history_raw']) & set(base_predictions['history']) & set(base_predictions['history_raw'])
        for identity in common:
            history, concepts = pi['history'][identity], pi['history_raw'][identity]
            bh, bc = base_predictions['history'][identity], base_predictions['history_raw'][identity]
            gain = abs(history['prediction']-history['deterioration'])-abs(concepts['prediction']-concepts['deterioration'])
            bgain = abs(bh['prediction']-bh['deterioration'])-abs(bc['prediction']-bc['deterioration'])
            groups[('concept_forecast_gain',history['outcome'],history['cohort_view'],history['activation_target'],history['profile'],history['design'])].append(
                dict(reference_year=history['reference_year'], year=history['forecast_origin_year'], difference=gain-bgain))
        for (question,outcome,cohort,activation,profile,design), values in groups.items():
            output.append(dict(system=system, baseline='reference_only_common', question=question, outcome=outcome,
                cohort_view=cohort, activation_target=activation, profile=profile, design=design,
                direction='window_minus_reference_only_common', **clustered_difference(values,'difference')))
    return output


def selected_scope(window_manifest):
    """Canonical selection recorded by the window run, or None for manifests that predate selections."""
    if window_manifest is None:
        return None
    manifest = checked_manifest(window_manifest)
    if 'selection' not in manifest:
        return None
    from temporal_window_concepts import canonical_window_jobs  # deferred: heavy module, only needed once a scope exists
    selection = [list(job) for job in canonical_window_jobs(manifest['selection'])]
    if selection != [list(job) for job in manifest['selection']]:
        raise ValueError('Window manifest selection is not canonical')
    systems = sorted({window for _, _, window in selection})
    if 'systems' in manifest and sorted(manifest['systems']) != systems:
        raise ValueError('Window manifest systems disagree with its selection')
    return dict(selection=selection, systems=systems,
                reference_years=sorted({ref for ref, _, _ in selection}),
                patient_split_seeds=sorted({seed for _, seed, _ in selection}),
                windows=systems, logical_jobs=len(selection), distinct_fits=manifest.get('distinct_fits'),
                window_manifest_sha256=file_sha256(window_manifest),
                completed_jobs=table(window_manifest, manifest, 'aliases'))


def _check_selected_scope(runs, scope, complete):
    contextual = {'original'}
    unexpected = sorted(set(runs) - contextual - set(scope['systems']))
    if unexpected:
        raise ValueError(f'Systems outside the selected scope are never added to a selected report: {unexpected}')
    if not complete:
        return
    missing = sorted((contextual | set(scope['systems'])) - set(runs))
    if missing:
        raise ValueError(f'Complete report is missing selected systems: {missing}')
    done = {(r['reference_year'], r['patient_split_seed'], r['window']) for r in scope.pop('completed_jobs')}
    absent = [job for job in scope['selection'] if tuple(job) not in done]
    if absent:
        raise ValueError(f'Selected jobs are not complete in the window manifest: {absent}')
    for system in scope['systems']:
        manifest = runs[system]['manifest']
        if manifest.get('sources', {}).get('window_concepts') != scope['window_manifest_sha256']:
            raise ValueError(f'Forecasts for {system} were not built from the supplied window manifest')
        covered = {(r['reference_year'], r['patient_split_seed']) for r in runs[system]['performance']}
        lacking = [job for job in scope['selection'] if job[2] == system and (job[0], job[1]) not in covered]
        if lacking:
            raise ValueError(f'Forecasting run for {system} lacks selected jobs: {lacking}')


def forecast_eligibility(runs):
    """Descriptive coverage and explicit forecast exclusions; ineligible forecasts are never predictions."""
    identity = ('system', 'reference_year', 'patient_split_seed', 'cohort_view')
    eligibility, exclusions = [], []
    for system, run in sorted(runs.items()):
        years = defaultdict(set)
        for row in run['performance']:
            years[key({**row, 'system': system}, identity)].add(row['test_year'])
        origins = defaultdict(set)
        for row in run['forecast_rows']:
            origins[key(row, identity)].add(row['forecast_origin_year'])
        excluded = defaultdict(set)
        for row in run['exclusions']:
            excluded[key(row, identity)].add(row['forecast_origin_year'])
            known = years.get(key(row, identity), set())
            missing = [y for y in (row['forecast_origin_year']-1, row['forecast_origin_year']+1) if y not in known] \
                if row['reason'] == 'missing_consecutive_performance' else []
            exclusions.append({**row, 'status': 'excluded_not_forecast', 'missing_performance_years': missing,
                               'explanation': EXCLUSION_EXPLANATIONS.get(row['reason'], row['reason'])})
        blocked = {key(r, identity + ('outcome', 'forecast_origin_year')) for r in run['exclusions']}
        for name in ('forecast_rows', 'predictions'):
            if any(key(r, identity + ('outcome', 'forecast_origin_year')) in blocked for r in run[name]):
                raise ValueError(f'{system}: ineligible forecast origins appear in {name}')
        for group, observed in years.items():
            if observed - origins.get(group, set()) - excluded.get(group, set()):
                raise ValueError(f'{system}: forecast origins {sorted(observed - origins.get(group, set()) - excluded.get(group, set()))} '
                                 f'for {group[1:]} were silently dropped instead of published as exclusions')
        counts = defaultdict(lambda: defaultdict(int))
        for name, field in (('performance', 'performance_rows'), ('measured_concepts', 'measured_concept_rows'),
                            ('forecast_rows', 'forecast_rows'), ('exclusions', 'exclusion_rows')):
            for row in run[name]:
                counts[(row['reference_year'], row['patient_split_seed'])][field] += 1
        for (reference, seed), values in sorted(counts.items()):
            eligible = values['forecast_rows'] > 0
            eligibility.append(dict(system=system, reference_year=reference, patient_split_seed=seed,
                performance_rows=values['performance_rows'], measured_concept_rows=values['measured_concept_rows'],
                performance_years=sorted({y for g, ys in years.items() if g[:3] == (system, reference, seed) for y in ys}),
                forecast_rows=values['forecast_rows'], exclusion_rows=values['exclusion_rows'],
                forecast_origin_years=sorted({y for g, ys in origins.items() if g[:3] == (system, reference, seed) for y in ys}),
                status='forecast_eligible' if eligible else 'descriptive_only',
                reason=None if eligible else 'no_origin_year_with_previous_current_and_future_performance'))
    return eligibility, exclusions


def build_report(manifests, output, *, stage_b_complete=False, window_manifest=None):
    """Publish explicit report inputs; notebook execution never discovers or launches fits."""
    output = Path(output).resolve()
    loaded = [load_forecasting(path) for path in manifests]
    systems = [r['manifest']['system'] for r in loaded]
    if len(set(systems)) != len(systems):
        raise ValueError('Supply one forecasting run per system')
    runs = dict(zip(systems, loaded))
    scope = selected_scope(window_manifest)
    if scope is not None:
        _check_selected_scope(runs, scope, stage_b_complete)
        scope.pop('completed_jobs', None)
    elif stage_b_complete and not LEGACY_SYSTEMS.issubset(runs):
        raise ValueError('Complete report requires all four systems')
    rows = matched_window_comparisons(runs)
    eligibility, exclusions = forecast_eligibility(runs)
    identity = [file_sha256(p) for p in manifests] + [stage_b_complete]
    if scope is not None:
        identity.append(dict(selection=scope['selection'], window_manifest_sha256=scope['window_manifest_sha256']))
    report_id = digest(identity)[:20]
    root = output / f'report_{report_id}'
    artifacts = {'window_comparisons': write_table(root, 'window_comparisons', rows),
                 'forecast_eligibility': write_table(root, 'forecast_eligibility', eligibility),
                 'forecast_exclusions': write_table(root, 'forecast_exclusions', exclusions)}
    descriptive = sorted({r['reference_year'] for r in eligibility if r['status'] == 'descriptive_only'})
    limitations = []
    if scope is not None and len(scope['patient_split_seeds']) == 1:
        limitations.append(f"Selected scope uses a single patient split ({scope['patient_split_seeds'][0]}); "
                           'it gives no between-split robustness evidence. SAE initialization seeds are a separate axis.')
    if descriptive:
        limitations.append(f'Reference years {descriptive} contribute performance and concept records but no forecast rows: '
                           'a forecast needs previous, current and future performance from the same frozen system. '
                           'Their ineligible origins are listed in forecast_exclusions, not scored as zero or failed forecasts.')
    manifest = dict(schema_version='temporal_concept_forecasting_report_v1', complete=True,
                    stage_b_complete=stage_b_complete,
                    scope=scope,
                    contextual_systems=['original'] if 'original' in runs else [],
                    forecast_eligibility=dict(descriptive_only_references=descriptive,
                        exclusion_rows=len(exclusions), forecast_rows=sum(r['forecast_rows'] for r in eligibility)),
                    limitations=limitations,
                    forecast_manifests=[dict(path=str(Path(p).resolve()),sha256=file_sha256(p)) for p in manifests],
                    window_manifest=None if window_manifest is None else dict(path=str(Path(window_manifest).resolve()),sha256=file_sha256(window_manifest)),
                    artifacts=artifacts,
                    interpretation='Training and concept discovery both expand; their separate contributions are not identified.')
    path = root / 'manifest.json'
    atomic_write_json(path, manifest)
    atomic_write_json(output/'report_manifest.json', dict(path=str(path), sha256=file_sha256(path)))
    return path
