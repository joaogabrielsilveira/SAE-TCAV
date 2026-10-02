"""Artifact-only report assembly and matched comparisons across fitted systems."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from artifact_storage import atomic_write_json, file_sha256
from temporal_concept_forecasting import (OUTCOMES, ROW_KEY, STRATA, digest, key,
    load_forecasting, unique_index, write_table)


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


def build_report(manifests, output, *, stage_b_complete=False, window_manifest=None):
    """Publish explicit report inputs; notebook execution never discovers or launches fits."""
    output = Path(output).resolve()
    loaded = [load_forecasting(path) for path in manifests]
    systems = [r['manifest']['system'] for r in loaded]
    if len(set(systems)) != len(systems):
        raise ValueError('Supply one forecasting run per system')
    runs = dict(zip(systems, loaded))
    if stage_b_complete and not {'original','reference_only_common','last_3','all_history'}.issubset(runs):
        raise ValueError('Complete report requires all four systems')
    rows = matched_window_comparisons(runs)
    report_id = digest([file_sha256(p) for p in manifests]+[stage_b_complete])[:20]
    root = output / f'report_{report_id}'
    artifacts = {'window_comparisons': write_table(root, 'window_comparisons', rows)}
    manifest = dict(schema_version='temporal_concept_forecasting_report_v1', complete=True,
                    stage_b_complete=stage_b_complete,
                    forecast_manifests=[dict(path=str(Path(p).resolve()),sha256=file_sha256(p)) for p in manifests],
                    window_manifest=None if window_manifest is None else dict(path=str(Path(window_manifest).resolve()),sha256=file_sha256(window_manifest)),
                    artifacts=artifacts,
                    interpretation='Training and concept discovery both expand; their separate contributions are not identified.')
    path = root / 'manifest.json'
    atomic_write_json(path, manifest)
    atomic_write_json(output/'report_manifest.json', dict(path=str(path), sha256=file_sha256(path)))
    return path
