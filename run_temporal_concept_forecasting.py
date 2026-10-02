"""Run both planned stages, publish the report, and execute its artifact-only notebook.

Restart this command after interruption to reuse validated completed work.
"""
from temporal_memory_recovery import configure_allocator
configure_allocator()

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from artifact_storage import file_sha256, atomic_write_json
from temporal_concept_forecasting import (ForecastConfig, build_forecasting, original_inputs,
                                         checked_manifest, table, configure_logging)
from temporal_window_concepts import run_window_concepts
from temporal_forecasting_report import build_report
from temporal_run_progress import progress_stage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, default=Path('stats/temporal_concept_forecasting'))
    parser.add_argument('--window-output', type=Path, default=Path('stats/temporal_window_concepts'))
    parser.add_argument('--device', choices=('auto','cpu','cuda'), default='cuda')
    parser.add_argument("--gpu-memory-safe", action="store_true", help="Bound GPU memory and stop on OOM instead of switching to CPU")
    from temporal_gpu_execution import add_gpu_arguments, apply_gpu_arguments
    from temporal_handoff import EmbeddingsReady, export_package
    add_gpu_arguments(parser)
    parser.add_argument('--stop-after-embeddings', action='store_true', help='Save a verified handoff and exit before SAE training')
    parser.add_argument('--import-handoff', type=Path, help='Explicitly reuse a completed fitted system across runtime identities')
    parser.add_argument('--resume-extraction', type=Path, help='run_identity.json for partial extraction; requires unchanged environment and GPU policy')
    parser.add_argument('--export-package', type=Path, help='Create portable package when stopping after embeddings')
    args = parser.parse_args(argv)
    if args.export_package and not args.stop_after_embeddings:
        parser.error('--export-package requires --stop-after-embeddings')
    if args.import_handoff and args.resume_extraction:
        parser.error('Choose completed handoff import or partial extraction resume')
    if (args.import_handoff or args.resume_extraction) and not args.gpu_memory_safe:
        parser.error('Explicit GPU handoff/resume requires --gpu-memory-safe')
    apply_gpu_arguments(args)
    os.chdir(args.repo.resolve())
    handoff_options = dict(stop_after_embeddings=args.stop_after_embeddings, import_handoff=args.import_handoff,
                           resume_extraction=args.resume_extraction)
    configure_logging(args.output)
    log = logging.getLogger(__name__)
    status_path = args.output / 'execution_status.json'
    status = {'status': 'running', 'pid': os.getpid(), 'device': args.device, 'gpu_memory_safe': args.gpu_memory_safe,
              'started_at': datetime.now(timezone.utc).isoformat()}
    atomic_write_json(status_path, status)
    try:
        config = ForecastConfig()
        paths = []
        if args.resume_extraction and args.stop_after_embeddings:
            log.info('Resuming extraction only; existing Stage A results retained')
        else:
            log.info('Stage A: original fitted model, existing measured concepts')
            with progress_stage("Stage A"):
                paths = [build_forecasting(original_inputs(args.repo), args.output, config)]
            build_report(paths, args.output)
        log.info('Stage B pilot: reference-only and largest all-history system')
        with progress_stage("Stage B pilot"):
            run_window_concepts(args.repo, args.window_output, pilot=True, device=args.device, gpu_memory_safe=args.gpu_memory_safe, **handoff_options)
        log.info('Stage B: complete three-window grid (pilot checkpoints reused)')
        with progress_stage("Stage B complete grid"):
            windows = run_window_concepts(args.repo, args.window_output, device=args.device, gpu_memory_safe=args.gpu_memory_safe, **handoff_options)
        manifest = checked_manifest(windows)
        for system in manifest['systems']:
            bundle = {n: table(windows, manifest, f'{system}_{n}') for n in ('performance','factors','tcav','universe','membership_audit')}
            bundle.update(system=system, sources={'window_concepts':file_sha256(windows)})
            with progress_stage(f"Forecasting {system}"):
                paths.append(build_forecasting(bundle, args.output, config))
        report = build_report(paths, args.output, stage_b_complete=True, window_manifest=windows)
        log.info('Executing artifact-only notebook from report %s', report)
        import nbformat
        from nbclient import NotebookClient
        notebook_path = args.repo/'temporal_concept_forecasting_analysis.ipynb'
        notebook = nbformat.read(notebook_path, as_version=4)
        # Set an explicit manifest even when a custom output directory is requested.
        notebook.cells.insert(0, nbformat.v4.new_code_cell(f'REPORT_MANIFEST = {str(report)!r}', metadata={'tags':['parameters']}))
        NotebookClient(notebook, timeout=600, kernel_name='python3', resources={'metadata':{'path':str(args.repo)}}).execute()
        nbformat.write(notebook, args.output/'temporal_concept_forecasting_analysis.executed.ipynb')
        log.info('All stages complete; executed notebook and report in %s', args.output)
        atomic_write_json(status_path, {**status, 'status': 'complete', 'exit_code': 0,
                          'finished_at': datetime.now(timezone.utc).isoformat()})
    except EmbeddingsReady as ready:
        atomic_write_json(status_path, {**status, 'status':'embeddings_ready', 'exit_code':0,
            'handoff_manifest':str(ready.manifest), 'finished_at':datetime.now(timezone.utc).isoformat()})
        log.info('Clean stop after embeddings: %s; SAE training deferred to destination', ready.manifest)
        if args.export_package:
            try:
                with progress_stage("Packaging handoff"):
                    export_package(args.repo, ready.manifest, args.export_package)
            except BaseException as error:
                atomic_write_json(status_path, {**status,'status':'export_failed','exit_code':1,
                    'handoff_manifest':str(ready.manifest),'error':str(error)})
                raise
            atomic_write_json(status_path, {**status,'status':'handoff_ready','exit_code':0,
                'handoff_manifest':str(ready.manifest),'package':str(args.export_package),
                'finished_at':datetime.now(timezone.utc).isoformat()})
    except BaseException as error:
        atomic_write_json(status_path, {**status, "status": "failed", "exit_code": 1,
                          "error": str(error), "finished_at": datetime.now(timezone.utc).isoformat()})
        log.exception('Run stopped; inspect the named stage and restart to resume completed checkpoints')
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
