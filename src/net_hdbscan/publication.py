
"""Stage, inspect, hash, and roll back complete net-hdbscan output generations.

The flat output layout is preserved. The directory swap uses a two-phase
backup/journal protocol, NOT a multi-file atomic transaction on all platforms.
"""
from __future__ import annotations
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import pandas as pd


class OutputPublicationError(RuntimeError):
    """Publication or integrity failure; no partial generation is accepted."""


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(value):
    import math
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, item: dict) -> None:
    tmp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('w', encoding='utf-8', newline='\n') as file:
            json.dump(_json_safe(item), file, indent=2, allow_nan=False)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _verify_generation(folder: Path, *, expected_inputs=None) -> dict:
    import geopandas as gpd
    manifest = folder / 'manifest.json'
    if not manifest.is_file():
        raise OutputPublicationError('manifest.json is missing')
    doc = json.loads(manifest.read_text(encoding='utf-8'))
    if doc.get('run') not in ('single', 'grouped'):
        raise OutputPublicationError('unknown manifest run type')
    paths = {'summary.csv', 'distance_trace.csv'}
    if doc['run'] == 'single':
        paths |= {'clustered_points.parquet', 'cluster_table.parquet', 'cluster_hierarchy.parquet'}
    else:
        for group, info in doc.get('groups', {}).items():
            files = info['files']
            if set(files) != {'points','clusters','hierarchy'}:
                raise OutputPublicationError(f'incomplete group file references: {group}')
            paths.update(files.values())
    output_info = {}
    for rel in sorted(paths):
        relative = Path(rel)
        if relative.is_absolute() or '..' in relative.parts:
            raise OutputPublicationError(f'unsafe output reference: {rel}')
        path = folder / relative
        if not path.is_file():
            raise OutputPublicationError(f'missing artifact: {rel}')
        if path.suffix == '.csv':
            frame = pd.read_csv(path)
        elif 'points' in relative.parts or relative.name == 'clustered_points.parquet':
            frame = gpd.read_parquet(path)
        else:
            frame = pd.read_parquet(path)
        if not isinstance(frame, pd.DataFrame):
            raise OutputPublicationError(f'invalid output: {rel}')
        output_info[rel] = {'sha256': _hash(path), 'rows': len(frame)}
    if expected_inputs is not None:
        if doc.get('input_sha256') != expected_inputs:
            raise OutputPublicationError('input checksum mismatch')
    if 'output_integrity' in doc and doc['output_integrity'] != output_info:
        raise OutputPublicationError('output integrity mismatch')
    # Verify partition/report cardinalities, including valid zero-row group outputs.
    summary = pd.read_csv(folder / 'summary.csv', keep_default_na=False, dtype={'group': str})
    if doc['run'] == 'single':
        points = gpd.read_parquet(folder / 'clustered_points.parquet')
        if len(summary) != 1 or int(summary.iloc[0]['n_points']) != len(points):
            raise OutputPublicationError('single-run cardinality disagreement')
        clusters = pd.read_parquet(folder / 'cluster_table.parquet')
        hierarchy = pd.read_parquet(folder / 'cluster_hierarchy.parquet')
        _verify_partition(points, clusters, hierarchy)
    else:
        if len(summary) != len(doc.get('groups', {})):
            raise OutputPublicationError('group-count disagreement')
        for group, info in doc['groups'].items():
            fs = info['files']
            points = gpd.read_parquet(folder / fs['points'])
            clusters = pd.read_parquet(folder / fs['clusters'])
            hierarchy = pd.read_parquet(folder / fs['hierarchy'])
            _verify_partition(points, clusters, hierarchy)
            row = summary.loc[summary['group'] == group]
            if len(row) != 1 or int(row.iloc[0]['n_points']) != len(points):
                raise OutputPublicationError(f'group count disagreement: {group}')
    return output_info


def _verify_partition(points, clusters, hierarchy):
    required = {'cluster_id', 'is_noise','membership','core_distance','snap_distance','snapped_x','snapped_y'}
    if not required.issubset(points.columns):
        raise OutputPublicationError(f'missing point columns: {required-set(points.columns)}')
    if not {'cluster_id','n_points','n_positions','stability','birth_distance','singleton'}.issubset(clusters.columns):
        raise OutputPublicationError('missing cluster columns')
    if not {'hierarchy_id','parent_id','selected','cluster_id'}.issubset(hierarchy.columns):
        raise OutputPublicationError('missing hierarchy columns')
    obs_counts = points['cluster_id'].dropna().value_counts()
    table = clusters.set_index('cluster_id')
    if not table.index.is_unique or set(obs_counts.index) != set(table.index):
        raise OutputPublicationError('cluster labels differ between point and cluster tables')
    for key, count in obs_counts.items():
        if int(table.loc[key, 'n_points']) != int(count):
            raise OutputPublicationError(f'cluster size differs for {key}')
    noise = points['is_noise'].astype(bool)
    membership = points['membership'].to_numpy(dtype=float)
    import numpy as np
    if not np.isfinite(membership).all() or (membership < 0).any() or (membership > 1 + 1e-12).any():
        raise OutputPublicationError('invalid membership values')
    if (membership[noise.to_numpy()] != 0).any():
        raise OutputPublicationError('noise has nonzero membership')
    if len(hierarchy):
        ids = set(hierarchy['hierarchy_id'].astype(int))
        for row in hierarchy.itertuples(index=False):
            if pd.notna(row.parent_id) and (int(row.parent_id) not in ids or int(row.parent_id) >= int(row.hierarchy_id)):
                raise OutputPublicationError('invalid hierarchy parent')


def _input_checksums(kwargs: dict) -> dict[str, str]:
    return {name: _hash(Path(kwargs[name])) for name in ('points_path', 'network_path')}


def _journal_path(destination: Path) -> Path:
    return destination.parent / ('.' + destination.name + '.publication-journal.json')


def _recover_if_needed(destination: Path) -> None:
    journal = _journal_path(destination)
    if not journal.exists():
        return
    data = json.loads(journal.read_text(encoding='utf-8'))
    if data.get('destination') != str(destination.resolve()) or data.get('version') != 1:
        raise OutputPublicationError('unsafe or incompatible publication journal')
    backup = Path(data['backup'])
    if backup.parent != destination.parent or not backup.name.startswith('.'+destination.name+'.backup-'):
        raise OutputPublicationError('publication journal references unexpected backup')
    if backup.exists() and not destination.exists():
        _verify_generation(backup)
        os.replace(backup, destination)
    elif backup.exists() and destination.exists():
        try:
            _verify_generation(destination)
        except Exception:
            # Candidate was interrupted or corrupted: restore verified predecessor.
            _verify_generation(backup)
            shutil.rmtree(destination)
            os.replace(backup, destination)
        else:
            shutil.rmtree(backup)
    elif not destination.exists():
        raise OutputPublicationError('missing both destination and recovery backup')
    else:
        _verify_generation(destination)
    journal.unlink()


def transactional_file_runner(original):
    """Decorate file pipelines; preserve public signatures and return values."""
    import inspect
    signature = inspect.signature(original)
    @wraps(original)
    def call(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        kw = dict(bound.arguments)
        dest = Path(kw['output_dir']).resolve()
        if dest.name in ('', '.', '..') or dest == dest.parent:
            raise OutputPublicationError('unsafe output directory')
        dest.parent.mkdir(parents=True, exist_ok=True)
        _recover_if_needed(dest)
        if dest.exists() and any(dest.iterdir()):
            if not kw.get('force', False):
                raise FileExistsError(f'output exists: {dest}; use force=True or --force')
            # Never delete arbitrary user files through --force.
            _verify_generation(dest)
            prior_doc = json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
            allowed = set(prior_doc.get('output_integrity', {}))
            if not allowed:
                if prior_doc.get('run') == 'single':
                    allowed = {'clustered_points.parquet','cluster_table.parquet',
                               'cluster_hierarchy.parquet','summary.csv','distance_trace.csv'}
                else:
                    allowed = {'summary.csv','distance_trace.csv'}
                    for group in prior_doc.get('groups', {}).values():
                        allowed.update(group['files'].values())
            present = {str(x.relative_to(dest)).replace(os.sep, '/') for x in dest.rglob('*') if x.is_file()}
            if present != allowed | {'manifest.json'}:
                raise OutputPublicationError('refusing --force on directory containing unknown or missing files')
        # Preserve the file-pipeline contract: missing optional Parquet support
        # must fail before any attempt to read or hash input files.  Resolve
        # through the original function's module globals so test monkeypatches
        # and runtime dependency checks observe the same validator.
        original.__globals__['_require_pyarrow']()
        inputs = _input_checksums(kw)
        stage = Path(tempfile.mkdtemp(prefix='.'+dest.name+'.stage-', dir=dest.parent))
        backup = dest.parent / ('.'+dest.name+'.backup-'+uuid.uuid4().hex)
        journal = _journal_path(dest)
        committed = False
        moved_old = False
        try:
            kw['output_dir'] = stage
            kw['force'] = False
            result = original(**kw)
            if _input_checksums(kw) != inputs:
                raise OutputPublicationError('input files changed during clustering')
            _verify_generation(stage)
            docpath = stage / 'manifest.json'
            doc = json.loads(docpath.read_text(encoding='utf-8'))
            doc['manifest_schema_version'] = 2
            doc['package'] = 'net-hdbscan'
            doc['input_sha256'] = inputs
            doc['output_integrity'] = _verify_generation(stage)
            _write_json(docpath, doc)
            _verify_generation(stage, expected_inputs=inputs)
            if dest.exists():
                _write_json(journal, {'version': 1, 'destination': str(dest), 'backup': str(backup)})
                os.replace(dest, backup)
                moved_old = True
            os.replace(stage, dest)
            committed = True
            _verify_generation(dest, expected_inputs=inputs)
            if backup.exists():
                shutil.rmtree(backup)
            if journal.exists():
                journal.unlink()
            return result
        except BaseException:
            if moved_old and backup.exists():
                # Roll back while the original verified generation still exists.
                if dest.exists():
                    shutil.rmtree(dest)
                os.replace(backup, dest)
            elif committed and not moved_old and dest.exists():
                # A brand-new run had no predecessor to restore.
                shutil.rmtree(dest)
            # If backup was already cleaned, the new generation remains intact.
            if journal.exists() and (dest.exists() or not moved_old):
                journal.unlink()
            raise
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return call
