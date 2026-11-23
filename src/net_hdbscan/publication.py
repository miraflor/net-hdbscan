
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
    """Publication or integrity failure requiring the caller to stop."""


class PostCommitCleanupError(OutputPublicationError):
    """Verified output is published, but recoverable housekeeping failed.

    This error must stop automation; it does *not* mean publication was rolled
    back.  The journal is retained for safe, idempotent recovery.
    """


# Files GDAL reads to build a shapefile's geometry, attributes, encoding and
# CRS. Unrelated datasets sharing a name prefix (roads.old.shp), spatial
# indexes (.qix, .sbn, .sbx) and metadata (.shp.xml) are excluded.
_SHAPEFILE_MEMBER_SUFFIXES = frozenset({".shp", ".shx", ".dbf", ".prj", ".cpg"})


def _hash(path: Path) -> str:
    """SHA-256 over the dataset actually opened by read_vector.

    Top-level symlinks are resolved both here and in read_vector, ensuring
    that sidecars are enumerated beside the same target .shp read by GDAL.
    Nested directory symlinks are rejected: rglob would otherwise omit their
    contents and create an incomplete hash. File symlinks are allowed and
    their target bytes are hashed under the member's relative name.
    """
    supplied = Path(path)
    if not supplied.exists():
        raise FileNotFoundError(f"input dataset is not a regular file or directory: {supplied}")
    path = supplied.resolve(strict=True)
    if path.is_dir():
        root = path
        entries = list(root.rglob("*"))
        nested_dir_links = [p for p in entries if p.is_symlink() and p.is_dir()]
        if nested_dir_links:
            raise ValueError(f"nested directory symlink cannot be checksummed safely: {nested_dir_links[0]}")
        files = sorted((p for p in entries if p.is_file() or p.is_symlink()),
                       key=lambda p: p.relative_to(root).as_posix())
        multifile = True
    elif path.is_file() and path.suffix.casefold() == ".shp":
        root = path.parent
        files = sorted((p for p in root.iterdir()
                        if p.stem.casefold() == path.stem.casefold()
                        and p.suffix.casefold() in _SHAPEFILE_MEMBER_SUFFIXES
                        and (p.is_file() or p.is_symlink())),
                       key=lambda p: p.name)
        # Identically named sidecars differing only by case are ambiguous
        # across Windows/Linux and must never be silently hashed as one set.
        names = [(p.stem.casefold(), p.suffix.casefold()) for p in files]
        if len(names) != len(set(names)):
            raise ValueError(f"ambiguous duplicate shapefile members: {path}")
        multifile = True
    elif path.is_file():
        root = path.parent
        files = [path]
        multifile = False
    else:
        raise FileNotFoundError(f"input dataset is not a regular file or directory: {supplied}")

    if not files:
        raise ValueError(f"input dataset is empty: {supplied}")
    h = hashlib.sha256()
    for member in files:
        if not member.is_file():
            raise ValueError(f"unreadable or non-regular dataset member: {member}")
        if multifile:
            relative = member.relative_to(root).as_posix().encode("utf-8")
            h.update(len(relative).to_bytes(8, "big"))
            h.update(relative)
            h.update(member.stat().st_size.to_bytes(8, "big"))
        with member.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
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
    """Verify each artifact once, keeping only one group's tables in memory.

    Preserves complete hash, schema, cardinality and partition checks without
    retaining every group's Parquet frames simultaneously in large runs.
    """
    import geopandas as gpd
    manifest = folder / 'manifest.json'
    if not manifest.is_file():
        raise OutputPublicationError('manifest.json is missing')
    doc = json.loads(manifest.read_text(encoding='utf-8'))
    if doc.get('run') not in ('single', 'grouped'):
        raise OutputPublicationError('unknown manifest run type')
    output_info = {}

    def read_checked(rel: str):
        relative = Path(rel)
        if relative.is_absolute() or '..' in relative.parts:
            raise OutputPublicationError(f'unsafe output reference: {rel}')
        path = folder / relative
        if not path.is_file():
            raise OutputPublicationError(f'missing artifact: {rel}')
        if path.suffix == '.csv':
            frame = (pd.read_csv(path, keep_default_na=False, dtype={'group': str})
                     if rel == 'summary.csv' else pd.read_csv(path))
        elif 'points' in relative.parts or relative.name == 'clustered_points.parquet':
            frame = gpd.read_parquet(path)
        else:
            frame = pd.read_parquet(path)
        if not isinstance(frame, pd.DataFrame):
            raise OutputPublicationError(f'invalid output: {rel}')
        output_info[rel] = {'sha256': _hash(path), 'rows': len(frame)}
        return frame

    summary = read_checked('summary.csv')
    read_checked('distance_trace.csv')
    if doc['run'] == 'single':
        points = read_checked('clustered_points.parquet')
        clusters = read_checked('cluster_table.parquet')
        hierarchy = read_checked('cluster_hierarchy.parquet')
        if len(summary) != 1 or int(summary.iloc[0]['n_points']) != len(points):
            raise OutputPublicationError('single-run cardinality disagreement')
        _verify_partition(points, clusters, hierarchy)
    else:
        groups = doc.get('groups', {})
        if len(summary) != len(groups):
            raise OutputPublicationError('group-count disagreement')
        for group, group_info in groups.items():
            files = group_info['files']
            if set(files) != {'points', 'clusters', 'hierarchy'}:
                raise OutputPublicationError(f'incomplete group file references: {group}')
            # Frames are discarded on the next loop iteration; no across-group
            # cache is retained regardless of the number of groups.
            points = read_checked(files['points'])
            clusters = read_checked(files['clusters'])
            hierarchy = read_checked(files['hierarchy'])
            _verify_partition(points, clusters, hierarchy)
            row = summary.loc[summary['group'] == group]
            if len(row) != 1 or int(row.iloc[0]['n_points']) != len(points):
                raise OutputPublicationError(f'group count disagreement: {group}')
    if expected_inputs is not None and doc.get('input_sha256') != expected_inputs:
        raise OutputPublicationError('input checksum mismatch')
    if 'output_integrity' in doc and doc['output_integrity'] != output_info:
        raise OutputPublicationError('output integrity mismatch')
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


def _input_resolved_paths(kwargs: dict) -> dict[str, str]:
    """Absolute input paths after following symbolic links, for provenance."""
    return {name: str(Path(kwargs[name]).resolve()) for name in ('points_path', 'network_path')}


def _journal_path(destination: Path) -> Path:
    return destination.parent / ('.' + destination.name + '.publication-journal.json')


def _journal_record(destination: Path, backup: Path, *, previous: bool,
                    phase: str) -> dict:
    return {
        'version': 2,
        'destination': str(destination),
        'backup': str(backup),
        'previous_generation': bool(previous),
        'phase': phase,
    }


def _read_journal(destination: Path) -> tuple[Path, bool, str]:
    """Validate the journal *before* trusting any path it names.

    Version 1 was used before post-commit cleanup was distinguished from
    rollback. Treat its state as prepared, and never delete a verified
    destination simply because a possibly partial backup exists.
    """
    journal = _journal_path(destination)
    try:
        data = json.loads(journal.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError) as exc:
        raise OutputPublicationError('unreadable publication journal') from exc
    if not isinstance(data, dict) or data.get('destination') != str(destination.resolve()):
        raise OutputPublicationError('unsafe or incompatible publication journal')
    if data.get('version') == 1:
        previous, phase = True, 'prepared'
    elif data.get('version') == 2:
        previous, phase = data.get('previous_generation'), data.get('phase')
        if not isinstance(previous, bool) or phase not in ('prepared', 'committed'):
            raise OutputPublicationError('invalid publication journal state')
    else:
        raise OutputPublicationError('unsupported publication journal version')
    try:
        backup = Path(data['backup'])
    except (KeyError, TypeError) as exc:
        raise OutputPublicationError('journal is missing the backup path') from exc
    if (not backup.is_absolute() or backup.parent != destination.parent
            or not backup.name.startswith('.' + destination.name + '.backup-')
            or backup.name == '.' + destination.name + '.backup-'):
        raise OutputPublicationError('publication journal references unexpected backup')
    return backup, previous, phase


def _remove_committed_backup(destination: Path, backup: Path) -> None:
    """Irreversible cleanup: must NEVER be followed by predecessor rollback."""
    if backup.exists():
        shutil.rmtree(backup)
    _journal_path(destination).unlink()


def _recover_if_needed(destination: Path) -> None:
    """Recover a crash journal; keep every verified generation intact.

    A committed journal is authoritative: its backup may already be partially
    deleted and therefore is NEVER a rollback candidate. Prepared journals
    favor a verified new generation or restore a *verified* predecessor.
    """
    journal = _journal_path(destination)
    if not journal.exists():
        return
    backup, previous, phase = _read_journal(destination)
    if phase == 'committed':
        if not destination.exists():
            raise OutputPublicationError('committed destination missing; refusing incomplete-backup rollback')
        _verify_generation(destination)
        try:
            _remove_committed_backup(destination, backup)
        except BaseException as exc:
            raise PostCommitCleanupError(
                f'published generation at {destination} is intact; cleanup is pending'
            ) from exc
        return

    # An interrupted reversible phase. Never discard a candidate until the
    # predecessor has passed integrity verification.
    if previous and backup.exists():
        if destination.exists():
            try:
                _verify_generation(destination)
            except Exception:
                _verify_generation(backup)
                shutil.rmtree(destination)
                os.replace(backup, destination)
                journal.unlink()
                return
            # Candidate was verified and published before the crash. Record
            # commitment durably *before* any irreversible backup deletion.
            _write_json(journal, _journal_record(destination, backup, previous=True,
                                                  phase='committed'))
            _recover_if_needed(destination)
            return
        _verify_generation(backup)
        os.replace(backup, destination)
        journal.unlink()
        return

    if destination.exists():
        _verify_generation(destination)
        # Either old->backup never happened or this is a first publication.
        if not previous:
            _write_json(journal, _journal_record(destination, backup, previous=False,
                                                  phase='committed'))
            _recover_if_needed(destination)
        else:
            journal.unlink()
        return
    if previous:
        raise OutputPublicationError('missing destination and recovery backup')
    # First publication was interrupted before candidate installation.
    journal.unlink()


def _rollback_precommit(destination: Path, backup: Path, *, previous: bool,
                        moved_old: bool, published: bool) -> None:
    """Rollback only while the backup is known complete and unmodified."""
    journal = _journal_path(destination)
    if journal.exists():
        recorded_backup, recorded_previous, phase = _read_journal(destination)
        if recorded_backup != backup or recorded_previous != previous:
            raise OutputPublicationError('rollback journal differs from expected transaction')
        if phase == 'committed':
            # An error may have occurred after the atomic journal replacement
            # but before its temporary-file cleanup. Never restore the backup.
            raise PostCommitCleanupError(
                f'published generation at {destination}; cleanup is pending'
            )
    if moved_old:
        _verify_generation(backup)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(backup, destination)
    elif previous:
        # Backup rename never succeeded. Do not touch an unverified directory.
        if not destination.exists():
            raise OutputPublicationError('predecessor missing before its backup was moved')
        _verify_generation(destination)
    elif published and destination.exists():
        # A failed *first* publication has no previous generation to restore.
        shutil.rmtree(destination)
    if journal.exists():
        journal.unlink()


def transactional_file_runner(original):
    """Stage, validate, and publish file outputs with fail-fast recovery."""
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
        # An empty user directory is not a verified output generation either.
        if dest.exists():
            if not kw.get('force', False):
                raise FileExistsError(f'output exists: {dest}; use force=True or --force')
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
        # The native dependency check must precede input access/hashing.
        original.__globals__['_require_pyarrow']()
        resolved_inputs = _input_resolved_paths(kw)
        inputs = _input_checksums(kw)
        if _input_resolved_paths(kw) != resolved_inputs:
            raise OutputPublicationError('input symlink targets changed during checksum')
        stage = Path(tempfile.mkdtemp(prefix='.'+dest.name+'.stage-', dir=dest.parent))
        backup = dest.parent / ('.'+dest.name+'.backup-'+uuid.uuid4().hex)
        journal = _journal_path(dest)
        previous = dest.exists()
        moved_old = False
        published = False
        try:
            kw['output_dir'] = stage
            kw['force'] = False
            result = original(**kw)
            if _input_resolved_paths(kw) != resolved_inputs or _input_checksums(kw) != inputs:
                raise OutputPublicationError('input files or symlink targets changed during clustering')
            info = _verify_generation(stage)
            docpath = stage / 'manifest.json'
            doc = json.loads(docpath.read_text(encoding='utf-8'))
            doc['manifest_schema_version'] = 2
            doc['package'] = 'net-hdbscan'
            doc['input_sha256'] = inputs
            doc['input_resolved_paths'] = resolved_inputs
            doc['output_integrity'] = info
            _write_json(docpath, doc)
            _verify_generation(stage, expected_inputs=inputs)

            # Persist the prepared journal before changing the visible output.
            _write_json(journal, _journal_record(dest, backup, previous=previous,
                                                  phase='prepared'))
            if previous:
                os.replace(dest, backup)
                moved_old = True
            os.replace(stage, dest)
            published = True
            _verify_generation(dest, expected_inputs=inputs)

            # The irreversible boundary: from here on NEVER roll back a
            # predecessor that may be partially removed by cleanup.
            _write_json(journal, _journal_record(dest, backup, previous=previous,
                                                  phase='committed'))
        except BaseException:
            # If the committed journal exists, _rollback_precommit fails safely
            # without touching the published generation.
            _rollback_precommit(dest, backup, previous=previous,
                                moved_old=moved_old, published=published)
            raise
        finally:
            if stage.exists():
                shutil.rmtree(stage)

        try:
            _remove_committed_backup(dest, backup)
        except BaseException as exc:
            raise PostCommitCleanupError(
                f'published generation at {dest} is intact; cleanup is pending'
            ) from exc
        return result
    return call
