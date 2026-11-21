"""Fault-injection regression tests for publication, rollback, and crash recovery.

Synthetic generations intentionally isolate the filesystem protocol from Parquet
I/O. The full pipeline tests independently exercise real data validation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from net_hdbscan import publication as pub


def _require_pyarrow():
    """Used by the synthetic writer to satisfy the public wrapper contract."""


def _writer(*, points_path, network_path, output_dir, force=False):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'payload.txt').write_text(str(points_path), encoding='utf-8')
    (folder / 'manifest.json').write_text(json.dumps({'run': 'single'}), encoding='utf-8')
    return 'writer-ok'


def _generation(folder: Path, payload: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'payload.txt').write_text(payload, encoding='utf-8')
    (folder / 'manifest.json').write_text(json.dumps({
        'run': 'single', 'output_integrity': {'payload.txt': {'rows': 1, 'sha256': 'synthetic'}},
    }), encoding='utf-8')


def _fake_verify(folder: Path, *, expected_inputs=None) -> dict:
    if not (folder / 'manifest.json').is_file():
        raise pub.OutputPublicationError('missing manifest')
    if not (folder / 'payload.txt').is_file():
        raise pub.OutputPublicationError('missing payload')
    doc = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if expected_inputs is not None and doc.get('input_sha256') != expected_inputs:
        raise pub.OutputPublicationError('input checksum mismatch')
    return {'payload.txt': {'rows': 1, 'sha256': 'synthetic'}}


@pytest.fixture
def publication_env(monkeypatch, tmp_path):
    monkeypatch.setattr(pub, '_input_checksums', lambda _: {'source': 'synthetic'})
    monkeypatch.setattr(pub, '_verify_generation', _fake_verify)
    out = tmp_path / 'run'
    _generation(out, 'OLD')
    runner = pub.transactional_file_runner(_writer)
    kwargs = {'points_path': 'NEW', 'network_path': 'ignored', 'output_dir': out, 'force': True}
    return out, runner, kwargs


def _payload(out):
    return (out / 'payload.txt').read_text(encoding='utf-8')


def _backups(out):
    return list(out.parent.glob('.run.backup-*'))


def _journal(out):
    return pub._journal_path(out)


def test_overwrite_happy_path_and_recovery_is_idempotent(publication_env):
    out, runner, kwargs = publication_env
    assert runner(**kwargs) == 'writer-ok'
    assert _payload(out) == 'NEW'
    assert not _backups(out)
    assert not _journal(out).exists()
    pub._recover_if_needed(out)
    pub._recover_if_needed(out)
    assert _payload(out) == 'NEW'


def test_partial_backup_deletion_preserves_published_output(publication_env, monkeypatch):
    out, runner, kwargs = publication_env
    actual = pub.shutil.rmtree
    triggered = []
    def partial_delete(folder, *args, **kw):
        folder = Path(folder)
        if '.run.backup-' in folder.name and not triggered:
            (folder / 'payload.txt').unlink()
            triggered.append(True)
            raise OSError('injected partial backup deletion')
        return actual(folder, *args, **kw)
    with monkeypatch.context() as m:
        m.setattr(pub.shutil, 'rmtree', partial_delete)
        with pytest.raises(pub.PostCommitCleanupError, match='cleanup is pending'):
            runner(**kwargs)
    assert triggered
    assert _payload(out) == 'NEW'
    assert len(_backups(out)) == 1  # Incomplete backup; NEVER a rollback candidate.
    assert json.loads(_journal(out).read_text())['phase'] == 'committed'
    pub._recover_if_needed(out)
    pub._recover_if_needed(out)
    assert _payload(out) == 'NEW'
    assert not _backups(out)
    assert not _journal(out).exists()


def test_journal_unlink_failure_preserves_committed_generation(publication_env, monkeypatch):
    out, runner, kwargs = publication_env
    unlink = Path.unlink
    triggered = []
    def broken_unlink(path, *args, **kw):
        if path == _journal(out) and not triggered:
            triggered.append(True)
            raise OSError('injected journal cleanup failure')
        return unlink(path, *args, **kw)
    with monkeypatch.context() as m:
        m.setattr(Path, 'unlink', broken_unlink)
        with pytest.raises(pub.PostCommitCleanupError, match='cleanup is pending'):
            runner(**kwargs)
    assert triggered
    assert _payload(out) == 'NEW'
    assert not _backups(out)
    assert json.loads(_journal(out).read_text())['phase'] == 'committed'
    pub._recover_if_needed(out)
    pub._recover_if_needed(out)
    assert _payload(out) == 'NEW'
    assert not _journal(out).exists()


def test_stage_verification_failure_does_not_touch_predecessor(publication_env, monkeypatch):
    out, runner, kwargs = publication_env
    def verifier(folder, *, expected_inputs=None):
        if '.run.stage-' in folder.name:
            raise pub.OutputPublicationError('injected stage validation')
        return _fake_verify(folder, expected_inputs=expected_inputs)
    monkeypatch.setattr(pub, '_verify_generation', verifier)
    with pytest.raises(pub.OutputPublicationError, match='stage validation'):
        runner(**kwargs)
    assert _payload(out) == 'OLD'
    assert not _backups(out)
    assert not _journal(out).exists()


@pytest.mark.parametrize('failure_point', ['old_to_backup', 'stage_to_dest'])
def test_directory_swap_failure_rolls_back(publication_env, monkeypatch, failure_point):
    out, runner, kwargs = publication_env
    replace = pub.os.replace
    triggered = []
    def fail_once(src, dst):
        src, dst = Path(src), Path(dst)
        hits = ((failure_point == 'old_to_backup' and src == out and '.run.backup-' in dst.name)
                or (failure_point == 'stage_to_dest' and '.run.stage-' in src.name and dst == out))
        if hits and not triggered:
            triggered.append(True)
            raise OSError('injected directory swap failure')
        return replace(src, dst)
    with monkeypatch.context() as m:
        m.setattr(pub.os, 'replace', fail_once)
        with pytest.raises(OSError, match='directory swap failure'):
            runner(**kwargs)
    assert triggered
    assert _payload(out) == 'OLD'
    assert not _backups(out)
    assert not _journal(out).exists()


def test_candidate_verification_failure_rolls_back(publication_env, monkeypatch):
    out, runner, kwargs = publication_env
    def verifier(folder, *, expected_inputs=None):
        if folder == out and _payload(folder) == 'NEW':
            raise pub.OutputPublicationError('injected candidate verification')
        return _fake_verify(folder, expected_inputs=expected_inputs)
    monkeypatch.setattr(pub, '_verify_generation', verifier)
    with pytest.raises(pub.OutputPublicationError, match='candidate verification'):
        runner(**kwargs)
    assert _payload(out) == 'OLD'
    assert not _backups(out)
    assert not _journal(out).exists()


def test_failed_restore_leaves_complete_backup_for_recovery(publication_env, monkeypatch):
    out, runner, kwargs = publication_env
    replace = pub.os.replace
    count = {'candidate': 0, 'restore': 0}
    def fail(src, dst):
        src, dst = Path(src), Path(dst)
        if '.run.stage-' in src.name and dst == out:
            count['candidate'] += 1
            raise OSError('injected candidate rename')
        if '.run.backup-' in src.name and dst == out:
            count['restore'] += 1
            raise OSError('injected backup restore')
        return replace(src, dst)
    with monkeypatch.context() as m:
        m.setattr(pub.os, 'replace', fail)
        with pytest.raises(OSError, match='backup restore'):
            runner(**kwargs)
    assert count == {'candidate': 1, 'restore': 1}
    assert not out.exists()
    assert len(_backups(out)) == 1
    assert _payload(_backups(out)[0]) == 'OLD'
    assert _journal(out).exists()
    pub._recover_if_needed(out)
    pub._recover_if_needed(out)
    assert _payload(out) == 'OLD'
    assert not _journal(out).exists()


def test_crash_recovery_prepared_missing_destination(publication_env):
    out, _, _ = publication_env
    backup = out.parent / '.run.backup-pretend'
    pub._write_json(_journal(out), pub._journal_record(out, backup, previous=True, phase='prepared'))
    os.replace(out, backup)
    pub._recover_if_needed(out)
    pub._recover_if_needed(out)
    assert _payload(out) == 'OLD'
    assert not _backups(out)
    assert not _journal(out).exists()


def test_crash_recovery_after_candidate_installed_before_commit(publication_env):
    out, _, _ = publication_env
    backup = out.parent / '.run.backup-pretend'
    pub._write_json(_journal(out), pub._journal_record(out, backup, previous=True, phase='prepared'))
    os.replace(out, backup)
    _generation(out, 'NEW')
    pub._recover_if_needed(out)
    pub._recover_if_needed(out)
    assert _payload(out) == 'NEW'
    assert not _backups(out)
    assert not _journal(out).exists()


def test_committed_crash_with_partial_backup_never_restores_it(publication_env):
    out, _, _ = publication_env
    backup = out.parent / '.run.backup-pretend'
    pub._write_json(_journal(out), pub._journal_record(out, backup, previous=True, phase='committed'))
    os.replace(out, backup)
    _generation(out, 'NEW')
    (backup / 'payload.txt').unlink()
    pub._recover_if_needed(out)
    pub._recover_if_needed(out)
    assert _payload(out) == 'NEW'
    assert not _backups(out)
    assert not _journal(out).exists()


def test_committed_missing_destination_refuses_incomplete_backup(publication_env):
    out, _, _ = publication_env
    backup = out.parent / '.run.backup-pretend'
    pub._write_json(_journal(out), pub._journal_record(out, backup, previous=True, phase='committed'))
    os.replace(out, backup)
    (backup / 'payload.txt').unlink()
    with pytest.raises(pub.OutputPublicationError, match='committed destination missing'):
        pub._recover_if_needed(out)
    assert backup.exists()
    assert _journal(out).exists()


def test_empty_existing_directory_is_not_overwritten(monkeypatch, tmp_path):
    monkeypatch.setattr(pub, '_input_checksums', lambda _: {'source': 'synthetic'})
    monkeypatch.setattr(pub, '_verify_generation', _fake_verify)
    dest = tmp_path / 'run'
    dest.mkdir()
    with pytest.raises(pub.OutputPublicationError, match='missing manifest'):
        pub.transactional_file_runner(_writer)(points_path='NEW', network_path='ignored',
                                                 output_dir=dest, force=True)
    assert not list(dest.iterdir())


def test_fresh_run_directory_swap_failure_leaves_no_result(monkeypatch, tmp_path):
    monkeypatch.setattr(pub, '_input_checksums', lambda _: {'source': 'synthetic'})
    monkeypatch.setattr(pub, '_verify_generation', _fake_verify)
    dest = tmp_path / 'run'
    replace = pub.os.replace
    def broken(src, dst):
        if '.run.stage-' in Path(src).name and Path(dst) == dest:
            raise OSError('injected first publication failure')
        return replace(src, dst)
    with monkeypatch.context() as m:
        m.setattr(pub.os, 'replace', broken)
        with pytest.raises(OSError, match='first publication failure'):
            pub.transactional_file_runner(_writer)(points_path='NEW',network_path='ignored',output_dir=dest)
    assert not dest.exists()
    assert not _journal(dest).exists()


@pytest.mark.parametrize('after_write', [False, True])
def test_committed_journal_write_exception_is_state_aware(publication_env, monkeypatch, after_write):
    out, runner, kwargs = publication_env
    write = pub._write_json
    triggered = []
    def fail_commit_marker(path, payload):
        if path == _journal(out) and payload.get('phase') == 'committed' and not triggered:
            triggered.append(True)
            if after_write:
                write(path, payload)
            raise OSError('injected commit-journal error')
        return write(path, payload)
    with monkeypatch.context() as m:
        m.setattr(pub, '_write_json', fail_commit_marker)
        with pytest.raises(
            pub.PostCommitCleanupError if after_write else OSError,
            match='cleanup is pending' if after_write else 'commit-journal error',
        ):
            runner(**kwargs)
    assert triggered
    if after_write:
        assert _payload(out) == 'NEW'
        assert json.loads(_journal(out).read_text())['phase'] == 'committed'
    else:
        assert _payload(out) == 'OLD'
        assert not _journal(out).exists()
    pub._recover_if_needed(out)
    assert _payload(out) == ('NEW' if after_write else 'OLD')


def test_prepared_recovery_invalid_candidate_restores_verified_predecessor(publication_env):
    out, _, _ = publication_env
    backup = out.parent / '.run.backup-pretend'
    pub._write_json(_journal(out), pub._journal_record(out, backup, previous=True, phase='prepared'))
    os.replace(out, backup)
    _generation(out, 'NEW')
    (out / 'payload.txt').unlink()
    pub._recover_if_needed(out)
    pub._recover_if_needed(out)
    assert _payload(out) == 'OLD'
    assert not _backups(out)
    assert not _journal(out).exists()


def test_fresh_published_output_is_recovered_after_precommit_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(pub, '_verify_generation', _fake_verify)
    out = tmp_path / 'run'
    backup = tmp_path / '.run.backup-notused'
    _generation(out, 'NEW')
    pub._write_json(_journal(out), pub._journal_record(out, backup, previous=False, phase='prepared'))
    pub._recover_if_needed(out)
    pub._recover_if_needed(out)
    assert _payload(out) == 'NEW'
    assert not _journal(out).exists()
