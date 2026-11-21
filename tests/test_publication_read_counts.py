"""Verifies integrity checks decode each artifact once per complete scan."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import pytest
from net_hdbscan import publication as pub


def test_single_verification_reuses_parsed_artifacts(tmp_path, monkeypatch):
    folder = tmp_path / 'run'
    folder.mkdir()
    paths = ('summary.csv', 'distance_trace.csv', 'clustered_points.parquet',
             'cluster_table.parquet', 'cluster_hierarchy.parquet')
    for name in paths:
        (folder / name).write_bytes(b'dummy')
    (folder / 'manifest.json').write_text(json.dumps({'run': 'single'}))
    counts = {}
    def note(path):
        name = Path(path).name
        counts[name] = counts.get(name, 0) + 1
        if name == 'summary.csv':
            return pd.DataFrame({'n_points': [0]})
        if name == 'distance_trace.csv':
            return pd.DataFrame()
        if name == 'clustered_points.parquet':
            return pd.DataFrame({key: pd.Series(dtype=dtype) for key, dtype in {
                'cluster_id': object, 'is_noise': bool, 'membership': float,
                'core_distance': float, 'snap_distance': float,
                'snapped_x': float, 'snapped_y': float}.items()})
        if name == 'cluster_table.parquet':
            return pd.DataFrame(columns=['cluster_id', 'n_points', 'n_positions',
                                         'stability', 'birth_distance', 'singleton'])
        if name == 'cluster_hierarchy.parquet':
            return pd.DataFrame(columns=['hierarchy_id', 'parent_id', 'selected', 'cluster_id'])
        raise AssertionError(name)
    monkeypatch.setattr(pd, 'read_csv', lambda path, **kwargs: note(path))
    monkeypatch.setattr(pd, 'read_parquet', lambda path, **kwargs: note(path))
    import geopandas as gpd
    monkeypatch.setattr(gpd, 'read_parquet', lambda path, **kwargs: note(path))
    info = pub._verify_generation(folder)
    assert set(info) == set(paths)
    assert counts == {name: 1 for name in paths}
