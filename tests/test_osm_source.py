"""Offline, fail-closed regression tests for the optional OSM network adapter."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString

from net_hdbscan.osm_source import fetch_osm_network, parse_bbox


@pytest.mark.parametrize('bbox', [
    '', '1,2,3', '1,2,3,4,5', 'nan,0,2,3', '1,0,1,3',
    '4,0,3,2', '-181,0,1,2', '0,-91,2,3', '170,-3,-170,3',
    '0,0,3,3',
])
def test_bbox_rejects_invalid_and_oversized_requests(bbox):
    with pytest.raises(ValueError):
        parse_bbox(bbox)


def test_bbox_wsen_order_is_explicit():
    assert tuple(parse_bbox('120.5,14.4,120.6,14.5')) == (120.5, 14.4, 120.6, 14.5)


@pytest.mark.parametrize('profile,custom_filter', [
    (None, None), ('walk', '[waterway=river]'), ('hippo', None),
    (None, 'waterway=river'), (None, '[waterway=river];out;'),
    (None, '[waterway=river]\n[out:json]'),
])
def test_selector_requires_single_explicit_network_semantics(profile, custom_filter, tmp_path):
    with pytest.raises(ValueError):
        fetch_osm_network(bbox='120,14,120.01,14.01', profile=profile,
                          custom_filter=custom_filter, cache_dir=tmp_path / 'cache')
    assert not (tmp_path / 'cache').exists()


@pytest.fixture
def fake_osmnx(monkeypatch):
    calls = []
    projected_lines = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(10, 0), (20, 0)])],
        crs='EPSG:3857',
    )
    graph = types.SimpleNamespace(nodes={1: None, 2: None, 3: None}, edges=[(1, 2), (2, 3)])
    def get_graph(bbox, **opts):
        calls.append((bbox, opts))
        return graph
    module = types.SimpleNamespace(
        __version__='2-test',
        graph=types.SimpleNamespace(graph_from_bbox=get_graph),
        projection=types.SimpleNamespace(project_graph=lambda graph: graph),
        convert=types.SimpleNamespace(graph_to_gdfs=lambda graph, **opts: projected_lines),
    )
    monkeypatch.setitem(sys.modules, 'osmnx', module)
    return calls


def test_network_is_persistent_cached_and_contains_provenance(tmp_path, fake_osmnx, monkeypatch):
    root = tmp_path / 'cache'
    opts = dict(bbox='120.51,14.58,120.52,14.59', profile='walk', cache_dir=root)
    first = fetch_osm_network(**opts)
    assert first.is_file() and first.name == 'network.gpkg'
    data = json.loads(first.with_name('source.json').read_text(encoding='utf-8'))
    assert data['request']['bbox_epsg4326_wsen'] == [120.51, 14.58, 120.52, 14.59]
    assert data['request']['distance_model'] == 'undirected_projected_polyline_length'
    assert data['license'] == 'ODbL 1.0'
    assert len(gpd.read_file(first)) == 2
    assert fake_osmnx == [((120.51, 14.58, 120.52, 14.59),
                           {'simplify': False, 'retain_all': True, 'network_type': 'walk'})]
    monkeypatch.delitem(sys.modules, 'osmnx')
    second = fetch_osm_network(**opts)
    assert first == second and len(fake_osmnx) == 1


def test_custom_filter_is_passed_explicitly(tmp_path, fake_osmnx):
    fetch_osm_network(bbox='120,14,120.01,14.01', custom_filter='[waterway~"river|stream"]',
                      cache_dir=tmp_path)
    assert fake_osmnx[0][1] == {
        'simplify': False, 'retain_all': True,
        'custom_filter': '[waterway~"river|stream"]',
    }


def test_corrupt_cache_is_never_silently_replaced(tmp_path, fake_osmnx):
    opts = dict(bbox='120,14,120.01,14.01', profile='all', cache_dir=tmp_path)
    first = fetch_osm_network(**opts)
    first.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='checksum mismatch'):
        fetch_osm_network(**opts)
    assert len(fake_osmnx) == 1


def test_failed_write_leaves_no_visible_cache_entry(tmp_path, fake_osmnx, monkeypatch):
    def fail_to_file(*args, **kwargs):
        raise OSError('injected disk failure')
    monkeypatch.setattr(gpd.GeoDataFrame, 'to_file', fail_to_file)
    with pytest.raises(OSError, match='injected'):
        fetch_osm_network(bbox='120,14,120.01,14.01', profile='walk', cache_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_cli_works_without_network_file_only_with_explicit_osm_selection(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from net_hdbscan import cli
    calls = []
    network_file = tmp_path / 'cached-network.gpkg'
    def fake_fetch(**kw):
        calls.append(('fetch', kw))
        return network_file
    def fake_cluster_files(**kw):
        calls.append(('cluster', kw))
        raise RuntimeError('pipeline reached')
    monkeypatch.setattr(cli, 'fetch_osm_network', fake_fetch)
    monkeypatch.setattr(cli, 'cluster_files', fake_cluster_files)
    base = ['cluster', '--points', 'observations.gpkg', '--output-dir', str(tmp_path / 'out'),
            '--max-distance', '100', '--min-cluster-size', '5']
    runner = CliRunner()
    missing = runner.invoke(cli.app, base)
    assert missing.exit_code != 0 and calls == []
    unsafe = runner.invoke(cli.app, base + ['--network-source', 'osm', '--bbox', '120,14,120.1,14.1'])
    assert unsafe.exit_code != 0 and calls == []
    valid = runner.invoke(cli.app, base + ['--network-source', 'osm', '--bbox', '120,14,120.1,14.1',
                                          '--osm-profile', 'walk'])
    assert valid.exit_code != 0 and isinstance(valid.exception, RuntimeError)
    assert str(valid.exception) == 'pipeline reached'
    assert calls[0][0] == 'fetch'
    assert calls[1][1]['network_path'] == network_file


@pytest.mark.parametrize('command', ['cluster', 'fetch-network'])
@pytest.mark.parametrize('options', [
    [],
    ['--osm-profile', 'imaginary'],
    ['--osm-profile', 'walk', '--osm-filter', '[waterway=river]'],
    ['--osm-filter', 'waterway=river'],
])
def test_cli_rejects_ambiguous_or_invalid_osm_selection_before_fetch(
        monkeypatch, tmp_path, command, options):
    from typer.testing import CliRunner
    from net_hdbscan import cli
    seen = []
    def forbidden_fetch(**kwargs):
        seen.append(kwargs)
        raise AssertionError('source fetch should not have been attempted')
    monkeypatch.setattr(cli, 'fetch_osm_network', forbidden_fetch)
    if command == 'cluster':
        args = ['cluster', '--points', 'observations.gpkg',
                '--output-dir', str(tmp_path / 'output'), '--max-distance', '100',
                '--network-source', 'osm', '--bbox', '120,14,120.1,14.1']
    else:
        args = ['fetch-network', '--source', 'osm', '--bbox', '120,14,120.1,14.1']
    result = CliRunner().invoke(cli.app, args + options)
    assert result.exit_code != 0
    assert seen == []
    assert not isinstance(result.exception, AssertionError)


def test_cli_disallows_mixed_sources_and_domain_assumptions(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    from net_hdbscan import cli
    monkeypatch.setattr(cli, 'fetch_osm_network', lambda **kwargs: pytest.fail('must not fetch'))
    base = ['cluster', '--points', 'observations.gpkg', '--network', 'user.gpkg',
            '--max-distance', '100', '--output-dir', str(tmp_path/'out')]
    result = CliRunner().invoke(cli.app, base + ['--network-source', 'osm',
                              '--bbox', '120,14,120.1,14.1', '--osm-profile', 'walk'])
    assert result.exit_code != 0


def test_missing_osmnx_fails_before_cache_creation(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'osmnx', None)
    folder = tmp_path / 'absent'
    with pytest.raises(ImportError, match='OSMnx 2.x'):
        fetch_osm_network(bbox='120,14,120.01,14.01', profile='walk', cache_dir=folder)
    assert not folder.exists()


def test_missing_cache_metadata_does_not_retry_online(tmp_path, fake_osmnx):
    first = fetch_osm_network(bbox='120,14,120.01,14.01', profile='walk', cache_dir=tmp_path)
    first.with_name('source.json').unlink()
    with pytest.raises(ValueError, match='incomplete or unreadable OSM cache'):
        fetch_osm_network(bbox='120,14,120.01,14.01', profile='walk', cache_dir=tmp_path)
    assert len(fake_osmnx) == 1
