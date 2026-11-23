"""Offline tests for named OSM waterways and geodesic circle acquisition."""
from __future__ import annotations

import json
import math
import sys
import types

import geopandas as gpd
import pytest
from pyproj import Geod
from shapely.geometry import LineString, MultiLineString
from typer.testing import CliRunner

from net_hdbscan.region import Circle, circle_polygon, clip_lines_to_circle, parse_circle
from net_hdbscan.osm_source import WATERWAY_FILTER, fetch_osm_network


def test_center_order_radius_and_geodesic_circle():
    circle = parse_circle('121.04,14.65', 10)
    assert circle == Circle(121.04, 14.65, 10)
    polygon = circle_polygon(circle)
    assert len(polygon.exterior.coords) == 361
    geod = Geod(ellps='WGS84')
    for x, y in list(polygon.exterior.coords)[:-1]:
        _, _, m = geod.inv(circle.longitude, circle.latitude, x, y)
        assert m == pytest.approx(10000, abs=1e-5)


@pytest.mark.parametrize('center,radius', [
    ('', 10), ('1', 10), ('1,2,3', 10), ('nan,2', 1),
    ('181,0', 1), ('0,91', 1), ('0,90', 1), ('0,-90', 1),
    ('0,0', 0), ('0,0', -1), ('0,0', math.inf),
    ('0,0', 100), ('179.999,0', 1), ('0,89.999', 1),
])
def test_bad_circles_rejected(center, radius):
    with pytest.raises(ValueError):
        parse_circle(center, radius)


def test_circle_clip_is_not_bounding_rectangle():
    polygon = circle_polygon(parse_circle('0,0', 1))
    lines = gpd.GeoDataFrame(geometry=[
        LineString([(-2000, 0), (2000, 0)]),
        LineString([(900, 900), (1100, 1100)]),
        MultiLineString([[(-3000, 100), (3000, 100)], [(2000, 2000), (2500, 2500)]]),
    ], crs='EPSG:3857')
    clipped = clip_lines_to_circle(lines, polygon)
    assert len(clipped) == 2
    assert set(clipped.geom_type) == {'LineString'}
    boundary = gpd.GeoSeries([polygon], crs='EPSG:4326').to_crs('EPSG:3857').iloc[0]
    for line in clipped.geometry:
        assert boundary.buffer(1e-7).covers(line)
        assert line.length > 1000
    outside = gpd.GeoDataFrame(geometry=[LineString([(2000, 2000), (3000, 3000)])], crs='EPSG:3857')
    with pytest.raises(ValueError, match='no line geometry'):
        clip_lines_to_circle(outside, polygon)


@pytest.fixture
def fake_osmnx(monkeypatch):
    calls = []
    projected = gpd.GeoDataFrame(geometry=[
        LineString([(-2000, 0), (2000, 0)]),
        LineString([(900, 900), (1100, 1100)]),
    ], crs='EPSG:3857')
    graph = types.SimpleNamespace(nodes={1: None, 2: None}, edges=[(1, 2)])

    def get_polygon(polygon, **opts):
        calls.append(('polygon', polygon, opts))
        return graph

    def get_bbox(bbox, **opts):
        calls.append(('bbox', bbox, opts))
        return graph

    mock = types.SimpleNamespace(
        __version__='2-test',
        graph=types.SimpleNamespace(graph_from_bbox=get_bbox, graph_from_polygon=get_polygon),
        projection=types.SimpleNamespace(project_graph=lambda graph: graph),
        convert=types.SimpleNamespace(graph_to_gdfs=lambda graph, **kwargs: projected),
    )
    monkeypatch.setitem(sys.modules, 'osmnx', mock)
    return calls


def test_waterway_profile_uses_custom_filter_with_bbox(tmp_path, fake_osmnx):
    fetch_osm_network(bbox='0,0,0.01,0.01', profile='waterways', cache_dir=tmp_path)
    assert fake_osmnx[0] == ('bbox', (0.0, 0.0, 0.01, 0.01), {
        'simplify': False, 'retain_all': True, 'custom_filter': WATERWAY_FILTER,
    })


def test_circle_fetch_uses_polygon_clips_and_caches(tmp_path, fake_osmnx, monkeypatch):
    args = dict(center='0,0', radius_km=1, profile='waterways', cache_dir=tmp_path)
    gpkg = fetch_osm_network(**args)
    assert gpkg.is_file()
    assert len(fake_osmnx) == 1
    how, polygon, opts = fake_osmnx[0]
    assert how == 'polygon'
    assert opts == {'simplify': False, 'retain_all': True,
                    'truncate_by_edge': True, 'custom_filter': WATERWAY_FILTER}
    assert len(polygon.exterior.coords) == 361
    data = json.loads(gpkg.with_name('source.json').read_text(encoding='utf-8'))
    assert data['request']['circle_epsg4326_lon_lat_radius_km'] == [0.0, 0.0, 1.0]
    assert data['request']['osm_profile'] == 'waterways'
    assert data['clipped_to_circle'] is True
    lines = gpd.read_file(gpkg)
    assert len(lines) == 1
    assert lines.geometry.iloc[0].length < 2100
    monkeypatch.setitem(sys.modules, 'osmnx', None)
    assert fetch_osm_network(**args) == gpkg
    assert len(fake_osmnx) == 1


@pytest.mark.parametrize('kwargs', [
    {},
    {'bbox': '0,0,0.01,0.01', 'center': '0,0', 'radius_km': 1},
    {'center': '0,0'},
    {'radius_km': 1},
    {'center': '0,0', 'radius_km': 0},
])
def test_invalid_region_fails_before_cache_or_network(tmp_path, kwargs):
    cache = tmp_path / 'cache'
    with pytest.raises(ValueError):
        fetch_osm_network(profile='waterways', cache_dir=cache, **kwargs)
    assert not cache.exists()


@pytest.mark.parametrize('command', ['cluster', 'fetch-network'])
@pytest.mark.parametrize('flags', [
    ['--bbox', '0,0,0.01,0.01', '--center', '0,0', '--radius-km', '1'],
    ['--center', '0,0'],
    ['--radius-km', '1'],
    ['--center', '0,0', '--radius-km', '0'],
    ['--center', '0,0', '--radius-km', '1', '--osm-profile', 'walk', '--osm-filter', '[waterway=river]'],
])
def test_cli_invalid_region_fails_before_fetch(command, flags, monkeypatch, tmp_path):
    from net_hdbscan import cli
    called = []
    monkeypatch.setattr(cli, 'fetch_osm_network', lambda **kwargs: called.append(kwargs))
    if command == 'cluster':
        base = ['cluster', '--points', 'observations.gpkg', '--network-source', 'osm',
                '--max-distance', '100', '--output-dir', str(tmp_path / 'out')]
    else:
        base = ['fetch-network', '--source', 'osm']
    if '--osm-profile' not in flags and '--osm-filter' not in flags:
        base += ['--osm-profile', 'waterways']
    result = CliRunner().invoke(cli.app, base + flags)
    assert result.exit_code != 0
    assert called == []


def test_cli_circle_accepts_waterways(monkeypatch, tmp_path):
    from net_hdbscan import cli
    calls = []
    fake_gpkg = tmp_path / 'network.gpkg'

    def fake_fetch(**kwargs):
        calls.append(kwargs)
        return fake_gpkg

    def stop_at_pipeline(**kwargs):
        raise RuntimeError('pipeline reached')

    monkeypatch.setattr(cli, 'fetch_osm_network', fake_fetch)
    monkeypatch.setattr(cli, 'cluster_files', stop_at_pipeline)
    base = ['--center', '121.04,14.65', '--radius-km', '10', '--osm-profile', 'waterways']
    runner = CliRunner()
    cluster = runner.invoke(cli.app, ['cluster', '--points', 'points.gpkg',
        '--network-source', 'osm', '--max-distance', '100',
        '--output-dir', str(tmp_path / 'output')] + base)
    assert isinstance(cluster.exception, RuntimeError)
    assert calls[-1]['bbox'] is None
    assert calls[-1]['center'] == '121.04,14.65'
    assert calls[-1]['radius_km'] == 10
    fetched = runner.invoke(cli.app, ['fetch-network', '--source', 'osm'] + base)
    assert fetched.exit_code == 0
    assert calls[-1]['center'] == '121.04,14.65'
