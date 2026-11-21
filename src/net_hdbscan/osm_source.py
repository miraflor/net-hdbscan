"""Optional acquisition of *undirected geometric* OSM line networks.

This adapter is deliberately outside the HDBSCAN core. It retrieves and caches
spatial lines; it does not turn road access, flow direction, or travel time into
a distance model. The existing network-space pipeline uses symmetric lengths.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from datetime import datetime, timezone
from typing import NamedTuple

import geopandas as gpd

OSM_PROFILES = frozenset({'all', 'all_public', 'bike', 'drive', 'drive_service', 'walk'})
_MAX_BBOX_KM2 = 10_000.0  # Guard public Overpass services; use regional extracts for larger jobs.
_ADAPTER_SCHEMA = 1
_FILTER_PATTERN = re.compile(r'(?:\[[^\[\]{};\r\n]{1,200}\])+\Z')


class BoundingBox(NamedTuple):
    west: float
    south: float
    east: float
    north: float


def parse_bbox(value: str | tuple[float, float, float, float]) -> BoundingBox:
    """Parse EPSG:4326 `west,south,east,north`; dateline crossing unsupported."""
    try:
        coords = ([float(part.strip()) for part in value.split(',')]
                  if isinstance(value, str) else [float(part) for part in value])
    except (ValueError, TypeError) as exc:
        raise ValueError('bbox must be west,south,east,north in longitude/latitude degrees') from exc
    if len(coords) != 4 or not all(math.isfinite(c) for c in coords):
        raise ValueError('bbox must contain four finite longitude/latitude numbers')
    west, south, east, north = coords
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError('bbox coordinates must be west < east and south < north, within world bounds; antimeridian crossing is unsupported')
    # Spherical area, adequate for rejecting accidentally enormous public API queries.
    spherical_area = (6371.0088 ** 2 * math.radians(east-west)
                      * (math.sin(math.radians(north))-math.sin(math.radians(south))))
    if spherical_area > _MAX_BBOX_KM2:
        raise ValueError(f'bbox area is approximately {spherical_area:,.0f} km2; public OSM requests are capped at {_MAX_BBOX_KM2:,.0f} km2; use a regional extract or a smaller bbox')
    return BoundingBox(west, south, east, north)


def _validate_selector(profile: str | None, custom_filter: str | None) -> tuple[str | None, str | None]:
    if (profile is None) == (custom_filter is None):
        raise ValueError('select exactly one of --osm-profile or --osm-filter; network semantics cannot be inferred')
    if profile is not None and profile not in OSM_PROFILES:
        raise ValueError(f'unknown OSM profile {profile!r}; choose from {sorted(OSM_PROFILES)}')
    if custom_filter is not None:
        custom_filter = custom_filter.strip()
        if len(custom_filter) > 500 or not _FILTER_PATTERN.fullmatch(custom_filter):
            raise ValueError('OSM filter must be a bracketed Overpass way-tag expression (no braces, semicolons, or newlines)')
    return profile, custom_filter


def _default_cache_dir() -> Path:
    local = os.environ.get('LOCALAPPDATA')
    return (Path(local) / 'net-hdbscan' / 'osm' if local
            else Path.home() / '.cache' / 'net-hdbscan' / 'osm')


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _request(bbox: BoundingBox, profile: str | None, custom_filter: str | None) -> dict:
    return {
        'adapter_schema': _ADAPTER_SCHEMA,
        'source': 'openstreetmap',
        'bbox_epsg4326_wsen': list(bbox),
        'osm_profile': profile,
        'osm_custom_filter': custom_filter,
        # Explicit: an access-filtered OSM graph, not a directional routing model.
        'distance_model': 'undirected_projected_polyline_length',
    }


def _validate_cache(entry: Path, request: dict) -> Path:
    if entry.is_symlink():
        raise ValueError(f'refusing symlinked OSM cache entry: {entry}')
    network = entry / 'network.gpkg'
    metadata = entry / 'source.json'
    try:
        data = json.loads(metadata.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f'incomplete or unreadable OSM cache; refusing automatic replacement: {entry}') from exc
    if data.get('request') != request or not network.is_file():
        raise ValueError(f'OSM cache does not match the requested network: {entry}')
    if data.get('network_sha256') != _sha256(network):
        raise ValueError(f'OSM cache checksum mismatch; refusing to use it: {network}')
    return network


def fetch_osm_network(*, bbox: str | tuple[float, float, float, float],
                      profile: str | None = None, custom_filter: str | None = None,
                      cache_dir: str | Path | None = None) -> Path:
    """Return an immutable cached GPKG line network; fetch once if absent.

    Internet and OSMnx are required only on a cache miss. This does not create
    HDBSCAN inputs or output datasets, nor silently update an existing extract.
    """
    area = parse_bbox(bbox)
    profile, custom_filter = _validate_selector(profile, custom_filter)
    request = _request(area, profile, custom_filter)
    key = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]
    root = Path(cache_dir).expanduser().resolve() if cache_dir else _default_cache_dir().resolve()
    entry = root / key
    if entry.exists() or entry.is_symlink():
        return _validate_cache(entry, request)
    try:
        import osmnx as ox
    except ImportError as exc:
        raise ImportError('OSM retrieval requires OSMnx 2.x; install with: pip install -e ".[osm]"') from exc
    if getattr(ox, '__version__', '2.0').split('.')[0] != '2' and not str(getattr(ox, '__version__', '')).startswith('2-'):
        raise ImportError('OSM acquisition requires OSMnx 2.x; install net-hdbscan[osm]')
    root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{key}.stage-', dir=root))
    try:
        opts = {'simplify': False, 'retain_all': True}
        if custom_filter is not None:
            opts['custom_filter'] = custom_filter
        else:
            opts['network_type'] = profile
        # OSMnx 2.x accepts (west, south, east, north) in the bbox argument.
        graph = ox.graph.graph_from_bbox(tuple(area), **opts)
        if not len(graph.nodes) or not len(graph.edges):
            raise ValueError('OSM returned no connected line features inside the requested bbox')
        projected = ox.projection.project_graph(graph)
        edges = ox.convert.graph_to_gdfs(projected, nodes=False, edges=True, fill_edge_geometry=True)
        if edges.empty or edges.crs is None or not edges.crs.is_projected:
            raise ValueError('OSM retrieval failed to produce nonempty projected network geometry')
        # Never write OSM one-way metadata into a pipeline that uses a symmetric metric.
        # Write only the geometry; osm access filtering is not vehicle routing.
        lines = gpd.GeoDataFrame(geometry=edges.geometry.reset_index(drop=True), crs=edges.crs)
        from .io import prepare_network
        prepare_network(lines)
        gpkg = stage / 'network.gpkg'
        lines.to_file(gpkg, driver='GPKG', layer='network', index=False)
        # Read-back checks the actual output serialization, not only the in-memory graph.
        prepare_network(gpd.read_file(gpkg, layer='network'))
        metadata = {
            'request': request,
            'retrieved_utc': datetime.now(timezone.utc).isoformat(),
            'provider': 'OpenStreetMap contributors',
            'license': 'ODbL 1.0',
            'attribution_url': 'https://www.openstreetmap.org/copyright',
            'provider_software': f'osmnx {getattr(ox, "__version__", "unknown")}',
            'network_crs': str(lines.crs),
            'line_features': len(lines),
            'network_sha256': _sha256(gpkg),
            'interpretation': 'undirected geometric line network; directions, flow and travel restrictions are not modeled',
        }
        (stage / 'source.json').write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        # An immutable directory becomes visible only after both validated files exist.
        if entry.exists() or entry.is_symlink():
            raise FileExistsError(f'OSM cache was populated concurrently; verify it and retry: {entry}')
        os.replace(stage, entry)
        return _validate_cache(entry, request)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
