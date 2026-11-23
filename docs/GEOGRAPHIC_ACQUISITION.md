# Geographic network acquisition: roads, waterways, boxes and circles

This guide describes the **implemented** opt-in OpenStreetMap (OSM) acquisition adapter in `net-hdbscan`, including its named `waterways` profile and geodesic center-plus-radius regions. For the science of the clustering algorithm itself, see the [algorithm guide](https://miraflor.github.io/net-hdbscan/). For source/caching contracts, see [Network sources](NETWORK_SOURCES.md); for scientific applicability, see [Applications](APPLICATIONS.md).

> **Fundamental distinction:** A geographic region limits **which line-network geometry is fetched**. It does **not** filter the observation points passed to `cluster`. The line network represents a **two-dimensional, undirected, geometric shortest-path distance**, regardless of whether OSM selected paths, roads, or waterways. Neither a source profile nor a circular boundary changes HDBSCAN*.

## 1. Select the connectivity representation deliberately

| Input | Required choices | Meaning |
| --- | --- | --- |
| Your own projected line layer | `--network PATH` | Cluster along the supplied LineString/MultiLineString network; OSM selection is not used |
| OSM road/path network | `--network-source osm`, one region, `--osm-profile walk` (or another profile) | OSM ways matching an OSMnx access profile; clustering still uses undirected line length |
| OSM waterway network | `--network-source osm`, one region, `--osm-profile waterways` | OSM ways tagged river, stream, canal or tidal channel; not guaranteed navigable or directionally valid |
| OSM custom network | `--network-source osm`, one region, `--osm-filter '...'` | OSM ways matching the explicitly supplied way-tag filter |
| Nongeographic/specialized distances | `hdbscan()` / `fit_hierarchy()` sparse API | Independently constructed scientifically defensible **symmetric** distances; no OSM or region |

Exactly **one** of `--osm-profile` and `--osm-filter` is required for OSM acquisition. No domain or travel mode is inferred from the observations or region. `--network` cannot be combined with OSM-only acquisition flags.

The available named profiles are `walk`, `bike`, `drive`, `drive_service`, `all`, `all_public`, and **`waterways`**. The first six select OSMnx's established `network_type` access profiles; `waterways` is a convenience profile implemented using the custom OSM way filter:

```text
[waterway~"^(river|stream|canal|tidal_channel)$"]
```

The filter selects OSM *ways* with those tag values. It is not a claim that all such ways are connected, passable, complete or valid for waterborne travel or biophysical dispersal. It does not include waterbody polygons, lakes or coastal open-water routing, and it does not create missing confluences. For a different selection use `--osm-filter`, which must contain a validated bracketed way-tag expression, *instead of* `--osm-profile`.

## 2. Geographic region: one of two exclusive forms

| Option | Input order / unit | Validation | Acquisition behavior |
| --- | --- | --- | --- |
| `--bbox "W,S,E,N"` | Four **longitude/latitude** numbers in EPSG:4326 | Strict west < east, south < north; world bounds, no dateline crossing; approximate spherical area at most **10,000 km²** | Existing `OSMnx graph_from_bbox` path, with no new post-fetch circle clip |
| `--center "lon,lat" --radius-km R` | Center longitude first, latitude second; **positive kilometers** | Finite coordinates, finite positive radius, WGS84 geodesic area at most **10,000 km²**; reject poles and antimeridian crossing | `OSMnx graph_from_polygon`, followed by actual polygon clipping of projected line geometry |

For a center/radius region, `net-hdbscan` constructs **360 vertices** by sending geodesics of length `R × 1,000` meters from the center at successive bearings on the WGS84 ellipsoid (`pyproj.Geod`). The vertices form a **360-sided polygonal approximation** of the geodesic circle, not a mathematically perfect continuous disk; boundary chords can differ slightly from a true circle. The projected network edge geometry is intersected with this polygon, with only nonempty, positive-length line fragments retained. A feature crossing the circle boundary may be truncated; a feature inside the enclosing bbox but outside the circle is excluded.

The polygon acquisition uses `graph_from_polygon(..., truncate_by_edge=True, simplify=False, retain_all=True)`, then projects the graph before clipping its edges to the polygon. Bbox acquisition continues to use `graph_from_bbox(..., simplify=False, retain_all=True)`. Neither workflow silently repairs arbitrary topology or snaps disconnected waterways together.

**The circle radius and the clustering horizon are different parameters:**

- `--radius-km` is the **geographic extent of network acquisition**, in kilometers, measured geodesically from the center.
- `--max-distance` is the **maximum shortest-path distance measured on the network**, in the projected network CRS's length units (often meters). It is not DBSCAN `eps` and may be longer or shorter than the acquisition radius.
- `--max-snap-distance` is an optional limit for how far a **supplied observation** may lie away from the resulting line network before the run fails. It is not a study-area point filter.

**Points are not filtered to the bbox/circle.** Every supplied point remains eligible unless optional `--group-col` missing-group policy excludes it. A point outside the acquisition region can still be snapped to the acquired network if it is accepted under the current snap-distance settings. If you want a study-area eligibility rule, spatially subset points **upstream** and document it explicitly.

## 3. Copyable CLI examples (PowerShell)

These are usage examples; they do not imply a completed live Overpass smoke test. Replace the sample input path with your actual point dataset. A backtick at the end of a PowerShell line continues the command.

### A. Existing OSM walking network with bbox

```powershell
net-hdbscan cluster `
  --points "data\observations.parquet" `
  --network-source osm `
  --bbox "120.97,14.58,121.05,14.66" `
  --osm-profile walk `
  --point-id-col observation_id `
  --max-distance 1500 `
  --min-cluster-size 10 `
  --output-dir "output\walk_clusters"
```

### B. New OSM waterway network with bbox

```powershell
net-hdbscan cluster `
  --points "data\aquatic_observations.parquet" `
  --network-source osm `
  --bbox "120.97,14.58,121.05,14.66" `
  --osm-profile waterways `
  --point-id-col observation_id `
  --max-distance 3000 `
  --min-cluster-size 10 `
  --output-dir "output\waterway_bbox"
```

### C. New OSM waterways within a 10 km geodesic circle

```powershell
net-hdbscan cluster `
  --points "data\aquatic_observations.parquet" `
  --network-source osm `
  --center "121.04,14.65" `
  --radius-km 10 `
  --osm-profile waterways `
  --point-id-col observation_id `
  --max-distance 3000 `
  --min-cluster-size 10 `
  --output-dir "output\waterway_circle"
```

### D. Fetch and inspect a circle first; then cluster the cached network

```powershell
net-hdbscan fetch-network `
  --source osm `
  --center "121.04,14.65" `
  --radius-km 10 `
  --osm-profile waterways
```

The command prints an absolute path to the cached `network.gpkg` and its sibling `source.json`. Inspect them, then reuse the cached path **as a user-supplied projected network** (no OSM flags on the second command):

```powershell
net-hdbscan cluster `
  --points "data\aquatic_observations.parquet" `
  --network "REPLACE_WITH_PRINTED_NETWORK_GPKG_PATH" `
  --point-id-col observation_id `
  --max-distance 3000 `
  --min-cluster-size 10 `
  --output-dir "output\waterway_cached"
```

Or repeat `--network-source osm`, the same region and selector on `cluster`; a verified cache hit will avoid a network request. If a cached entry is damaged, the adapter **raises an error** instead of downloading over it.

### E. Custom waterway subset (alternative to the named profile)

```powershell
net-hdbscan fetch-network `
  --source osm `
  --bbox "120.97,14.58,121.05,14.66" `
  --osm-filter '[waterway~"river|stream"]'
```

Do not supply `--osm-profile waterways` at the same time. A custom expression takes the place of the profile, not a second selector combined with it.

## 4. Python acquisition API

The optional acquisition adapter is also callable directly:

```python
from net_hdbscan.osm_source import fetch_osm_network

# Bbox: west, south, east, north
road_gpkg = fetch_osm_network(
    bbox=(120.97, 14.58, 121.05, 14.66), profile="walk"
)

# Geodesic circle: longitude, latitude and radius in km
river_gpkg = fetch_osm_network(
    center=(121.04, 14.65), radius_km=10, profile="waterways"
)

# Read with GeoPandas, then pass to the existing geospatial API if desired.
import geopandas as gpd
network = gpd.read_file(river_gpkg)
```

`fetch_osm_network` returns a `pathlib.Path` to a projected GeoPackage line layer; it is an acquisition utility, not a clustering call. `parse_bbox`, `parse_circle`, and `circle_polygon` are implementation/validation helpers, not a replacement for the documented clustering API.

## 5. Cache, metadata and provenance

The opt-in OSM adapter caches each explicit request under an immutable, SHA-256-keyed directory:

| Platform | Default cache root |
| --- | --- |
| Windows | `%LOCALAPPDATA%\net-hdbscan\osm\` |
| Other platforms | `~/.cache/net-hdbscan/osm/` |

You may override the root with `--osm-cache-dir` or the Python `cache_dir` argument. Every entry contains **`network.gpkg`** (projected geometry) and **`source.json`** (provenance and checksum). Cache keys distinguish region, source, selector, and distance interpretation. **Existing bbox request keys are preserved** to retain compatibility with older cached extracts. Circular requests use a new request-key schema and include:

```json
{
  "adapter_schema": 2,
  "source": "openstreetmap",
  "circle_epsg4326_lon_lat_radius_km": [121.04, 14.65, 10.0],
  "circle_boundary_segments": 360,
  "osm_profile": "waterways",
  "osm_custom_filter": null,
  "distance_model": "undirected_projected_polyline_length"
}
```

This is the **`source.json` `request` object**, not the complete `source.json` file. The actual metadata also includes a UTC retrieval timestamp, OSMnx version, projected CRS, feature count, ODbL attribution, SHA-256 of `network.gpkg`, the effective `resolved_osm_filter`, whether `clipped_to_circle` was used, and an explicit modeling-interpretation statement. Bbox requests continue to use `adapter_schema: 1` and `bbox_epsg4326_wsen`.

A cache hit validates the stored request and file checksum and performs **no OSMnx/Overpass retrieval**. Entries are written into a staging directory and only published after serialization and read-back validation. Missing metadata, corruption, or a conflicting existing entry makes the operation fail closed; there is no silent replacement/refresh. A new cache directory can be used deliberately to obtain a fresh extract. Preserve `source.json` beside `network.gpkg` for provenance; the clustering output `manifest.json` separately records its input network file and checksum.

OpenStreetMap data are subject to the **ODbL**; credit OpenStreetMap contributors and review [OSM attribution requirements](https://www.openstreetmap.org/copyright). OSM/Overpass availability, completeness and rate limits vary. The package enforces a maximum estimated query area of 10,000 km², but smaller requests may still be computationally or operationally expensive. Regional PBF downloads, HydroRIVERS ingestion and other third-party hydrology adapters are **not implemented** by this OSM acquisition feature; preprocess those into a projected line network and provide `--network` instead.

## 6. Interpretation and scientific safeguards

An acquired OSM line network is only a geometrical representation. It is **not** a scientifically validated river network, hydrologic connectivity model, boat route network, flood flow model or traffic routing engine. Specifically:

- The geospatial distance is **symmetric, undirected two-dimensional arc length**, even when the OSM way itself has directional tags. There is no downstream/upstream, travel time, vessel speed, river current, elevation/3-D length, capacity, turn restriction, flow resistance or barrier-cost model.
- `waterways` may include non-navigable streams and artificial canals and may exclude lakes, reservoirs, estuaries, or relevant unmapped reaches. OSM tagging may be incomplete or inconsistent.
- Disconnected components remain disconnected. Two visible crossing lines do not necessarily connect; the graph construction needs shared vertices (after coordinate canonicalization). Projection, line clipping, original digitization and OSM topology may influence connections at the circle edge.
- Snapping a survey observation to a waterway does not demonstrate actual access or dispersal. Check `snap_distance`, `--max-snap-distance`, network components, core-distance truncation and the resulting hierarchy before interpreting clusters.
- The finite `--max-distance` horizon means pairs beyond the measured horizon are *unmeasured*, not necessarily unreachable. Compare the summary's structural disconnections with potentially resolvable truncated core distances.
- If flow direction or hydraulic properties are essential, build and justify an external **symmetric** distance representation for the sparse HDBSCAN* API. The present core rejects asymmetric distance graphs rather than silently converting them.

A responsible report should preserve input point source, extraction region and selector, cache provenance, network CRS and units, topology decisions, snap-distance distribution, `max_distance`, clustering parameters, and whether undirected length has a defensible scientific interpretation.

## 7. Validation checklist and limitations of the current tests

The offline regression suite contains checks for malformed and oversized circles, longitude/latitude order, geodesic radius, circle clipping vs bbox clipping, custom filter selection, OSMnx polygon vs bbox acquisition, cache read-back and reuse, CLI source exclusivity and fail-fast invalid argument handling. Its mocked OSMnx responses **do not prove live Overpass availability, OSM tag coverage or real-world navigability**.

From the `net-hdbscan` repository with the intended Python environment:

```powershell
python -m pip install -e ".[dev,osm]"
if ($LASTEXITCODE -ne 0) { throw "Install failed" }
python -B -m pytest -q -p no:cacheprovider
if ($LASTEXITCODE -ne 0) { throw "Tests failed; do not commit or merge" }
```

To perform a live smoke test, choose a **small area with verified OSM waterway coverage**, run `fetch-network` with `--osm-profile waterways`, inspect the resulting GeoPackage and metadata, and repeat the identical request to confirm the checksum-verified cache hit. A live smoke test is **not** part of the offline test suite and should not be reported as completed unless actually executed. Merge/deploy only after required CI checks for the current commit have finished successfully; failed, skipped, missing or pending checks block merging.

## 8. Common errors

| Symptom | What to check |
| --- | --- |
| `choose --bbox or --center...` | Specify **one** region; remove bbox when using center/radius |
| `requires --bbox or both --center and --radius-km` | Supply both circle arguments or a bbox |
| `radius-km must be positive and finite` | Radius must be a finite positive number, not degrees |
| `circle area ... capped` | Use a smaller circle or preprocess a regional line extract |
| `crosses the antimeridian` / `reaches a pole` | Current polygon adapter does not support wrapped/polar circles |
| `select exactly one of --osm-profile or --osm-filter` | Provide one—and only one—network selector |
| `OSM returned no line geometry within the bounding circle` | The selected OSM ways may not cross the circle (or may be absent/incomplete) |
| `checksum mismatch` / `incomplete ... cache` | Do not delete or silently replace corrupted entries during automated runs; investigate or use a deliberate new cache root |
| `max_snap_distance` failure | Input observations may be far from the acquired network; **not** automatically filtered |

For the full API and network interpretation, see [the README](../README.md), [network sources and provenance](NETWORK_SOURCES.md), [applications](APPLICATIONS.md) and [development tests](DEVELOPMENT.md).
