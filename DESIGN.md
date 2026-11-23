# net-hdbscan — design

## Core invariant

```text
points + supplied projected line network (or explicit OSM-cached linework)
  → snap points to network arcs
  → compress identical snapped positions
  → sparse shortest-path distances up to a bounded horizon
  → HDBSCAN*
  → clustered points + cluster table + hierarchy + diagnostics
```

Study-area or other eligibility filtering is deliberately outside the package; `net-hdbscan` clusters every supplied point.

## Package scope

`net-hdbscan` owns network-space HDBSCAN clustering. It does not own territory generation, Voronoi construction, domain classification or topology repair. An optional acquisition adapter downloads and caches OSM linework without modifying the clustering core. The package is self-contained at the application level and exposes its own network, clustering, pipeline and diagnostic interfaces.

The in-memory core depends on GeoPandas, NumPy, pandas, Pyogrio, SciPy and Shapely. PyArrow is an optional file extra and Typer is an optional CLI extra.

## Network distance engine

Input lines are split into straight arcs between consecutive vertices. Coordinates may be canonicalized to `vertex_digits` significant digits; two lines connect only when they then share a vertex. Geometric crossings without a shared vertex remain disconnected.

Points are snapped to continuous network positions. Pairwise clustering distance is the shortest path along the network between snapped positions; point-to-network snap distance is QA metadata and is not added to the clustering metric.

The bounded neighbour search uses local spatial batches and bounded Dijkstra searches. Pair-count and internal candidate-chunk limits are memory-safety controls, not model parameters. Unusually dense single-source searches are processed in bounded slices while retaining the minimum distance to each unique target.

## Sparse HDBSCAN contract

The lower-level `hdbscan()` routine consumes a symmetric sparse distance graph. Each ordered pair may be stored at most once; stored distances must be finite and non-negative; missing entries mean unknown/farther than the supplied horizon. Optional weights are positive integer multiplicities. Validation occurs before clustering so malformed sparse representations cannot silently change core distances.

Core distances count each position's own weight. Long sparse rows use partial selection because only the smallest `min_samples - own_weight` neighbour entries can affect the answer. Mutual reachability and the minimum spanning forest are then constructed from finite-core pairs.

Equal mutual-reachability distances are handled simultaneously as multifurcations. This avoids arbitrary binary tie ordering and makes the hierarchy invariant to observation order and point-ID naming under the package's deterministic conventions.

## Grouped data

Grouped runs share network preparation and point snapping. Missing and blank groups are internal sentinels rather than ordinary strings. The user-facing tokens `__null__` and `__blank__` denote actual missing values; literal categories with those names remain valid and are displayed as `literal:__null__` and `literal:__blank__` where disambiguation is required.

When missing groups are excluded, they are removed before reprojection and snapping. Planned output names are checked with case-insensitive semantics so a run is safe on Windows even when prepared on another platform.

## Adaptive horizon

`max_distance` is a search/truncation horizon, not a DBSCAN epsilon. Fixed mode uses the requested horizon directly. Adaptive mode tries a geometric radius ladder and may stop when two consecutive successful trials have the same flat clustering/membership within tolerance and the share of otherwise-resolvable truncated core distances is sufficiently small.

That stopping rule demonstrates local stability on the tested ladder only. Every trial is recorded in `distance_trace.csv` so the decision remains auditable.

## Reproducibility

Manifests record package version, input paths and selected layers, clustering configuration, `vertex_digits`, analysis CRS and linear unit, and graph vertex/arc/component counts. Cluster IDs and output ordering use deterministic conventions.

## Domain-neutral design and acquisition boundary

The abstract `hdbscan`/`fit_hierarchy` core accepts a scientifically justified
**symmetric sparse distance matrix**; this is not restricted to geography.
The CLI/file workflow is a **two-dimensional geospatial** network-distance
adapter: it snaps point features to projected linework and computes undirected
shortest-path lengths. The optional `osm_source` acquisition adapter only
provides that linework from an **explicit** OSM request; it cannot choose a
scientific distance model. Directional flow/travel semantics and true 3-D
lengths are not implied by any source profile.

OSM requests must specify exactly one explicit region—an EPSG:4326
`west,south,east,north` bbox **or** `longitude,latitude` center and positive
`radius_km`—and exactly one network selector (named profile or validated
bracketed custom way-tag filter). Named `waterways` maps to the exact filter
`[waterway~"^(river|stream|canal|tidal_channel)$"]`. This selects OSM ways;
it does not supply downstream direction, navigability, flow cost or topology
repair. No selection is inferred from the data.

**Region semantics:** `parse_bbox` retains the earlier strict ordering and
spherical area bound. `region.parse_circle` checks finite coordinates and
radius, rejects polar and antimeridian-spanning circles, and delegates to a
360-vertex WGS84 geodesic polygon built with `pyproj.Geod`. A geodesic polygon
is a finite approximation of the ideal circle. Circle acquisition uses
`OSMnx graph_from_polygon(..., truncate_by_edge=True)` while legacy bbox
acquisition continues to use `graph_from_bbox`. After projection, circular
network linework is intersected with the projected polygon and nonempty,
positive-length line segments retained; bbox linework is not circle-clipped.
Both region types limit line-network acquisition, **not point eligibility**.
No eligibility filter is reintroduced in the CLI or pipeline.

**Distance distinction:** `radius_km` is the geodesic download extent,
`max_distance` is the geospatial algorithm's shortest-path discovery horizon
in the projected network CRS's linear units, and `max_snap_distance` is
an optional QA failure limit for observation-to-network snapping. These
parameters serve separate purposes. Polygon clipping can change topology and
shortest paths; no automatic buffer, gap repair or confluence repair occurs.

**Cache and provenance:** old bbox `_request` metadata and adapter schema 1
are retained byte-for-byte in the request key. Circle requests use adapter
schema 2 with center/radius and boundary-segment count 360. Request hashes
identify immutable cache entries; cache hits verify the exact request and
SHA-256 of `network.gpkg`, with no network call. `source.json` stores UTC
retrieval, OSMnx version, source license, projected CRS, feature count,
clipping flag, effective waterway filter and network checksum. New entries
are prepared and read-back-validated in a temporary directory before atomic
per-directory publication; incomplete/corrupt entries are **not** silently
replaced. Regional PBF extracts, HydroRIVERS downloads and flow-routing
models are outside the current adapter scope.

Optional `--group-col` performs independent clustering by arbitrary category.
It never implies an industry, species, or social category as the default, and
cross-category clusters are excluded when it is used.

See `docs/NETWORK_SOURCES.md` and `docs/APPLICATIONS.md`.
