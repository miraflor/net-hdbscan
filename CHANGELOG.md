# Changelog

## 0.3.1 — corrective hardening (release candidate; tag after verification)

- Correct weighted duplicate stack condensation in the simultaneous and binary hierarchy paths.
- Bound Dijkstra output allocations including source nodes. Fail fast with a clear value error if even one source exceeds the budget; no implicit memory overrun.
- Hash full directory datasets and exact shapefile sidecars (not nearby datasets); support top-level symlinked inputs with resolved-path provenance and reject unsafe nested directory links.
- Canonicalize numeric groups and CSV overrides exactly using decimal arithmetic (including large integers and scientific notation); preserve distinct blank/null groups; reject dense distance arrays and improve CLI errors. Numeric group output names can change compared with v0.3.0.
- Fix version provenance, declare pyproj directly, extend CI to Python 3.10–3.13.
- Clarify frequent tied-distance differences from scikit-learn and zero-strength membership for stacked points.
- Retain historical `groups_outside_universe` warning semantics in the patch release.
- Keep the existing `v0.3.0` tag immutable.


## 0.3.0 — 2026-11-23 (OSM waterways and geodesic circular network acquisition)

- Provenance: packages built from the immutable `v0.3.0` tag at `0c7568b` report version `0.2.0` in their generated manifests; identify them by tag/commit rather than the recorded package version.
- Add the named `--osm-profile waterways` shortcut for OSM `river`, `stream`, `canal` and `tidal_channel` ways, without introducing hydrological or directional routing semantics.
- Add `--center longitude,latitude --radius-km R` to both `cluster` and `fetch-network` as a mutually exclusive alternative to `--bbox`. Build a 360-sided WGS84 geodesic circle and clip the retrieved projected line network to its polygon boundary.
- Preserve the existing bbox acquisition flow, cached request-key schema and clustering core; circle requests use a separate immutable cache-key schema and record effective filter, geometry clipping, source attribution and checksum.
- Add offline regression/failure tests for circle parsing, area/pole/antimeridian rejection, clipped geometry, OSMnx polygon acquisition, CLI exclusivity and cache behavior. Live OSM acquisition remains an independent smoke test, not covered by mock-based unit tests.
- Thoroughly document usage, network-versus-observation bounds, radii versus shortest-path horizons, provenance, limits and reproducibility in the README, source guide, design contract, applications, development guide and geographic acquisition runbook.

- Add opt-in OpenStreetMap/OSMnx line-network acquisition with an explicit
  bbox, network way-profile or user-provided tag filter, bounded public queries,
  immutable checksum-verified GPKG cache, and acquisition provenance metadata.
- Keep user-supplied projected line networks and symmetric sparse distance
  matrices as the fundamental data contracts; no OSM dependency in the core.
- Document ecology, sociology, infrastructure, health and other possible
  representations with explicit scientific validity and directionality limits.
- Clarify that grouping is optional independent stratification across arbitrary
  categories—not a required industry or other domain-specific attribute.
- Add offline tests for source selection, malformed requests, cache errors and
  negative paths; do not alter core HDBSCAN extraction or default behavior.

## 0.2.0 — 2026-09-30

- Breaking: remove study-area boundary filtering from the CLI and Python/file APIs. `net-hdbscan` now clusters every supplied point; geographic eligibility filtering belongs upstream.
- Add a visual algorithm guide covering network distance, snapping, core distance, mutual reachability, the minimum spanning forest, hierarchy construction, condensation, stability, cluster extraction, and the `max_distance` search horizon.
- Publish the documentation through GitHub Pages.
- Keep the clustering model, grouped execution, adaptive distance search, diagnostics, hierarchy outputs, and sparse-distance safeguards unchanged.

## 0.1.0 — 2026-09-27

Initial standalone release of `net-hdbscan`.

### Core

- HDBSCAN* over shortest-path distance on a supplied projected spatial line network.
- Direct Shapely/SciPy network graph, snapping and bounded local shortest-path neighbour search; no dependency on `spaghetti`, scikit-learn at runtime, or an external HDBSCAN package.
- Weighted compression of duplicate snapped positions, deterministic tie handling, hierarchy, membership strengths and core distances.
- Fixed and adaptive distance horizons, pair-budget handling, per-trial trace output and structural-vs-truncation core-distance diagnostics.
- Partial selection for long sparse rows when computing core distances, preserving results while avoiding unnecessary full-row sorting.

### Interfaces and outputs

- Distribution/repository name `net-hdbscan`, import package `net_hdbscan`, command `net-hdbscan`.
- Public `NetworkGraph`, `build_network_graph`, `snap_points`, `distinct_positions`, `neighbor_graph` and sparse `hdbscan` interfaces.
- Optional boundary; without one, every input point is eligible.
- Per-cluster table in addition to point labels and hierarchy.
- Grouped execution with missing-group policies, declared group universes and per-group parameter overrides.
- PyArrow and Typer are optional extras (`files` and `cli`); development installs use `.[dev]`.

### Hardening before first release

- Sparse HDBSCAN input validation rejects asymmetric graphs, duplicate sparse coordinates, negative/non-finite distances and non-integer weights instead of silently reinterpreting them.
- Literal group values named `__null__` or `__blank__` no longer collide with missing-group sentinels; summaries/files disambiguate them with a `literal:` prefix, and group-parameter CSVs can use the same prefix to escape reserved missing-group tokens.
- Group output filenames are checked with case-insensitive semantics so `A` and `a` cannot overwrite one another on Windows.
- Excluded null/blank groups are removed before reprojection/boundary filtering/snapping.
- Oversized single-source candidate searches and dense same-arc rows are processed in bounded slices rather than defeating the advertised internal chunk limits.
- File-based runs check for PyArrow before expensive input processing.
- Manifests record selected input layers, analysis CRS/unit and network graph counts.
- Public diagnostics use neutral `network_components_used` terminology; `vertex_digits` is validated as a positive integer or `None`.
