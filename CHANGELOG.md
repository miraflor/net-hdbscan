# Changelog

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
