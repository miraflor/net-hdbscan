# net-hdbscan — design

## Core invariant

```text
points + supplied projected line network
  → snap points to network arcs
  → compress identical snapped positions
  → sparse shortest-path distances up to a bounded horizon
  → HDBSCAN*
  → clustered points + cluster table + hierarchy + diagnostics
```

Study-area or other eligibility filtering is deliberately outside the package; `net-hdbscan` clusters every supplied point.

## Package scope

`net-hdbscan` owns network-space HDBSCAN clustering. It does not own territory generation, Voronoi construction, network downloading, domain classification or topology repair. The package is self-contained at the application level and exposes its own network, clustering, pipeline and diagnostic interfaces.

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
