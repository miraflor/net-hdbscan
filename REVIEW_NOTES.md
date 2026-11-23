# Review notes

`net-hdbscan` 0.1.0 is the initial standalone release of a network-space HDBSCAN* package. It is not presented as a new HDBSCAN algorithm; the package contribution is the integration of sparse HDBSCAN* with bounded shortest-path distance on supplied spatial line networks, together with reproducible grouped workflows and diagnostics.

The package contains a direct Shapely/SciPy network graph and snapping engine, bounded sparse shortest-path neighbour search, weighted HDBSCAN* implementation, deterministic equal-distance hierarchy treatment, fixed/adaptive distance modes, grouped-data policies and diagnostic outputs. Territory/Voronoi construction is deliberately absent.

Fixed and adaptive regression runs were checked for stable clustered point fields, hierarchy, diagnostics and trial records apart from runtime fields. The faster long-row core-distance implementation was also checked against the full-sort implementation and produced identical values on tie-heavy and large sparse cases.

Before this first release, the public sparse HDBSCAN boundary was tightened: malformed asymmetric or duplicate sparse distance graphs are rejected, and weights must be actual positive integers rather than values that happen to become integers after casting. Group sentinels are now internal rather than ordinary strings, preventing real categories named `__null__` or `__blank__` from being treated as missing. Windows-safe filename collision checks, pre-snap exclusion of missing groups, hard internal chunk bounds, early PyArrow checks, richer manifests and neutral network terminology were added as operational corrections.

The repository currently collects 112 tests. In the build environment used for this revision, all exercised non-Parquet tests pass. PyArrow is not installed there, so tests that physically read/write GeoParquet cannot be completed in that environment; the `dev` extra includes PyArrow and the complete suite should be run in the target development environment before committing/tagging the release.

Adaptive `distance_status = "converged"` means local flat-result stability plus low resolvable core truncation under the tested geometric radius ladder. It is deliberately not described as proof that every larger horizon would preserve the result.

## 0.3.0 adapter note

The initial release statement that network downloading was outside package
scope described the original package only. The opt-in OSM adapter, released
in 0.3.0, adds bounded acquisition and verified local caching without making OSM a
required dependency or changing the symmetric-distance HDBSCAN* core.
Applications outside geographic point/line networks continue to use the
low-level sparse distance API. OSM one-way restrictions, hydrologic direction,
physical 3-D paths, and travel-time costs are **not** silently implemented.
The fixed historical test count above should not be interpreted as the current
regression count. See `docs/APPLICATIONS.md` and `docs/NETWORK_SOURCES.md`.

## 0.3.0 acquisition extension

The opt-in OSM adapter now offers an explicit **waterways** way-tag profile and a **center-plus-radius** geodesic polygon region as an alternative to a bounding box. The region constrains **acquired line geometry only**; study-area filtering of input observations remains out of scope. Circle geometry uses a 360-segment WGS84 geodesic polygon and projected line clipping. Bbox cache identities remain compatible while circular requests have separately keyed metadata. The symmetric line-length model and HDBSCAN* core are unchanged.

A local Windows checkout reported a passing full pytest suite after applying the implementation overlay. This is **not** evidence of a completed live OSM/Overpass smoke test or of current GitHub CI completion. Re-run all tests and verify required CI for the current PR head before merging. See `docs/GEOGRAPHIC_ACQUISITION.md` for detailed usage, source and modeling limits.
