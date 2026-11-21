# net-hdbscan

`net-hdbscan` provides **HDBSCAN\*** over **symmetric sparse network distances**. Its geospatial workflow clusters point observations along supplied projected line networks; an **optional, explicitly selected** OpenStreetMap adapter can obtain linework from a bounding box. It does not assume that the network is a road, that the observations are businesses, or that clustering must be stratified.

It is a clustering-only package. It does not construct Voronoi territories or other downstream partitions.

For a visual, step-by-step explanation of the algorithm—including network distance, snapping, sparse neighbour search, mutual reachability, the minimum spanning forest, tie-invariant hierarchy construction, condensation and cluster extraction—see the **[net-hdbscan documentation site](https://miraflor.github.io/net-hdbscan/)**.

```text
point layer + supplied or explicitly retrieved projected line network
    ↓
snap points to the line network
    ↓
compute sparse shortest-path distances up to a bounded horizon
    ↓
HDBSCAN*
    ↓
clustered points + hierarchy + diagnostics
```

Version 0.2.0 removes study-area boundary handling so the package has a narrower contract: the supplied points and spatial network are the complete clustering inputs. Geographic eligibility filtering belongs upstream. The package retains its network-distance engine, sparse HDBSCAN* implementation, grouped execution pipeline, adaptive distance search, hierarchy outputs and diagnostics.

## Install

From a source checkout, with everything needed for development and the command line:

```powershell
python -m pip install -e ".[dev]"
```

The core (network distances, HDBSCAN* and the in-memory pipeline `cluster_geodataframes`) needs only GeoPandas, NumPy, pandas, Pyogrio, SciPy and Shapely 2.1 or later. Three optional extras add the rest: `files` adds PyArrow, which reading and writing (Geo)Parquet needs (`cluster_files`, `cluster_files_by_column`, `.parquet` inputs); `cli` adds PyArrow and Typer for the `net-hdbscan` command. The `dev` extra contains the file and CLI extras, plus pytest and scikit-learn (used only by validation tests). The independent `osm` extra adds OSMnx 2.x for explicit network acquisition; it is not required for clustering or cached network reuse. Without an extra, the functions that need it stop with a message that names the extra to install, for example `pip install "net-hdbscan[cli]"`. The command can also be run as `python -m net_hdbscan`.

## Choosing a network and a scientific distance

**No network source is chosen implicitly.** Supply a 2-D projected line network
(`--network`) or select the optional OSM adapter with `--network-source osm`, an
explicit EPSG:4326 bbox and either an OSM profile or a way-tag filter. OSM
acquisition is cached with source metadata and SHA-256. For non-geographic
relational networks, use the existing **symmetric sparse-distance HDBSCAN***
Python API; a bounding box is irrelevant. Directional flow, travel-time costs
and true 3-D network geometry are not implemented in the geospatial pipeline.

**`--group-col` is optional stratification.** It runs separate cluster analyses
within user-defined categories and prohibits cross-category clusters by design.
There is no required industry, species, demographic, or other domain column.

See [network-source and OSM guidance](docs/NETWORK_SOURCES.md) and
[illustrative scientific applications and limitations](docs/APPLICATIONS.md).

To enable open-data retrieval: `python -m pip install -e ".[dev,osm]"`.

## Quick start

```powershell
net-hdbscan cluster `
  --points "data\points.parquet" `
  --network "data\lines.gpkg" --network-layer lines `
  --point-id-col canonical_id `
  --max-distance 3000 `
  --min-cluster-size 20 `
  --output-dir "output\hdbscan"
```

Alternatively, download an explicit OSM network without a local network file:

```powershell
net-hdbscan cluster `
  --points "data\observations.parquet" `
  --network-source osm --bbox "120.97,14.58,121.05,14.66" `
  --osm-profile walk --point-id-col observation_id `
  --max-distance 1500 --min-cluster-size 10 `
  --output-dir "output\observations"
```

The `walk` profile selects an OSM **way filter**, not a directional travel model.
Use `--osm-filter '[waterway~"river|stream"]'` instead of `--osm-profile`
for exploratory OSM waterways, after checking the resulting topology and
scientific validity. OSMnx 2.x is required only for a new download.

For **optional independently stratified** data:

```powershell
net-hdbscan cluster `
  --points "data\points.parquet" `
  --network "data\lines.gpkg" --network-layer lines `
  --point-id-col canonical_id `
  --group-col stratum `
  --max-distance 5000 `
  --distance-mode adaptive `
  --min-cluster-size 5 `
  --min-samples 5 `
  --output-dir "output\by_industry"
```

## Network distance

If observation `i` snaps to network position `s_i` and observation `j` snaps to `s_j`, the clustering distance is the shortest-path length on the supplied network between those snapped positions. Point-to-network snap distance is not added to this metric; it is reported separately for quality assurance. Observations on disconnected components have no finite path between them.

The network CRS must be projected. All distance parameters are expressed in that CRS's linear units. Points are reprojected to the network CRS.

Lines join only where they share a vertex after coordinate canonicalization. A geometric crossing without a shared vertex is not connected. MultiLineString parts are exploded and do not acquire artificial links to one another.

## HDBSCAN* model

`min_cluster_size` is the smallest group retained as a cluster. `min_samples` controls the core-distance density estimate and defaults to `min_cluster_size`. The package supports `eom` and `leaf` cluster selection, `cluster_selection_epsilon`, `max_cluster_size`, `allow_single_cluster`, and a `core_distance_floor` for limiting the influence of very large stacks of co-located observations.

Several observations at one exact snapped network position are compressed to one position during the network search. With `--duplicates count` (default), their multiplicity is retained as HDBSCAN weight. With `--duplicates once`, each distinct position counts once.

The HDBSCAN* implementation is included in this package because a truncated sparse network-distance graph can be disconnected and can leave rows with fewer than `min_samples` stored neighbours. The implementation uses a deterministic multifurcation treatment for equal mutual-reachability distances rather than allowing arbitrary binary tie ordering to determine the hierarchy.

## `max_distance`

`max_distance` is a **search/truncation horizon**, not DBSCAN's `eps`. Network pairs farther apart are not measured. The hierarchy is exact below that horizon; the program has no distance information above it.

In `--distance-mode fixed`, the specified value is used exactly. In `--distance-mode adaptive`, it is a hard ceiling. The search begins at a smaller radius and grows geometrically. It can stop when two consecutive successful radii give the same flat cluster IDs/noise labels and membership values while the share of otherwise-resolvable observations with truncated core distance is at or below `--core-truncation-tolerance`.

The adaptive status `converged` therefore means **local flat-result stability under that diagnostic rule**. It is not a proof that every larger radius would give the same clustering. `distance_trace.csv` records every trial so the selection can be audited.

The summary distinguishes:

- `share_core_beyond_max_distance`: all observations with infinite core distance;
- `share_core_structurally_unreachable`: observations on components whose total effective weight is below `min_samples`;
- `share_core_truncated`: otherwise-resolvable observations whose core distance lies beyond the current horizon.

`--max-neighbor-pairs` is a **fatal hard memory-safety limit** in both fixed and adaptive modes. Exceeding it at any attempted radius raises `NeighborPairLimitError`; no earlier smaller-radius result is silently accepted. No new successful result has `distance_status = pair_cap_limited`. File outputs are staged and checked before publication; an existing verified output generation is preserved if the new run fails.

## Grouped data

With `--group-col`, each group is clustered independently while the input files, prepared network, network graph and point snapping are shared across the run. When null/blank groups are excluded, those rows are removed before snapping.

Null and blank group values are **excluded by default**. Use `--missing-group-policy include` only when they are genuine categories, or `error` when incomplete classification should stop the run. Missing values use the user-facing tokens `__null__` and `__blank__`; literal categories with those exact names remain valid and are shown as `literal:__null__` and `literal:__blank__` where disambiguation is needed. In a group-parameter CSV, prefix a reserved literal with `literal:`.

`--group-universe` accepts a one-column CSV declaring expected categories. Declared-but-absent groups receive zero-row outputs and summary rows. Observed groups outside the declared universe are still processed and explicitly reported.

`--group-params` accepts per-group overrides. Supported fields include `max_distance`, adaptive-distance settings, `min_cluster_size`, `min_samples`, cluster-selection settings, `core_distance_floor`, and `duplicates`.

Example:

```text
stratum,max_distance,min_cluster_size,min_samples
31,5000,,
55,1500,5,5
63,1500,5,5
```

## Outputs

A single run writes:

```text
clustered_points.parquet
cluster_table.parquet
cluster_hierarchy.parquet
summary.csv
distance_trace.csv
manifest.json
```

A grouped run writes:

```text
points/group_<value>.parquet
clusters/group_<value>.parquet
hierarchy/group_<value>.parquet
summary.csv
distance_trace.csv
manifest.json
```

The point output preserves input attributes and adds:

| field | meaning |
|---|---|
| `cluster_id` | Stable public cluster ID (`C000001`, ...); null for noise unless singleton noise is requested |
| `is_noise` | Whether HDBSCAN* labels the observation as noise |
| `membership` | Membership strength from 0 to 1; 0 for noise |
| `core_distance` | HDBSCAN core distance; `inf` when unresolved within the search horizon |
| `snap_distance` | Straight-line distance from the original point to its snapped network position |
| `snapped_x`, `snapped_y` | Snapped network coordinates |

The cluster table has one row per public cluster ID: `cluster_id`, `n_points` (observations), `n_positions` (distinct snapped positions), `stability`, `birth_distance` (network distance at which the cluster separated from a larger group; `inf` if it was already separate at the search horizon), and `singleton` (true for the one-position IDs that `--noise-policy singleton` gives to noise; their stability and birth distance are empty). A group without clusters gets an empty table with the same columns.

The hierarchy output contains `hierarchy_id`, `parent_id`, `birth_distance`, `size`, `stability`, `selected`, and the public `cluster_id` for selected clusters.

`summary.csv` contains point/position counts, pair counts, core-distance truncation diagnostics, network-component and snap-distance diagnostics, cluster counts/noise shares, actual/requested search distances, adaptive status, pair-limit state, parameters used, group-universe metadata, and runtime.

`distance_trace.csv` contains one row per fixed/adaptive trial with radius, outcome, pair count, cluster count, noise share, core truncation, stability against the immediately preceding trial, and runtime.

`manifest.json` records package version, input paths and selected layers, clustering parameters, `vertex_digits`, analysis CRS/linear unit, network graph counts and output file layout.

## Python API

```python
import geopandas as gpd
from net_hdbscan import HDBSCANConfig, cluster_geodataframes

out = cluster_geodataframes(
    points=gpd.read_parquet("points.parquet"),
    network=gpd.read_file("roads.gpkg"),
    config=HDBSCANConfig(max_distance=3000, min_cluster_size=20),
    point_id_col="canonical_id",
)

out.points
out.clusters
out.hierarchy
out.summary
out.trace
```

Lower-level functions are also public: `build_network_graph` (returns a `NetworkGraph`), `snap_points`, `distinct_positions`, `neighbor_graph` (sparse matrix of network distances up to a limit), and `hdbscan` (HDBSCAN* on any symmetric sparse distance matrix, with optional positive-integer weights). The implementation module is `net_hdbscan.sparse_hdbscan`.

## Determinism and validation

The implementation is designed so the computed network distances, flat partition, membership strengths and core distances do not depend on input row order or arbitrary point-ID names. Public cluster IDs are numbered by the first canonical member and can therefore be renamed without changing the underlying partition.

Core distances use a partial selection on long rows: a position needs at most `min_samples` minus its own weight neighbours, so only that many smallest entries of its row are ordered. The result is bitwise identical to sorting whole rows; on a graph with 6.8 million pairs the step took 0.3 s instead of 4.1 s.

The lower-level sparse HDBSCAN API validates the sparse representation before clustering: the matrix must be square and symmetric (including stored zero entries), each ordered pair may be stored at most once, stored distances must be finite and non-negative, and optional weights must be finite positive integers. This avoids silent duplicate summation or weight truncation.

The repository test suite covers network topology and snapping against independent brute-force references, bounded shortest-path neighbours, pair/chunk limits, HDBSCAN* comparisons with scikit-learn, malformed sparse graph rejection, weighted duplicate positions, tie-heavy cases, row/ID invariance, adaptive-radius behavior, grouped missing-value policies and sentinel collisions, group universes, per-group overrides, manifests, and CLI/file outputs.

The release was validated with fixed and adaptive regression runs on a production-scale spatial-network dataset. Point labels, noise flags, membership, core distances, snapped positions, hierarchy and adaptive trial records were stable across the reference runs; runtime fields were treated as non-deterministic diagnostics.

## Scope

`net-hdbscan` clusters every supplied point. Study-area filtering or other eligibility filtering belongs upstream of the package.

`net-hdbscan` does not build a network from an external service, infer or repair uncertain topology, construct Voronoi territories, choose substantive group categories, or claim that adaptive local stability proves global invariance to larger search horizons.

## License

MIT.


## Reusable hierarchy and extraction (improvement API)

Fit the truncated distance hierarchy once and compare extraction methods without
recomputing network distances or the minimum-spanning forest:

```python
from net_hdbscan import fit_hierarchy, extract_clusters, compare_extractions
fitted = fit_hierarchy(sparse_network_distance_graph, min_cluster_size=5)
eom = extract_clusters(fitted, method="eom")
leaf = extract_clusters(fitted, method="leaf")
report = compare_extractions(fitted)
```

`report.summary` contains effective-weight shares, cluster counts and extraction
runtime; `report.eom_parent_to_leaf_descendants` maps selected hierarchy nodes.
Comparison does not change the primary selected clustering. The fitted hierarchy
is an in-memory object; checkpoint serialization is not supported yet.

### Reliability and diagnostics

Group-level summary columns now describe core-distance truncation, structural
unreachability, size feasibility, retention and cluster dominance separately.
The manifest includes SHA-256 digests for inputs and staged output artifacts.
A whole output generation is staged, re-read and checked before replacing the
flat output directory. Replacement uses a rollback journal and two renames;
**it is not a truly atomic multi-file transaction on every operating system**.
A crash can temporarily leave the output directory absent until journal recovery
on the next run. `--force` refuses to delete unrecognized output files.

#### Recovery semantics and post-commit failures

A versioned publication journal distinguishes the **prepared** (reversible)
phase from a verified **committed** output. Before commitment, a failed
replacement restores only an independently verified, complete predecessor.
After commitment, backup deletion and journal removal are housekeeping: if
either fails, `PostCommitCleanupError` stops execution while preserving the
verified new generation and the journal for idempotent recovery on the next
invocation. A partially deleted backup is **never** used for rollback. An
unreadable journal or a missing/corrupted committed destination requires
explicit investigation; recovery does not silently destroy a generation.

Output publication is not an atomic directory-tree swap across platforms;
readers should not treat the rename interval as an uninterrupted snapshot.
`--force` does not overwrite arbitrary directories, including empty ones.
The feasibility field `can_form_two_min_size_groups` is necessary, not
sufficient, for two extracted clusters. `zero_cluster_explanation` lists
co-occurring diagnostics rather than a unique proven cause.


## Author and Citation

**James Matthew Miraflor**\
[Scientific Computing Laboratory](https://scl.dcs.upd.edu.ph/)\
[Department of Computer Science](https://dcs.upd.edu.ph/)\
College of Engineering\
University of the Philippines Diliman

**net-hdbscan** is authored and maintained by James Matthew Miraflor, with AI assistance in software development, debugging, testing, and documentation. Responsibility for the project's methodological decisions, scientific validity, and published outputs rests with the author.

### Performance verification and CI

EOM-to-leaf ancestor mapping now uses a single topological pass through the
condensed hierarchy, followed by assignments for selected leaves. It avoids
pairwise walks through every selected EOM/leaf cluster combination. The
selected clustering itself is unchanged.

Each publication integrity scan decodes each output artifact once and reuses
that in-memory table for its cross-artifact checks. Integrity SHA-256 hashes
and partition/cardinality checks remain mandatory. The output generation may
still be verified multiple times at distinct publication/recovery phases; this
is deliberate because the on-disk state changes between those phases.

To profile integrity verification on an **existing, realistically sized**
output generation, run the read-only benchmark:

```bash
python -B benchmarks/benchmark_publication.py PATH_TO_OUTPUT --repeats 3
```

Compare identical input generations, hardware, Python, and dependencies for
before/after measurements. The benchmark reports actual file sizes, duration,
and Python-traced peak allocations. Native NumPy/Arrow allocations are not
fully captured by `tracemalloc`; these metrics must not be interpreted as
complete process peak RSS. No runtime or memory speedup is guaranteed without
measurements on a representative workload.

The GitHub Actions workflow runs the complete regression and fault-injection
suite on Windows and Linux with supported Python versions. All applicable
checks must finish successfully before merging a pull request; missing or
pending checks do not count as successful.
