# net-hdbscan

`net-hdbscan` clusters geospatial point observations with **HDBSCAN\*** using **shortest-path distance along a supplied spatial line network** rather than Euclidean distance.

It is a clustering-only package. It does not construct Voronoi territories or other downstream partitions.

For a visual, step-by-step explanation of the algorithm—including network distance, snapping, sparse neighbour search, mutual reachability, the minimum spanning forest, tie-invariant hierarchy construction, condensation and cluster extraction—see the **[net-hdbscan documentation site](https://miraflor.github.io/net-hdbscan/)**.

```text
point layer + supplied line network
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

The core (network distances, HDBSCAN* and the in-memory pipeline `cluster_geodataframes`) needs only GeoPandas, NumPy, pandas, Pyogrio, SciPy and Shapely 2.1 or later. Two optional extras add the rest: `files` adds PyArrow, which reading and writing (Geo)Parquet needs (`cluster_files`, `cluster_files_by_column`, `.parquet` inputs); `cli` adds PyArrow and Typer for the `net-hdbscan` command. The `dev` extra contains both, plus pytest and scikit-learn (used only by validation tests). Without an extra, the functions that need it stop with a message that names the extra to install, for example `pip install "net-hdbscan[cli]"`. The command can also be run as `python -m net_hdbscan`.

## Quick start

```powershell
net-hdbscan cluster `
  --points "data\points.parquet" `
  --network "data\roads.gpkg" --network-layer roads `
  --point-id-col canonical_id `
  --max-distance 3000 `
  --min-cluster-size 20 `
  --output-dir "output\hdbscan"
```

For grouped data:

```powershell
net-hdbscan cluster `
  --points "data\points.parquet" `
  --network "data\roads.gpkg" --network-layer roads `
  --point-id-col canonical_id `
  --group-col industry_code `
  --max-distance 5000 `
  --distance-mode adaptive `
  --min-cluster-size 5 `
  --min-samples 5 `
  --output-dir "output\by_industry"
```

## Network distance

If observation $`i`$ snaps to network position $`s_i`$ and observation $`j`$ snaps to $`s_j`$, the clustering distance is the shortest-path length on the supplied network between those snapped positions. Point-to-network snap distance is not added to this metric; it is reported separately for quality assurance. Observations on disconnected components have no finite path between them.

In symbols, with $`x_i`$ the original point, $`N`$ the set of all locations on the network, and $`d_N`$ shortest-path length on the network:

```math
s_i = \arg\min_{z \in N} \lVert x_i - z \rVert, \qquad d(i, j) = d_N(s_i, s_j),
```

where $`d(i, j) = \infty`$ when $`s_i`$ and $`s_j`$ lie on different network components, and the reported snap distance is $`\lVert x_i - s_i \rVert`$. Each point is snapped to the nearest point of the nearest arc; ties between equally near arcs go to the smallest arc index.

The network CRS must be projected. All distance parameters are expressed in that CRS's linear units. Points are reprojected to the network CRS.

Lines join only where they share a vertex after coordinate canonicalization. A geometric crossing without a shared vertex is not connected. MultiLineString parts are exploded and do not acquire artificial links to one another.

## HDBSCAN* model

`min_cluster_size` is the smallest group retained as a cluster. `min_samples` controls the core-distance density estimate and defaults to `min_cluster_size`. The package supports `eom` and `leaf` cluster selection, `cluster_selection_epsilon`, `max_cluster_size`, `allow_single_cluster`, and a `core_distance_floor` for limiting the influence of very large stacks of co-located observations.

### Positions and weights

Several observations at one exact snapped network position are compressed to one position during the network search. With `--duplicates count` (default), their multiplicity is retained as HDBSCAN weight. With `--duplicates once`, each distinct position counts once.

Each distinct position $`p`$ therefore has a weight $`w_p`$: the number of observations at $`p`$ with `--duplicates count`, or $`w_p = 1`$ with `--duplicates once`. Every cluster size below, including the sizes compared with `min_cluster_size` and `max_cluster_size`, is a sum of these weights.

### Core distance

The core distance of a position is the smallest network distance within which the total weight, counting the position's own weight, reaches $`k`$ = `min_samples`:

```math
\mathrm{core}(p) = \min \Bigl\{\, r \ge 0 \;:\; \sum_{q \,:\, d(p, q) \le r} w_q \;\ge\; k \,\Bigr\}.
```

It is $`0`$ when $`w_p \ge k`$, and $`\infty`$ when the total weight within the search horizon stays below $`k`$ (see [`max_distance`](#max_distance)). A positive `core_distance_floor` $`f`$ raises every smaller core distance to $`f`$, so each core distance becomes $`\max\bigl(\mathrm{core}(p), f\bigr)`$. With the default $`f = 0`$, the result is standard HDBSCAN*.

### Mutual reachability and the hierarchy

For two positions within the search horizon of each other, the mutual reachability distance is

```math
d_{\mathrm{mr}}(p, q) = \max\bigl(\mathrm{core}(p),\ \mathrm{core}(q),\ d(p, q)\bigr).
```

Pairs farther apart than the horizon, and pairs that involve an infinite core distance, have no mutual-reachability edge. The hierarchy is built from the minimum spanning forest of these edges and is expressed with $`\lambda = 1/\text{distance}`$ ($`\lambda = \infty`$ at distance 0). It is a forest rather than a tree because a truncated graph can be disconnected; all its components are joined under one root at $`\lambda = 0`$.

The HDBSCAN* implementation is included in this package because a truncated sparse network-distance graph can be disconnected and can leave rows with fewer than `min_samples` stored neighbours. The implementation uses a deterministic multifurcation treatment for equal mutual-reachability distances rather than allowing arbitrary binary tie ordering to determine the hierarchy.

### Cluster selection and membership

The stability of a cluster $`C`$ sums, over its positions, how long each stays in $`C`$ on the $`\lambda`$ scale:

```math
S(C) = \sum_{p \in C} w_p \bigl(\lambda_p - \lambda_{\mathrm{birth}}(C)\bigr),
```

where $`\lambda_{\mathrm{birth}}(C)`$ is the $`\lambda`$ at which $`C`$ appears and $`\lambda_p`$ is the $`\lambda`$ at which $`p`$ leaves $`C`$, either as noise or into a child cluster. With `eom`, a cluster is selected when its stability is at least the summed stability of its selected descendants, and a cluster larger than `max_cluster_size` is not selected. With `leaf`, the clusters without child clusters are selected. `cluster_selection_epsilon` and `allow_single_cluster` have scikit-learn's meaning.

The membership strength of an observation at position $`p`$ in its selected cluster $`C`$ is

```math
\mathrm{membership}(p) = \frac{\min\bigl(\lambda_p,\ \lambda_{\max}(C)\bigr)}{\lambda_{\max}(C)},
```

where $`\lambda_{\max}(C)`$ is the largest $`\lambda`$ reached in $`C`$. It is 1 when $`\lambda_{\max}(C) = 0`$ or $`\lambda_p = \infty`$, and 0 for noise.

## `max_distance`

`max_distance` is a **search/truncation horizon**, not DBSCAN's `eps`. Network pairs farther apart are not measured. The hierarchy is exact below that horizon; the program has no distance information above it.

In symbols, a search with radius $`r`$ stores exactly the pairs of positions with $`d(p, q) \le r`$; every other pair is treated as infinitely far apart.

In `--distance-mode fixed`, the specified value is used exactly. In `--distance-mode adaptive`, it is a hard ceiling. The search begins at a smaller radius and grows geometrically. It can stop when two consecutive successful radii give the same flat cluster IDs/noise labels and membership values while the share of otherwise-resolvable observations with truncated core distance is at or below `--core-truncation-tolerance`.

The adaptive radii are

```math
r_0 = \frac{R}{g^{\,s-1}}, \qquad r_{t+1} = \min\bigl(R,\; g\, r_t\bigr),
```

where $`R`$ is `max_distance`, $`g`$ is `--distance-growth` (default 1.5) and $`s`$ is `--distance-steps` (default 4). `--min-distance` replaces $`r_0`$ when it is given, and a positive `core_distance_floor` keeps $`r_0`$ above that floor. With the defaults, the radii are $`0.30R`$, $`0.44R`$, $`0.67R`$ and $`R`$. The default `--core-truncation-tolerance` is 0.01, and membership values are compared within `--stability-tolerance` (default $`10^{-12}`$).

The adaptive status `converged` therefore means **local flat-result stability under that diagnostic rule**. It is not a proof that every larger radius would give the same clustering. `distance_trace.csv` records every trial so the selection can be audited.

The summary distinguishes:

- `share_core_beyond_max_distance`: all observations with infinite core distance;
- `share_core_structurally_unreachable`: observations on components whose total effective weight is below `min_samples`;
- `share_core_truncated`: otherwise-resolvable observations whose core distance lies beyond the current horizon.

`--max-neighbor-pairs` is a hard memory-safety limit. In fixed mode exceeding it raises `NeighborPairLimitError`. In adaptive mode, if a larger trial exceeds the limit after at least one successful trial, the last successful result is kept with `distance_status = pair_cap_limited`.

## Grouped data

With `--group-col`, each group is clustered independently while the input files, prepared network, network graph and point snapping are shared across the run. When null/blank groups are excluded, those rows are removed before snapping.

Null and blank group values are **excluded by default**. Use `--missing-group-policy include` only when they are genuine categories, or `error` when incomplete classification should stop the run. Missing values use the user-facing tokens `__null__` and `__blank__`; literal categories with those exact names remain valid and are shown as `literal:__null__` and `literal:__blank__` where disambiguation is needed. In a group-parameter CSV, prefix a reserved literal with `literal:`.

`--group-universe` accepts a one-column CSV declaring expected categories. Declared-but-absent groups receive zero-row outputs and summary rows. Observed groups outside the declared universe are still processed and explicitly reported.

`--group-params` accepts per-group overrides. Supported fields include `max_distance`, adaptive-distance settings, `min_cluster_size`, `min_samples`, cluster-selection settings, `core_distance_floor`, and `duplicates`.

Example:

```text
industry_code,max_distance,min_cluster_size,min_samples
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

## Provenance

`net-hdbscan` implements published methods. It does not propose a new clustering algorithm. The clustering model is HDBSCAN* (Campello et al., 2013a; Campello et al., 2015), applied to the shortest-path length between positions on a network, as in earlier network-constrained clustering (Yiu and Mamoulis, 2004). The HDBSCAN* definitions use only pairwise distances, through the core distance and the mutual reachability distance. Replacing Euclidean distance with network distance therefore changes the input, not the method. The conventions follow scikit-learn's `HDBSCAN` (Pedregosa et al., 2011) and the `hdbscan` library (McInnes et al., 2017), so that results can be compared.

With the default settings (`distance_mode="fixed"`, `core_distance_floor=0`, `cluster_selection_method="eom"`), the package computes standard HDBSCAN* on the network distance truncated at `max_distance`. Two optional settings are not taken from a published source; they are described under [What is specific to this package](#what-is-specific-to-this-package).

### Why the package has its own HDBSCAN* implementation

scikit-learn's `HDBSCAN` accepts a sparse precomputed distance matrix only when every row stores at least `min_samples` neighbours (or a fill value for missing distances is supplied) and the graph is connected. A network-distance graph truncated at `max_distance` meets neither condition in general. The package therefore computes the same algorithm on such a graph. This extends the input that the algorithm accepts; it does not change the model. If every pair that was not measured is given infinite distance, the standard HDBSCAN* definitions already give the two rules stated under [Mutual reachability and the hierarchy](#mutual-reachability-and-the-hierarchy): a position that cannot reach `min_samples` weight within the horizon has an infinite core distance, and separate components join only at infinite distance, which is $`\lambda = 0`$.

The DBSCAN* clustering at every distance up to `max_distance` is the same as without truncation. A cluster that is already separate at `max_distance` is born at $`\lambda = 0`$, because the program has no distance information above that value; this is the standard definition applied to the truncated distance.

### Source of each step

| Step in `net-hdbscan` | Published source or earlier implementation |
|---|---|
| Observations lie on network edges, and the distance is the shortest-path length along the network | Yiu and Mamoulis (2004). Snapping each point to its nearest road segment and splitting the segment at that location: Wang et al. (2019). |
| Bounded shortest-path search up to `max_distance`, with exact results below that distance | OPTICS uses a generating distance as an upper limit on the neighbourhood search to reduce computation, and clusterings for thresholds up to that limit can be extracted from its result (Ankerst et al., 1999). Shortest paths: Dijkstra (1959), computed with SciPy (Virtanen et al., 2020). |
| Core distance, mutual reachability distance, hierarchy, condensed tree, stability and Excess-of-Mass selection | HDBSCAN* (Campello et al., 2013a; Campello et al., 2015). Stability-based selection of clusters from a cluster tree: Campello et al. (2013b). A minimum spanning tree gives the single-linkage hierarchy (Gower and Ross, 1969). |
| Co-located observations compressed into one weighted position | Equivalent to HDBSCAN* on the uncompressed observations. Observations at one position are at distance 0 from each other, so each counts in the others' core-distance neighbourhood, and all of them leave a cluster at the same $`\lambda`$. |
| Equal mutual-reachability distances processed together, which can split a cluster into more than two parts at one level | The HDBSCAN* hierarchy is the set of DBSCAN* clusterings at every distance level (Campello et al., 2015). Below a distance shared by several edges, all of those edges are absent at once. A binary merge tree must put such edges in an arbitrary order; this is why results can differ from scikit-learn in cases with many ties. |
| `leaf` selection, `max_cluster_size`, `allow_single_cluster` and membership strength | Conventions of scikit-learn's `HDBSCAN` and the `hdbscan` library (McInnes et al., 2017). |
| `cluster_selection_epsilon` | HDBSCAN($`\hat{\varepsilon}`$) of Malzer and Baum (2020), with scikit-learn's meaning. |
| Partial selection of the smallest row entries when computing core distances | Selection algorithm (Hoare, 1961). The result is identical to sorting whole rows. |

### What is specific to this package

Two optional settings are the package's own work. Both are off by default.

**`core_distance_floor`.** A positive floor $`f`$ raises every core distance below $`f`$ to $`f`$. It exists because co-located observations whose total weight reaches `min_samples` have core distance 0, so $`\lambda`$ is infinite for them and any cluster that contains them receives infinite stability. At every distance level $`\varepsilon \ge f`$ the floor changes nothing, because $`\max(\mathrm{core}(p), f) \le \varepsilon`$ exactly when $`\mathrm{core}(p) \le \varepsilon`$. Below $`f`$, no position is core. The result is therefore the standard HDBSCAN* hierarchy cut at distance $`f`$, with stability accumulated only up to $`\lambda = 1/f`$. A split that the standard hierarchy shows only at distances below $`f`$ does not appear. Because the stability values change, the selected clusters can change, so a positive floor modifies the published model. No published source for this rule was found in the provenance review of September 2026. The closest published option, `cluster_selection_epsilon` (Malzer and Baum, 2020), changes which clusters are selected and does not modify the hierarchy.

**Adaptive distance mode** (`distance_mode="adaptive"`). The search radius grows geometrically up to `max_distance`, and the search stops when two consecutive radii give the same flat labels and membership values and the share of truncated core distances is within the tolerance. Its parts have precedents: a bounded neighbourhood search, as in OPTICS (Ankerst et al., 1999), and a search bound that grows geometrically, as in unbounded search (Bentley and Yao, 1976). The stopping rule itself is the package's own. It only selects a parameter: for the selected radius, the output is HDBSCAN* on the network distance truncated at that radius. As stated under [`max_distance`](#max_distance), the `converged` status means local stability, not a proof.

The other package-specific parts are engineering: canonical vertex and arc identity, deterministic snapping, batching and memory limits in the neighbour search, validation of sparse graphs, grouped execution, and diagnostics. None of them defines a new clustering model.

## References

Ankerst, M., Breunig, M. M., Kriegel, H.-P., & Sander, J. (1999). OPTICS: Ordering points to identify the clustering structure. In *Proceedings of the 1999 ACM SIGMOD International Conference on Management of Data*, 49–60. https://doi.org/10.1145/304181.304187

Bentley, J. L., & Yao, A. C.-C. (1976). An almost optimal algorithm for unbounded searching. *Information Processing Letters*, 5(3), 82–87.

Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013a). Density-based clustering based on hierarchical density estimates. In *Advances in Knowledge Discovery and Data Mining (PAKDD 2013)*, Lecture Notes in Computer Science 7819, 160–172. Springer. https://doi.org/10.1007/978-3-642-37456-2_14

Campello, R. J. G. B., Moulavi, D., Zimek, A., & Sander, J. (2013b). A framework for semi-supervised and unsupervised optimal extraction of clusters from hierarchies. *Data Mining and Knowledge Discovery*, 27(3), 344–371. https://doi.org/10.1007/s10618-013-0311-4

Campello, R. J. G. B., Moulavi, D., Zimek, A., & Sander, J. (2015). Hierarchical density estimates for data clustering, visualization, and outlier detection. *ACM Transactions on Knowledge Discovery from Data*, 10(1), 5:1–5:51. https://doi.org/10.1145/2733381

Dijkstra, E. W. (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1, 269–271.

Gower, J. C., & Ross, G. J. S. (1969). Minimum spanning trees and single linkage cluster analysis. *Journal of the Royal Statistical Society, Series C (Applied Statistics)*, 18(1), 54–64.

Hoare, C. A. R. (1961). Algorithm 65: Find. *Communications of the ACM*, 4(7), 321–322.

Malzer, C., & Baum, M. (2020). A hybrid approach to hierarchical density-based cluster selection. In *2020 IEEE International Conference on Multisensor Fusion and Integration for Intelligent Systems (MFI)*, 223–228. https://doi.org/10.48550/arXiv.1911.02282

McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. *Journal of Open Source Software*, 2(11), 205. https://doi.org/10.21105/joss.00205

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2

Wang, T., Ren, C., Luo, Y., & Tian, J. (2019). NS-DBSCAN: A density-based clustering algorithm in network space. *ISPRS International Journal of Geo-Information*, 8(5), 218. https://doi.org/10.3390/ijgi8050218

Yiu, M. L., & Mamoulis, N. (2004). Clustering objects on a spatial network. In *Proceedings of the 2004 ACM SIGMOD International Conference on Management of Data*, 443–454. https://doi.org/10.1145/1007568.1007619

## License

MIT.
