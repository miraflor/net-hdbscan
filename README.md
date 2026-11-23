# net-hdbscan

**Deterministic HDBSCAN\* clustering over symmetric network distances.**

`net-hdbscan` detects density-based clusters using distances defined by connectivity rather than assuming Euclidean proximity. The core accepts a validated, symmetric sparse distance graph; the geospatial workflow derives distances from shortest paths along a projected line network. An optional OpenStreetMap (OSM) adapter retrieves geographic linework when explicitly requested.

It is a clustering package, **not** a network-inference, topology-repair, transport-routing, or territory-generation package. Its methods are applicable wherever the supplied network distance has a defensible interpretation.

**[Algorithm walkthrough](https://miraflor.github.io/net-hdbscan/)** · **[Geographic acquisition: roads, waterways, circles](docs/GEOGRAPHIC_ACQUISITION.md)** · **[Network sources](docs/NETWORK_SOURCES.md)** · **[Scientific applications](docs/APPLICATIONS.md)** · **[Development and verification](docs/DEVELOPMENT.md)**

## Installation

From a source checkout:

```powershell
python -m pip install -e ".[dev]"
```

The core uses GeoPandas, NumPy, pandas, Pyogrio, SciPy, and Shapely. Optional extras are:

| Extra | Adds |
| --- | --- |
| `files` | PyArrow for GeoParquet/Parquet input and output |
| `cli` | Typer and PyArrow for the `net-hdbscan` command |
| `osm` | OSMnx 2.x for explicitly requested network downloads |
| `dev` | File/CLI requirements, pytest, and scikit-learn for validation |

To use OSM acquisition in a development checkout, install `python -m pip install -e ".[dev,osm]"`. OSMnx is **not** required for the clustering core or to reuse an already cached network. The CLI can also be invoked with `python -m net_hdbscan`.

## Choose your input model

`net-hdbscan` has two complementary entry points:

| Input | What you provide | Distance used | Entry point |
| --- | --- | --- | --- |
| **Geospatial line network** | Point observations and a projected LineString/MultiLineString network | Undirected shortest-path length along network geometry | CLI or `cluster_geodataframes` |
| **Sparse distance graph** | A symmetric sparse matrix of scientifically defined nonnegative distances | The supplied distances; absent entries mean unknown/out of the modeled horizon | `hdbscan`, `fit_hierarchy`, `extract_clusters` |

The first workflow can take a user-supplied network (roads, river centerlines, ecological corridors, cables, etc.) or retrieve OSM ways through the optional adapter. OSM is a **data source**, not the meaning of the network. A bounding box is never interpreted as a request for roads by default.

The sparse-distance API does not require geographic coordinates, bounding boxes, or OSM. For example, a relational network or a precomputed ecological connectivity model may be appropriate **if its distances are symmetric and meaningful**. Edge strength, probability, and direction are not interchangeable with distance without an explicit model.

**Current modeling boundary:** the geospatial workflow measures *two-dimensional undirected geometric length*. It does not honor one-way access, travel time, river-current direction, flow resistance, temporal connections, or true 3-D length. Do not interpret OSM's `drive` or `walk` profiles as implementing those costs; they select ways for acquisition. See [network sources and limitations](docs/NETWORK_SOURCES.md).

### Is a grouping column needed?

**No.** By default, all supplied observations participate in one clustering analysis. `--group-col` is optional **stratification**: it runs a separate clustering within each value of an arbitrary user-chosen column, so clusters cannot cross categories. This could be species, experimental condition, survey stratum, or something else—not a prescribed industry or other domain field. Omit it if between-category clusters are important.

## Quick start: spatial networks

### A. Use your own network

You need a geospatial point layer with a unique observation ID and a projected line-network layer. The network's coordinate reference system (CRS) determines the distance units; point coordinates are reprojected to it.

```powershell
net-hdbscan cluster `
  --points "data\observations.parquet" `
  --network "data\network.gpkg" --network-layer lines `
  --point-id-col observation_id `
  --max-distance 3000 `
  --min-cluster-size 20 `
  --output-dir "output\clusters"
```

`--network-layer lines` is needed only when that GeoPackage layer is named `lines`; omit it for a single-layer GeoPackage or other supported line format. If the projected CRS uses meters, `3000` denotes a 3 km network-distance horizon.

### B. Retrieve an explicitly selected OSM network

```powershell
net-hdbscan cluster `
  --points "data\observations.parquet" `
  --network-source osm `
  --bbox "120.97,14.58,121.05,14.66" `
  --osm-profile walk `
  --point-id-col observation_id `
  --max-distance 1500 `
  --min-cluster-size 10 `
  --output-dir "output\osm_clusters"
```

`--bbox` is **west,south,east,north** in EPSG:4326 longitude/latitude. Choose exactly one supported `--osm-profile` or `--osm-filter`; the package will not infer the network type. A custom way filter such as `--osm-filter '[waterway~"river|stream"]'` is possible, but retrieved OSM waterways are not automatically a validated hydrological or ecological connectivity model.

Alternatively, use `--center "121.04,14.65" --radius-km 10` instead of `--bbox`. The center and radius define a geodesic acquisition circle: retrieved network linework is clipped to its boundary. The named `--osm-profile waterways` selects rivers, streams, canals, and tidal channels. The bounding region limits the fetched network, **not** point eligibility; `--radius-km` and `--max-distance` have separate meanings.

OSM requests use an immutable, checksum-verified local cache with acquisition provenance. Public-service availability and coverage vary. For regional extracts or non-OSM data, prepare your own projected line network and use `--network`. See [retrieval, provenance, and limitations](docs/NETWORK_SOURCES.md).

### C. Acquire OSM waterways inside a bounding box

The named `waterways` profile selects OSM `river`, `stream`, `canal` and `tidal_channel` ways; this is **way selection**, not hydrological routing, navigability or flow direction.

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

### D. Acquire OSM waterways inside a geodesic circle

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

`--center` is **longitude,latitude** in EPSG:4326; `--radius-km` is the geographic acquisition radius in kilometers. The circle is approximated by **360 WGS84 geodesic boundary vertices** and the downloaded lines are clipped to that polygon, not to its bounding rectangle. Use either `--bbox` **or** the **pair** `--center` and `--radius-km`, never both. The OSM adapter rejects overly large, invalid, polar and antimeridian-crossing regions.

**Three distinct distances:** `--radius-km` is the geographic download extent; `--max-distance` is the shortest-path search horizon in the network's projected length units; `--max-snap-distance` optionally fails when an observation is too far from the network. Neither bbox nor circle filters the observation rows. See the [complete acquisition guide](docs/GEOGRAPHIC_ACQUISITION.md) for `fetch-network`, cache keys, provenance, compatibility, known limitations and troubleshooting.

### E. Optional independent stratification

Add `--group-col stratum` to a spatial clustering command when you explicitly want a separate HDBSCAN* run for each value of the `stratum` column. Group-specific overrides, missing-category rules, and declared category universes are described below.

## How the clustering works

**Geospatial pipeline**

```text
observation points + supplied / explicitly retrieved linework
       ↓
snap points to continuous network positions
       ↓
compress identical snapped positions (with effective weights)
       ↓
compute bounded, sparse network shortest-path distances
       ↓
HDBSCAN*: core distance → mutual reachability → spanning forest
       ↓
condensed hierarchy → EOM or leaf extraction
       ↓
labels, membership strengths, hierarchy, diagnostics
```

The observation-to-network snap distance is reported for quality assurance but **is not added** to the network shortest-path clustering distance. Disconnected network components have no finite path between them. Line crossings connect only at shared network vertices after coordinate canonicalization; visible crossings alone do not create intersections.

### Density and selection

- `--min-cluster-size` sets the smallest eligible cluster; `--min-samples` controls the core-distance density estimate and defaults to the minimum cluster size.
- `--cluster-selection-method eom` (default) and `leaf` extract flat clusters from the condensed hierarchy. Equal mutual-reachability distances are handled using deterministic, simultaneous multifurcations rather than arbitrary binary tie ordering.
- `--duplicates count` (default) retains the multiplicity of observations snapped to one position; `--duplicates once` treats each distinct position as one effective observation.
- `--noise-policy exclude` (default) leaves noise unassigned. `singleton` assigns public identifiers to individual noise positions.

### The distance horizon is not DBSCAN epsilon

`--max-distance` is the **maximum distance measured**, not a DBSCAN `eps` neighborhood threshold. Unmeasured pairs beyond this horizon do not become known distances.

With `--distance-mode fixed`, the requested horizon is used directly. With `adaptive`, it is a hard ceiling: a sequence of increasing radii may stop after consecutive trials give stable flat results with sufficiently low *otherwise-resolvable* core-distance truncation. A status of `converged` establishes **local stability under that rule**, not global invariance to every larger horizon. Examine `distance_trace.csv` before interpreting it.

`--max-neighbor-pairs` is a fatal resource limit, including in adaptive mode: exceeding it aborts rather than silently returning a smaller-radius result.

### Grouping details (optional)

With `--group-col`, group runs share network preparation and point snapping while performing separate cluster extraction. Null/blank groups are excluded by default (`--missing-group-policy exclude`); choose `include` or `error` deliberately. Reserved display tokens `__null__` and `__blank__` distinguish missing group values, while literal categories with those names are preserved using `literal:` disambiguation.

`--group-universe` can declare expected categories, including absent ones, in a one-column CSV. `--group-params` supplies per-group overrides such as distance horizon, cluster size, density settings, or selection settings:

```csv
stratum,max_distance,min_cluster_size,min_samples
A,5000,,
B,1500,5,5
```

## Outputs and diagnostics

The CLI writes a complete output generation into `--output-dir`:

| File | Meaning |
| --- | --- |
| `clustered_points.parquet` | Input attributes, stable public cluster IDs, noise flags, membership, core and snap distances, snapped coordinates |
| `cluster_table.parquet` | Cluster sizes (observations and distinct snapped positions), stability, birth distance, singleton flag |
| `cluster_hierarchy.parquet` | Condensed hierarchy, ancestry, stability, selected status, public IDs |
| `summary.csv` | Cluster counts, retention/noise, group and component counts, truncation diagnostics, selected horizon, parameters and runtime |
| `distance_trace.csv` | Per-radius outcomes, pair counts, cluster/noise summaries, stability and runtime |
| `manifest.json` | Input/configuration metadata, CRS, file references, input and output SHA-256 integrity data |

For stratified runs, the point, cluster, and hierarchy files instead appear under `points/`, `clusters/`, and `hierarchy/` as per-group Parquet files, alongside the shared summary, trace, and manifest.

In particular, distinguish core distances that remain infinite because of **structural disconnection** from those **truncated by the chosen search horizon**. The diagnostics report both; a zero-cluster explanation lists co-occurring indicators, not a uniquely proven cause. The `can_form_two_min_size_groups` flag is a necessary condition only.

## Python API

### Geospatial objects

```python
import geopandas as gpd
from net_hdbscan import HDBSCANConfig, cluster_geodataframes

result = cluster_geodataframes(
    points=gpd.read_parquet("observations.parquet"),
    network=gpd.read_file("network.gpkg"),
    config=HDBSCANConfig(max_distance=3000, min_cluster_size=20),
    point_id_col="observation_id",
)

result.points      # observations and labels
result.clusters    # cluster-level table
result.hierarchy   # condensed hierarchy
result.summary     # run diagnostics
result.trace       # tested distance radii
```

### Optional OSM network acquisition in Python

```python
from net_hdbscan.osm_source import fetch_osm_network

# A verified immutable cache hit does not download again.
network_path = fetch_osm_network(
    center=(121.04, 14.65),
    radius_km=10,
    profile="waterways",
)
```

The acquisition function returns a path to a projected `network.gpkg`. Read it with GeoPandas and pass the line layer into `cluster_geodataframes`, or use the CLI with `--network`. OSMnx 2.x is needed for cache misses (`.[osm]`), not for the clustering core or cache hits. Detailed behavior: [Geographic acquisition](docs/GEOGRAPHIC_ACQUISITION.md).

### Sparse network distances and reusable hierarchy

```python
from net_hdbscan import hdbscan, fit_hierarchy, extract_clusters, compare_extractions

# sparse_distances: validated, symmetric scipy.sparse matrix of distances
result = hdbscan(sparse_distances, min_cluster_size=5)

fitted = fit_hierarchy(sparse_distances, min_cluster_size=5)
eom = extract_clusters(fitted, method="eom")
leaf = extract_clusters(fitted, method="leaf")
comparison = compare_extractions(fitted)
```

Fit once to reuse the same network-distance hierarchy for multiple extraction policies. `comparison.summary` reports selected-cluster counts, effective-weight shares, stability, and extraction time; `comparison.eom_parent_to_leaf_descendants` maps selected hierarchy nodes. No automatic EOM/leaf method selection is performed; the hierarchy is an in-memory object, not a serialized checkpoint.

The sparse input must be **square, symmetric, finite and nonnegative at stored positions**, with no duplicate stored coordinate pairs. Optional weights are positive integer multiplicities. Missing off-diagonal entries denote unknown/out-of-horizon distances, not zero-length edges. Construct a scientifically justified symmetric graph before using this API; see [application examples](docs/APPLICATIONS.md).

Other public functions include `build_network_graph`, `snap_points`, `distinct_positions`, and `neighbor_graph`.

## Reproducibility and operational safety

The implementation uses deterministic tie handling and canonical conventions to avoid dependence of the underlying flat partition on arbitrary input row order or observation-ID names, for an otherwise identical input and configuration. Changing the network, distance model, horizon, or preprocessing can change results; public label *names* may also change without changing the partition.

File-based runs stage and verify a complete generation before replacing an existing output. The publication journal distinguishes reversible preparation from a committed generation, uses checksums and integrity checks, and supports recoverable post-commit cleanup. **This is not an atomic directory-tree transaction on every operating system**, and a crash can briefly leave the destination unavailable. `--force` refuses to replace arbitrary or unrecognized directories. Recovery never intentionally restores a partially deleted predecessor backup.

Spatial study-area eligibility filtering is outside the package: every supplied point is eligible unless optional stratification explicitly excludes missing-category rows. The software does not infer topology or scientifically validate the user's proposed distance. In particular, directed, temporal and true 3-D network-distance models are **not** provided by the current geospatial pipeline.

## Documentation and development

- [Illustrated algorithm guide](https://miraflor.github.io/net-hdbscan/)
- [Geographic acquisition: examples, selection, circles, clipping and errors](docs/GEOGRAPHIC_ACQUISITION.md)
- [Network acquisition, cache and provenance](docs/NETWORK_SOURCES.md)
- [Applications and modeling boundaries](docs/APPLICATIONS.md)
- [Development, testing, CI, failure-path verification and benchmarking](docs/DEVELOPMENT.md)
- [Design and internal contracts](DESIGN.md)
- [Changelog](CHANGELOG.md)

## License and citation

Released under the [MIT License](LICENSE).

**James Matthew Miraflor** — [Scientific Computing Laboratory](https://scl.dcs.upd.edu.ph/), [Department of Computer Science](https://dcs.upd.edu.ph/), College of Engineering, University of the Philippines Diliman

**net-hdbscan** is authored and maintained by James Matthew Miraflor, with AI assistance in software development, debugging, testing, and documentation. Responsibility for methodological decisions, scientific validity, and published outputs rests with the author.
