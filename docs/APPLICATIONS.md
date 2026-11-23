# Domain-neutral applications and validity boundaries

`net-hdbscan` is **HDBSCAN*** driven by a scientifically meaningful *symmetric network distance*. The name refers to a network, not to transport, land use, or an economic sector. Applications below are examples of possible representations—not claims of validation in those sciences.

| Area | Observed entities | Candidate network | Distance/model judgment | Appropriate entry point |
| --- | --- | --- | --- | --- |
| Urban and rural geography | Places, stations, events | Walkable paths, roads, rail linework | Undirected geometric path length is not car travel time | Geospatial line pipeline or OSM adapter |
| Freshwater ecology | Species samples or sensors | River/stream linework | Flow direction, connectivity and barriers may invalidate symmetric lengths; verify source topology | Validated linework; named OSM `waterways` profile or custom tag filter **only exploratory** |
| Landscape ecology | Organism observations | Ecological corridors | Must define corridor cost and permeability; maps alone do not imply movement | User-provided projected lines or sparse distances |
| Sociology | People, organizations | Relational/contact graph | Edge meaning and distance transform must be specified; no bbox is meaningful | Symmetric sparse-distance Python API |
| Epidemiology | Cases or locations | Mobility/contact network | Temporal exposure and direction may dominate; shortest path is not transmission probability | External modeling then sparse-distance API |
| Biology and anatomy | Sample locations | Vessel/duct centerlines | Flow direction, 3-D lengths, diameter/resistance are omitted by 2-D geospatial pipeline | External validated distance matrix when appropriate |
| Infrastructure science | Sensors, defects, incidents | Pipeline/cable/water-distribution network | Check physical barriers, valves, direction and weights | Projected lines if symmetric length is valid |

## Waterway observations: acquisition is not a flow model

The named OSM `waterways` profile selects ways tagged `river`, `stream`, `canal` or `tidal_channel`. A bounding box or center-plus-radius geodesic circle constrains the **acquired line network**; it does not select or exclude aquatic observations. The circle clips network centerline geometry to a 360-sided geodesic boundary approximation. The resulting graph is still a **two-dimensional undirected geometric network**.

For ecological observations, sensors or river-adjacent facilities, first justify whether **symmetric distance measured along centerlines** represents the process under study. Upstream/downstream flow, hydraulic resistance, seasonal inundation, waterfalls, dams, riverbank access, navigability, waterbody polygons and estuarine/coastal connectivity are not encoded. Disconnected branches and line crossings do not become connected merely because they look adjacent on a map. OSM map coverage is not evidence of validated hydrological topology.

A circular network cut may truncate a reach connecting observations through geometry **outside** the circle; that can alter shortest paths and components. If the scientific object is connectivity of observations *within* a region rather than travel paths *confined to* that region, decide and document whether the network should be acquired with a wider buffer. Point-subset eligibility remains an explicit upstream decision. A custom prevalidated network can be supplied using `--network`; a hydrologically defined **symmetric** distance graph can be given to the sparse API if the scientific problem demands it.

See the [geographic acquisition guide](GEOGRAPHIC_ACQUISITION.md) and the [network provenance contract](NETWORK_SOURCES.md).

## Stratification is not a default

`--group-col` means **run separate analyses by category**. It does not change the distance rule and does not learn cross-category clusters. It should be omitted when all observations belong in one clustering analysis. There is no required `industry_code`, `species`, `department`, or other domain field.

## Different notions of a network

**Geographic line networks** encode metric-constrained movement along 2-D polylines. The file/CLI workflows snap points and calculate undirected shortest-path lengths up to a bounded horizon. Network intersections are based on shared projected vertices rather than visible crossings, and the recorded snap distance is quality-control metadata, not part of the clustering distance.

**Abstract graphs** (e.g., person-to-person ties, cell signaling, communication networks) require a different interpretation. Construct an appropriate symmetric sparse distance graph externally and pass it to `hdbscan` or `fit_hierarchy`; OSM, bounding boxes, snapping and GeoPackages are irrelevant here. A tie weight representing *strength* is not automatically a metric *distance*.

**Directed networks and temporal networks** require additional modeling. The current core rejects asymmetric distance matrices; using OSM's one-way or waterway direction as though implemented would be scientifically misleading. No automatic conversion is performed.

## Example: a relational graph without geographic coordinates

The network-distance core does not require longitude, latitude, line geometry,
OSM or a `--group-col`. Consider an undirected, unweighted relation graph;
one possible (and contestable) distance is a weighted shortest path
between nodes. The bridge between triangles has length two:

```python
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import shortest_path
from net_hdbscan import hdbscan

# Observations could be individuals, organizations, cells or devices.
# This toy graph defines symmetric ties, NOT their scientific validity.
edges = np.array([(0, 1), (1, 2), (2, 0),
                  (3, 4), (4, 5), (5, 3), (2, 3)])
row, col = edges.T
n = 6
edge_lengths = np.where((row == 2) & (col == 3), 2.0, 1.0)
adj = coo_matrix((np.r_[edge_lengths, edge_lengths],
                  (np.r_[row, col], np.r_[col, row])), shape=(n, n)).tocsr()
d = shortest_path(adj, directed=False, unweighted=False)

# The sparse clustering input stores only known finite distances
# within a declared two-hop horizon; absent pairs remain absent.
keep = np.isfinite(d) & (d > 0) & (d <= 2)
i, j = np.nonzero(keep)
distance_graph = csr_matrix((d[i, j], (i, j)), shape=(n, n))
result = hdbscan(distance_graph, min_cluster_size=2, min_samples=2)
print(result.labels)
```

This is a deliberately small API illustration, **not** a general-purpose
social-network distance model. You must separately decide whether hop count,
weighted path cost, resistance distance or another symmetric distance is
appropriate. The sparse graph must be symmetric and nonnegative as required
by the existing HDBSCAN API.

## Minimum validity checks

Before interpreting a cluster, document your observations, network source/version, topology, coordinate system and units, connectivity and disconnections, shortest-path cost, truncation horizon, optional stratification policy, and whether symmetric distance is defensible. Examine snap distances and core-distance truncation diagnostics. Compare results across plausible network models when scientific conclusions are sensitive to the model.

See [network sources and reproducibility](NETWORK_SOURCES.md) and the top-level [README](../README.md).
