# Network sources and provenance

`net-hdbscan` accepts **an explicit distance model**, never a presumed domain.

There are three ways to supply connectivity:

1. **User-supplied projected line network:** `--network lines.gpkg`, the most general geospatial workflow. The linework may represent roads, waterways, ecological corridors, vessel centerlines, or any other *two-dimensional undirected geometric network*, provided the data actually justify that interpretation.
2. **Explicit OpenStreetMap acquisition:** `--network-source osm --bbox W,S,E,N --osm-profile walk` (or another profile), optionally using `--osm-filter '[waterway~"river|stream"]'` *instead of* a profile. OSMnx is an optional adapter; OSM is **never inferred** from a missing `--network` argument.
3. **Symmetric sparse-distance graph:** call the low-level `hdbscan(...)`, `fit_hierarchy(...)`, or `compare_extractions(...)` Python API. It can represent abstract relationship networks after distances have been defined and computed externally. No geographic bbox or snapping is needed.

The first two use the geospatial line-network pipeline. The third uses the generic sparse HDBSCAN* core. These are different input contracts, not interchangeable representations.

## Install

```powershell
python -m pip install -e ".[dev,osm]"
```

`[osm]` adds OSMnx 2.x for **cache misses only**. No runtime network request is made on importing `net_hdbscan` or running the sparse-distance core.

## Fetch from OpenStreetMap, then cluster

An explicit, reproducible workflow:

```powershell
net-hdbscan fetch-network --source osm --bbox "120.97,14.58,121.05,14.66" --osm-profile walk
net-hdbscan cluster --points observations.parquet --network "PATH_PRINTED_BY_FETCH_COMMAND" --point-id-col observation_id --max-distance 1500 --min-cluster-size 10 --output-dir results
```

Or use one command and let it reuse the persistent network cache:

```powershell
net-hdbscan cluster --points observations.parquet --network-source osm --bbox "120.97,14.58,121.05,14.66" --osm-profile walk --point-id-col observation_id --max-distance 1500 --min-cluster-size 10 --output-dir results
```

`--bbox` uses **EPSG:4326 longitude/latitude** and order **west,south,east,north** (not the order of many GIS APIs). The adapter rejects inverted boxes, antimeridian crossing, and requests estimated to exceed 10,000 km² to avoid accidental oversized public-service requests. Smaller regions can still be expensive; network availability depends on OSM/Overpass. The package never performs network retrieval without the explicit `osm` source.

`--osm-profile` must be one of `walk`, `bike`, `drive`, `drive_service`, `all`, `all_public`. It selects **which OSM ways to download**, not directional routing semantics. OSMnx's graph is projected, retained disconnected components are kept, and its edge geometry is saved to `network.gpkg`. The clustering pipeline then computes *undirected geometric arc-length* shortest paths. **One-way restrictions, travel time, turn restrictions, capacities, and direction of water or blood flow are not respected.** Calling this a driving-time or downstream-flow distance would be incorrect.

For explicitly tagged non-street ways, you can instead supply a custom filter:

```powershell
net-hdbscan cluster --points aquatic_observations.parquet --network-source osm --bbox "120.97,14.58,121.05,14.66" --osm-filter '[waterway~"river|stream"]' --max-distance 1000 --min-cluster-size 5 --output-dir aquatic_results
```

**A way-tag filter does not guarantee a scientifically valid river network.** Waterway tagging, crossing topology, flow direction, digitizing gaps, and segment connectivity must be checked. Where possible, bring your own independently validated line network instead.

## Persistent network cache

OSM downloads are saved to an immutable, request-keyed cache. By default:

- Windows: `%LOCALAPPDATA%\net-hdbscan\osm\<request-key>\`
- Other platforms: `~/.cache/net-hdbscan/osm/<request-key>/`

Use `--osm-cache-dir` for an explicit project-controlled location. Each entry holds `network.gpkg` and `source.json`. The JSON records the bounding box, selected OSM filter, retrieval time (UTC), OSMnx version, projected CRS, ODbL attribution, source interpretation, and SHA-256 of the network. Cache hits check the checksum and make **no network request**. Corrupted or partial entries fail rather than being overwritten silently. Use a fresh cache directory if you intentionally want a new OSM snapshot; retain the earlier source for reproducibility.

The standard `manifest.json` records the resolved cached `network.gpkg` path and its input checksum. Preserve the cache's adjacent `source.json` to reproduce acquisition details. Cached network extracts use OpenStreetMap data subject to the [ODbL](https://www.openstreetmap.org/copyright). Cite OpenStreetMap contributors as appropriate and comply with attribution and use requirements.

Large regional extracts (such as **Geofabrik** PBFs) are **not downloaded or parsed by this adapter**. For larger projects, preprocess a suitable regional extract into a projected line network and supply it via `--network`; this avoids abusive public Overpass queries and exposes the topology decisions to review.

## Domain-specific distance is the user's scientific decision

Do not infer a transport mode from a bbox, a grouping attribute, or the domain label. A source file answers *where the network came from*, while the distance model answers *what a connection means*.

- Geographic line networks presently support **undirected projected length**.
- Abstract sparse distance inputs support any scientifically justified **symmetric finite non-negative** pair distances with missing entries representing unknown/out-of-horizon, after careful external construction.
- Directed, asymmetric and flow-weighted distances are **not yet part of the core contract**. Symmetrization, if undertaken upstream, must be declared and justified scientifically.
- True 3-D geometric line networks are **not supported by the 2-D line pipeline**. A 3-D distance model may be converted externally into an appropriate symmetric sparse distance matrix; z coordinates must not be silently discarded while claiming physical length.

## Optional stratification (`--group-col`)

A group column is never required. If supplied, points are clustered **independently** within each category while using a shared network. This excludes between-category clusters by construction; it is not a category-informed distance or multi-layer network method. The group may be taxon, site, population, experimental condition, or any column chosen by the researcher. The input must also contain a unique point identifier (`--point-id-col`, default `point_id`). No particular industry or subject-area field is assumed.
