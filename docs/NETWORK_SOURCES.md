# Network sources and provenance

`net-hdbscan` accepts **an explicit distance model**, never a presumed domain.

There are three ways to supply connectivity:

1. **User-supplied projected line network:** `--network lines.gpkg`, the most general geospatial workflow. The linework may represent roads, waterways, ecological corridors, vessel centerlines, or any other *two-dimensional undirected geometric network*, provided the data actually justify that interpretation.
2. **Explicit OpenStreetMap acquisition:** select `--network-source osm`, exactly one OSM way selector (`--osm-profile walk`, `--osm-profile waterways`, or custom `--osm-filter`), and exactly one geographic region: `--bbox W,S,E,N` **or** `--center lon,lat --radius-km R`. OSMnx is an optional acquisition adapter; OSM is **never inferred** from a missing `--network` argument.
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

`--bbox` uses **EPSG:4326 longitude/latitude** and order **west,south,east,north** (not the order of many GIS APIs). The adapter rejects inverted boxes, antimeridian crossing, and requests estimated to exceed 10,000 km² to avoid accidental oversized public-service requests. Smaller regions can still be expensive; network availability depends on OSM/Overpass. The package never performs network retrieval without the explicit `osm` source. Bbox acquisition retains its existing behavior; a geodesic circle is a new, alternative network-acquisition region. Both bound **network geometry**, not input point eligibility.

`--osm-profile` must be one of `walk`, `bike`, `drive`, `drive_service`, `all`, `all_public`, **`waterways`**. It selects **which OSM ways to download**, not directional routing semantics. OSMnx's graph is projected, retained disconnected components are kept, and its edge geometry is saved to `network.gpkg`. The clustering pipeline then computes *undirected geometric arc-length* shortest paths. **One-way restrictions, travel time, turn restrictions, capacities, and direction of water or blood flow are not respected.** Calling this a driving-time or downstream-flow distance would be incorrect.

For explicitly tagged non-street ways, you can instead supply a custom filter:

```powershell
net-hdbscan cluster --points aquatic_observations.parquet --network-source osm --bbox "120.97,14.58,121.05,14.66" --osm-filter '[waterway~"river|stream"]' --max-distance 1000 --min-cluster-size 5 --output-dir aquatic_results
```

**A way-tag filter does not guarantee a scientifically valid river network.** Waterway tagging, crossing topology, flow direction, digitizing gaps, and segment connectivity must be checked. Where possible, bring your own independently validated line network instead.

## Named waterways and circle acquisitions

`--osm-profile waterways` is a shorthand for the bracketed OSM way filter:

```text
[waterway~"^(river|stream|canal|tidal_channel)$"]
```

It selects ways tagged `river`, `stream`, `canal` or `tidal_channel`; it does **not** imply navigability, flow direction, line connectivity or a validated watershed. OSM custom filters remain supported **instead of** a profile. Do not use `--osm-profile` and `--osm-filter` together.

To request a **center-plus-radius** region, replace `--bbox` with `--center "121.04,14.65" --radius-km 10`. This uses a 360-segment WGS84 geodesic polygon (a finite polygonal approximation, not a perfect continuous circle), queried with `OSMnx graph_from_polygon(..., truncate_by_edge=True)`. The projected line geometry is then clipped to the **actual polygon**, not the enclosing rectangle; only positive-length lines are retained. Invalid/missing coordinates, nonpositive/nonfinite radii, antimeridian-crossing or pole-reaching circles, and requests exceeding **10,000 km²** are rejected before a remote request. `--bbox` and `--center`/`--radius-km` are mutually exclusive.

```powershell
net-hdbscan fetch-network --source osm --center "121.04,14.65" --radius-km 10 --osm-profile waterways
net-hdbscan cluster --points aquatic_observations.parquet --network-source osm --center "121.04,14.65" --radius-km 10 --osm-profile waterways --max-distance 3000 --min-cluster-size 5 --output-dir aquatic_results
```

**Do not confuse the limits:** `--radius-km` is an acquisition extent in kilometers; `--max-distance` limits measured **network shortest-path distance** in the projected line layer's units; `--max-snap-distance` optionally fails on points too far from the network. Neither circle nor bbox excludes observation rows. Study-area point eligibility filtering remains **upstream** of the package.

The full [geographic acquisition guide](GEOGRAPHIC_ACQUISITION.md) includes a configuration matrix, cache metadata schema, clipping details, Python API, usage examples, common failures and QA recommendations.

## Persistent network cache

OSM downloads are saved to an immutable, request-keyed cache. By default:

- Windows: `%LOCALAPPDATA%\net-hdbscan\osm\<request-key>\`
- Other platforms: `~/.cache/net-hdbscan/osm/<request-key>/`

Use `--osm-cache-dir` for an explicit project-controlled location. Each entry holds `network.gpkg` and `source.json`. The JSON records the **bbox or geodesic circle**, selected OSM profile or custom filter, retrieval time (UTC), OSMnx version, projected CRS, ODbL attribution, source interpretation, and SHA-256 of the network. Circular requests store `circle_epsg4326_lon_lat_radius_km`, `circle_boundary_segments: 360` and `adapter_schema: 2` inside the `request` object; existing bbox cache request keys remain on `adapter_schema: 1` and are unchanged. Source metadata also records `clipped_to_circle` and `resolved_osm_filter`. Cache hits check the checksum and make **no network request**. Corrupted or partial entries fail rather than being overwritten silently. Use a fresh cache directory if you intentionally want a new OSM snapshot; retain the earlier source for reproducibility.

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
