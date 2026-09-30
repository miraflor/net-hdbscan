"""End-to-end network-space HDBSCAN*: points -> clusters and hierarchy."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from ._version import __version__
from .config import (
    BLANK_GROUP_KEY,
    NULL_GROUP_KEY,
    HDBSCANConfig,
    group_display,
    group_key,
    normalize_group_params,
    resolve_group_config,
)
from .sparse_hdbscan import hdbscan
from .io import (
    canonical_order,
    prepare_network,
    prepare_points,
    read_vector,
    _require_pyarrow,
    write_csv,
    write_geoparquet,
    write_json,
    write_parquet,
)
from .network import (
    DEFAULT_VERTEX_DIGITS,
    NetworkGraph,
    Snaps,
    build_network_graph,
    distinct_positions,
    empty_snaps,
    neighbor_graph,
    snap_points,
    NeighborPairLimitError,
)

SUMMARY_COLUMNS = [
    "group",
    "n_points_input",
    "n_points",
    "n_positions",
    "max_position_weight",
    "stacked_positions",
    "n_pairs",
    "share_core_beyond_max_distance",
    "share_core_structurally_unreachable",
    "share_core_truncated",
    "network_components_used",
    "share_on_largest_component",
    "snap_distance_median",
    "snap_distance_p95",
    "snap_distance_max",
    "n_clusters",
    "noise_share",
    "largest_cluster_share",
    "max_distance",
    "requested_max_distance",
    "distance_mode",
    "distance_status",
    "distance_steps_run",
    "distance_stable",
    "pair_limit_encountered",
    "pair_limit_distance",
    "core_truncation_tolerance",
    "min_cluster_size",
    "min_samples",
    "cluster_selection_method",
    "cluster_selection_epsilon",
    "core_distance_floor",
    "duplicates",
    "declared_in_universe",
    "missing_group_policy",
    "n_missing_group_input",
    "seconds",
]


CLUSTER_COLUMNS = ["cluster_id", "n_points", "n_positions", "stability", "birth_distance", "singleton"]


def _cluster_table(table: pd.DataFrame) -> pd.DataFrame:
    """The per-cluster table with fixed columns and types (also when empty)."""
    types = {"cluster_id": object, "n_points": "int64", "n_positions": "int64", "stability": float, "birth_distance": float, "singleton": bool}
    out = pd.DataFrame({column: pd.Series(dtype=kind) for column, kind in types.items()})
    if table is None or len(table) == 0 or "n_points" not in table.columns:
        return out
    return table[CLUSTER_COLUMNS].astype(types).reset_index(drop=True)


@dataclass
class ClusterOutputs:
    """Results for one clustering unit (a whole run or one group)."""

    points: gpd.GeoDataFrame
    hierarchy: pd.DataFrame
    summary: dict[str, Any] = field(default_factory=dict)
    trace: pd.DataFrame | None = None
    clusters: pd.DataFrame | None = None
    analysis: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# One clustering unit
# ---------------------------------------------------------------------------


def _cluster_ids(n_obs: int, position: np.ndarray, pos_label: np.ndarray, singleton_noise: bool) -> np.ndarray:
    """Public cluster IDs ``C000001, ...`` numbered by each cluster's first observation.

    Observations are in canonical order (sorted by point ID as text), so the
    numbering does not depend on the input row order.
    """
    obs_label = pos_label[position]
    unit = np.where(obs_label >= 0, obs_label, -1).astype(np.int64)
    if singleton_noise:
        # One singleton per noise *position*: co-located observations share it.
        offset = int(pos_label.max()) + 1 if len(pos_label) and pos_label.max() >= 0 else 0
        noise = obs_label < 0
        unit[noise] = offset + position[noise]
    ids = np.full(n_obs, None, dtype=object)
    valid = unit >= 0
    if not valid.any():
        return ids
    units = unit[valid]
    unique_units, first = np.unique(units, return_index=True)
    rank = np.empty(len(unique_units), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(unique_units))
    names = np.array([f"C{k + 1:06d}" for k in range(len(unique_units))], dtype=object)
    ids[valid] = names[rank[np.searchsorted(unique_units, units)]]
    return ids


def _empty_hierarchy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hierarchy_id": pd.Series(dtype="int64"),
            "parent_id": pd.Series(dtype="Int64"),
            "birth_distance": pd.Series(dtype=float),
            "size": pd.Series(dtype="int64"),
            "stability": pd.Series(dtype=float),
            "selected": pd.Series(dtype=bool),
            "cluster_id": pd.Series(dtype=object),
        }
    )


def _cluster_unit(
    points: gpd.GeoDataFrame,
    snaps: Snaps,
    graph: NetworkGraph,
    config: HDBSCANConfig,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Cluster points in canonical order."""
    n = len(points)
    out = points.copy()
    stats: dict[str, Any] = {"n_points": n}
    if n == 0:
        for column, dtype in [("cluster_id", object), ("is_noise", bool), ("membership", float), ("core_distance", float)]:
            out[column] = pd.Series(dtype=dtype)
        for column in ["snap_distance", "snapped_x", "snapped_y"]:
            out[column] = pd.Series(dtype=float)
        return out, pd.DataFrame(), _empty_hierarchy(), stats

    position, representative = distinct_positions(snaps)
    n_pos = len(representative)
    counts = np.bincount(position, minlength=n_pos)
    weights = counts if config.duplicates == "count" else np.ones(n_pos, dtype=np.int64)
    pos_snaps = snaps.subset(representative)
    distances = neighbor_graph(
        graph, pos_snaps, max_distance=config.max_distance, max_pairs=int(config.max_neighbor_pairs)
    )
    result = hdbscan(
        distances,
        weights,
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        cluster_selection_method=config.cluster_selection_method,
        cluster_selection_epsilon=config.cluster_selection_epsilon,
        max_cluster_size=config.max_cluster_size,
        allow_single_cluster=config.allow_single_cluster,
        core_distance_floor=config.core_distance_floor,
    )

    ids = _cluster_ids(n, position, result.labels, config.noise_policy == "singleton")
    obs_label = result.labels[position]
    out["cluster_id"] = pd.Series(ids, index=out.index, dtype=object)
    out["is_noise"] = obs_label < 0
    out["membership"] = result.probabilities[position]
    out["core_distance"] = result.core_distances[position]
    out["snap_distance"] = snaps.snap_distance
    out["snapped_x"] = snaps.snapped_xy[:, 0]
    out["snapped_y"] = snaps.snapped_xy[:, 1]

    # Public ID of each selected hierarchy cluster.
    k = len(result.cluster_parent)
    public_of_cluster = np.full(k, None, dtype=object)
    labelled = np.flatnonzero(obs_label >= 0)
    if len(labelled):
        labels_seen, first = np.unique(obs_label[labelled], return_index=True)
        public = dict(zip(labels_seen.tolist(), ids[labelled[first]].tolist()))
        for c in np.flatnonzero(result.cluster_selected):
            public_of_cluster[c] = public[int(result.cluster_label[c])]
    with np.errstate(divide="ignore"):
        birth_distance = np.where(result.cluster_birth_lambda > 0, 1.0 / result.cluster_birth_lambda, np.inf)
    hierarchy = pd.DataFrame(
        {
            "hierarchy_id": np.arange(k, dtype=np.int64),
            "parent_id": pd.array([None if p < 0 else int(p) for p in result.cluster_parent], dtype="Int64"),
            "birth_distance": birth_distance,
            "size": result.cluster_size.astype(np.int64),
            "stability": result.cluster_stability,
            "selected": result.cluster_selected,
            "cluster_id": pd.Series(public_of_cluster, dtype=object),
        }
    )

    # Attributes of every public cluster (selected clusters and singletons).
    members = out.loc[out["cluster_id"].notna()]
    table = pd.DataFrame({"cluster_id": sorted(members["cluster_id"].unique().tolist())})
    if len(table):
        n_points = members.groupby("cluster_id").size()
        n_positions = pd.Series(position[out["cluster_id"].notna().to_numpy()], index=members["cluster_id"].to_numpy())
        n_positions = n_positions.groupby(level=0).nunique()
        by_id = hierarchy.dropna(subset=["cluster_id"]).set_index("cluster_id")
        table["n_points"] = table["cluster_id"].map(n_points).astype("int64")
        table["n_positions"] = table["cluster_id"].map(n_positions).astype("int64")
        table["stability"] = table["cluster_id"].map(by_id["stability"]).astype(float)
        table["birth_distance"] = table["cluster_id"].map(by_id["birth_distance"]).astype(float)
        table["singleton"] = ~table["cluster_id"].isin(by_id.index)

    pos_component = graph.component[pos_snaps.u]
    obs_component = pos_component[position]
    comp_counts = np.unique(obs_component, return_counts=True)[1]
    _, comp_inverse = np.unique(pos_component, return_inverse=True)
    comp_weight = np.bincount(comp_inverse, weights=weights)
    unreachable_pos = comp_weight[comp_inverse] < config.effective_min_samples
    unreachable_obs = unreachable_pos[position]
    infinite_obs = np.isinf(result.core_distances[position])
    cluster_counts = np.bincount(obs_label[obs_label >= 0]) if (obs_label >= 0).any() else np.zeros(0, dtype=np.int64)
    stats.update(
        n_positions=int(n_pos),
        max_position_weight=int(counts.max()),
        stacked_positions=int((counts >= config.effective_min_samples).sum()),
        n_pairs=int(distances.nnz // 2),
        share_core_beyond_max_distance=float(infinite_obs.mean()),
        share_core_structurally_unreachable=float(unreachable_obs.mean()),
        share_core_truncated=float((infinite_obs & ~unreachable_obs).mean()),
        network_components_used=int(len(comp_counts)),
        share_on_largest_component=float(comp_counts.max() / n),
        snap_distance_median=float(np.median(snaps.snap_distance)),
        snap_distance_p95=float(np.quantile(snaps.snap_distance, 0.95)),
        snap_distance_max=float(snaps.snap_distance.max()),
        n_clusters=int(result.n_clusters),
        noise_share=float((obs_label < 0).mean()),
        largest_cluster_share=float(cluster_counts.max() / n) if len(cluster_counts) else 0.0,
    )
    return out, table, hierarchy, stats


def _check_snap_distance(ids, distances: np.ndarray, limit: float | None) -> None:
    if limit is None or math.isinf(limit) or len(distances) == 0:
        return
    too_far = distances > limit
    if too_far.any():
        k = int(np.flatnonzero(too_far)[0])
        raise ValueError(
            f"point {ids[k]!r} lies {distances[k]:g} units from the nearest network; max_snap_distance={limit:g} "
            f"({int(too_far.sum())} point(s) exceed it)"
        )


def _snap_points_to_network(network: gpd.GeoDataFrame, points: gpd.GeoDataFrame, vertex_digits: int | None):
    """Build the network graph and snap the points (graph is None when there are no points)."""
    if len(points) == 0:
        return None, empty_snaps()
    graph = build_network_graph(network.geometry.to_numpy(), vertex_digits=vertex_digits)
    return graph, snap_points(graph, shapely.get_coordinates(points.geometry.to_numpy()))


def _adaptive_start_distance(config: HDBSCANConfig) -> float:
    """First search radius for adaptive mode.

    Without an explicit minimum, choose a geometric ladder whose final step
    is the requested ``max_distance``. The start is also kept above a
    positive core-distance floor, because a trial config must remain valid.
    """
    if config.min_distance is not None:
        start = float(config.min_distance)
    else:
        start = float(config.max_distance) / (float(config.distance_growth) ** (int(config.distance_steps) - 1))
    if config.core_distance_floor > 0:
        start = max(start, float(config.core_distance_floor) * (1.0 + 1e-12))
    return min(start, float(config.max_distance))


def _flat_result_stable(previous, current, tolerance: float) -> bool:
    """Whether two trials have the same public flat clustering and membership.

    net-hdbscan gives cluster IDs deterministic numbering, so exact
    public IDs are meaningful across distance trials on the same observations.
    Core distances are intentionally *not* compared: increasing the search
    horizon is allowed to resolve previously infinite core distances without
    changing the selected flat clustering.
    """
    a = previous[0]
    b = current[0]
    if len(a) != len(b):
        return False
    aid = a["cluster_id"].fillna("__noise__").astype(str).to_numpy()
    bid = b["cluster_id"].fillna("__noise__").astype(str).to_numpy()
    if not np.array_equal(aid, bid):
        return False
    if not np.array_equal(a["is_noise"].to_numpy(), b["is_noise"].to_numpy()):
        return False
    return bool(
        np.allclose(
            a["membership"].to_numpy(float),
            b["membership"].to_numpy(float),
            rtol=0.0,
            atol=float(tolerance),
            equal_nan=True,
        )
    )


# Maximum number of times an automatically chosen first radius is reduced
# when even that radius exceeds max_neighbor_pairs.
_MAX_START_REDUCTIONS = 12

# Columns of the per-trial record (the grouped writer adds a "group" column).
TRACE_COLUMNS = [
    "trial",
    "radius",
    "outcome",
    "n_pairs",
    "n_clusters",
    "noise_share",
    "share_core_truncated",
    "same_as_previous",
    "seconds",
]


def _trace_row(trial: int, radius: float, outcome: str, stats: dict[str, Any] | None, same, seconds: float) -> dict[str, Any]:
    stats = stats or {}
    return {
        "trial": int(trial),
        "radius": float(radius),
        "outcome": outcome,
        "n_pairs": stats.get("n_pairs", np.nan),
        "n_clusters": stats.get("n_clusters", np.nan),
        "noise_share": stats.get("noise_share", np.nan),
        "share_core_truncated": stats.get("share_core_truncated", np.nan),
        "same_as_previous": same,
        "seconds": round(float(seconds), 3),
    }


def _cluster_unit_with_distance_search(
    points: gpd.GeoDataFrame,
    snaps: Snaps,
    graph: NetworkGraph,
    config: HDBSCANConfig,
):
    """Run one unit in fixed or adaptive distance mode.

    Returns ``(points, table, hierarchy, stats, trace)``; ``trace`` has one
    row per clustering trial (see ``TRACE_COLUMNS``).

    Adaptive mode treats ``config.max_distance`` as a hard ceiling. It grows
    a geometric search radius and stops when (1) the flat clustering is
    unchanged from the previous successful radius and (2) no more than
    ``core_truncation_tolerance`` of observations have an *otherwise
    reachable* core distance beyond the current radius. Condition (1) can
    hold at a radius where the result still changes at larger radii; the
    trace shows the history.

    The radii are ``start, start * growth, ...`` up to the ceiling, so the
    ceiling is always tried unless the search converges or reaches the pair
    budget first. If a trial exceeds ``max_neighbor_pairs``, the last
    successful result is returned and marked ``pair_cap_limited``. If even
    the first automatically chosen radius is too dense, that radius is
    reduced (at most ``_MAX_START_REDUCTIONS`` times) until one trial fits;
    a radius that already exceeded the budget is not tried again. An
    explicitly supplied ``min_distance`` is never reduced.
    """
    trace: list[dict[str, Any]] = []

    def finish(result, *, radius, status, steps, stable, pair_hit, pair_distance):
        points, table, hierarchy, stats = result
        stats.update(
            max_distance=float(radius),
            requested_max_distance=float(config.max_distance),
            distance_mode=config.distance_mode,
            distance_status=status,
            distance_steps_run=int(steps),
            distance_stable=stable,
            pair_limit_encountered=bool(pair_hit),
            pair_limit_distance=pair_distance,
            core_truncation_tolerance=float(config.core_truncation_tolerance),
        )
        return points, table, hierarchy, stats, pd.DataFrame(trace, columns=TRACE_COLUMNS)

    if config.distance_mode == "fixed" or len(points) == 0:
        began = time.perf_counter()
        result = _cluster_unit(points, snaps, graph, config)
        if len(points):
            trace.append(_trace_row(1, config.max_distance, "ok", result[3], np.nan, time.perf_counter() - began))
        return finish(
            result,
            radius=config.max_distance,
            status="fixed" if len(points) else "empty",
            steps=1 if len(points) else 0,
            stable=np.nan,
            pair_hit=False,
            pair_distance=np.nan,
        )

    ceiling = float(config.max_distance)
    growth = float(config.distance_growth)
    tolerance = float(config.core_truncation_tolerance)
    radius = _adaptive_start_distance(config)
    explicit_floor = float(config.min_distance) if config.min_distance is not None else 0.0
    absolute_floor = max(explicit_floor, float(config.core_distance_floor) * (1.0 + 1e-12))

    previous = None
    last_success = None
    last_radius = None
    last_stable = False
    successes = 0
    reductions = 0
    pair_hit = False
    pair_distance = np.nan
    smallest_failed = np.inf
    trial = 0

    while True:
        if np.isclose(radius, ceiling, rtol=1e-12, atol=0.0):
            radius = ceiling
        if last_success is not None and radius >= smallest_failed * (1.0 - 1e-9):
            # This radius (or a smaller one) already exceeded the pair budget.
            # The relative margin absorbs rounding: shrinking by ``growth`` and
            # growing again does not always give back exactly the same number.
            return finish(
                last_success, radius=last_radius, status="pair_cap_limited", steps=successes,
                stable=bool(last_stable), pair_hit=True, pair_distance=pair_distance,
            )
        trial += 1
        began = time.perf_counter()
        try:
            current = _cluster_unit(points, snaps, graph, config.replace(max_distance=radius, distance_mode="fixed"))
        except NeighborPairLimitError as exc:
            pair_hit = True
            pair_distance = float(exc.max_distance)
            smallest_failed = min(smallest_failed, radius)
            trace.append(_trace_row(trial, radius, "pair_limit", {"n_pairs": exc.count}, np.nan, time.perf_counter() - began))
            if last_success is not None:
                return finish(
                    last_success, radius=last_radius, status="pair_cap_limited", steps=successes,
                    stable=bool(last_stable), pair_hit=True, pair_distance=pair_distance,
                )
            smaller = radius / growth
            if config.min_distance is not None or smaller <= absolute_floor or reductions >= _MAX_START_REDUCTIONS:
                raise
            reductions += 1
            radius = max(smaller, absolute_floor)
            continue

        successes += 1
        stats = current[3]
        stable = previous is not None and _flat_result_stable(previous, current, config.stability_tolerance)
        trace.append(_trace_row(trial, radius, "ok", stats, bool(stable) if previous is not None else np.nan, time.perf_counter() - began))
        last_success, last_radius, last_stable = current, radius, stable

        # Every observation lies on a network component whose total weight is
        # below min_samples: no radius can produce a cluster.
        if (
            stats.get("share_core_structurally_unreachable", 0.0) >= 1.0 - 1e-15
            and stats.get("share_core_truncated", 0.0) <= 1e-15
        ):
            return finish(current, radius=radius, status="structurally_unreachable", steps=successes,
                          stable=True, pair_hit=pair_hit, pair_distance=pair_distance)

        if stable and stats.get("share_core_truncated", 1.0) <= tolerance:
            return finish(current, radius=radius, status="converged", steps=successes,
                          stable=True, pair_hit=pair_hit, pair_distance=pair_distance)

        if radius >= ceiling:
            return finish(current, radius=radius, status="ceiling_reached", steps=successes,
                          stable=bool(stable), pair_hit=pair_hit, pair_distance=pair_distance)

        previous = current
        radius = min(ceiling, radius * growth)


def _run_unit(
    points: gpd.GeoDataFrame,
    snaps: Snaps,
    graph: NetworkGraph,
    config: HDBSCANConfig,
    point_id_col: str,
) -> ClusterOutputs:
    start = time.perf_counter()
    order = canonical_order(points[point_id_col].tolist()) if len(points) else np.empty(0, dtype=np.int64)
    points = points.iloc[order].reset_index(drop=True)
    snaps = snaps.subset(order)
    _check_snap_distance(points[point_id_col].tolist(), snaps.snap_distance, config.max_snap_distance)
    points, table, hierarchy, stats, trace = _cluster_unit_with_distance_search(points, snaps, graph, config)
    stats.update(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.effective_min_samples,
        cluster_selection_method=config.cluster_selection_method,
        cluster_selection_epsilon=config.cluster_selection_epsilon,
        core_distance_floor=config.core_distance_floor,
        duplicates=config.duplicates,
        seconds=round(time.perf_counter() - start, 3),
    )
    return ClusterOutputs(points=points, hierarchy=hierarchy, summary=stats, trace=trace, clusters=_cluster_table(table))


def _analysis_metadata(crs, graph: NetworkGraph | None) -> dict[str, Any]:
    """Stable network metadata useful for reproducing a run."""
    try:
        unit = crs.axis_info[0].unit_name or None
    except (AttributeError, IndexError):
        unit = None
    return {
        "crs": str(crs),
        "distance_unit": unit,
        "graph_built": graph is not None,
        "network_vertices": None if graph is None else int(graph.n_vertices),
        "network_arcs": None if graph is None else int(graph.n_arcs),
        "network_components": None if graph is None else int(graph.n_components),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cluster_geodataframes(
    points: gpd.GeoDataFrame,
    network: gpd.GeoDataFrame,
    config: HDBSCANConfig,
    *,
    point_id_col: str = "point_id",
    vertex_digits: int | None = DEFAULT_VERTEX_DIGITS,
) -> ClusterOutputs:
    """Cluster all supplied points as one HDBSCAN* unit.

    Output points are reprojected to the network CRS and sorted by point ID
    as text. Study-area or other eligibility filtering belongs upstream.
    """
    if not isinstance(config, HDBSCANConfig):
        raise TypeError("config must be an HDBSCANConfig")
    network = prepare_network(network)
    crs = network.crs
    work = prepare_points(points, crs, point_id_col)
    graph, snaps = _snap_points_to_network(network, work, vertex_digits)
    outputs = _run_unit(work, snaps, graph, config, point_id_col)
    outputs.summary = {"group": None, "n_points_input": int(len(points)), **outputs.summary}
    outputs.analysis = _analysis_metadata(crs, graph)
    return outputs


def _summary_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in SUMMARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[SUMMARY_COLUMNS]


def _trace_frame(parts) -> pd.DataFrame:
    """Stack per-unit trial records into one table with a ``group`` column."""
    frames = []
    for group, trace in parts:
        if trace is None or len(trace) == 0:
            continue
        frame = trace.copy()
        frame.insert(0, "group", group)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["group", *TRACE_COLUMNS])
    return pd.concat(frames, ignore_index=True)[["group", *TRACE_COLUMNS]]


def _manifest(kind: str, inputs: dict[str, Any], config: HDBSCANConfig, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "net_hdbscan_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": kind,
        "inputs": inputs,
        "clustering": config.as_dict(),
        **extra,
    }


def cluster_files(
    *,
    points_path: str | Path,
    network_path: str | Path,
    output_dir: str | Path,
    config: HDBSCANConfig,
    point_id_col: str = "point_id",
    points_layer: str | None = None,
    network_layer: str | None = None,
    vertex_digits: int | None = DEFAULT_VERTEX_DIGITS,
    force: bool = False,
) -> ClusterOutputs:
    """Cluster one point file and write points, hierarchy and diagnostics.

    Files: ``clustered_points.parquet``, ``cluster_table.parquet`` (one row
    per cluster), ``cluster_hierarchy.parquet``,
    ``summary.csv``, ``distance_trace.csv`` and ``manifest.json``.
    """
    output_dir = Path(output_dir)
    targets = {
        "points": output_dir / "clustered_points.parquet",
        "clusters": output_dir / "cluster_table.parquet",
        "hierarchy": output_dir / "cluster_hierarchy.parquet",
        "summary": output_dir / "summary.csv",
        "trace": output_dir / "distance_trace.csv",
        "manifest": output_dir / "manifest.json",
    }
    _refuse_existing(targets.values(), force)
    # Every file-based run writes Parquet, so fail before input reading or
    # clustering if the optional file extra is unavailable.
    _require_pyarrow()
    outputs = cluster_geodataframes(
        read_vector(points_path, name="points", layer=points_layer),
        read_vector(network_path, name="network", layer=network_layer),
        config,
        point_id_col=point_id_col,
        vertex_digits=vertex_digits,
    )
    write_geoparquet(targets["points"], outputs.points, force=force)
    write_parquet(targets["clusters"], outputs.clusters, force=force)
    write_parquet(targets["hierarchy"], outputs.hierarchy, force=force)
    write_csv(targets["summary"], _summary_frame([outputs.summary]), force=force)
    write_csv(targets["trace"], _trace_frame([(None, outputs.trace)]), force=force)
    inputs = {
        "points": str(points_path),
        "points_layer": points_layer,
        "network": str(network_path),
        "network_layer": network_layer,
        "point_id_col": point_id_col,
    }
    write_json(
        targets["manifest"],
        _manifest(
            "single",
            inputs,
            config,
            {"vertex_digits": vertex_digits, "analysis": outputs.analysis, "summary": outputs.summary},
        ),
        force=force,
    )
    return outputs


def group_output_token(value) -> str:
    """Filesystem-safe, unambiguous token of a group value or internal key."""
    return quote(group_display(group_key(value)), safe="-_.~")


def _refuse_existing(paths, force: bool) -> None:
    if force:
        return
    for path in paths:
        if Path(path).exists():
            raise FileExistsError(f"output already exists: {path}; pass force=True or --force to replace it")


def cluster_files_by_column(
    *,
    points_path: str | Path,
    network_path: str | Path,
    output_dir: str | Path,
    group_col: str,
    config: HDBSCANConfig,
    group_params: Mapping[str, Mapping[str, Any]] | None = None,
    group_universe: list[str] | None = None,
    missing_group_policy: str = "exclude",
    point_id_col: str = "point_id",
    points_layer: str | None = None,
    network_layer: str | None = None,
    vertex_digits: int | None = DEFAULT_VERTEX_DIGITS,
    force: bool = False,
) -> pd.DataFrame:
    """Cluster each value of ``group_col`` separately and write one set of files per group.

    The network is read and built once and all retained points are snapped
    once. Each group is then clustered on its own,
    with the run-wide ``config`` changed by ``group_params[group_key]`` where
    given (see ``net_hdbscan.config.read_group_params``). Missing/blank group
    values are excluded by default; choose ``missing_group_policy='include'``
    to treat ``__null__`` and ``__blank__`` as explicit groups, or ``'error'``
    to reject them. ``group_universe`` can declare expected groups so absent
    groups still receive zero-row outputs and summary rows.

    Output layout: ``points/group_<token>.parquet``,
    ``clusters/group_<token>.parquet`` (one row per cluster),
    ``hierarchy/group_<token>.parquet``, ``summary.csv`` and
    ``manifest.json``. Returns the summary table.
    """
    if not isinstance(config, HDBSCANConfig):
        raise TypeError("config must be an HDBSCANConfig")
    # Grouped file runs always write Parquet; fail before reading potentially
    # large inputs if the optional file dependency is unavailable.
    _require_pyarrow()
    points = read_vector(points_path, name="points", layer=points_layer)
    network = read_vector(network_path, name="network", layer=network_layer)
    if group_col not in points.columns:
        raise ValueError(f"group column {group_col!r} not found")
    if group_col == point_id_col:
        raise ValueError("group column and point id column must differ")

    if missing_group_policy not in {"exclude", "include", "error"}:
        raise ValueError("missing_group_policy must be 'exclude', 'include', or 'error'")

    raw = points[group_col]
    keys = raw.map(group_key)
    missing_mask = keys.isin({NULL_GROUP_KEY, BLANK_GROUP_KEY})
    n_missing_input = int(missing_mask.sum())
    if missing_group_policy == "error" and n_missing_input:
        null_n = int((keys == NULL_GROUP_KEY).sum())
        blank_n = int((keys == BLANK_GROUP_KEY).sum())
        raise ValueError(
            f"group column {group_col!r} has {n_missing_input:,} missing/blank value(s) "
            f"({null_n:,} null, {blank_n:,} blank); choose missing_group_policy='exclude' or 'include'"
        )

    observed = keys if missing_group_policy == "include" else keys.loc[~missing_mask]
    def _group_sort_key(k):
        if k == BLANK_GROUP_KEY:
            return (0, "")
        if k == NULL_GROUP_KEY:
            return (2, "")
        return (1, str(k))

    observed_values = sorted(observed.unique().tolist(), key=_group_sort_key)
    if group_universe is not None:
        universe = [group_key(v) for v in group_universe]
        if any(v in {NULL_GROUP_KEY, BLANK_GROUP_KEY} for v in universe):
            raise ValueError("group_universe cannot contain null/blank values")
        if len(universe) != len(set(universe)):
            raise ValueError("group_universe lists a group more than once")
        values = universe + [v for v in observed_values if v not in set(universe)]
        group_universe_keys = universe
    else:
        values = observed_values

    normalized_group_params = normalize_group_params(group_params)
    if normalized_group_params:
        unknown = sorted(set(normalized_group_params) - set(values))
        if unknown:
            shown = [group_display(k) for k in unknown[:10]]
            raise ValueError(f"group parameter keys match no group in {group_col!r}: {shown}")
    configs = {key: resolve_group_config(config, normalized_group_params, key) for key in values}

    output_dir = Path(output_dir)
    planned: dict[str, dict[str, Path]] = {}
    seen: dict[str, str] = {}
    for key in values:
        token = group_output_token(key)
        name = f"group_{token}.parquet"
        # Guard Windows' case-insensitive filename semantics even when tests
        # are run on a case-sensitive platform.
        collision_key = name.casefold()
        if collision_key in seen:
            raise ValueError(
                f"group values {group_display(seen[collision_key])!r} and {group_display(key)!r} "
                f"map to colliding file names on case-insensitive filesystems ({name!r})"
            )
        seen[collision_key] = key
        planned[key] = {
            "points": output_dir / "points" / name,
            "clusters": output_dir / "clusters" / name,
            "hierarchy": output_dir / "hierarchy" / name,
        }
    run_files = [output_dir / "summary.csv", output_dir / "distance_trace.csv", output_dir / "manifest.json"]
    _refuse_existing([p for paths in planned.values() for p in paths.values()] + run_files, force)

    network = prepare_network(network)
    crs = network.crs

    # Missing/blank groups that are excluded can never contribute to an output
    # group. Remove them before reprojection and especially snapping, which can
    # be the dominant preprocessing cost on large inputs.
    if missing_group_policy == "exclude":
        eligible = ~missing_mask.to_numpy()
        points_for_run = points.loc[eligible].copy().reset_index(drop=True)
        keys_for_run = keys.to_numpy()[eligible]
    else:
        points_for_run = points
        keys_for_run = keys.to_numpy()

    work = prepare_points(points_for_run, crs, point_id_col)
    work_keys = keys_for_run
    graph, snaps_all = _snap_points_to_network(network, work, vertex_digits)
    analysis = _analysis_metadata(crs, graph)

    declared = set(group_universe_keys) if group_universe is not None else None
    outside_universe_keys = [v for v in values if declared is not None and v not in declared]
    outside_universe = [group_display(v) for v in outside_universe_keys]
    rows = []
    traces = []
    for key in values:
        rows_in = np.flatnonzero(work_keys == key)
        subset = work.iloc[rows_in].reset_index(drop=True)
        group_snaps = snaps_all.subset(rows_in)
        try:
            outputs = _run_unit(subset, group_snaps, graph, configs[key], point_id_col)
        except Exception as exc:
            n_positions = len(distinct_positions(group_snaps)[1]) if len(group_snaps) else 0
            context = (
                f"group {group_display(key)!r} failed "
                f"(n_points_input={int((keys == key).sum()):,}, "
                f"n_points={len(subset):,}, n_positions={n_positions:,})"
            )
            if hasattr(exc, "add_note"):  # Python 3.11+: keep the original error type
                exc.add_note(context)
                raise
            if isinstance(exc, ValueError):
                raise ValueError(f"{context}: {exc}") from exc
            raise RuntimeError(f"{context}: {exc}") from exc
        summary = {
            "group": group_display(key),
            "n_points_input": int((keys == key).sum()),
            **outputs.summary,
            "declared_in_universe": (key in declared) if declared is not None else None,
            "missing_group_policy": missing_group_policy,
            "n_missing_group_input": n_missing_input,
        }
        rows.append(summary)
        traces.append((group_display(key), outputs.trace))
        write_geoparquet(planned[key]["points"], outputs.points, force=force)
        write_parquet(planned[key]["clusters"], outputs.clusters, force=force)
        write_parquet(planned[key]["hierarchy"], outputs.hierarchy, force=force)

    summary = _summary_frame(rows)
    summary.attrs["missing_group_policy"] = missing_group_policy
    summary.attrs["n_missing_group_input"] = n_missing_input
    summary.attrs["groups_outside_universe"] = outside_universe
    write_csv(output_dir / "summary.csv", summary, force=force)
    write_csv(output_dir / "distance_trace.csv", _trace_frame(traces), force=force)
    inputs = {
        "points": str(points_path),
        "points_layer": points_layer,
        "network": str(network_path),
        "network_layer": network_layer,
        "group_col": group_col,
        "point_id_col": point_id_col,
    }
    extra = {
        "vertex_digits": vertex_digits,
        "analysis": analysis,
        "group_params": {group_display(k): dict(v) for k, v in normalized_group_params.items()},
        "group_universe": list(group_universe) if group_universe is not None else None,
        "missing_group_policy": missing_group_policy,
        "n_missing_group_input": n_missing_input,
        "groups_outside_universe": outside_universe,
        "groups": {group_display(key): {"files": {k: str(p.relative_to(output_dir)) for k, p in planned[key].items()}} for key in values},
    }
    write_json(output_dir / "manifest.json", _manifest("grouped", inputs, config, extra), force=force)
    return summary
