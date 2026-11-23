"""Command line interface for ``net-hdbscan cluster ...``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from ._version import __version__
from .config import HDBSCANConfig, read_group_params, read_group_universe
from .network import DEFAULT_MAX_NEIGHBOR_PAIRS, DEFAULT_VERTEX_DIGITS
from .pipeline import _summary_frame, cluster_files, cluster_files_by_column
from .osm_source import fetch_osm_network, parse_bbox, _validate_selector
from .region import parse_circle

app = typer.Typer(add_completion=False, no_args_is_help=True, rich_markup_mode=None)

_SHOWN = [
    ("group", "group"),
    ("n_points", "points"),
    ("n_clusters", "clusters"),
    ("noise_share", "noise"),
    ("largest_cluster_share", "largest"),
    ("share_core_truncated", "core trunc"),
    ("max_distance", "distance"),
    ("distance_status", "status"),
    ("max_position_weight", "max stack"),
    ("seconds", "seconds"),
]


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show the version and exit."),
) -> None:
    """Cluster points with HDBSCAN* using shortest-path distance on a spatial network."""


def _print_summary(summary: pd.DataFrame) -> None:
    table = summary[[c for c, _ in _SHOWN]].copy()
    table.columns = [name for _, name in _SHOWN]
    for column in ["noise", "largest", "core trunc"]:
        table[column] = table[column].map(lambda v: "" if pd.isna(v) else f"{100 * v:.1f}%")
    table["group"] = table["group"].map(lambda v: "(all)" if v is None or pd.isna(v) else str(v))
    typer.echo(table.to_string(index=False))


@app.command('fetch-network')
def fetch_network_command(
    source: str = typer.Option(..., '--source', help="Explicit source; currently only 'osm'."),
    bbox: str | None = typer.Option(None, '--bbox', help='EPSG:4326 west,south,east,north; alternative to --center.'),
    center: str | None = typer.Option(None, '--center', help='EPSG:4326 longitude,latitude; requires --radius-km.'),
    radius_km: float | None = typer.Option(None, '--radius-km', help='Geodesic circle radius in kilometers; requires --center.'),
    osm_profile: str | None = typer.Option(None, '--osm-profile', help='OSM way-access profile (not a travel-time model).'),
    osm_filter: str | None = typer.Option(None, '--osm-filter', help='Alternative bracketed OSM way-tag filter.'),
    osm_cache_dir: Path | None = typer.Option(None, '--osm-cache-dir', help='Immutable cached network root.'),
) -> None:
    """Explicitly fetch and cache network line geometry without clustering."""
    if source != 'osm':
        raise typer.BadParameter("only --source osm is currently implemented")
    try:
        if bbox is not None:
            if center is not None or radius_km is not None:
                raise ValueError('choose --bbox or --center with --radius-km, not both')
            area = parse_bbox(bbox)
        else:
            if center is None or radius_km is None:
                raise ValueError('provide --bbox or both --center and --radius-km')
            parse_circle(center, radius_km)
            area = None
        profile, custom_filter = _validate_selector(osm_profile, osm_filter)
        path = fetch_osm_network(bbox=area, center=center, radius_km=radius_km,
                                 profile=profile, custom_filter=custom_filter,
                                 cache_dir=osm_cache_dir)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f'Network: {path}')
    typer.echo(f'Provenance: {path.with_name("source.json")}')


def _validated_config(**kwargs) -> HDBSCANConfig:
    try:
        return HDBSCANConfig(**kwargs)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def cluster(
    points: Path = typer.Option(..., "--points", help="Point layer (GeoParquet, GeoPackage, ...)."),
    network: Path | None = typer.Option(None, "--network", help="Explicit projected line network; alternative to --network-source."),
    network_source: str | None = typer.Option(None, '--network-source', help="Explicit network provider: 'osm' only."),
    bbox: str | None = typer.Option(None, '--bbox', help='For OSM: EPSG:4326 west,south,east,north.'),
    center: str | None = typer.Option(None, '--center', help='For OSM: EPSG:4326 longitude,latitude.'),
    radius_km: float | None = typer.Option(None, '--radius-km', help='For OSM: geodesic radius in kilometers.'),
    osm_profile: str | None = typer.Option(None, '--osm-profile', help='OSM way-access profile, not directional routing.'),
    osm_filter: str | None = typer.Option(None, '--osm-filter', help='Alternative OSM way-tag filter.'),
    osm_cache_dir: Path | None = typer.Option(None, '--osm-cache-dir', help='Network cache location.'),
    output_dir: Path = typer.Option(..., "--output-dir", help="Folder for clustered points, hierarchy and diagnostics."),
    max_distance: float = typer.Option(..., "--max-distance", help="Fixed search radius, or hard ceiling in adaptive mode (CRS units)."),
    min_cluster_size: int = typer.Option(5, "--min-cluster-size", help="Smallest number of observations in a cluster."),
    min_samples: int | None = typer.Option(None, "--min-samples", help="Density smoothing; default = min cluster size."),
    cluster_selection_method: str = typer.Option("eom", "--cluster-selection-method", help="'eom' or 'leaf'."),
    cluster_selection_epsilon: float = typer.Option(0.0, "--cluster-selection-epsilon", help="Do not split clusters below this distance."),
    max_cluster_size: int | None = typer.Option(None, "--max-cluster-size", help="With 'eom', do not select larger clusters."),
    allow_single_cluster: bool = typer.Option(False, "--allow-single-cluster", help="Allow one all-including cluster."),
    core_distance_floor: float = typer.Option(0.0, "--core-distance-floor", help="Raise smaller core distances to this value."),
    duplicates: str = typer.Option("count", "--duplicates", help="'count' each observation or count each position 'once'."),
    noise_policy: str = typer.Option("exclude", "--noise-policy", help="'exclude' or 'singleton' (one ID per noise position)."),
    max_snap_distance: float | None = typer.Option(None, "--max-snap-distance", help="Fail if a point is farther from the network."),
    max_neighbor_pairs: int = typer.Option(DEFAULT_MAX_NEIGHBOR_PAIRS, "--max-neighbor-pairs", help="Safety limit on stored position pairs."),
    distance_mode: str = typer.Option("fixed", "--distance-mode", help="'fixed' or 'adaptive'. Adaptive searches up to --max-distance."),
    min_distance: float | None = typer.Option(None, "--min-distance", help="Optional first radius in adaptive mode."),
    distance_growth: float = typer.Option(1.5, "--distance-growth", help="Geometric radius growth in adaptive mode (>1)."),
    distance_steps: int = typer.Option(4, "--distance-steps", help="Nominal adaptive radius steps (>=2)."),
    core_truncation_tolerance: float = typer.Option(0.01, "--core-truncation-tolerance", help="Max resolvable core-distance share beyond the adaptive radius."),
    stability_tolerance: float = typer.Option(1e-12, "--stability-tolerance", help="Membership tolerance in the adaptive stability check."),
    point_id_col: str = typer.Option("point_id", "--point-id-col"),
    group_col: str | None = typer.Option(None, "--group-col", help="Cluster each value of this column independently."),
    group_params: Path | None = typer.Option(None, "--group-params", help="CSV of per-group parameter overrides."),
    group_universe: Path | None = typer.Option(None, "--group-universe", help="One-column CSV declaring expected group values, including absent groups."),
    missing_group_policy: str = typer.Option("exclude", "--missing-group-policy", help="'exclude', 'include', or 'error' for null/blank group values."),
    vertex_digits: int = typer.Option(DEFAULT_VERTEX_DIGITS, "--vertex-digits", help="Significant digits for joining network vertices."),
    points_layer: str | None = typer.Option(None, "--points-layer"),
    network_layer: str | None = typer.Option(None, "--network-layer"),
    force: bool = typer.Option(False, "--force", help="Replace existing outputs."),
) -> None:
    """Cluster points and write point labels, hierarchy and diagnostics."""
    config = _validated_config(
        max_distance=max_distance,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        max_cluster_size=max_cluster_size,
        allow_single_cluster=allow_single_cluster,
        core_distance_floor=core_distance_floor,
        duplicates=duplicates,
        noise_policy=noise_policy,
        max_snap_distance=max_snap_distance,
        max_neighbor_pairs=max_neighbor_pairs,
        distance_mode=distance_mode,
        min_distance=min_distance,
        distance_growth=distance_growth,
        distance_steps=distance_steps,
        core_truncation_tolerance=core_truncation_tolerance,
        stability_tolerance=stability_tolerance,
    )
    # Source selection is an explicit data-model choice, not inferred from
    # missing network input, observational categories or geographic location.
    if network is not None:
        if any(value is not None for value in (network_source, bbox, center, radius_km, osm_profile, osm_filter, osm_cache_dir)):
            raise typer.BadParameter('--network cannot be combined with OSM source options')
    else:
        if network_source != 'osm':
            raise typer.BadParameter('supply --network, or explicitly select --network-source osm')
        if bbox is not None:
            if center is not None or radius_km is not None:
                raise typer.BadParameter('choose --bbox or --center with --radius-km, not both')
        elif center is None or radius_km is None:
            raise typer.BadParameter('--network-source osm requires --bbox or both --center and --radius-km')
        if network_layer is not None:
            raise typer.BadParameter('--network-layer is only valid with a user-supplied --network')
        try:
            area = parse_bbox(bbox) if bbox is not None else None
            if bbox is None:
                parse_circle(center, radius_km)
            profile, custom_filter = _validate_selector(osm_profile, osm_filter)
            network = fetch_osm_network(bbox=area, center=center, radius_km=radius_km,
                                        profile=profile, custom_filter=custom_filter,
                                        cache_dir=osm_cache_dir)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(f'Using cached OSM network: {network}')
    common = dict(
        points_path=points,
        network_path=network,
        output_dir=output_dir,
        config=config,
        point_id_col=point_id_col,
        points_layer=points_layer,
        network_layer=network_layer,
        vertex_digits=vertex_digits,
        force=force,
    )
    if group_col is None:
        if group_params is not None:
            raise typer.BadParameter("--group-params requires --group-col")
        if group_universe is not None:
            raise typer.BadParameter("--group-universe requires --group-col")
        outputs = cluster_files(**common)
        summary = _summary_frame([outputs.summary])
    else:
        overrides = read_group_params(group_params, group_col) if group_params is not None else None
        universe = read_group_universe(group_universe, group_col) if group_universe is not None else None
        summary = cluster_files_by_column(
            group_col=group_col,
            group_params=overrides,
            group_universe=universe,
            missing_group_policy=missing_group_policy,
            **common,
        )
    _print_summary(summary)
    excluded = int(summary.attrs.get("n_missing_group_input", 0)) if hasattr(summary, "attrs") else 0
    if excluded and summary.attrs.get("missing_group_policy") == "exclude":
        typer.echo(f"Excluded {excluded:,} input row(s) with null/blank group values.")
    outside = list(summary.attrs.get("groups_outside_universe") or [])
    if outside:
        shown = ", ".join(outside[:10]) + (" ..." if len(outside) > 10 else "")
        typer.echo(
            f"WARNING: {len(outside)} observed group(s) are not in --group-universe and were processed anyway "
            f"(see declared_in_universe in summary.csv): {shown}"
        )
    if "distance_status" in summary.columns:
        group_tolerance = summary["core_truncation_tolerance"].fillna(core_truncation_tolerance)
        unresolved = (
            summary["distance_status"].eq("ceiling_reached")
            & summary["share_core_truncated"].fillna(0).gt(group_tolerance)
        ).sum()
        if unresolved:
            typer.echo(
                f"WARNING: {int(unresolved)} group(s) reached the adaptive distance ceiling with "
                "resolvable core truncation above tolerance; inspect summary.csv."
            )
    typer.echo(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    app()
