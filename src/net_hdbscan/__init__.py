"""net-hdbscan: symmetric sparse-distance HDBSCAN* with optional geospatial network workflows."""

from ._version import __version__
from .config import GROUP_OVERRIDE_FIELDS, HDBSCANConfig, read_group_params, read_group_universe
from .sparse_hdbscan import HDBSCANResult, FittedHierarchy, fit_hierarchy, extract_clusters, hdbscan
from .network import (
    NeighborPairLimitError,
    NetworkGraph,
    Snaps,
    build_network_graph,
    distinct_positions,
    neighbor_graph,
    snap_points,
)
from .comparison import ExtractionComparison, compare_extractions
from .osm_source import fetch_osm_network, parse_bbox, _validate_selector
from .pipeline import ClusterOutputs, cluster_files, cluster_files_by_column, cluster_geodataframes

__all__ = [
    "GROUP_OVERRIDE_FIELDS",
    "ClusterOutputs",
    "HDBSCANConfig",
    "HDBSCANResult",
    "FittedHierarchy",
    "fit_hierarchy",
    "extract_clusters",
    "ExtractionComparison",
    "compare_extractions",
    "NeighborPairLimitError",
    "NetworkGraph",
    "Snaps",
    "__version__",
    "build_network_graph",
    "cluster_files",
    "cluster_files_by_column",
    "cluster_geodataframes",
    "distinct_positions",
    "hdbscan",
    "neighbor_graph",
    "read_group_params",
    "read_group_universe",
    "snap_points",
    "fetch_osm_network",
    "parse_bbox",
]
