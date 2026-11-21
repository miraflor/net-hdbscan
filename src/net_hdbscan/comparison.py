"""Opt-in comparison of EOM/leaf extractions from one frozen hierarchy.

Shares use effective position weights (not observation counts with duplicates='once').
No selection is changed automatically.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
from time import perf_counter
import numpy as np
from .sparse_hdbscan import FittedHierarchy, HDBSCANResult, extract_clusters


@dataclass(frozen=True)
class ExtractionComparison:
    results: Mapping[str, HDBSCANResult]
    summary: tuple[dict, ...]
    eom_parent_to_leaf_descendants: Mapping[int, tuple[int, ...]]


def compare_extractions(
    fitted: FittedHierarchy, *, methods: tuple[str, ...] = ('eom', 'leaf'),
    cluster_selection_epsilon: float = 0.0,
    max_cluster_size: int | None = None,
    allow_single_cluster: bool = False,
) -> ExtractionComparison:
    if not methods or len(set(methods)) != len(methods):
        raise ValueError('methods must be a nonempty sequence without duplicates')
    if not set(methods) <= {'eom', 'leaf'}:
        raise ValueError('methods must contain only eom and/or leaf')
    results = {}
    summary = []
    for method in methods:
        begin = perf_counter()
        result = extract_clusters(
            fitted, method=method,
            cluster_selection_epsilon=cluster_selection_epsilon,
            max_cluster_size=max_cluster_size,
            allow_single_cluster=allow_single_cluster,
        )
        seconds = perf_counter() - begin
        clustered = result.labels >= 0
        retained = int(fitted.weights[clustered].sum())
        total = int(fitted.weights.sum())
        selected_size = {}
        for label in np.unique(result.labels[clustered]):
            selected_size[int(label)] = int(fitted.weights[result.labels == label].sum())
        summary.append({
            'method': method,
            'n_clusters': result.n_clusters,
            'retained_effective_weight': retained,
            'noise_effective_weight': total - retained,
            'retained_share': retained / total if total else 0.0,
            'largest_cluster_share': max(selected_size.values(), default=0) / total if total else 0.0,
            'selected_stability_sum': float(result.cluster_stability[result.cluster_selected].sum()),
            'extraction_seconds': seconds,
        })
        results[method] = result
    descendant_map = {}
    if 'eom' in results and 'leaf' in results:
        eom = results['eom']
        leaf = results['leaf']
        for root in np.flatnonzero(eom.cluster_selected):
            descendants = []
            for node in np.flatnonzero(leaf.cluster_selected):
                ancestor = int(node)
                while ancestor > 0 and ancestor != root:
                    ancestor = int(fitted.parent[ancestor])
                if ancestor == root:
                    descendants.append(int(node))
            descendant_map[int(root)] = tuple(descendants)
    return ExtractionComparison(results, tuple(summary), descendant_map)
