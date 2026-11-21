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


def _map_selected_descendants(parent: np.ndarray, eom_selected: np.ndarray,
                              leaf_selected: np.ndarray) -> dict[int, tuple[int, ...]]:
    """Map selected leaf nodes to their nearest selected EOM ancestor in O(K + L).

    Hierarchy node IDs are topological: a non-root parent's ID precedes its
    children's IDs, as produced by the condensed-hierarchy builder.
    """
    parent = np.asarray(parent, dtype=np.int64)
    eom_selected = np.asarray(eom_selected, dtype=bool)
    leaf_selected = np.asarray(leaf_selected, dtype=bool)
    k = len(parent)
    if eom_selected.shape != (k,) or leaf_selected.shape != (k,):
        raise ValueError('selection flags must match hierarchy length')
    owner = np.full(k, -1, dtype=np.int64)
    mapping: dict[int, list[int]] = {int(c): [] for c in np.flatnonzero(eom_selected)}
    for c in range(k):
        if eom_selected[c]:
            owner[c] = c
        elif c:
            p = int(parent[c])
            if p < 0 or p >= c:
                raise ValueError('hierarchy parent must precede child')
            owner[c] = owner[p]
        if leaf_selected[c] and owner[c] >= 0:
            mapping[int(owner[c])].append(c)
    return {root: tuple(leaves) for root, leaves in mapping.items()}


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
        descendant_map = _map_selected_descendants(
            fitted.parent, results['eom'].cluster_selected,
            results['leaf'].cluster_selected,
        )
    return ExtractionComparison(results, tuple(summary), descendant_map)
