"""Pure, inexpensive adequacy diagnostics for bounded network HDBSCAN*."""
from __future__ import annotations
import numpy as np


def group_diagnostics(*, n_observations, position, weights, position_components,
                      labels, core_distances, cluster_stability, cluster_selected,
                      effective_min_samples, min_cluster_size, allow_single_cluster):
    n = int(n_observations)
    weights = np.asarray(weights, dtype=np.int64)
    position = np.asarray(position, dtype=np.int64)
    components = np.asarray(position_components)
    core_distances = np.asarray(core_distances)
    labels = np.asarray(labels)
    if len(position) != n or len(labels) != n or len(weights) != len(components):
        raise ValueError("diagnostics: inconsistent input lengths")
    effective_weight = int(weights.sum())
    if len(weights):
        _, inverse = np.unique(components, return_inverse=True)
        comp_weight = np.bincount(inverse, weights=weights)
        feasible_components = int((comp_weight >= effective_min_samples).sum())
        structural_pos = comp_weight[inverse] < effective_min_samples
    else:
        feasible_components = 0
        structural_pos = np.zeros(0, dtype=bool)
    unresolved_pos = np.isinf(core_distances)
    structural_obs = structural_pos[position]
    unresolved_obs = unresolved_pos[position]
    truncated_obs = unresolved_obs & ~structural_obs
    valid = labels >= 0
    retained = int(valid.sum())
    counts = np.unique(labels[valid], return_counts=True)[1] if retained else np.empty(0,dtype=int)
    largest = int(counts.max()) if len(counts) else 0
    selected = np.asarray(cluster_stability)[np.asarray(cluster_selected, dtype=bool)]
    selected_sum = float(np.sum(selected)) if len(selected) else 0.0
    feasible_two = effective_weight >= 2 * min_cluster_size
    reasons = []
    if n == 0 or effective_weight < min_cluster_size:
        reasons.append("insufficient_effective_weight")
    if not allow_single_cluster and not feasible_two:
        reasons.append("insufficient_weight_for_two_branches")
    if structural_obs.any():
        reasons.append("structurally_unreachable_observations")
    if truncated_obs.any():
        reasons.append("truncated_core_distances")
    if not valid.any():
        reasons.append("no_clusters_selected_by_extraction")
    return {
        "n_components_with_points": int(len(np.unique(components))),
        "n_components_core_feasible": feasible_components,
        "n_effective_weight": effective_weight,
        "n_positions_core_unresolved": int(unresolved_pos.sum()),
        "n_observations_core_unresolved": int(unresolved_obs.sum()),
        "n_observations_structurally_unreachable": int(structural_obs.sum()),
        "n_observations_core_truncated": int(truncated_obs.sum()),
        "can_form_two_min_size_groups": bool(feasible_two),
        "selected_cluster_count": int(len(counts)),
        "retained_observation_share": retained / n if n else 0.0,
        "largest_cluster_share_total": largest / n if n else 0.0,
        "largest_cluster_share_retained": largest / retained if retained else None,
        "selected_stability_sum": selected_sum,
        "zero_cluster_explanation": ";".join(reasons) if not retained else "",
    }
