"""Unreleased v0.3.1-follow-up regression tests. No network retrieval."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from net_hdbscan import hdbscan
from net_hdbscan.sparse_hdbscan import _validate_weights


def test_unconnected_unweighted_positions_are_not_one_cluster():
    result = hdbscan(
        csr_matrix((6, 6)), min_cluster_size=5, min_samples=1,
        allow_single_cluster=True,
    )
    assert result.n_clusters == 0
    assert (result.labels == -1).all()


def test_small_disconnected_pairs_do_not_become_virtual_root_cluster():
    values = np.zeros((6, 6), dtype=float)
    for i in (0, 2, 4):
        values[i, i + 1] = values[i + 1, i] = 1.0
    result = hdbscan(csr_matrix(values), min_cluster_size=5,
                     min_samples=1, allow_single_cluster=True)
    assert result.n_clusters == 0
    assert (result.labels == -1).all()


def test_connected_component_can_be_a_single_cluster():
    values = np.ones((6, 6)) - np.eye(6)
    result = hdbscan(csr_matrix(values), min_cluster_size=5,
                     min_samples=1, allow_single_cluster=True)
    assert result.n_clusters == 1
    assert (result.labels == 0).all()


def test_supported_duplicate_stack_remains_valid_but_unresolved_stack_is_noise():
    graph = csr_matrix((1, 1))
    valid = hdbscan(graph, np.array([3]), min_cluster_size=2,
                    min_samples=2, allow_single_cluster=True)
    invalid = hdbscan(graph, np.array([3]), min_cluster_size=2,
                      min_samples=4, allow_single_cluster=True)
    assert valid.n_clusters == 1 and valid.labels.tolist() == [0]
    assert invalid.n_clusters == 0 and invalid.labels.tolist() == [-1]


def test_max_cluster_size_includes_all_root_members():
    values = np.ones((6, 6)) - np.eye(6)
    result = hdbscan(csr_matrix(values), min_cluster_size=5, min_samples=1,
                     max_cluster_size=5, allow_single_cluster=True)
    assert result.n_clusters == 0
    assert (result.labels == -1).all()


@pytest.mark.parametrize('bad', [
    np.array([float(2**63)]),
    np.array([float(2**53)]),
    np.array([float(2**24)], dtype=np.float32),
    np.array([1 + 2j]),
    np.array(['9223372036854775808']),
    np.array(['1.25']),
    np.array(['nan']),
    np.array([True]),
])
def test_invalid_or_imprecise_weights_rejected(bad):
    with pytest.raises(ValueError):
        _validate_weights(bad, 1)


def test_large_integer_and_decimal_string_weights_remain_exact():
    high = 9007199254740993
    assert _validate_weights(np.array([high], dtype=np.int64), 1).tolist() == [high]
    assert _validate_weights([str(high) + '.0'], 1).tolist() == [high]
    assert _validate_weights(['9.007199254740993e15'], 1).tolist() == [high]
    assert _validate_weights([str(2**63 - 1)], 1).tolist() == [2**63 - 1]
    assert _validate_weights(np.array([1.0, 2.0]), 2).tolist() == [1, 2]


def test_sum_of_individually_valid_weights_cannot_overflow_int64():
    values = np.array([2**62, 2**62], dtype=np.int64)
    with pytest.raises(ValueError, match='sum of weights'):
        _validate_weights(values, 2)
    with pytest.raises(ValueError, match='sum of weights'):
        hdbscan(csr_matrix((2, 2)), weights=values, min_cluster_size=2)

# Regression discovered after v3: disconnected noise must not count against
# the EOM cap of a single connected cluster selected at the synthetic root.
def test_root_cap_excludes_unrelated_disconnected_noise():
    distances = np.zeros((9, 9), dtype=float)
    distances[:6, :6] = np.ones((6, 6)) - np.eye(6)
    graph = csr_matrix(distances)
    options = dict(min_cluster_size=5, min_samples=1, allow_single_cluster=True)
    uncapped = hdbscan(graph, **options)
    at_cap = hdbscan(graph, max_cluster_size=6, **options)
    below_cap = hdbscan(graph, max_cluster_size=5, **options)

    assert uncapped.n_clusters == at_cap.n_clusters == 1
    assert np.array_equal(uncapped.labels, at_cap.labels)
    assert (at_cap.labels[:6] >= 0).all()
    assert (at_cap.labels[6:] == -1).all()
    assert below_cap.n_clusters == 0
    assert (below_cap.labels == -1).all()
    # Root hierarchy population remains the whole input, not labelled size.
    assert at_cap.cluster_size[0] == 9


def test_root_cap_counts_duplicate_stack_but_not_disconnected_noise():
    graph = csr_matrix((2, 2))
    options = dict(weights=np.array([3, 1], dtype=np.int64),
                   min_cluster_size=2, min_samples=2, allow_single_cluster=True)
    at_cap = hdbscan(graph, max_cluster_size=3, **options)
    below_cap = hdbscan(graph, max_cluster_size=2, **options)
    assert at_cap.n_clusters == 1
    assert at_cap.labels.tolist() == [0, -1]
    assert below_cap.n_clusters == 0
    assert (below_cap.labels == -1).all()
    assert at_cap.cluster_size[0] == 4
