"""Correctness and structural scaling tests for EOM-to-leaf ancestry mapping."""
import numpy as np
import pytest
from net_hdbscan.comparison import _map_selected_descendants


def _reference(parent, eom, leaf):
    result = {}
    for root in np.flatnonzero(eom):
        matches = []
        for node in np.flatnonzero(leaf):
            ancestor = int(node)
            while ancestor > 0 and ancestor != root:
                ancestor = int(parent[ancestor])
            if ancestor == root:
                matches.append(int(node))
        result[int(root)] = tuple(matches)
    return result


@pytest.mark.parametrize('seed', range(20))
def test_mapping_matches_original_nested_ancestor_walk(seed):
    rng = np.random.default_rng(seed)
    n = 300
    parent = np.concatenate(([-1], np.array([rng.integers(0, i) for i in range(1, n)])))
    # EOM extraction yields an antichain: no selected node may be an
    # ancestor of another selected node.
    eom = np.zeros(n, dtype=bool)
    selected_above = np.zeros(n, dtype=bool)
    for i in range(n):
        if i:
            selected_above[i] = selected_above[parent[i]] or eom[parent[i]]
        if not selected_above[i] and rng.random() < 0.16:
            eom[i] = True
    leaf = rng.random(n) < 0.22
    assert _map_selected_descendants(parent, eom, leaf) == _reference(parent, eom, leaf)


def test_mapping_handles_root_and_no_selected_nodes():
    parent = np.array([-1, 0, 0, 1, 1, 2], dtype=np.int64)
    eom = np.array([True, False, False, False, False, False])
    leaf = np.array([True, False, True, True, True, True])
    assert _map_selected_descendants(parent, eom, leaf) == {0: (0, 2, 3, 4, 5)}
    assert _map_selected_descendants(parent, np.zeros(6, bool), leaf) == {}


def test_mapping_rejects_invalid_hierarchy():
    with pytest.raises(ValueError, match='parent must precede child'):
        _map_selected_descendants(np.array([-1, 2, 0]),
                                  np.array([False, False, False]),
                                  np.array([False, False, True]))
    with pytest.raises(ValueError, match='selection flags'):
        _map_selected_descendants(np.array([-1, 0]),
                                  np.array([True]), np.array([False, True]))


def test_mapping_large_hierarchy_without_quadratic_pairwise_scan():
    n = 100_000
    parent = np.arange(n, dtype=np.int64) - 1
    eom = np.zeros(n, bool)
    eom[::1000] = True
    leaf = np.zeros(n, bool)
    leaf[1::2] = True
    mapping = _map_selected_descendants(parent, eom, leaf)
    assert len(mapping) == 100
    assert sum(map(len, mapping.values())) == leaf.sum()
    assert mapping[0][:3] == (1, 3, 5)
