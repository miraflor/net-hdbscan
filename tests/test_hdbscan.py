import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

from net_hdbscan.sparse_hdbscan import (
    _from_linkage,
    _from_simultaneous_hierarchy,
    _SimultaneousHierarchy,
    core_distances,
    hdbscan,
)


def blobs(seed=0, n=240, noise=24):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 100, size=(4, 2))
    spread = rng.uniform(1.5, 6, 4)
    label = rng.integers(0, 4, n)
    xy = centers[label] + rng.normal(size=(n, 2)) * spread[label, None]
    return np.vstack([xy, rng.uniform(0, 100, size=(noise, 2))])


def same_partition(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not np.array_equal(a < 0, b < 0):
        return False
    m = a >= 0
    return len(set(zip(a[m], b[m]))) == len(set(a[m])) == len(set(b[m]))


def truncated(xy, radius):
    d = cdist(xy, xy)
    d[d > radius] = 0
    return csr_matrix(d)


def test_weighted_core_distance_equals_expanded():
    rng = np.random.default_rng(4)
    xy = rng.uniform(0, 50, size=(60, 2))
    w = rng.integers(1, 4, len(xy))
    expanded = np.repeat(xy, w, axis=0)
    owner = np.repeat(np.arange(len(xy)), w)
    for k in [1, 2, 5, 9]:
        weighted = core_distances(csr_matrix(cdist(xy, xy)), w, k)
        d = cdist(expanded, expanded)
        reference = np.sort(d, axis=1)[:, k - 1]  # the observation itself is at distance 0
        assert np.allclose(weighted[owner], reference)


def test_core_distance_is_infinite_without_enough_neighbours():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [50.0, 0.0]])
    core = core_distances(truncated(xy, 5.0), np.ones(3), 2)
    assert np.allclose(core[:2], 1.0) and np.isinf(core[2])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("method", ["eom", "leaf"])
@pytest.mark.parametrize("mcs", [5, 20])
def test_same_as_scikit_learn_without_ties(seed, method, mcs):
    """With min_samples=1 no distances tie, so results must equal scikit-learn's exactly."""
    sklearn = pytest.importorskip("sklearn.cluster")
    xy = blobs(seed)
    d = cdist(xy, xy)
    ref = sklearn.HDBSCAN(min_cluster_size=mcs, min_samples=1, metric="precomputed", cluster_selection_method=method, copy=True).fit(d)
    mine = hdbscan(csr_matrix(d), min_cluster_size=mcs, min_samples=1, cluster_selection_method=method)
    assert same_partition(ref.labels_, mine.labels)
    assert np.allclose(ref.probabilities_, mine.probabilities)


def test_same_as_scikit_learn_given_the_same_tree():
    """Condensing, selection, labels and membership match scikit-learn for its own single-linkage tree."""
    pytest.importorskip("sklearn")
    try:
        from sklearn.cluster._hdbscan._tree import tree_to_labels
        from sklearn.cluster._hdbscan.hdbscan import _hdbscan_brute
    except ImportError:
        pytest.skip("scikit-learn internals changed")
    xy = blobs(5)
    d = cdist(xy, xy)
    n = len(xy)
    checked = 0
    for ms in [1, 4, 10]:
        slt = _hdbscan_brute(d.copy(), min_samples=ms, alpha=1.0, metric="precomputed")
        size = np.concatenate([np.ones(n, np.int64), slt["cluster_size"].astype(np.int64)])
        linkage = (slt["left_node"].astype(np.int64), slt["right_node"].astype(np.int64), slt["value"].astype(float), size, np.array([2 * n - 2]))
        core = core_distances(csr_matrix(d), np.ones(n, np.int64), ms)
        for mcs, method, eps, single in [(5, "eom", 0.0, False), (15, "leaf", 0.0, False), (10, "eom", 2.0, False), (40, "eom", 0.0, True)]:
            try:
                ref_labels, ref_prob = tree_to_labels(slt, mcs, method, single, eps, None)
            except (TypeError, KeyError):  # known scikit-learn failures in some settings
                continue
            mine = _from_linkage(
                n, linkage, np.ones(n, np.int64), core, min_cluster_size=mcs, cluster_selection_method=method,
                cluster_selection_epsilon=eps, max_cluster_size=None, allow_single_cluster=single,
            )
            assert np.array_equal(ref_labels, mine.labels)
            assert np.allclose(ref_prob, mine.probabilities)
            checked += 1
    assert checked >= 6


def test_multifurcation_code_reproduces_scikit_learn_on_its_own_tree():
    """The condensing code the package uses gives scikit-learn's labels when given scikit-learn's binary tree."""
    pytest.importorskip("sklearn")
    try:
        from sklearn.cluster._hdbscan._tree import tree_to_labels
        from sklearn.cluster._hdbscan.hdbscan import _hdbscan_brute
    except ImportError:
        pytest.skip("scikit-learn internals changed")
    xy = blobs(5)
    d = cdist(xy, xy)
    n = len(xy)
    checked = 0
    for ms in [1, 4, 10]:
        slt = _hdbscan_brute(d.copy(), min_samples=ms, alpha=1.0, metric="precomputed")
        hierarchy = _SimultaneousHierarchy(
            children=tuple(np.array([l, r], dtype=np.int64) for l, r in zip(slt["left_node"], slt["right_node"])),
            distance=slt["value"].astype(float),
            size=np.concatenate([np.ones(n, np.int64), slt["cluster_size"].astype(np.int64)]),
            tops=np.array([2 * n - 2]),
        )
        core = core_distances(csr_matrix(d), np.ones(n, np.int64), ms)
        for mcs, method, eps, single in [(5, "eom", 0.0, False), (15, "leaf", 0.0, False), (10, "eom", 2.0, False), (40, "eom", 0.0, True)]:
            try:
                ref_labels, ref_prob = tree_to_labels(slt, mcs, method, single, eps, None)
            except (TypeError, KeyError):  # known scikit-learn failures in some settings
                continue
            mine = _from_simultaneous_hierarchy(
                n, hierarchy, np.ones(n, np.int64), core, min_cluster_size=mcs, cluster_selection_method=method,
                cluster_selection_epsilon=eps, max_cluster_size=None, allow_single_cluster=single,
            )
            assert np.array_equal(ref_labels, mine.labels)
            assert np.allclose(ref_prob, mine.probabilities)
            checked += 1
    assert checked >= 6


def test_weighted_positions_equal_expanded_observations():
    sklearn = pytest.importorskip("sklearn.cluster")
    rng = np.random.default_rng(8)
    xy = blobs(3, n=150, noise=10)
    w = np.ones(len(xy), np.int64)
    w[rng.choice(len(xy), 12, replace=False)] = rng.integers(2, 7, 12)
    expanded = np.repeat(xy, w, axis=0)
    owner = np.repeat(np.arange(len(xy)), w)
    ref = sklearn.HDBSCAN(min_cluster_size=10, min_samples=1, metric="precomputed", copy=True).fit(cdist(expanded, expanded))
    mine = hdbscan(csr_matrix(cdist(xy, xy)), w, min_cluster_size=10, min_samples=1)
    assert same_partition(ref.labels_, mine.labels[owner])


def test_truncated_graph_gives_a_forest_of_clusters():
    rng = np.random.default_rng(2)
    a = rng.normal([0, 0], 1, size=(40, 2))
    b = rng.normal([100, 0], 1, size=(40, 2))
    lonely = np.array([[50.0, 50.0], [-60.0, 30.0]])
    xy = np.vstack([a, b, lonely])
    result = hdbscan(truncated(xy, 10.0), min_cluster_size=10, min_samples=5)
    assert result.n_clusters == 2
    assert len(set(result.labels[:40])) == 1 and len(set(result.labels[40:80])) == 1
    assert result.labels[0] != result.labels[40]
    assert (result.labels[80:] == -1).all()
    assert np.isinf(result.core_distances[80:]).all()
    # Clusters that are separate pieces at the search radius are born at lambda 0.
    assert np.all(result.cluster_birth_lambda[result.cluster_selected] == 0)


def test_empty_and_tiny_inputs():
    assert hdbscan(csr_matrix((0, 0))).labels.shape == (0,)
    one = hdbscan(csr_matrix((1, 1)), np.array([3]), min_cluster_size=2)
    assert list(one.labels) == [-1]
    single = hdbscan(csr_matrix((1, 1)), np.array([3]), min_cluster_size=2, allow_single_cluster=True)
    assert list(single.labels) == [0]


def stacked_example():
    rng = np.random.default_rng(5)
    xy = np.vstack([rng.normal(size=(300, 2)) * 5, rng.normal(size=(300, 2)) * 5 + [60, 0], [[2.0, 1.0]]])
    w = np.ones(len(xy), np.int64)
    w[-1] = 40  # forty observations at one position inside the first group
    return csr_matrix(cdist(xy, xy)), w


def test_a_stack_splits_its_cluster_unless_limited():
    graph, w = stacked_example()
    counted = hdbscan(graph, w, min_cluster_size=20, min_samples=10)
    assert counted.n_clusters == 3  # same result as scikit-learn on the expanded observations
    once = hdbscan(graph, None, min_cluster_size=20, min_samples=10)
    floored = hdbscan(graph, w, min_cluster_size=20, min_samples=10, core_distance_floor=0.5)
    for result in (once, floored):
        assert result.n_clusters == 2
        assert (result.labels >= 0).all()


def test_epsilon_and_max_size_and_single_cluster():
    xy = blobs(1)
    graph = csr_matrix(cdist(xy, xy))
    base = hdbscan(graph, min_cluster_size=5, cluster_selection_method="leaf")
    merged = hdbscan(graph, min_cluster_size=5, cluster_selection_method="leaf", cluster_selection_epsilon=15.0)
    assert merged.n_clusters < base.n_clusters
    big = hdbscan(graph, min_cluster_size=5)
    limit = int(np.bincount(big.labels[big.labels >= 0]).max()) - 1
    capped = hdbscan(graph, min_cluster_size=5, max_cluster_size=limit)
    sizes = np.bincount(capped.labels[capped.labels >= 0])
    assert sizes.max() <= limit
    one_blob = np.random.default_rng(0).normal(size=(200, 2))
    single = hdbscan(csr_matrix(cdist(one_blob, one_blob)), min_cluster_size=150, allow_single_cluster=True)
    assert single.n_clusters == 1


def test_results_do_not_depend_on_position_order():
    xy = blobs(7)
    order = np.random.default_rng(1).permutation(len(xy))
    a = hdbscan(csr_matrix(cdist(xy, xy)), min_cluster_size=8, min_samples=1)
    b = hdbscan(csr_matrix(cdist(xy[order], xy[order])), min_cluster_size=8, min_samples=1)
    assert same_partition(a.labels[order], b.labels)


def test_tied_distances_do_not_depend_on_position_order():
    """Equal mutual-reachability distances must be one simultaneous level.

    This deliberately uses integer coordinates and min_samples > 1, which
    create many exact ties and therefore exercise the simultaneous-merge rule.
    """
    rng = np.random.default_rng(1)
    centers = np.array([[0, 0], [12, 0], [6, 10]])
    which = rng.integers(0, 3, 24)
    xy = np.rint(centers[which] + rng.normal(scale=2.2, size=(24, 2))).astype(float)
    order = rng.permutation(len(xy))
    d = cdist(xy, xy)
    a = hdbscan(csr_matrix(d), min_cluster_size=4, min_samples=3)
    b = hdbscan(csr_matrix(d[np.ix_(order, order)]), min_cluster_size=4, min_samples=3)
    assert same_partition(a.labels[order], b.labels)
    assert np.allclose(a.probabilities[order], b.probabilities)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_cluster_size": 1},
        {"min_samples": 0},
        {"cluster_selection_method": "best"},
        {"cluster_selection_epsilon": -1.0},
        {"max_cluster_size": 0},
        {"core_distance_floor": -1.0},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        hdbscan(csr_matrix((3, 3)), **kwargs)


def test_negative_distances_rejected():
    with pytest.raises(ValueError):
        hdbscan(csr_matrix(np.array([[0.0, -1.0], [-1.0, 0.0]])), min_cluster_size=2)


def test_fractional_weights_are_rejected_instead_of_truncated():
    graph = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="integer"):
        hdbscan(graph, np.array([1.9, 1.9]), min_cluster_size=2)


def test_boolean_weights_are_rejected():
    graph = csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="integer"):
        hdbscan(graph, np.array([True, True]), min_cluster_size=2)


def test_asymmetric_sparse_graph_is_rejected_including_stored_pattern():
    graph = csr_matrix(([1.0], ([0], [1])), shape=(2, 2))
    with pytest.raises(ValueError, match="symmetric"):
        hdbscan(graph, min_cluster_size=2)


def test_duplicate_sparse_entries_are_rejected_before_scipy_can_sum_them():
    # Row 0 stores column 1 twice. Construct CSR directly so duplicates survive.
    graph = csr_matrix(
        (
            np.array([1.0, 1.0, 1.0]),
            np.array([1, 1, 0]),
            np.array([0, 2, 3]),
        ),
        shape=(2, 2),
    )
    assert not graph.has_canonical_format
    with pytest.raises(ValueError, match="duplicate stored entry"):
        hdbscan(graph, min_cluster_size=2)
