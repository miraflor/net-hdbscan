"""HDBSCAN* on a sparse, truncated distance graph with weighted positions.

Why this module exists
----------------------
scikit-learn's ``HDBSCAN(metric="precomputed")`` accepts a sparse distance
matrix only if every row stores at least ``min_samples`` neighbours and the
graph is connected. A network-distance graph truncated at ``max_distance``
satisfies neither condition in general (isolated observations, islands,
disconnected network pieces). This module implements the same algorithm
(Campello, Moulavi and Sander 2013; McInnes, Healy and Astels 2017) directly
on such a graph.

Conventions follow scikit-learn so that results can be compared:

* ``min_samples`` counts the observation itself;
* lambda = 1 / distance, and lambda = inf for distance 0;
* the Excess-of-Mass rule selects a cluster when its stability is at least
  the summed stability of its selected descendants;
* ``cluster_selection_epsilon``, ``max_cluster_size`` and
  ``allow_single_cluster`` have scikit-learn's meaning.

Equal mutual-reachability distances are the one deliberate hierarchy-level
choice. They are processed simultaneously as a multifurcation. A binary MST
implementation has to order tied edges somehow, and that arbitrary order can
change zero-lifetime intermediate groups and, in turn, selected clusters. The
simultaneous rule makes the result invariant to row order and point-ID naming;
it can therefore differ from scikit-learn in tie-heavy cases.

Two extensions:

1. Weighted positions. Observations at one snapped position have distance 0
   to each other. They are represented once, with a weight equal to their
   count. A position with weight ``w`` behaves as ``w`` identical points:
   they stay together until the distance equals the position's core
   distance, where all of them leave their cluster at once.
2. Disconnected graphs. Pairs farther apart than ``max_distance`` (or on
   disconnected network components) are treated as infinitely far apart. The
   minimum spanning tree becomes a forest, and all forest components hang
   from one root at lambda = 0.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

_TINY = np.finfo(float).tiny


@dataclass(frozen=True)
class CondensedTree:
    """Edges of the condensed tree.

    ``child`` is a cluster number when ``child_is_cluster`` is true and a
    position number otherwise. ``child_size`` is the total weight of the
    child. Cluster 0 is the root.
    """

    parent: np.ndarray
    child: np.ndarray
    lambda_val: np.ndarray
    child_size: np.ndarray
    child_is_cluster: np.ndarray


@dataclass(frozen=True)
class HDBSCANResult:
    """HDBSCAN output for positions.

    ``labels`` holds -1 for noise and 0..k-1 for selected clusters, numbered
    in hierarchy order. The per-cluster arrays are indexed by hierarchy
    cluster number (0 = root).
    """

    labels: np.ndarray
    probabilities: np.ndarray
    core_distances: np.ndarray
    exit_lambda: np.ndarray
    tree: CondensedTree
    cluster_parent: np.ndarray
    cluster_birth_lambda: np.ndarray
    cluster_size: np.ndarray
    cluster_stability: np.ndarray
    cluster_selected: np.ndarray
    cluster_label: np.ndarray

    @property
    def n_clusters(self) -> int:
        return int(self.cluster_selected.sum())


@dataclass(frozen=True)
class _SimultaneousHierarchy:
    """Single-linkage hierarchy with equal-distance merges kept simultaneous.

    Leaves are positions ``0..n-1``.  Every internal node represents one
    connected component created at a *distinct* mutual-reachability distance
    and may therefore have more than two children.  This is important when
    distances tie: forcing a multifurcation into an arbitrary binary merge
    order can create zero-lifetime intermediate clusters whose sizes depend on
    edge/point ordering.
    """

    children: tuple[np.ndarray, ...]
    distance: np.ndarray
    size: np.ndarray
    tops: np.ndarray


def validate_parameters(
    *,
    min_cluster_size: int,
    min_samples: int | None,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    max_cluster_size: int | None,
) -> int:
    """Check parameters and return the effective ``min_samples``."""
    if isinstance(min_cluster_size, bool) or not isinstance(min_cluster_size, (int, np.integer)) or min_cluster_size < 2:
        raise ValueError("min_cluster_size must be an integer >= 2")
    if min_samples is None:
        min_samples = int(min_cluster_size)
    if isinstance(min_samples, bool) or not isinstance(min_samples, (int, np.integer)) or min_samples < 1:
        raise ValueError("min_samples must be an integer >= 1")
    if cluster_selection_method not in {"eom", "leaf"}:
        raise ValueError("cluster_selection_method must be 'eom' or 'leaf'")
    eps = float(cluster_selection_epsilon)
    if not (np.isfinite(eps) and eps >= 0):
        raise ValueError("cluster_selection_epsilon must be a finite number >= 0")
    if max_cluster_size is not None and (
        isinstance(max_cluster_size, bool) or not isinstance(max_cluster_size, (int, np.integer)) or max_cluster_size < 1
    ):
        raise ValueError("max_cluster_size must be None or an integer >= 1")
    return int(min_samples)


# Rows longer than this (and longer than 4 x the entries they need) are
# handled one at a time with a partial selection; shorter rows are sorted
# together, which is faster when there are many of them.
_LONG_ROW = 64


def core_distances(graph: csr_matrix, weights: np.ndarray, min_samples: int) -> np.ndarray:
    """Smallest distance at which each position has ``min_samples`` weight around it.

    The position's own weight counts. A position whose own weight reaches
    ``min_samples`` has core distance 0. A position that does not reach
    ``min_samples`` within the stored graph has core distance ``inf``.
    Diagonal entries of ``graph`` are ignored.

    Every weight is at least 1, so a position needs at most
    ``min_samples - own weight`` neighbours: only that many smallest entries
    of its row can matter. Long rows therefore use a partial selection
    (the ``need`` smallest values, without sorting the rest), and only short
    rows are sorted in full. Both paths give exactly the same value.
    """
    graph = csr_matrix(graph)
    n = graph.shape[0]
    weights = np.asarray(weights, dtype=np.int64)
    need = int(min_samples) - weights
    core = np.full(n, np.inf)
    core[need <= 0] = 0.0
    indptr, indices, data = graph.indptr, graph.indices, graph.data
    length = np.diff(indptr)
    active = (need > 0) & (length > 0)
    long_rows = np.flatnonzero(active & (length > np.maximum(_LONG_ROW, 4 * need)))
    short = active.copy()
    short[long_rows] = False

    # Short rows: sort their entries by (row, distance) and read the answer
    # from the cumulative weight of the neighbours.
    if short.any():
        selected = np.repeat(short, length)
        rows = np.repeat(np.arange(n), length)[selected]
        cols = indices[selected]
        dist = data[selected]
        off = rows != cols
        rows, cols, dist = rows[off], cols[off], dist[off]
        if len(rows):
            order = np.lexsort((dist, rows))
            rows, cols, dist = rows[order], cols[order], dist[order]
            cum = np.cumsum(weights[cols])
            which = np.flatnonzero(short)
            row_start = np.searchsorted(rows, which, side="left")
            row_end = np.searchsorted(rows, which, side="right")
            base = np.where(row_start > 0, cum[np.maximum(row_start - 1, 0)], 0)
            k = np.searchsorted(cum, base + need[which], side="left")
            ok = k < row_end
            core[which[ok]] = dist[k[ok]]

    # Long rows: the ``need`` smallest entries are enough (each weight >= 1).
    for r in long_rows:
        lo, hi = indptr[r], indptr[r + 1]
        dist = data[lo:hi]
        cols = indices[lo:hi]
        keep = cols != r
        if not keep.all():
            dist, cols = dist[keep], cols[keep]
        m = min(int(need[r]), len(dist))
        if m == 0:
            continue
        pick = np.argpartition(dist, m - 1)[:m] if m < len(dist) else np.arange(len(dist))
        pick = pick[np.argsort(dist[pick], kind="stable")]
        cum = np.cumsum(weights[cols[pick]])
        j = int(np.searchsorted(cum, need[r], side="left"))
        if j < len(cum):
            core[r] = dist[pick[j]]
    return core


def mutual_reachability_forest(graph: csr_matrix, core: np.ndarray):
    """Minimum spanning forest of the mutual reachability graph.

    For a stored pair, the mutual reachability distance is
    ``max(core_i, core_j, d_ij)``: the smallest distance at which both
    positions are core positions and within reach of each other. Pairs
    involving a position with infinite core distance are dropped.

    Returns edge arrays ``(a, b, weight)`` sorted by weight, then by ``a``
    and ``b``, with ``a < b``.
    """
    coo = coo_matrix(graph)
    upper = coo.row < coo.col
    r, c, d = coo.row[upper], coo.col[upper], coo.data[upper]
    m = np.maximum(d, np.maximum(core[r], core[c]))
    keep = np.isfinite(m)
    r, c, m = r[keep], c[keep], m[keep]
    n = graph.shape[0]
    if len(r) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=float)
    # scipy treats stored zeros as missing edges; lift zeros to the smallest
    # positive float for the tree search and restore them afterwards.
    lifted = np.maximum(m, _TINY)
    mst = minimum_spanning_tree(coo_matrix((lifted, (r, c)), shape=(n, n)).tocsr()).tocoo()
    a = np.minimum(mst.row, mst.col).astype(np.int64)
    b = np.maximum(mst.row, mst.col).astype(np.int64)
    w = mst.data.astype(float)
    w[w <= _TINY] = 0.0
    order = np.lexsort((b, a, w))
    return a[order], b[order], w[order]


def _simultaneous_hierarchy(
    n: int,
    a: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    weights: np.ndarray,
) -> _SimultaneousHierarchy:
    """Build the single-linkage hierarchy, batching every equal-distance level.

    The minimum spanning forest preserves the connected components of the
    mutual-reachability graph at every distance threshold.  At one distinct
    edge weight, all components joined by that level are therefore merged in
    one operation.  The resulting hierarchy depends on distances and
    connectivity, not on the arbitrary order of tied MST edges.
    """
    parent = list(range(n))
    node_of = list(range(n))
    sizes: list[int] = [int(x) for x in weights]
    children: list[np.ndarray] = []
    distance: list[float] = []

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x: int, y: int) -> int:
        rx, ry = find(x), find(y)
        if rx == ry:
            return rx
        # Rank is unnecessary here; choose a deterministic representative.
        if ry < rx:
            rx, ry = ry, rx
        parent[ry] = rx
        return rx

    start = 0
    while start < len(w):
        stop = start + 1
        level = float(w[start])
        while stop < len(w) and w[stop] == w[start]:
            stop += 1

        # Current DSU components touched by edges at this exact distance.
        level_edges: list[tuple[int, int]] = []
        touched: set[int] = set()
        for i in range(start, stop):
            ra, rb = find(int(a[i])), find(int(b[i]))
            if ra == rb:
                continue
            level_edges.append((ra, rb))
            touched.add(ra)
            touched.add(rb)

        if level_edges:
            # Temporary DSU over the *pre-level* components.  This discovers
            # every multifurcation at the level before mutating the main DSU.
            temp_parent = {r: r for r in touched}

            def tfind(x: int) -> int:
                root = x
                while temp_parent[root] != root:
                    root = temp_parent[root]
                while temp_parent[x] != root:
                    temp_parent[x], x = root, temp_parent[x]
                return root

            def tunion(x: int, y: int) -> None:
                rx, ry = tfind(x), tfind(y)
                if rx == ry:
                    return
                if ry < rx:
                    rx, ry = ry, rx
                temp_parent[ry] = rx

            for ra, rb in level_edges:
                tunion(ra, rb)

            groups: dict[int, list[int]] = {}
            for r in touched:
                groups.setdefault(tfind(r), []).append(r)

            # Group iteration order may affect hierarchy IDs, but not the
            # partition. Sorting keeps output reproducible for a fixed input.
            for roots in sorted(groups.values(), key=lambda xs: min(xs)):
                roots = sorted(set(roots))
                if len(roots) < 2:
                    continue
                child_nodes = np.asarray([node_of[r] for r in roots], dtype=np.int64)
                new_node = len(sizes)
                children.append(child_nodes)
                distance.append(level)
                sizes.append(int(sum(sizes[c] for c in child_nodes)))

                rep = roots[0]
                for r in roots[1:]:
                    rep = union(rep, r)
                rep = find(rep)
                node_of[rep] = new_node

        start = stop

    roots = {find(x) for x in range(n)}
    tops = np.asarray(sorted((node_of[r] for r in roots)), dtype=np.int64)
    return _SimultaneousHierarchy(
        children=tuple(children),
        distance=np.asarray(distance, dtype=float),
        size=np.asarray(sizes, dtype=np.int64),
        tops=tops,
    )


def _single_linkage(n: int, a: np.ndarray, b: np.ndarray, w: np.ndarray, weights: np.ndarray):
    """Merge list (left, right, distance, size) and the top node of each component."""
    parent = list(range(n))
    node_of = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    m = len(a)
    left = np.empty(m, dtype=np.int64)
    right = np.empty(m, dtype=np.int64)
    size = np.empty(n + m, dtype=np.int64)
    size[:n] = weights
    min_leaf = np.empty(n + m, dtype=np.int64)
    min_leaf[:n] = np.arange(n)
    for t in range(m):
        ra, rb = find(int(a[t])), find(int(b[t]))
        na, nb = node_of[ra], node_of[rb]
        left[t], right[t] = na, nb
        size[n + t] = size[na] + size[nb]
        min_leaf[n + t] = min(min_leaf[na], min_leaf[nb])
        parent[rb] = ra
        node_of[ra] = n + t
    tops = sorted({node_of[find(x)] for x in range(n)}, key=lambda node: min_leaf[node])
    return left, right, w.astype(float), size, np.asarray(tops, dtype=np.int64)


def _condense(n, left, right, distance, size, tops, weights, core, min_cluster_size):
    """Condensed tree records: (parent cluster, child, lambda, child size, child is cluster)."""
    parents: list[int] = []
    children: list[int] = []
    lambdas: list[float] = []
    sizes: list[int] = []
    is_cluster: list[bool] = []

    def record(p, c, lam, s, flag):
        parents.append(p)
        children.append(c)
        lambdas.append(lam)
        sizes.append(s)
        is_cluster.append(flag)

    def fall_out(node, cluster, lam):
        stack = [node]
        while stack:
            x = stack.pop()
            if x < n:
                record(cluster, x, lam, int(weights[x]), False)
            else:
                stack.append(int(right[x - n]))
                stack.append(int(left[x - n]))

    relabel: dict[int, int] = {}
    queue: deque[int] = deque()
    next_cluster = 1
    big = [int(t) for t in tops if size[t] >= min_cluster_size]
    if len(big) >= 2:
        for t in big:
            relabel[t] = next_cluster
            record(0, next_cluster, 0.0, int(size[t]), True)
            next_cluster += 1
            queue.append(t)
    elif len(big) == 1:
        relabel[big[0]] = 0
        queue.append(big[0])
    for t in tops:
        if size[t] < min_cluster_size:
            fall_out(int(t), 0, 0.0)

    while queue:
        node = queue.popleft()
        cluster = relabel[node]
        if node < n:
            # A position that is a cluster by itself: its observations share
            # one location and leave together at the position's core distance.
            lam = np.inf if core[node] == 0 else 1.0 / core[node]
            record(cluster, node, lam, int(weights[node]), False)
            continue
        l, r = int(left[node - n]), int(right[node - n])
        d = distance[node - n]
        if d <= 0:
            # Distance 0 between distinct positions: indistinguishable, so no split.
            fall_out(node, cluster, np.inf)
            continue
        lam = 1.0 / d
        big_l, big_r = size[l] >= min_cluster_size, size[r] >= min_cluster_size
        if big_l and big_r:
            for child in (l, r):
                relabel[child] = next_cluster
                record(cluster, next_cluster, lam, int(size[child]), True)
                next_cluster += 1
                queue.append(child)
        elif big_l:
            relabel[l] = cluster
            queue.append(l)
            fall_out(r, cluster, lam)
        elif big_r:
            relabel[r] = cluster
            queue.append(r)
            fall_out(l, cluster, lam)
        else:
            fall_out(l, cluster, lam)
            fall_out(r, cluster, lam)

    tree = CondensedTree(
        parent=np.asarray(parents, dtype=np.int64),
        child=np.asarray(children, dtype=np.int64),
        lambda_val=np.asarray(lambdas, dtype=float),
        child_size=np.asarray(sizes, dtype=np.int64),
        child_is_cluster=np.asarray(is_cluster, dtype=bool),
    )
    return tree, next_cluster


def _condense_simultaneous(
    n: int,
    hierarchy: _SimultaneousHierarchy,
    weights: np.ndarray,
    core: np.ndarray,
    min_cluster_size: int,
):
    """Condense a multifurcating hierarchy without ordering tied splits.

    A node created at distance ``d`` splits into *all* of its children at
    lambda ``1/d``.  Children smaller than ``min_cluster_size`` fall out at
    that same lambda; one large child continues the current cluster; two or
    more large children create new child clusters.  This is the direct
    multifurcation analogue of ``_condense``.
    """
    parents: list[int] = []
    children_out: list[int] = []
    lambdas: list[float] = []
    sizes_out: list[int] = []
    is_cluster: list[bool] = []

    def record(p, c, lam, s, flag):
        parents.append(p)
        children_out.append(c)
        lambdas.append(lam)
        sizes_out.append(s)
        is_cluster.append(flag)

    def node_children(node: int) -> np.ndarray:
        if node < n:
            return np.empty(0, dtype=np.int64)
        return hierarchy.children[node - n]

    def fall_out(node: int, cluster: int, lam: float) -> None:
        stack = [node]
        while stack:
            x = int(stack.pop())
            if x < n:
                record(cluster, x, lam, int(weights[x]), False)
            else:
                stack.extend(int(c) for c in node_children(x))

    relabel: dict[int, int] = {}
    queue: deque[int] = deque()
    next_cluster = 1
    big = [int(t) for t in hierarchy.tops if hierarchy.size[t] >= min_cluster_size]
    if len(big) >= 2:
        for t in big:
            relabel[t] = next_cluster
            record(0, next_cluster, 0.0, int(hierarchy.size[t]), True)
            next_cluster += 1
            queue.append(t)
    elif len(big) == 1:
        relabel[big[0]] = 0
        queue.append(big[0])
    for t in hierarchy.tops:
        if hierarchy.size[t] < min_cluster_size:
            fall_out(int(t), 0, 0.0)

    while queue:
        node = int(queue.popleft())
        cluster = relabel[node]
        if node < n:
            lam = np.inf if core[node] == 0 else 1.0 / core[node]
            record(cluster, node, lam, int(weights[node]), False)
            continue

        d = float(hierarchy.distance[node - n])
        if d <= 0:
            # Distinct positions joined at zero distance cannot be separated
            # below any finite scale.
            fall_out(node, cluster, np.inf)
            continue
        lam = 1.0 / d
        child_nodes = [int(c) for c in node_children(node)]
        big_children = [c for c in child_nodes if hierarchy.size[c] >= min_cluster_size]

        if len(big_children) >= 2:
            big_set = set(big_children)
            for child in child_nodes:
                if child in big_set:
                    relabel[child] = next_cluster
                    record(cluster, next_cluster, lam, int(hierarchy.size[child]), True)
                    next_cluster += 1
                    queue.append(child)
                else:
                    fall_out(child, cluster, lam)
        elif len(big_children) == 1:
            keep = big_children[0]
            relabel[keep] = cluster
            queue.append(keep)
            for child in child_nodes:
                if child != keep:
                    fall_out(child, cluster, lam)
        else:
            for child in child_nodes:
                fall_out(child, cluster, lam)

    tree = CondensedTree(
        parent=np.asarray(parents, dtype=np.int64),
        child=np.asarray(children_out, dtype=np.int64),
        lambda_val=np.asarray(lambdas, dtype=float),
        child_size=np.asarray(sizes_out, dtype=np.int64),
        child_is_cluster=np.asarray(is_cluster, dtype=bool),
    )
    return tree, next_cluster


def _stability(tree: CondensedTree, n_clusters: int):
    birth = np.zeros(n_clusters)
    cluster_parent = np.full(n_clusters, -1, dtype=np.int64)
    cluster_size = np.zeros(n_clusters, dtype=np.int64)
    rows = tree.child_is_cluster
    birth[tree.child[rows]] = tree.lambda_val[rows]
    cluster_parent[tree.child[rows]] = tree.parent[rows]
    cluster_size[tree.child[rows]] = tree.child_size[rows]
    # Size of the root as scikit-learn counts it for max_cluster_size: the
    # summed size of its child clusters.
    cluster_size[0] = int(tree.child_size[(tree.parent == 0) & tree.child_is_cluster].sum())
    with np.errstate(invalid="ignore"):
        contribution = (tree.lambda_val - birth[tree.parent]) * tree.child_size
    stability = np.zeros(n_clusters)
    np.add.at(stability, tree.parent, contribution)
    return birth, cluster_parent, cluster_size, stability


def _children_lists(cluster_parent: np.ndarray) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(len(cluster_parent))]
    for c in range(1, len(cluster_parent)):
        children[int(cluster_parent[c])].append(c)
    return children


def _descendants(children: list[list[int]], c: int) -> list[int]:
    out, stack = [], list(children[c])
    while stack:
        x = stack.pop()
        out.append(x)
        stack.extend(children[x])
    return out


def _epsilon_search(leaves, birth, cluster_parent, children, epsilon, allow_single_cluster) -> set[int]:
    """scikit-learn's HDBSCAN(epsilon-hat) rule (Malzer and Baum 2020)."""

    def birth_distance(c: int) -> float:
        return np.inf if birth[c] == 0 else 1.0 / birth[c]

    def traverse_upwards(leaf: int) -> int:
        while True:
            parent = int(cluster_parent[leaf])
            if parent == 0:
                return 0 if allow_single_cluster else leaf
            if birth_distance(parent) > epsilon:
                return parent
            leaf = parent

    selected: list[int] = []
    processed: set[int] = set()
    for leaf in sorted(leaves):
        if birth_distance(leaf) < epsilon:
            if leaf not in processed:
                chosen = traverse_upwards(leaf)
                selected.append(chosen)
                processed.update(x for x in _descendants(children, chosen) if x != chosen)
        else:
            selected.append(leaf)
    return set(selected)


def _select(
    birth, cluster_parent, cluster_size, stability, *, method, epsilon, max_cluster_size, allow_single_cluster
) -> set[int]:
    k = len(birth)
    children = _children_lists(cluster_parent)
    has_cluster_children = k > 1
    if method == "eom":
        limit = np.inf if max_cluster_size is None else max_cluster_size
        stab = stability.copy()
        choose = np.zeros(k, dtype=bool)
        for c in range(k - 1, -1, -1):
            if c == 0 and not allow_single_cluster:
                continue
            subtree = float(sum(stab[ch] for ch in children[c]))
            if subtree > stab[c] or cluster_size[c] > limit:
                stab[c] = subtree
            else:
                choose[c] = True
        selected: set[int] = set()
        stack = [0] if allow_single_cluster else list(children[0])
        while stack:
            c = stack.pop()
            if choose[c]:
                selected.add(c)
            else:
                stack.extend(children[c])
        if epsilon != 0.0 and has_cluster_children:
            if selected == {0}:
                selected = {0} if allow_single_cluster else set()
            else:
                selected = _epsilon_search(selected, birth, cluster_parent, children, epsilon, allow_single_cluster)
        return selected
    leaves = {c for c in range(1, k) if not children[c]}
    if epsilon != 0.0:
        return _epsilon_search(leaves, birth, cluster_parent, children, epsilon, allow_single_cluster)
    return leaves


def _validate_sparse_distance_graph(graph) -> csr_matrix:
    """Return a canonical CSR distance graph after validating its metric representation.

    Stored entries are known finite distances. The sparse pattern itself is part
    of the data contract, so duplicate coordinates are rejected rather than
    silently summed, and symmetry includes explicitly stored zero-distance
    entries. Diagonal entries are allowed and ignored by the HDBSCAN core.
    """
    try:
        coo = coo_matrix(graph, dtype=float)
    except Exception as exc:
        raise ValueError("graph must be a two-dimensional sparse distance matrix") from exc
    if len(coo.shape) != 2 or coo.shape[0] != coo.shape[1]:
        raise ValueError("graph must be square")
    if len(coo.data) and (not np.isfinite(coo.data).all() or (coo.data < 0).any()):
        raise ValueError("stored distances must be finite and >= 0")

    # Duplicates must be detected before COO -> CSR conversion, because SciPy
    # otherwise combines duplicate entries by summing their values.
    if coo.nnz:
        order = np.lexsort((coo.col, coo.row))
        row = coo.row[order]
        col = coo.col[order]
        duplicate = (row[1:] == row[:-1]) & (col[1:] == col[:-1])
        if duplicate.any():
            k = int(np.flatnonzero(duplicate)[0]) + 1
            raise ValueError(
                f"graph contains duplicate stored entry ({int(row[k])}, {int(col[k])}); "
                "each ordered pair must be stored at most once"
            )

        # Compare the complete stored triplets, not graph - graph.T: sparse
        # arithmetic can erase an unmatched explicit zero.
        forward = np.lexsort((coo.col, coo.row))
        backward = np.lexsort((coo.row, coo.col))
        same_pairs = np.array_equal(coo.row[forward], coo.col[backward]) and np.array_equal(
            coo.col[forward], coo.row[backward]
        )
        same_values = same_pairs and np.array_equal(coo.data[forward], coo.data[backward])
        if not same_values:
            raise ValueError("graph must be symmetric, including its stored zero entries")

    out = coo.tocsr()
    out.sort_indices()
    return out


def _validate_weights(weights, n: int) -> np.ndarray:
    """Return positive integer multiplicities without silently truncating input."""
    if weights is None:
        return np.ones(n, dtype=np.int64)
    raw = np.asarray(weights)
    if raw.shape != (n,) or np.issubdtype(raw.dtype, np.bool_):
        raise ValueError("weights must hold one integer >= 1 per position")

    limit = np.iinfo(np.int64).max
    if np.issubdtype(raw.dtype, np.integer):
        if (raw < 1).any():
            raise ValueError("weights must hold one finite integer >= 1 per position")
        if np.issubdtype(raw.dtype, np.unsignedinteger) and (raw > limit).any():
            raise ValueError("weights exceed the supported int64 range")
        return raw.astype(np.int64, copy=False)

    try:
        numeric = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must hold one integer >= 1 per position") from exc
    valid = np.isfinite(numeric) & (numeric >= 1) & (numeric <= float(limit)) & (numeric == np.floor(numeric))
    if not valid.all():
        raise ValueError("weights must hold one finite integer >= 1 per position")
    return numeric.astype(np.int64)


def hdbscan(
    graph: csr_matrix,
    weights: np.ndarray | None = None,
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_method: str = "eom",
    cluster_selection_epsilon: float = 0.0,
    max_cluster_size: int | None = None,
    allow_single_cluster: bool = False,
    core_distance_floor: float = 0.0,
) -> HDBSCANResult:
    """Run HDBSCAN* on a symmetric sparse distance graph.

    Stored entries of ``graph`` are the known pairwise distances; missing
    entries mean "farther than the search radius". ``weights`` gives the
    number of observations at each position (default 1).

    ``core_distance_floor`` raises every core distance below it to it, and
    so every mutual reachability distance too. With the default 0 the
    result is standard HDBSCAN*. A positive floor limits the density of
    stacked observations (see the README).
    """
    min_samples = validate_parameters(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=cluster_selection_epsilon,
        max_cluster_size=max_cluster_size,
    )
    graph = _validate_sparse_distance_graph(graph)
    n = graph.shape[0]
    weights = _validate_weights(weights, n)

    floor = float(core_distance_floor)
    if not (np.isfinite(floor) and floor >= 0):
        raise ValueError("core_distance_floor must be a finite number >= 0")
    core = core_distances(graph, weights, min_samples)
    if floor > 0:
        core = np.maximum(core, floor)
    a, b, w = mutual_reachability_forest(graph, core)
    hierarchy = _simultaneous_hierarchy(n, a, b, w, weights)
    return _from_simultaneous_hierarchy(
        n,
        hierarchy,
        weights,
        core,
        min_cluster_size=int(min_cluster_size),
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        max_cluster_size=max_cluster_size,
        allow_single_cluster=allow_single_cluster,
    )


def _from_simultaneous_hierarchy(
    n,
    hierarchy,
    weights,
    core,
    *,
    min_cluster_size,
    cluster_selection_method,
    cluster_selection_epsilon,
    max_cluster_size,
    allow_single_cluster,
) -> HDBSCANResult:
    """Condense and select clusters from the tie-invariant hierarchy."""
    tree, k = _condense_simultaneous(n, hierarchy, weights, core, int(min_cluster_size))
    return _finalize_condensed_tree(
        n,
        tree,
        k,
        weights,
        core,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        max_cluster_size=max_cluster_size,
        allow_single_cluster=allow_single_cluster,
    )


def _from_linkage(
    n,
    linkage,
    weights,
    core,
    *,
    min_cluster_size,
    cluster_selection_method,
    cluster_selection_epsilon,
    max_cluster_size,
    allow_single_cluster,
) -> HDBSCANResult:
    """Condense a single-linkage tree, select clusters and label positions."""
    left, right, distance, size, tops = linkage
    tree, k = _condense(n, left, right, distance, size, tops, weights, core, int(min_cluster_size))
    return _finalize_condensed_tree(
        n,
        tree,
        k,
        weights,
        core,
        cluster_selection_method=cluster_selection_method,
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        max_cluster_size=max_cluster_size,
        allow_single_cluster=allow_single_cluster,
    )


def _finalize_condensed_tree(
    n: int,
    tree: CondensedTree,
    k: int,
    weights: np.ndarray,
    core: np.ndarray,
    *,
    cluster_selection_method: str,
    cluster_selection_epsilon: float,
    max_cluster_size: int | None,
    allow_single_cluster: bool,
) -> HDBSCANResult:
    """Select clusters, assign labels and compute membership from a condensed tree."""
    epsilon = float(cluster_selection_epsilon)
    birth, cluster_parent, cluster_size, stability = _stability(tree, k)
    selected = _select(
        birth,
        cluster_parent,
        cluster_size,
        stability,
        method=cluster_selection_method,
        epsilon=epsilon,
        max_cluster_size=max_cluster_size,
        allow_single_cluster=allow_single_cluster,
    )

    cluster_selected = np.zeros(k, dtype=bool)
    cluster_selected[list(selected)] = True
    cluster_label = np.full(k, -1, dtype=np.int64)
    for label, c in enumerate(sorted(selected)):
        cluster_label[c] = label

    # Each position leaves exactly one cluster (its last one) at one lambda.
    points = ~tree.child_is_cluster
    exit_cluster = np.full(n, -1, dtype=np.int64)
    exit_lambda = np.zeros(n)
    exit_cluster[tree.child[points]] = tree.parent[points]
    exit_lambda[tree.child[points]] = tree.lambda_val[points]

    # Nearest selected ancestor (inclusive) of every hierarchy cluster.
    owner = np.full(k, -1, dtype=np.int64)
    for c in range(k):
        if cluster_selected[c]:
            owner[c] = c
        elif c > 0:
            owner[c] = owner[int(cluster_parent[c])]

    labels = np.full(n, -1, dtype=np.int64)
    if n:
        own = owner[exit_cluster]
        labels[own > 0] = cluster_label[own[own > 0]]
        if selected == {0} and allow_single_cluster:
            # scikit-learn's rule when the root is the only cluster: keep the
            # observations that stay until the root's largest lambda (or until
            # distance cluster_selection_epsilon).
            if epsilon != 0.0:
                threshold = 1.0 / epsilon
            else:
                root_rows = tree.parent == 0
                threshold = tree.lambda_val[root_rows].max() if root_rows.any() else np.inf
            labels[(own == 0) & (exit_lambda >= threshold)] = cluster_label[0]

    # Membership strength, scikit-learn's convention.
    deaths = np.zeros(k)
    if len(tree.parent):
        np.maximum.at(deaths, tree.parent, tree.lambda_val)
    probabilities = np.zeros(n)
    labelled = labels >= 0
    if labelled.any():
        label_cluster = np.flatnonzero(cluster_selected)[np.argsort(cluster_label[cluster_selected])]
        max_lambda = deaths[label_cluster[labels[labelled]]]
        lam = exit_lambda[labelled]
        with np.errstate(divide="ignore", invalid="ignore"):
            prob = np.minimum(lam, max_lambda) / max_lambda
        prob = np.where((max_lambda == 0) | np.isinf(lam), 1.0, prob)
        probabilities[labelled] = prob

    reported_size = cluster_size.copy()
    reported_size[0] = int(weights.sum())
    return HDBSCANResult(
        labels=labels,
        probabilities=probabilities,
        core_distances=core,
        exit_lambda=exit_lambda,
        tree=tree,
        cluster_parent=cluster_parent,
        cluster_birth_lambda=birth,
        cluster_size=reported_size,
        cluster_stability=stability,
        cluster_selected=cluster_selected,
        cluster_label=cluster_label,
    )
