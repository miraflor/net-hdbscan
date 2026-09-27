"""Network graph, snapping, and bounded shortest-path neighbour search.

Distance definition
-------------------
Every observation is snapped to the nearest point of the nearest network arc
(Euclidean distance). The distance between two observations is the shortest
path along the network between their snapped positions. The snap
distance itself is not added; it is reported as quality-assurance data.
Observations on disconnected parts of the network have no finite distance.

The implementation uses Shapely and SciPy directly for network construction,
snapping and shortest-path search.

Network topology
----------------
Each LineString is split into straight arcs between consecutive vertices.
Two lines are joined only where they share a vertex after rounding each
coordinate to ``vertex_digits`` significant digits (default 11, the same
default as ``spaghetti``). No other connection is inferred: two lines that
cross without a shared vertex (for example a bridge over a road) are not
connected. MultiLineString parts are never joined to each other.

Neighbour search
----------------
Only pairs with network distance ``<= max_distance`` are needed. A path of
length ``d`` never leaves the Euclidean disc of radius ``d`` around its start,
so the search for a batch of nearby observations only needs the network
vertices inside the batch's bounding box enlarged by ``max_distance``. The
searches therefore run on small local subgraphs, and the cost does not grow
with the size of the whole network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

DEFAULT_VERTEX_DIGITS = 11

# Upper limit on stored neighbour pairs (unordered pairs of distinct snapped
# positions). One pair needs roughly 100-200 bytes at peak.
DEFAULT_MAX_NEIGHBOR_PAIRS = 20_000_000

# Memory budget (number of float64 values) for the dense result of one batch
# of Dijkstra searches: sources x local nodes.
_SEARCH_BATCH_VALUES = 8_000_000

# Target search work (sources x local network vertices) of one spatial batch.
_BATCH_WORK = 2_000_000

# Candidate (source, target) entries formed at one time when search results
# are joined with the observations attached to the reached vertices.
_CANDIDATE_CHUNK = 4_000_000

# Same-arc pairs produced at one time.
_SAME_ARC_BATCH_PAIRS = 1_000_000

_INDEX = np.int64


# ---------------------------------------------------------------------------
# Network graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NetworkGraph:
    """Straight network arcs between rounded vertices.

    ``arc_u[k] < arc_v[k]`` for every arc; each vertex pair appears once.
    ``component[i]`` is the connected component of vertex ``i``.
    """

    vertex_xy: np.ndarray
    arc_u: np.ndarray
    arc_v: np.ndarray
    arc_length: np.ndarray
    adjacency: csr_matrix
    component: np.ndarray
    arc_tree: shapely.STRtree
    vertex_tree: shapely.STRtree

    @property
    def n_vertices(self) -> int:
        return len(self.vertex_xy)

    @property
    def n_arcs(self) -> int:
        return len(self.arc_u)

    @property
    def n_components(self) -> int:
        return int(self.component.max()) + 1 if len(self.component) else 0


def round_significant(values: np.ndarray, digits: int) -> np.ndarray:
    """Round every value to a positive integer number of significant digits."""
    if isinstance(digits, bool) or not isinstance(digits, (int, np.integer)) or digits < 1:
        raise ValueError(f"vertex_digits must be a positive integer or None, got {digits!r}")
    digits = int(digits)
    values = np.asarray(values, dtype=float)
    out = values.copy()
    nonzero = values != 0
    if nonzero.any():
        exponent = np.floor(np.log10(np.abs(values[nonzero])))
        factor = 10.0 ** (digits - 1 - exponent)
        out[nonzero] = np.round(values[nonzero] * factor) / factor
    return out


def build_network_graph(lines, *, vertex_digits: int | None = DEFAULT_VERTEX_DIGITS) -> NetworkGraph:
    """Build the network graph from LineString / MultiLineString geometries.

    ``lines`` is any array-like of Shapely geometries in a projected CRS.
    Validation of CRS and geometry types is done by the caller
    (``net_hdbscan.io.prepare_network``).
    """
    if vertex_digits is not None and (
        isinstance(vertex_digits, bool)
        or not isinstance(vertex_digits, (int, np.integer))
        or vertex_digits < 1
    ):
        raise ValueError(f"vertex_digits must be a positive integer or None, got {vertex_digits!r}")
    geoms = np.asarray(lines, dtype=object)
    parts = shapely.get_parts(geoms)
    coords, part_index = shapely.get_coordinates(parts, return_index=True)
    if len(coords) == 0:
        raise ValueError("network has no coordinates")
    if vertex_digits is not None:
        coords = round_significant(coords, int(vertex_digits))

    vertex_xy, vertex_id = np.unique(coords, axis=0, return_inverse=True)
    vertex_id = np.asarray(vertex_id).reshape(-1)

    same_part = part_index[1:] == part_index[:-1]
    first = vertex_id[:-1][same_part]
    second = vertex_id[1:][same_part]
    proper = first != second
    u = np.minimum(first, second)[proper]
    v = np.maximum(first, second)[proper]
    if len(u) == 0:
        raise ValueError("network has no arcs of positive length")
    arcs = np.unique(np.column_stack([u, v]), axis=0)
    arc_u = arcs[:, 0].astype(_INDEX)
    arc_v = arcs[:, 1].astype(_INDEX)
    arc_length = np.hypot(*(vertex_xy[arc_v] - vertex_xy[arc_u]).T)

    n = len(vertex_xy)
    adjacency = coo_matrix(
        (np.concatenate([arc_length, arc_length]), (np.concatenate([arc_u, arc_v]), np.concatenate([arc_v, arc_u]))),
        shape=(n, n),
    ).tocsr()
    adjacency.sort_indices()
    _, component = connected_components(adjacency, directed=False)

    segments = shapely.linestrings(np.stack([vertex_xy[arc_u], vertex_xy[arc_v]], axis=1))
    return NetworkGraph(
        vertex_xy=vertex_xy,
        arc_u=arc_u,
        arc_v=arc_v,
        arc_length=arc_length,
        adjacency=adjacency,
        component=np.asarray(component, dtype=_INDEX),
        arc_tree=shapely.STRtree(segments),
        vertex_tree=shapely.STRtree(shapely.points(vertex_xy)),
    )



# ---------------------------------------------------------------------------
# Snapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Snaps:
    """Where observations lie on the network graph, in input order.

    Observation ``k`` lies on the arc from vertex ``u[k]`` to ``v[k]``
    (``u[k] < v[k]``) at distance ``offset[k]`` from ``u[k]``, on an arc of
    length ``arc_length[k]``. ``offset`` is exactly 0 or exactly
    ``arc_length`` when the observation snaps onto a vertex.
    """

    u: np.ndarray
    v: np.ndarray
    offset: np.ndarray
    arc_length: np.ndarray
    snap_distance: np.ndarray
    snapped_xy: np.ndarray

    def __len__(self) -> int:
        return len(self.u)

    def subset(self, index) -> Snaps:
        return Snaps(
            u=self.u[index],
            v=self.v[index],
            offset=self.offset[index],
            arc_length=self.arc_length[index],
            snap_distance=self.snap_distance[index],
            snapped_xy=self.snapped_xy[index],
        )


def empty_snaps() -> Snaps:
    ints = np.empty(0, dtype=_INDEX)
    floats = np.empty(0, dtype=float)
    return Snaps(ints, ints, floats, floats, floats, np.empty((0, 2), dtype=float))


def snap_points(graph: NetworkGraph, xy: np.ndarray) -> Snaps:
    """Snap each coordinate pair to the nearest point of the nearest arc.

    Ties between equally near arcs are broken by the smallest arc index, so
    the result is deterministic. When the nearest point is a shared vertex,
    all incident arcs are equally near and the snapped position is the same
    vertex whichever arc is chosen.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    n = len(xy)
    if n == 0:
        return empty_snaps()
    if not np.isfinite(xy).all():
        raise ValueError("point coordinates contain NaN or infinity")

    source, arc = graph.arc_tree.query_nearest(shapely.points(xy), all_matches=True)
    order = np.lexsort((arc, source))
    source, arc = source[order], arc[order]
    first = np.ones(len(source), dtype=bool)
    first[1:] = source[1:] != source[:-1]
    arc_of = np.full(n, -1, dtype=_INDEX)
    arc_of[source[first]] = arc[first]
    if (arc_of < 0).any():
        raise RuntimeError("some points were not assigned to a network arc")

    u = graph.arc_u[arc_of]
    v = graph.arc_v[arc_of]
    a = graph.vertex_xy[u]
    b = graph.vertex_xy[v]
    ab = b - a
    length = graph.arc_length[arc_of]
    t = np.einsum("ij,ij->i", xy - a, ab) / np.einsum("ij,ij->i", ab, ab)
    t = np.clip(t, 0.0, 1.0)
    at_u = t <= 0.0
    at_v = t >= 1.0
    snapped = a + ab * t[:, None]
    snapped[at_u] = a[at_u]
    snapped[at_v] = b[at_v]
    offset = t * length
    offset[at_u] = 0.0
    offset[at_v] = length[at_v]
    snap_distance = np.hypot(*(xy - snapped).T)
    return Snaps(
        u=u.astype(_INDEX),
        v=v.astype(_INDEX),
        offset=offset,
        arc_length=length,
        snap_distance=snap_distance,
        snapped_xy=snapped,
    )


def distinct_positions(snaps: Snaps) -> tuple[np.ndarray, np.ndarray]:
    """Group observations that snapped to exactly the same network position.

    Returns ``position`` (position number of each observation) and
    ``representative`` (one observation per position, indexed by position
    number).

    A vertex position is identified by its vertex ID, so one vertex reached
    through different incident arcs is one position. An interior position is
    identified by ``(u, v, offset)``. Interior positions of different arcs are
    never merged, even when their coordinates coincide, because such arcs can
    cross without being connected.

    Numbering and representatives depend only on where the observations lie
    on the network, never on their order or their IDs. Vertex positions come
    first, ordered by vertex ID (vertex IDs follow coordinates), then interior
    positions ordered by arc and offset. The representative of a position is
    the observation with the smallest ``(u, v, snap_distance, snapped x,
    snapped y)``; observations that are equal in all of these are
    interchangeable in every later computation. Because the neighbour search
    decides from the position numbers which position of a pair starts the
    shortest-path search, this rule makes all computed distances bitwise
    identical when observations are reordered or their IDs renamed.
    """
    n = len(snaps)
    if n == 0:
        empty = np.empty(0, dtype=_INDEX)
        return empty, empty
    at_u = snaps.offset == 0.0
    at_v = ~at_u & (snaps.offset == snaps.arc_length)
    is_vertex = at_u | at_v
    vertex = np.where(at_u, snaps.u, snaps.v)
    kind = (~is_vertex).astype(np.int8)
    a = np.where(is_vertex, vertex, snaps.u)
    b = np.where(is_vertex, -1, snaps.v)
    offset_key = np.where(is_vertex, 0.0, snaps.offset)
    order = np.lexsort((offset_key, b, a, kind))
    kind_s, a_s, b_s, offset_s = kind[order], a[order], b[order], offset_key[order]
    new = np.ones(n, dtype=bool)
    new[1:] = (kind_s[1:] != kind_s[:-1]) | (a_s[1:] != a_s[:-1]) | (b_s[1:] != b_s[:-1]) | (offset_s[1:] != offset_s[:-1])
    position = np.empty(n, dtype=_INDEX)
    position[order] = np.cumsum(new) - 1
    pick = np.lexsort(
        (snaps.snapped_xy[:, 1], snaps.snapped_xy[:, 0], snaps.snap_distance, snaps.v, snaps.u, position)
    )
    first = np.ones(n, dtype=bool)
    first[1:] = position[pick][1:] != position[pick][:-1]
    return position, pick[first].astype(_INDEX)


# ---------------------------------------------------------------------------
# Neighbour search
# ---------------------------------------------------------------------------


class NeighborPairLimitError(ValueError):
    """The bounded network-neighbour graph exceeded its configured pair budget."""

    def __init__(self, count: int, max_pairs: int, max_distance: float):
        self.count = int(count)
        self.max_pairs = int(max_pairs)
        self.max_distance = float(max_distance)
        super().__init__(
            f"at least {self.count:,} pairs of snapped positions are within "
            f"max_distance={self.max_distance:g}, above max_neighbor_pairs={self.max_pairs:,}. "
            "Use a smaller max_distance, adaptive distance mode, or raise max_neighbor_pairs "
            "(each pair needs roughly 100-200 bytes at peak)."
        )


def _too_many_pairs(count: int, max_pairs: int, max_distance: float) -> NeighborPairLimitError:
    return NeighborPairLimitError(count, max_pairs, max_distance)


def _chunks(weights: np.ndarray, limit: int):
    """Consecutive index ranges whose weights sum to at most ``limit``.

    An index whose own weight exceeds ``limit`` forms a range alone.
    """
    cum = np.concatenate([[0], np.cumsum(weights, dtype=np.int64)])
    first, n = 0, len(weights)
    while first < n:
        last = int(np.searchsorted(cum, cum[first] + limit, side="right")) - 1
        last = min(max(last, first + 1), n)
        yield slice(first, last)
        first = last


def _attachment(n_vertices: int, pos: Snaps):
    """For every vertex, the positions on arcs that end there, nearest first (CSR layout)."""
    n = len(pos)
    ends = np.concatenate([pos.u, pos.v])
    point = np.concatenate([np.arange(n), np.arange(n)])
    along = np.concatenate([pos.offset, pos.arc_length - pos.offset])
    order = np.lexsort((point, along, ends))
    ends, point, along = ends[order], point[order], along[order]
    start = np.searchsorted(ends, np.arange(n_vertices + 1))
    return start, point.astype(_INDEX), along


def _prefix_within(along: np.ndarray, lo: np.ndarray, hi: np.ndarray, reached: np.ndarray, limit: float) -> np.ndarray:
    """How many of ``along[lo:hi]`` (ascending) satisfy ``reached + along <= limit``."""
    first, lo, hi = lo.copy(), lo.copy(), hi.copy()
    while True:
        active = lo < hi
        if not active.any():
            return lo - first
        mid = (lo + hi) // 2
        ok = active & (reached + along[np.where(active, mid, 0)] <= limit)
        lo = np.where(ok, mid + 1, lo)
        hi = np.where(active & ~ok, mid, hi)


def _batches(pos: Snaps, sources: np.ndarray, vertex_tree: shapely.STRtree, limit: float, work: int) -> list[np.ndarray]:
    """Split source positions into spatially compact batches.

    A batch is split into four quadrants while its estimated search work
    (sources x network vertices within ``limit`` of its bounding box) exceeds
    ``work`` and its extent exceeds ``limit / 2``. Dense areas therefore get
    small batches and sparse areas large ones. Batching changes speed and
    memory use, never the result.
    """
    xy = pos.snapped_xy
    done: list[np.ndarray] = []
    stack = [sources] if len(sources) else []
    while stack:
        idx = stack.pop()
        p = xy[idx]
        lo, hi = p.min(axis=0), p.max(axis=0)
        if len(idx) == 1 or (hi - lo).max() <= limit / 2:
            done.append(idx)
            continue
        box = shapely.box(lo[0] - limit, lo[1] - limit, hi[0] + limit, hi[1] + limit)
        if len(idx) * len(vertex_tree.query(box)) <= work:
            done.append(idx)
            continue
        mid = (lo + hi) / 2
        east, north = p[:, 0] > mid[0], p[:, 1] > mid[1]
        for part in (~east & ~north, east & ~north, ~east & north, east & north):
            if part.any():
                stack.append(idx[part])
    return done


def _local_search_graph(graph: NetworkGraph, local: np.ndarray, local_of: np.ndarray, pos: Snaps, sources: np.ndarray, limit: float):
    """Local network subgraph plus one source node per searched position.

    Node layout: local network vertices ``0..L-1``, source nodes ``L..L+S-1``,
    and two dead-end nodes ``L+S`` and ``L+S+1`` without edges. A source node
    has two outgoing edges, to the two ends of its arc, weighted by the
    distances along the arc. An end that is outside the local area, or
    farther than ``limit`` along the arc, is replaced by a dead end: no path
    of length ``<= limit`` can use it.
    """
    sub = graph.adjacency[local][:, local]
    L = len(local)
    S = len(sources)
    lu = local_of[pos.u[sources]]
    lv = local_of[pos.v[sources]]
    to_u = pos.offset[sources]
    to_v = pos.arc_length[sources] - to_u
    use_u = (lu >= 0) & (to_u <= limit)
    use_v = (lv >= 0) & (to_v <= limit)
    first_target = np.where(use_u, lu, L + S)
    second_target = np.where(use_v, lv, L + S + 1)
    first_weight = np.where(use_u, to_u, 0.0)
    second_weight = np.where(use_v, to_v, 0.0)
    swap = first_target > second_target  # keep column indices sorted within each row
    first_target, second_target = np.where(swap, second_target, first_target), np.where(swap, first_target, second_target)
    first_weight, second_weight = np.where(swap, second_weight, first_weight), np.where(swap, first_weight, second_weight)
    indices = np.empty(2 * S, dtype=sub.indices.dtype)
    data = np.empty(2 * S, dtype=float)
    indices[0::2], indices[1::2] = first_target, second_target
    data[0::2], data[1::2] = first_weight, second_weight
    indptr = np.concatenate(
        [sub.indptr, sub.nnz + 2 * np.arange(1, S + 1), [sub.nnz + 2 * S, sub.nnz + 2 * S]]
    ).astype(sub.indptr.dtype)
    size = L + S + 2
    search = csr_matrix((np.concatenate([sub.data, data]), np.concatenate([sub.indices, indices]), indptr), shape=(size, size))
    return search


def _oversized_source_pairs(
    *,
    source_index: int,
    n_positions: int,
    lo: np.ndarray,
    count: np.ndarray,
    reached: np.ndarray,
    attached: np.ndarray,
    along: np.ndarray,
    found_before: int,
    max_pairs: int,
    max_distance: float,
):
    """Deduplicate one source whose endpoint candidates exceed the normal chunk.

    Candidate slices stay bounded by ``_CANDIDATE_CHUNK``. A target can be
    encountered through several reached vertices or both ends of its arc, so
    the best distance is accumulated before the source's unique pairs are
    yielded. This path is intentionally slower and is used only for unusually
    dense individual searches.
    """
    best = np.full(n_positions, np.inf, dtype=float)
    touched_parts: list[np.ndarray] = []
    n_touched = 0
    for entry in range(len(count)):
        first = int(lo[entry])
        stop = first + int(count[entry])
        for p0 in range(first, stop, _CANDIDATE_CHUNK):
            p1 = min(stop, p0 + _CANDIDATE_CHUNK)
            j = attached[p0:p1]
            keep = j > source_index
            if not keep.any():
                continue
            j = j[keep]
            d = reached[entry] + along[p0:p1][keep]
            unique_j = np.unique(j)
            new_j = unique_j[np.isinf(best[unique_j])]
            if len(new_j):
                n_touched += len(new_j)
                if found_before + n_touched > max_pairs:
                    raise _too_many_pairs(found_before + n_touched, max_pairs, max_distance)
                touched_parts.append(new_j.astype(_INDEX, copy=False))
            np.minimum.at(best, j, d)
    if not touched_parts:
        empty = np.empty(0, dtype=_INDEX)
        return empty, empty, np.empty(0, dtype=float)
    target = np.concatenate(touched_parts).astype(_INDEX, copy=False)
    distance = best[target].copy()
    source = np.full(len(target), source_index, dtype=_INDEX)
    return source, target, distance


def _oversized_reached_source_pairs(
    *,
    source_index: int,
    distance_row: np.ndarray,
    local: np.ndarray,
    start: np.ndarray,
    attached: np.ndarray,
    along: np.ndarray,
    n_positions: int,
    max_distance: float,
    found_before: int,
    max_pairs: int,
):
    """Process one source with too many reached vertices without a large join."""
    reached_local = np.flatnonzero(distance_row <= max_distance)
    best = np.full(n_positions, np.inf, dtype=float)
    touched_parts: list[np.ndarray] = []
    n_touched = 0
    reached_chunk = max(1, _CANDIDATE_CHUNK // 4)
    for r0 in range(0, len(reached_local), reached_chunk):
        y_loc = reached_local[r0 : r0 + reached_chunk]
        reached = distance_row[y_loc]
        y = local[y_loc]
        lo = start[y]
        count = _prefix_within(along, lo, start[y + 1], reached, max_distance)
        for entry in range(len(count)):
            first = int(lo[entry])
            stop = first + int(count[entry])
            for p0 in range(first, stop, _CANDIDATE_CHUNK):
                p1 = min(stop, p0 + _CANDIDATE_CHUNK)
                j = attached[p0:p1]
                keep = j > source_index
                if not keep.any():
                    continue
                j = j[keep]
                d = reached[entry] + along[p0:p1][keep]
                unique_j = np.unique(j)
                new_j = unique_j[np.isinf(best[unique_j])]
                if len(new_j):
                    n_touched += len(new_j)
                    if found_before + n_touched > max_pairs:
                        raise _too_many_pairs(found_before + n_touched, max_pairs, max_distance)
                    touched_parts.append(new_j.astype(_INDEX, copy=False))
                np.minimum.at(best, j, d)
    if not touched_parts:
        empty = np.empty(0, dtype=_INDEX)
        return empty, empty, np.empty(0, dtype=float)
    target = np.concatenate(touched_parts).astype(_INDEX, copy=False)
    distance = best[target].copy()
    source = np.full(len(target), source_index, dtype=_INDEX)
    return source, target, distance


def _pairs_through_vertices(graph: NetworkGraph, pos: Snaps, max_distance: float, max_pairs: int, *, batch_work: int | None = None):
    """Pairs whose shortest path leaves the first position's arc through one of its ends.

    Yields ``(i, j, d)`` arrays with ``i < j``; each unordered pair is
    produced once, by the search of its lower index ``i``.
    """
    n = len(pos)
    start, attached, along = _attachment(graph.n_vertices, pos)
    local_of = np.full(graph.n_vertices, -1, dtype=np.int64)
    work = _BATCH_WORK if batch_work is None else int(batch_work)
    found = 0

    # Positions farther than max_distance from both ends of their arc can
    # only have neighbours on the same arc; they need no search.
    reach = (pos.offset <= max_distance) | (pos.arc_length - pos.offset <= max_distance)
    for batch in _batches(pos, np.flatnonzero(reach), graph.vertex_tree, max_distance, work):
        xy = pos.snapped_xy[batch]
        lo = xy.min(axis=0) - max_distance
        hi = xy.max(axis=0) + max_distance
        local = np.sort(graph.vertex_tree.query(shapely.box(lo[0], lo[1], hi[0], hi[1])))
        if len(local) == 0:
            continue
        local_of[local] = np.arange(len(local))
        per_chunk = max(1, _SEARCH_BATCH_VALUES // (len(local) + 2))
        for c0 in range(0, len(batch), per_chunk):
            sources = batch[c0 : c0 + per_chunk]
            S = len(sources)
            search = _local_search_graph(graph, local, local_of, pos, sources, max_distance)
            L = len(local)
            dist = dijkstra(search, directed=True, indices=L + np.arange(S), limit=max_distance)[:, :L]
            within = dist <= max_distance
            per_source = within.sum(axis=1)
            reached_limit = max(1, _CANDIDATE_CHUNK // 4)
            for rows in _chunks(per_source, reached_limit):
                # _chunks emits an overweight single row when one source alone
                # exceeds the limit. Handle it separately so the temporary
                # reached-vertex and candidate arrays remain bounded.
                if rows.stop - rows.start == 1 and per_source[rows.start] > reached_limit:
                    source_index = int(sources[rows.start])
                    i, j, d = _oversized_reached_source_pairs(
                        source_index=source_index,
                        distance_row=dist[rows.start],
                        local=local,
                        start=start,
                        attached=attached,
                        along=along,
                        n_positions=n,
                        max_distance=max_distance,
                        found_before=found,
                        max_pairs=max_pairs,
                    )
                    if len(i):
                        found += len(i)
                        yield i, j, d
                    continue

                s_idx, y_loc = np.nonzero(within[rows])
                s_idx = s_idx + rows.start
                reached = dist[s_idx, y_loc]
                y = local[y_loc]
                lo_a = start[y]
                count = _prefix_within(along, lo_a, start[y + 1], reached, max_distance)
                edges = np.searchsorted(s_idx, np.arange(rows.start, rows.stop + 1))
                before = np.concatenate([[0], np.cumsum(count, dtype=np.int64)])
                candidates_per_source = before[edges[1:]] - before[edges[:-1]]
                for group in _chunks(candidates_per_source, _CANDIDATE_CHUNK):
                    t0, t1 = int(edges[group.start]), int(edges[group.stop])
                    c = count[t0:t1]
                    total = int(c.sum())
                    if total == 0:
                        continue
                    if total > _CANDIDATE_CHUNK:
                        if group.stop - group.start != 1:
                            raise RuntimeError("candidate chunking produced an oversized multi-source group")
                        source_index = int(sources[rows.start + group.start])
                        i, j, d = _oversized_source_pairs(
                            source_index=source_index,
                            n_positions=n,
                            lo=lo_a[t0:t1],
                            count=c,
                            reached=reached[t0:t1],
                            attached=attached,
                            along=along,
                            found_before=found,
                            max_pairs=max_pairs,
                            max_distance=max_distance,
                        )
                        if len(i):
                            found += len(i)
                            yield i, j, d
                        continue
                    entry = np.repeat(np.arange(t0, t1), c)
                    position = lo_a[entry] + (np.arange(total) - np.repeat(np.cumsum(c) - c, c))
                    i = sources[s_idx[entry]]
                    j = attached[position]
                    d = reached[entry] + along[position]
                    keep = j > i
                    i, j, d = i[keep], j[keep], d[keep]
                    if len(i) == 0:
                        continue
                    key = i * np.int64(n) + j
                    order = np.lexsort((d, key))
                    key, i, j, d = key[order], i[order], j[order], d[order]
                    first = np.ones(len(key), dtype=bool)
                    first[1:] = key[1:] != key[:-1]
                    found += int(first.sum())
                    if found > max_pairs:
                        raise _too_many_pairs(found, max_pairs, max_distance)
                    yield i[first], j[first], d[first]
            del dist, within
        local_of[local] = -1


def _reach_on_arc(offset: np.ndarray, stop: np.ndarray, limit: float) -> np.ndarray:
    """For each sorted position ``k``, the first index in ``(k, stop[k])`` beyond ``limit``."""
    base = offset
    lo, hi = np.arange(len(offset)) + 1, stop.copy()
    while True:
        active = lo < hi
        if not active.any():
            return lo
        mid = (lo + hi) // 2
        ok = active & (offset[np.where(active, mid, 0)] - base <= limit)
        lo = np.where(ok, mid + 1, lo)
        hi = np.where(active & ~ok, mid, hi)


def _pairs_on_same_arc(pos: Snaps, max_distance: float, max_pairs: int):
    """Pairs on one arc, measured directly along the arc.

    An arc is a straight segment, so no path that leaves the arc is shorter
    than the direct distance between two positions on it.
    """
    n = len(pos)
    if n < 2:
        return
    order = np.lexsort((pos.offset, pos.v, pos.u))
    u, v, offset = pos.u[order], pos.v[order], pos.offset[order]
    new_arc = np.ones(n, dtype=bool)
    new_arc[1:] = (u[1:] != u[:-1]) | (v[1:] != v[:-1])
    arc_start = np.flatnonzero(new_arc)
    arc_stop = np.append(arc_start[1:], n)
    reach = _reach_on_arc(offset, np.repeat(arc_stop, arc_stop - arc_start), max_distance)
    ahead = reach - np.arange(n) - 1
    total = int(ahead.sum(dtype=np.int64))
    if total > max_pairs:
        raise _too_many_pairs(total, max_pairs, max_distance)
    for rows in _chunks(ahead, _SAME_ARC_BATCH_PAIRS):
        if rows.stop - rows.start == 1 and ahead[rows.start] > _SAME_ARC_BATCH_PAIRS:
            p = rows.start
            for q0 in range(p + 1, int(reach[p]), _SAME_ARC_BATCH_PAIRS):
                q = np.arange(q0, min(int(reach[p]), q0 + _SAME_ARC_BATCH_PAIRS))
                yield (
                    np.full(len(q), order[p], dtype=_INDEX),
                    order[q].astype(_INDEX),
                    offset[q] - offset[p],
                )
            continue
        c = ahead[rows]
        count = int(c.sum())
        if count == 0:
            continue
        first = np.repeat(np.arange(rows.start, rows.stop), c)
        second = first + 1 + (np.arange(count) - np.repeat(np.cumsum(c) - c, c))
        yield order[first], order[second], offset[second] - offset[first]


def neighbor_graph(
    graph: NetworkGraph,
    pos: Snaps,
    *,
    max_distance: float,
    max_pairs: int = DEFAULT_MAX_NEIGHBOR_PAIRS,
    batch_work: int | None = None,
) -> csr_matrix:
    """All pairs of positions with network distance ``<= max_distance``.

    ``pos`` should hold one entry per distinct snapped position (see
    ``distinct_positions``). The result is a symmetric sparse ``(n, n)``
    matrix whose stored entries are exactly those pairs, with their
    shortest-path distances. Pairs farther apart, or on disconnected
    components, are absent.

    ``batch_work`` sets the target size of the spatial search batches
    (default 2,000,000); it changes speed and memory, never results.
    """
    max_distance = float(max_distance)
    if not (np.isfinite(max_distance) and max_distance > 0):
        raise ValueError(f"max_distance must be a positive finite number, got {max_distance!r}")
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, (int, np.integer)) or max_pairs < 1:
        raise ValueError(f"max_neighbor_pairs must be a positive integer, got {max_pairs!r}")
    n = len(pos)
    if n < 2:
        return csr_matrix((n, n), dtype=float)

    parts = list(_pairs_through_vertices(graph, pos, max_distance, int(max_pairs), batch_work=batch_work))
    if parts:
        low = np.concatenate([p[0] for p in parts])
        high = np.concatenate([p[1] for p in parts])
        distance = np.concatenate([p[2] for p in parts]).astype(float)
        key = low * np.int64(n) + high
        order = np.argsort(key, kind="stable")
        low, high, distance, key = low[order], high[order], distance[order], key[order]
    else:
        low = high = key = np.empty(0, dtype=np.int64)
        distance = np.empty(0, dtype=float)
    del parts

    added_low, added_high, added_distance = [], [], []
    n_unique = len(low)
    for first, second, direct in _pairs_on_same_arc(pos, max_distance, int(max_pairs)):
        a = np.minimum(first, second)
        b = np.maximum(first, second)
        direct_key = a * np.int64(n) + b
        if len(key):
            at = np.searchsorted(key, direct_key)
            overlap = at < len(key)
            overlap[overlap] &= key[at[overlap]] == direct_key[overlap]
            if overlap.any():
                np.minimum.at(distance, at[overlap], direct[overlap])
        else:
            overlap = np.zeros(len(direct_key), dtype=bool)
        new = ~overlap
        n_new = int(new.sum())
        if n_unique + n_new > max_pairs:
            raise _too_many_pairs(n_unique + n_new, int(max_pairs), max_distance)
        if n_new:
            added_low.append(a[new])
            added_high.append(b[new])
            added_distance.append(direct[new])
            n_unique += n_new
    if added_low:
        low = np.concatenate([low, *added_low])
        high = np.concatenate([high, *added_high])
        distance = np.concatenate([distance, *added_distance])

    matrix = coo_matrix(
        (np.concatenate([distance, distance]), (np.concatenate([low, high]), np.concatenate([high, low]))),
        shape=(n, n),
    ).tocsr()
    if matrix.nnz != 2 * len(low):
        raise RuntimeError("sparse conversion dropped or merged neighbour pairs")
    return matrix
