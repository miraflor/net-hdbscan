import numpy as np
import pytest
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from shapely.geometry import LineString, MultiLineString

from conftest import grid_segments, irregular_network_data
from net_hdbscan.network import (
    build_network_graph,
    distinct_positions,
    neighbor_graph,
    round_significant,
    snap_points,
)


def test_lines_join_only_at_shared_vertices():
    shared = build_network_graph([LineString([(0, 0), (10, 0)]), LineString([(10, 0), (10, 10)])])
    assert shared.n_components == 1
    # Two lines that cross without a shared vertex (a bridge) stay separate.
    crossing = build_network_graph([LineString([(0, 5), (10, 5)]), LineString([(5, 0), (5, 10)])])
    assert crossing.n_components == 2


def test_multilinestring_parts_are_not_joined():
    graph = build_network_graph([MultiLineString([[(0, 0), (10, 0)], [(20, 0), (30, 0)]])])
    assert graph.n_arcs == 2
    assert graph.n_components == 2


def test_rounding_joins_nearly_identical_vertices():
    lines = [LineString([(100000.0, 0), (100010.0, 0)]), LineString([(100010.0 + 1e-9, 0), (100020.0, 0)])]
    assert build_network_graph(lines).n_components == 1
    assert build_network_graph(lines, vertex_digits=None).n_components == 2


def test_round_significant():
    values = np.array([0.0, 1234567.891234, -0.000123456789])
    assert np.allclose(round_significant(values, 4), [0.0, 1235000.0, -0.0001235])


def test_zero_length_and_duplicate_arcs_are_dropped():
    graph = build_network_graph([LineString([(0, 0), (0, 0), (5, 0)]), LineString([(5, 0), (0, 0)])])
    assert graph.n_arcs == 1
    assert graph.arc_length[0] == 5


def test_snapping_matches_brute_force():
    rng = np.random.default_rng(0)
    segments = shapely.linestrings(rng.uniform(0, 1000, size=(300, 2, 2)))
    graph = build_network_graph(segments, vertex_digits=None)
    xy = rng.uniform(0, 1000, size=(500, 2))
    snaps = snap_points(graph, xy)
    arcs = shapely.linestrings(np.stack([graph.vertex_xy[graph.arc_u], graph.vertex_xy[graph.arc_v]], axis=1))
    brute = shapely.distance(shapely.points(xy)[:, None], arcs[None, :]).min(axis=1)
    assert np.allclose(snaps.snap_distance, brute, atol=1e-9)
    # The snapped point lies on its arc at the reported offset.
    start = graph.vertex_xy[snaps.u]
    assert np.allclose(np.hypot(*(snaps.snapped_xy - start).T), snaps.offset, atol=1e-9)


def test_snapping_beyond_an_arc_end_gives_the_exact_vertex():
    graph = build_network_graph([LineString([(0, 0), (10, 0)])])
    snaps = snap_points(graph, np.array([[-3.0, 1.0], [12.0, -1.0], [4.0, 2.0]]))
    assert snaps.offset[0] == 0.0 and snaps.offset[1] == snaps.arc_length[1]
    assert np.array_equal(snaps.snapped_xy[:2], [[0.0, 0.0], [10.0, 0.0]])
    assert snaps.offset[2] == pytest.approx(4.0)


def test_one_vertex_reached_from_different_arcs_is_one_position():
    graph = build_network_graph([LineString([(0, 0), (10, 0)]), LineString([(10, 0), (10, 10)])])
    snaps = snap_points(graph, np.array([[11.0, -1.0], [12.0, -3.0], [5.0, 1.0]]))
    position, representative = distinct_positions(snaps)
    assert position[0] == position[1] != position[2]
    assert list(representative) == [0, 2]


def _brute_force_distances(graph, pos):
    """All-pairs shortest paths on a graph where every position is inserted on its arc."""
    n_v = graph.n_vertices
    node = np.full(len(pos), -1, dtype=np.int64)
    at_u = pos.offset == 0
    at_v = pos.offset == pos.arc_length
    node[at_u] = pos.u[at_u]
    node[at_v & ~at_u] = pos.v[at_v & ~at_u]
    interior = np.flatnonzero(node < 0)
    node[interior] = n_v + np.arange(len(interior))
    rows, cols, weights = [], [], []
    arc_key = {(int(u), int(v)): k for k, (u, v) in enumerate(zip(graph.arc_u, graph.arc_v))}
    on_arc = {}
    for i in interior:
        on_arc.setdefault(arc_key[(int(pos.u[i]), int(pos.v[i]))], []).append(i)
    for k, (u, v, length) in enumerate(zip(graph.arc_u, graph.arc_v, graph.arc_length)):
        chain = sorted(on_arc.get(k, []), key=lambda i: pos.offset[i])
        stops = [int(u)] + [int(node[i]) for i in chain] + [int(v)]
        offsets = [0.0] + [float(pos.offset[i]) for i in chain] + [float(length)]
        for a, b, oa, ob in zip(stops[:-1], stops[1:], offsets[:-1], offsets[1:]):
            rows += [a, b]
            cols += [b, a]
            weights += [ob - oa, ob - oa]
    size = n_v + len(interior)
    matrix = coo_matrix((weights, (rows, cols)), shape=(size, size)).tocsr()
    full = dijkstra(matrix, directed=False, indices=node)
    return full[:, node]


@pytest.mark.parametrize("max_distance", [60.0, 250.0, 700.0])
@pytest.mark.parametrize("batch_work", [None, 50])
def test_neighbor_graph_matches_brute_force(max_distance, batch_work):
    rng = np.random.default_rng(1)
    segments = grid_segments(n_lines=11, step=100.0)
    keep = rng.random(len(segments)) > 0.15  # remove some arcs: dead ends and detours
    graph = build_network_graph(segments[keep])
    xy = np.vstack([rng.uniform(0, 1000, size=(150, 2)), [[300.0, 300.0], [300.0, 300.0]], graph.vertex_xy[:5] + 0.5])
    snaps = snap_points(graph, xy)
    position, representative = distinct_positions(snaps)
    pos = snaps.subset(representative)
    found = neighbor_graph(graph, pos, max_distance=max_distance, batch_work=batch_work).toarray()
    expected = _brute_force_distances(graph, pos)
    np.fill_diagonal(expected, np.inf)
    within = expected <= max_distance
    assert np.array_equal(found != 0, within)
    assert np.allclose(found[within], expected[within], atol=1e-9)
    assert np.allclose(found, found.T)


def test_disconnected_roads_have_no_pairs():
    graph = build_network_graph([LineString([(0, 0), (100, 0)]), LineString([(0, 10), (100, 10)])])
    snaps = snap_points(graph, np.array([[50.0, 1.0], [50.0, 9.0]]))
    assert neighbor_graph(graph, snaps, max_distance=1000).nnz == 0


def test_pair_limit_is_enforced():
    graph = build_network_graph([LineString([(0, 0), (100, 0)])])
    snaps = snap_points(graph, np.column_stack([np.linspace(1, 99, 30), np.ones(30)]))
    with pytest.raises(ValueError, match="max_neighbor_pairs"):
        neighbor_graph(graph, snaps, max_distance=1000, max_pairs=10)


def test_invalid_max_distance():
    graph = build_network_graph([LineString([(0, 0), (100, 0)])])
    snaps = snap_points(graph, np.array([[1.0, 1.0], [2.0, 2.0]]))
    for bad in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            neighbor_graph(graph, snaps, max_distance=bad)


def test_positions_and_distances_do_not_depend_on_observation_order():
    """Position numbers, representatives and distances are the same, bit for bit, after reordering."""
    points, _, roads = irregular_network_data(seed=0)
    graph = build_network_graph(roads.geometry.to_numpy())
    xy = shapely.get_coordinates(points.geometry.to_numpy())
    xy = np.vstack([xy, xy[:20], graph.vertex_xy[:15] + 0.1])  # stacked observations and vertex positions
    results = []
    for order in [np.arange(len(xy)), np.random.default_rng(5).permutation(len(xy))]:
        snaps = snap_points(graph, xy[order])
        position, representative = distinct_positions(snaps)
        where = np.empty(len(xy), dtype=np.int64)
        where[order] = position  # position of each original observation
        rep = snaps.subset(representative)
        matrix = neighbor_graph(graph, rep, max_distance=600).toarray()
        results.append((where, rep.u, rep.v, rep.offset, rep.snapped_xy, matrix))
    for a, b in zip(results[0], results[1]):
        assert np.array_equal(a, b)


def test_vertex_digits_must_be_positive_integer_or_none():
    lines = [LineString([(0, 0), (10, 0)])]
    for bad in [0, -1, 1.5, True]:
        with pytest.raises(ValueError, match="vertex_digits"):
            build_network_graph(lines, vertex_digits=bad)
    build_network_graph(lines, vertex_digits=None)


def test_same_arc_batches_are_hard_bounded(monkeypatch):
    import net_hdbscan.network as network_module

    graph = build_network_graph([LineString([(0, 0), (100, 0)])])
    snaps = snap_points(graph, np.column_stack([np.arange(1.0, 13.0), np.zeros(12)]))
    monkeypatch.setattr(network_module, "_SAME_ARC_BATCH_PAIRS", 5)
    parts = list(network_module._pairs_on_same_arc(snaps, 1000.0, 1000))
    assert parts
    assert max(len(part[0]) for part in parts) <= 5
    pairs = {(int(i), int(j)) for a, b, _ in parts for i, j in zip(a, b)}
    assert len(pairs) == 12 * 11 // 2


def test_tiny_internal_chunks_preserve_neighbor_graph(monkeypatch):
    import net_hdbscan.network as network_module

    lines = []
    for k in range(8):
        angle = 2 * np.pi * k / 8
        lines.append(LineString([(0, 0), (100 * np.cos(angle), 100 * np.sin(angle))]))
    graph = build_network_graph(lines, vertex_digits=None)
    xy = []
    for k in range(8):
        angle = 2 * np.pi * k / 8
        for radius in [10, 20, 30, 40]:
            xy.append((radius * np.cos(angle), radius * np.sin(angle)))
    snaps = snap_points(graph, np.asarray(xy))
    _, representative = distinct_positions(snaps)
    positions = snaps.subset(representative)
    expected = neighbor_graph(graph, positions, max_distance=1000, max_pairs=100000)

    monkeypatch.setattr(network_module, "_CANDIDATE_CHUNK", 8)
    monkeypatch.setattr(network_module, "_SAME_ARC_BATCH_PAIRS", 7)
    got = neighbor_graph(graph, positions, max_distance=1000, max_pairs=100000)
    assert (expected != got).nnz == 0
