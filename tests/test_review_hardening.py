"""Regression and failure-path tests for the v0.3.1 hardening."""
from pathlib import Path
import hashlib
import numpy as np
import pytest
from net_hdbscan.config import group_key
from net_hdbscan.publication import _hash
from net_hdbscan import network


def test_group_key_integral_float_matches_integer_but_not_string():
    assert group_key(1) == group_key(1.0) == "1"
    assert group_key(np.float64(2.0)) == "2"
    assert group_key("1.0") == "1.0"
    assert group_key(1.5) == "1.5"
    assert group_key(None) != "1"


def test_gdb_directory_integrity_and_failure_cases(tmp_path):
    folder = tmp_path / "sample.gdb"
    (folder / "nested").mkdir(parents=True)
    member = folder / "nested" / "part"
    member.write_bytes(b"original")
    digest = _hash(folder)
    assert _hash(folder) == digest
    member.write_bytes(b"modified")
    assert _hash(folder) != digest
    with pytest.raises(FileNotFoundError, match="not a regular file"):
        _hash(tmp_path / "missing.gdb")
    empty = tmp_path / "empty.gdb"
    empty.mkdir()
    with pytest.raises(ValueError, match="empty"):
        _hash(empty)


def test_shapefile_sidecar_changes_invalidate_checksum(tmp_path):
    shp = tmp_path / "roads.shp"
    shp.write_bytes(b"geometry")
    (tmp_path / "roads.shx").write_bytes(b"index")
    prj = tmp_path / "roads.prj"
    prj.write_bytes(b"epsg-a")
    digest = _hash(shp)
    prj.write_bytes(b"epsg-b")
    assert _hash(shp) != digest
    digest2 = _hash(shp)
    (tmp_path / "roads.dbf").write_bytes(b"attributes")
    assert _hash(shp) != digest2


def test_regular_file_checksum_is_standard_sha256(tmp_path):
    file = tmp_path / "points.gpkg"
    file.write_bytes(b"abc")
    assert _hash(file) == hashlib.sha256(b"abc").hexdigest()


def test_dijkstra_budget_exactly_includes_source_nodes():
    from math import isqrt
    B = network._SEARCH_BATCH_VALUES
    for L in (1, 16, 200, 1200, 5000):
        b = L + 2
        S = (isqrt(b * b + 4 * B) - b) // 2
        assert S >= 1
        assert S * (L + S + 2) <= B
        assert (S + 1) * (L + S + 3) > B


def test_binary_condensation_drops_isolated_underdense_stack():
    from net_hdbscan.sparse_hdbscan import _condense, _condense_simultaneous, _SimultaneousHierarchy
    weights = np.array([6, 1], dtype=np.int64)
    core = np.array([np.inf, np.inf])
    empty = np.empty(0, dtype=np.int64)
    tree1, n1 = _condense(2, empty, empty, np.empty(0), weights,
                           np.array([0, 1]), weights, core, 5)
    hierarchy = _SimultaneousHierarchy(children=(), distance=np.empty(0),
                                       size=weights, tops=np.array([0, 1]))
    tree2, n2 = _condense_simultaneous(2, hierarchy, weights, core, 5)
    assert n1 == n2 == 1
    assert not tree1.child_is_cluster.any()
    assert not tree2.child_is_cluster.any()


def test_impossible_one_source_dijkstra_budget_fails_before_search(monkeypatch):
    from shapely.geometry import LineString
    graph = network.build_network_graph([LineString([(0, 0), (10, 0)])])
    positions = network.snap_points(graph, np.array([[5.0, 0.0]]))
    monkeypatch.setattr(network, "_SEARCH_BATCH_VALUES", 1)
    with pytest.raises(MemoryError, match="even with one source"):
        list(network._pairs_through_vertices(graph, positions,
             max_distance=10.0, max_pairs=100))


def test_release_metadata_is_consistent():
    from net_hdbscan._version import __version__
    root = Path(__file__).resolve().parents[1]
    assert __version__ == "0.3.1"
    assert "## 0.3.1" in (root / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "<span>v0.3.1</span>" in (root / "docs/index.html").read_text(encoding="utf-8")
