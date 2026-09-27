"""Release-level tests for net-hdbscan."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import cdist
from shapely.geometry import Point

import net_hdbscan
from conftest import CRS
from net_hdbscan import HDBSCANConfig, cluster_files, cluster_files_by_column, cluster_geodataframes
from net_hdbscan.sparse_hdbscan import core_distances

CONFIG = HDBSCANConfig(max_distance=600, min_cluster_size=15)
BLOB_A = [f"q{i:04d}" for i in range(120)]


# --- module path and names -------------------------------------------------


def test_the_function_does_not_hide_the_module():
    module = importlib.import_module("net_hdbscan.sparse_hdbscan")
    assert callable(net_hdbscan.hdbscan) and net_hdbscan.hdbscan is module.hdbscan
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("net_hdbscan.hdbscan")


def test_public_network_names_are_exposed():
    assert net_hdbscan.NetworkGraph.__name__ == "NetworkGraph"
    assert callable(net_hdbscan.build_network_graph)


# --- core distances: the per-row path for long rows ------------------------


@pytest.mark.parametrize("min_samples", [2, 5, 20, 60])
def test_core_distances_on_long_rows_equal_expanded_observations(min_samples):
    """Rows of about 200 entries use the partial-selection path; compare with a brute-force reference."""
    rng = np.random.default_rng(min_samples)
    # Distinct integer coordinates: many equal distances, but no two positions at distance 0
    # (a dense matrix does not store zeros, and real positions at one place are merged).
    xy = np.unique(np.rint(rng.uniform(0, 40, size=(230, 2))), axis=0)
    w = rng.integers(1, 5, len(xy))
    d = cdist(xy, xy)
    graph = csr_matrix(d)
    graph.setdiag(0.25)  # stored diagonal entries must be ignored
    expanded = np.repeat(xy, w, axis=0)
    owner = np.repeat(np.arange(len(xy)), w)
    reference = np.sort(cdist(expanded, expanded), axis=1)[:, min_samples - 1][np.searchsorted(owner, np.arange(len(xy)))]
    reference = np.where(w >= min_samples, 0.0, reference)
    assert np.array_equal(core_distances(graph, w, min_samples), reference)


def test_core_distances_mix_long_and_short_rows():
    rng = np.random.default_rng(9)
    n = 3000
    rows = np.concatenate([np.zeros(500, int), rng.integers(1, n, 6000)])
    cols = np.concatenate([rng.integers(1, n, 500), rng.integers(1, n, 6000)])
    keep = rows != cols
    dist = rng.integers(1, 30, len(rows)).astype(float)[keep]
    graph = coo_matrix((np.concatenate([dist, dist]), (np.concatenate([rows[keep], cols[keep]]), np.concatenate([cols[keep], rows[keep]]))), shape=(n, n)).tocsr()
    graph.sum_duplicates()
    w = rng.integers(1, 3, n)
    dense = graph.toarray()
    dense[dense == 0] = np.inf
    np.fill_diagonal(dense, np.inf)
    for k in [2, 4, 9]:
        expected = np.full(n, np.inf)
        for r in range(n):
            if w[r] >= k:
                expected[r] = 0.0
                continue
            order = np.argsort(dense[r], kind="stable")
            cum = np.cumsum(w[order])
            j = np.searchsorted(cum, k - w[r])
            if j < n and np.isfinite(dense[r, order[j]]):
                expected[r] = dense[r, order[j]]
        assert np.array_equal(core_distances(graph, w, k), expected)


# --- per-cluster table -----------------------------------------------------


def test_cluster_table_describes_every_cluster(points, boundary, roads):
    out = cluster_geodataframes(points, boundary, roads, CONFIG.replace(noise_policy="singleton"))
    table, labelled = out.clusters, out.points
    assert list(table.columns) == ["cluster_id", "n_points", "n_positions", "stability", "birth_distance", "singleton"]
    counts = labelled["cluster_id"].value_counts()
    assert table.set_index("cluster_id")["n_points"].to_dict() == counts.to_dict()
    selected = table.loc[~table["singleton"]]
    assert len(selected) == out.summary["n_clusters"] and selected["stability"].notna().all()
    assert table.loc[table["singleton"], "n_positions"].eq(1).all()
    noise_ids = set(labelled.loc[labelled["is_noise"], "cluster_id"])
    assert set(table.loc[table["singleton"], "cluster_id"]) == noise_ids


def test_cluster_table_files(files, tmp_path, points, boundary, roads):
    cluster_files(points_path=files["points"], boundary_path=files["boundary"], network_path=files["roads"], output_dir=tmp_path / "one", config=CONFIG)
    table = pd.read_parquet(tmp_path / "one" / "cluster_table.parquet")
    assert len(table) == 2 and table["n_points"].sum() == (~gpd.read_parquet(tmp_path / "one" / "clustered_points.parquet")["is_noise"]).sum()
    data = points.copy()
    data.loc[:2, "sector"] = "outside"
    data.loc[:2, "geometry"] = [Point(5000, 5000)] * 3
    data.to_parquet(tmp_path / "grouped.parquet")
    cluster_files_by_column(
        points_path=tmp_path / "grouped.parquet", boundary_path=files["boundary"], network_path=files["roads"],
        output_dir=tmp_path / "many", group_col="sector", config=CONFIG,
    )
    names = sorted(p.name for p in (tmp_path / "many" / "clusters").iterdir())
    assert names == ["group_a.parquet", "group_b.parquet", "group_outside.parquet"]
    empty = pd.read_parquet(tmp_path / "many" / "clusters" / "group_outside.parquet")
    assert len(empty) == 0 and list(empty.columns) == ["cluster_id", "n_points", "n_positions", "stability", "birth_distance", "singleton"]


# --- optional boundary -----------------------------------------------------


def test_boundary_is_optional(points, boundary, roads, tmp_path):
    far = gpd.GeoDataFrame({"point_id": ["far"], "sector": ["a"]}, geometry=[Point(2040, 2040)], crs=CRS)
    data = pd.concat([points, far], ignore_index=True)
    with_boundary = cluster_geodataframes(data, boundary, roads, CONFIG)
    without = cluster_geodataframes(data, None, roads, CONFIG)
    assert without.summary["n_points"] == len(data) == with_boundary.summary["n_points"]
    small = gpd.GeoDataFrame(geometry=[boundary.geometry.iloc[0].buffer(-1000)], crs=CRS)
    assert cluster_geodataframes(data, small, roads, CONFIG).summary["n_points"] < len(data)
    pytest.importorskip("pyarrow")
    data.to_parquet(tmp_path / "p.parquet")
    roads.to_parquet(tmp_path / "r.parquet")
    out = cluster_files(points_path=tmp_path / "p.parquet", network_path=tmp_path / "r.parquet", output_dir=tmp_path / "out", config=CONFIG)
    assert out.summary["n_points"] == len(data)


# --- optional extras -------------------------------------------------------


def test_missing_pyarrow_gives_a_clear_message(monkeypatch):
    from net_hdbscan import io

    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(ImportError, match=r"net-hdbscan\[files\]"):
        io._require_pyarrow()


def test_missing_cli_extra_gives_a_clear_message(monkeypatch):
    from net_hdbscan import _cli_entry

    monkeypatch.setitem(sys.modules, "typer", None)
    monkeypatch.delitem(sys.modules, "net_hdbscan.cli", raising=False)
    with pytest.raises(SystemExit, match=r"net-hdbscan\[cli\]"):
        _cli_entry.main()


def test_python_dash_m_runs_the_command():
    src = str(Path(net_hdbscan.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
    result = subprocess.run([sys.executable, "-m", "net_hdbscan", "--version"], capture_output=True, text=True, env=env)
    assert result.returncode == 0 and result.stdout.strip() == net_hdbscan.__version__


def test_file_pipeline_checks_pyarrow_before_reading(monkeypatch, tmp_path):
    import net_hdbscan.pipeline as pipeline

    seen = []

    def missing():
        seen.append("pyarrow")
        raise ImportError("missing pyarrow")

    def should_not_read(*args, **kwargs):
        seen.append("read")
        raise AssertionError("read_vector should not be reached")

    monkeypatch.setattr(pipeline, "_require_pyarrow", missing)
    monkeypatch.setattr(pipeline, "read_vector", should_not_read)
    with pytest.raises(ImportError, match="missing pyarrow"):
        cluster_files(
            points_path="points.gpkg",
            network_path="network.gpkg",
            output_dir=tmp_path / "out",
            config=CONFIG,
        )
    assert seen == ["pyarrow"]


def test_manifest_records_layers_crs_unit_and_graph_counts(files, tmp_path):
    out_dir = tmp_path / "manifest"
    cluster_files(
        points_path=files["points"],
        boundary_path=files["boundary"],
        network_path=files["roads"],
        output_dir=out_dir,
        config=CONFIG,
    )
    import json

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["inputs"]["points_layer"] is None
    assert manifest["inputs"]["boundary_layer"] is None
    assert manifest["inputs"]["network_layer"] is None
    analysis = manifest["analysis"]
    assert analysis["crs"]
    assert analysis["network_vertices"] > 0
    assert analysis["network_arcs"] > 0
    assert analysis["network_components"] > 0
