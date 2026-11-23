"""Regression tests for the follow-up review of v0.3.1."""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from net_hdbscan import HDBSCANConfig, cluster_files, cluster_files_by_column
from net_hdbscan.cli import app
from net_hdbscan.config import numeric_group_key
from net_hdbscan.publication import _hash

CONFIG = HDBSCANConfig(max_distance=600, min_cluster_size=15)
runner = CliRunner()


def _symlink_or_skip(target, link):
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available on this platform/account")


# --- A. shapefile members --------------------------------------------------

def test_shapefile_checksum_ignores_other_datasets_with_the_same_prefix(tmp_path):
    for suffix in (".shp", ".shx", ".dbf", ".prj"):
        (tmp_path / f"roads{suffix}").write_bytes(b"roads" + suffix.encode())
    digest = _hash(tmp_path / "roads.shp")
    # Another dataset, metadata and a spatial index do not change the data.
    for name in ("roads.old.shp", "roads.old.dbf", "roads.shp.xml", "roads.qix", "roadside.dbf"):
        (tmp_path / name).write_bytes(b"unrelated")
    assert _hash(tmp_path / "roads.shp") == digest
    # A real member still invalidates the checksum.
    (tmp_path / "roads.cpg").write_bytes(b"UTF-8")
    assert _hash(tmp_path / "roads.shp") != digest


# --- B. symbolic links -----------------------------------------------------

def test_symlinked_file_hashes_like_its_target(tmp_path):
    target = tmp_path / "points.gpkg"
    target.write_bytes(b"abc")
    link = tmp_path / "link.gpkg"
    _symlink_or_skip(target, link)
    assert _hash(link) == hashlib.sha256(b"abc").hexdigest()


def test_symlinked_inputs_are_accepted_and_resolved_paths_recorded(tmp_path, points, roads):
    pytest.importorskip("pyarrow")
    real = tmp_path / "data"
    real.mkdir()
    points.to_parquet(real / "points.parquet")
    roads.to_parquet(real / "roads.parquet")
    link = tmp_path / "points_link.parquet"
    _symlink_or_skip(real / "points.parquet", link)
    cluster_files(points_path=link, network_path=real / "roads.parquet",
                  output_dir=tmp_path / "out", config=CONFIG)
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_resolved_paths"]["points_path"] == str((real / "points.parquet").resolve())
    assert manifest["input_sha256"]["points_path"] == _hash(real / "points.parquet")


# --- C. CLI input errors ---------------------------------------------------

def _cli_args(tmp_path, points, roads, out):
    pytest.importorskip("pyarrow")
    points.to_parquet(tmp_path / "p.parquet")
    roads.to_parquet(tmp_path / "r.parquet")
    return ["cluster", "--points", str(tmp_path / "p.parquet"), "--network", str(tmp_path / "r.parquet"),
            "--output-dir", str(out), "--max-distance", "600", "--min-cluster-size", "15"]


def test_cli_input_error_is_one_line_without_traceback(tmp_path, points, roads):
    args = _cli_args(tmp_path, points, roads, tmp_path / "out")
    result = runner.invoke(app, args + ["--point-id-col", "nope"])
    assert result.exit_code == 1
    assert "Error:" in result.output and "'nope'" in result.output
    assert "Traceback (most recent call last)" not in result.output


def test_cli_grouped_error_names_the_failing_group(tmp_path, points, roads):
    args = _cli_args(tmp_path, points, roads, tmp_path / "out")
    result = runner.invoke(app, args + ["--group-col", "sector", "--max-neighbor-pairs", "5"])
    assert result.exit_code == 1
    assert "group 'a' failed" in result.output
    assert "max_neighbor_pairs" in result.output
    assert "Traceback (most recent call last)" not in result.output


# --- D. numeric group keys -------------------------------------------------

def test_numeric_group_key_spelling():
    assert numeric_group_key("1") == numeric_group_key("1.0") == numeric_group_key("1e0") == "1"
    assert numeric_group_key("1.5") == "1.5"
    assert numeric_group_key("12345678901234567") == "12345678901234567"  # exact, no float rounding
    assert numeric_group_key("north") == "north"
    assert numeric_group_key("nan") == "nan"


def _float_coded_points(tmp_path, points):
    data = points.copy()
    data["code"] = np.where(np.arange(len(data)) % 2 == 0, 1.0, 2.0)
    data.loc[0, "code"] = np.nan  # a missing value keeps the column float64
    path = tmp_path / "coded.parquet"
    data.to_parquet(path)
    return path


def test_float_group_column_accepts_keys_written_as_floats(tmp_path, points, roads):
    pytest.importorskip("pyarrow")
    roads.to_parquet(tmp_path / "r.parquet")
    params = tmp_path / "params.csv"
    params.write_text("code,min_cluster_size\n1.0,12\n2,14\n")
    from net_hdbscan import read_group_params, read_group_universe
    universe = tmp_path / "universe.csv"
    universe.write_text("code\n1.0\n2\n3.0\n")
    summary = cluster_files_by_column(
        points_path=_float_coded_points(tmp_path, points), network_path=tmp_path / "r.parquet",
        output_dir=tmp_path / "out", group_col="code", config=CONFIG,
        group_params=read_group_params(params, "code"),
        group_universe=read_group_universe(universe, "code"),
    )
    by_group = summary.set_index("group")
    assert list(by_group.index) == ["1", "2", "3"]
    assert by_group.loc["1", "min_cluster_size"] == 12 and by_group.loc["2", "min_cluster_size"] == 14
    assert by_group.loc["3", "n_points"] == 0
    assert bool(by_group["declared_in_universe"].all())
    assert (tmp_path / "out" / "points" / "group_1.parquet").exists()


def test_two_keys_for_one_numeric_value_are_rejected(tmp_path, points, roads):
    pytest.importorskip("pyarrow")
    roads.to_parquet(tmp_path / "r.parquet")
    with pytest.raises(ValueError, match="more than once"):
        cluster_files_by_column(
            points_path=_float_coded_points(tmp_path, points), network_path=tmp_path / "r.parquet",
            output_dir=tmp_path / "out", group_col="code", config=CONFIG,
            group_params={"1": {"min_cluster_size": 12}, "1.0": {"min_cluster_size": 13}},
        )


def test_text_group_column_keeps_keys_as_written(tmp_path, points, roads):
    pytest.importorskip("pyarrow")
    roads.to_parquet(tmp_path / "r.parquet")
    data = points.copy()
    data["code"] = np.where(np.arange(len(data)) % 2 == 0, "1.0", "2.0")
    data.to_parquet(tmp_path / "text.parquet")
    with pytest.raises(ValueError, match="match no group"):
        cluster_files_by_column(
            points_path=tmp_path / "text.parquet", network_path=tmp_path / "r.parquet",
            output_dir=tmp_path / "out", group_col="code", config=CONFIG,
            group_params={"1": {"min_cluster_size": 12}},
        )


# --- Exactness and compatibility beyond the reviewer's original cases -------

def test_numeric_decimal_preserves_large_int64_and_scientific_spelling():
    assert numeric_group_key("9007199254740993.0") == "9007199254740993"
    assert numeric_group_key("9.007199254740993e15") == "9007199254740993"
    assert numeric_group_key("9223372036854775807.0") == "9223372036854775807"
    assert numeric_group_key("1e-05") == numeric_group_key("0.0000100") == "0.00001"
    assert numeric_group_key("0.100") == "0.1"
    assert numeric_group_key("-0.0") == "0"


def test_int64_codes_keep_exact_identity_through_csv_group_overrides(tmp_path, points, roads):
    pytest.importorskip("pyarrow")
    from net_hdbscan import read_group_params
    high = 9007199254740993
    data = points.copy()
    data["code"] = pd.Series([high if i % 2 == 0 else high + 1 for i in range(len(data))], dtype="Int64")
    data.to_parquet(tmp_path / "points.parquet")
    roads.to_parquet(tmp_path / "roads.parquet")
    params = tmp_path / "params.csv"
    params.write_text(f"code,min_cluster_size\n{high}.0,12\n{high + 1},14\n")
    overrides = read_group_params(params, "code")
    result = cluster_files_by_column(points_path=tmp_path / "points.parquet",
        network_path=tmp_path / "roads.parquet", output_dir=tmp_path / "out",
        group_col="code", config=CONFIG, group_params=overrides).set_index("group")
    assert set(result.index) == {str(high), str(high + 1)}
    assert result.loc[str(high), "min_cluster_size"] == 12
    assert result.loc[str(high + 1), "min_cluster_size"] == 14


def test_exact_shapefile_members_and_linked_shp_from_other_folder(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    for ext in (".shp", ".shx", ".dbf", ".prj"):
        (real / ("roads" + ext)).write_bytes(ext.encode())
    first = _hash(real / "roads.shp")
    (real / "roads.old.shp").write_bytes(b"other")
    (real / "roads.shp.xml").write_bytes(b"metadata")
    assert _hash(real / "roads.shp") == first
    link = tmp_path / "alias.shp"
    _symlink_or_skip(real / "roads.shp", link)
    assert _hash(link) == _hash(real / "roads.shp")
    (real / "roads.dbf").write_bytes(b"changed")
    assert _hash(link) != first


def test_directory_dataset_nested_symlink_is_not_silently_skipped(tmp_path):
    root = tmp_path / "data.gdb"
    root.mkdir()
    external = tmp_path / "elsewhere"
    external.mkdir()
    (external / "important").write_bytes(b"abc")
    _symlink_or_skip(external, root / "nested")
    with pytest.raises(ValueError, match="nested directory symlink"):
        _hash(root)


def test_budget_failure_is_expected_value_error_and_no_oversized_search(monkeypatch):
    from shapely.geometry import LineString
    from net_hdbscan import network
    g = network.build_network_graph([LineString([(0,0),(10,0)])])
    snaps = network.snap_points(g, np.array([[5.0,0.0]]))
    monkeypatch.setattr(network, "_SEARCH_BATCH_VALUES", 1)
    with pytest.raises(network.LocalSearchBudgetExceeded):
        list(network._pairs_through_vertices(g, snaps, max_distance=10.0, max_pairs=100))
    assert issubclass(network.LocalSearchBudgetExceeded, ValueError)


def test_search_budget_error_names_bounding_box_and_remedies(monkeypatch):
    from shapely.geometry import LineString
    from net_hdbscan import network

    graph = network.build_network_graph([LineString([(0, 0), (10, 0)])])
    snaps = network.snap_points(graph, np.array([[5.0, 0.0]]))
    monkeypatch.setattr(network, "_SEARCH_BATCH_VALUES", 1)
    with pytest.raises(network.LocalSearchBudgetExceeded) as error:
        list(network._pairs_through_vertices(graph, snaps, max_distance=10.0, max_pairs=100))
    message = str(error.value)
    assert "2 network vertices in the expanded search bounding box" in message
    assert "5 float64 values" in message
    assert "fixed budget is 1 values" in message
    assert "smaller max_distance" in message
    assert "_SEARCH_BATCH_VALUES" not in message
