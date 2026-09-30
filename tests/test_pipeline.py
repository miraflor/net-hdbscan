import inspect
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from conftest import CRS, irregular_network_data
from net_hdbscan import HDBSCANConfig, cluster_files, cluster_files_by_column, cluster_geodataframes
from net_hdbscan.config import read_group_params

CONFIG = HDBSCANConfig(max_distance=600, min_cluster_size=15)
BLOB_A = [f"q{i:04d}" for i in range(120)]
BLOB_B = [f"q{i:04d}" for i in range(120, 240)]


def ids_by_point(frame):
    return frame.set_index("point_id")["cluster_id"]


def same_partition(a, b):
    a = np.asarray(a, dtype=object)
    b = np.asarray(b, dtype=object)
    a_noise = pd.isna(a)
    b_noise = pd.isna(b)
    if not np.array_equal(a_noise, b_noise):
        return False
    keep = ~a_noise
    return len(set(zip(a[keep], b[keep]))) == len(set(a[keep])) == len(set(b[keep]))


def test_two_blobs_become_two_clusters(points, roads):
    out = cluster_geodataframes(points, roads, CONFIG)
    ids = ids_by_point(out.points)
    assert ids.loc[BLOB_A].nunique() == 1 and ids.loc[BLOB_B].nunique() == 1
    assert ids.loc[BLOB_A].iloc[0] != ids.loc[BLOB_B].iloc[0]
    assert out.summary["n_clusters"] == 2
    expected = {"cluster_id", "is_noise", "membership", "core_distance", "snap_distance", "snapped_x", "snapped_y"}
    assert expected <= set(out.points.columns)
    assert (out.points["is_noise"] == out.points["cluster_id"].isna()).all()
    assert out.points["membership"].between(0, 1).all()
    assert out.points["point_id"].is_monotonic_increasing
    assert out.hierarchy["selected"].sum() == 2


def test_results_do_not_depend_on_row_order(points, roads):
    first = ids_by_point(cluster_geodataframes(points, roads, CONFIG).points)
    shuffled = points.sample(frac=1.0, random_state=4).reset_index(drop=True)
    second = ids_by_point(cluster_geodataframes(shuffled, roads, CONFIG).points)
    pd.testing.assert_series_equal(first.sort_index(), second.sort_index())


def test_partition_does_not_depend_on_point_id_names(points, roads):
    data = points.copy()
    data["row_key"] = np.arange(len(data))
    first = cluster_geodataframes(data, roads, CONFIG).points.sort_values("row_key")
    renamed = data.copy()
    # Reverse the lexical ordering that the pipeline uses internally.
    renamed["point_id"] = [f"renamed_{i:05d}" for i in range(len(renamed), 0, -1)]
    second = cluster_geodataframes(renamed, roads, CONFIG).points.sort_values("row_key")
    assert same_partition(first["cluster_id"], second["cluster_id"])
    assert np.allclose(first["membership"], second["membership"])


@pytest.mark.parametrize("min_cluster_size, min_samples", [(15, 8), (20, 10)])
def test_renaming_point_ids_changes_no_result_on_an_irregular_network(min_cluster_size, min_samples):
    """Renaming IDs must not change the partition, membership or core distances, bit for bit.

    This data detects regressions where tie ordering or position numbering
    accidentally follows point IDs.
    """
    points, roads = irregular_network_data(seed=0)
    config = HDBSCANConfig(max_distance=900, min_cluster_size=min_cluster_size, min_samples=min_samples)
    first = cluster_geodataframes(points, roads, config).points
    renamed = points.copy()
    renamed["point_id"] = [f"r{k:04d}" for k in range(len(points), 0, -1)]  # reverses the ID order
    second = cluster_geodataframes(renamed, roads, config).points
    second["point_id"] = second["point_id"].map(dict(zip(renamed["point_id"], points["point_id"])))
    first = first.set_index("point_id").sort_index()
    second = second.set_index("point_id").sort_index()
    assert same_partition(first["cluster_id"], second["cluster_id"])
    assert np.array_equal(first["membership"].to_numpy(), second["membership"].to_numpy())
    assert np.array_equal(first["core_distance"].to_numpy(), second["core_distance"].to_numpy())


def test_singleton_noise_is_one_id_per_position(points, roads):
    lonely = gpd.GeoDataFrame(
        {"point_id": ["z1", "z2"], "sector": ["a", "a"]}, geometry=[Point(1900, 150), Point(1900, 150)], crs=CRS
    )
    data = pd.concat([points, lonely], ignore_index=True)
    config = CONFIG.replace(noise_policy="singleton")
    out = cluster_geodataframes(data, roads, config)
    ids = ids_by_point(out.points)
    assert ids.notna().all()
    assert ids["z1"] == ids["z2"]  # net-hdbscan 0.1 failed here with a RuntimeError


def test_snap_distance_limit(points, roads):
    far = gpd.GeoDataFrame({"point_id": ["far"], "sector": ["a"]}, geometry=[Point(2040, 2040)], crs=CRS)
    data = pd.concat([points, far], ignore_index=True)
    with pytest.raises(ValueError, match="max_snap_distance"):
        cluster_geodataframes(data, roads, CONFIG.replace(max_snap_distance=10))
    out = cluster_geodataframes(data, roads, CONFIG.replace(max_snap_distance=100))
    assert out.points.set_index("point_id").loc["far", "snap_distance"] == pytest.approx(40.0 * np.sqrt(2))  # nearest road point is the grid corner


def test_input_checks(points, roads):
    with pytest.raises(ValueError, match="reserved"):
        cluster_geodataframes(points.assign(cluster_id="x"), roads, CONFIG)
    with pytest.raises(ValueError, match="duplicate"):
        cluster_geodataframes(pd.concat([points, points.iloc[:1]], ignore_index=True), roads, CONFIG)
    with pytest.raises(ValueError, match="projected"):
        cluster_geodataframes(points, roads.to_crs("EPSG:4326"), CONFIG)
    with pytest.raises(ValueError):
        HDBSCANConfig(max_distance=100, core_distance_floor=100)


def test_python_api_has_no_boundary_parameter():
    assert "boundary" not in inspect.signature(cluster_geodataframes).parameters


def test_all_supplied_points_are_used(points, roads):
    extra = gpd.GeoDataFrame(
        {"point_id": ["far"], "sector": ["a"]},
        geometry=[Point(5000, 5000)],
        crs=CRS,
    )
    data = pd.concat([points, extra], ignore_index=True)
    out = cluster_geodataframes(data, roads, CONFIG)
    assert len(out.points) == len(data)
    assert out.summary["n_points_input"] == len(data)
    assert "far" in set(out.points["point_id"])
    assert set(BLOB_B) <= set(out.points["point_id"])


def test_single_run_files(files, tmp_path):
    out_dir = tmp_path / "out"
    cluster_files(points_path=files["points"], network_path=files["roads"], output_dir=out_dir, config=CONFIG)
    for name in ["clustered_points.parquet", "cluster_hierarchy.parquet", "summary.csv", "distance_trace.csv", "manifest.json"]:
        assert (out_dir / name).exists()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["clustering"]["max_distance"] == 600 and manifest["clustering"]["effective_min_samples"] == 15
    assert len(pd.read_csv(out_dir / "summary.csv")) == 1
    with pytest.raises(FileExistsError):
        cluster_files(points_path=files["points"], network_path=files["roads"], output_dir=out_dir, config=CONFIG)
    cluster_files(
        points_path=files["points"], network_path=files["roads"], output_dir=out_dir,
        config=CONFIG, force=True,
    )


def test_grouped_run_with_overrides(points, roads, tmp_path):
    pytest.importorskip("pyarrow")
    data = points.copy()
    data.loc[data.index[-4:-2], "sector"] = None
    data.loc[data.index[-2:], "sector"] = ""
    data.loc[data.index[-4:], "geometry"] = [Point(1800, 200), Point(1810, 200), Point(200, 1800), Point(210, 1800)]
    # The same point ID may appear in two groups.
    extra = data.iloc[[0]].copy()
    extra["sector"] = "b"
    data = pd.concat([data, extra], ignore_index=True)
    paths = {"points": tmp_path / "p.parquet", "roads": tmp_path / "r.parquet"}
    data.to_parquet(paths["points"])
    roads.to_parquet(paths["roads"])
    params = tmp_path / "params.csv"
    params.write_text("sector,min_cluster_size,min_samples,duplicates,allow_single_cluster\nb,30,10,once,\n__null__,2,1,,true\n")
    overrides = read_group_params(params, "sector")
    assert overrides == {
        "b": {"min_cluster_size": 30, "min_samples": 10, "duplicates": "once"},
        "__null__": {"min_cluster_size": 2, "min_samples": 1, "allow_single_cluster": True},
    }

    out_dir = tmp_path / "out"
    summary = cluster_files_by_column(
        points_path=paths["points"], network_path=paths["roads"], output_dir=out_dir,
        group_col="sector", config=CONFIG, group_params=overrides, missing_group_policy="include",
    )
    assert summary["group"].tolist() == ["__blank__", "a", "b", "__null__"]
    row = summary.set_index("group")
    assert row.loc["b", "min_cluster_size"] == 30 and row.loc["b", "duplicates"] == "once"
    assert row.loc["a", "min_cluster_size"] == 15
    assert row.loc["__null__", "n_clusters"] == 1
    for folder in ["points", "hierarchy"]:
        names = sorted(p.name for p in (out_dir / folder).iterdir())
        assert names == ["group___blank__.parquet", "group___null__.parquet", "group_a.parquet", "group_b.parquet"]
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["group_params"]["b"]["min_cluster_size"] == 30


def test_group_parameter_file_errors(tmp_path, files):
    bad = tmp_path / "bad.csv"
    bad.write_text("sector,eps\na,3\n")
    with pytest.raises(ValueError, match="unknown columns"):
        read_group_params(bad, "sector")
    unknown = tmp_path / "unknown.csv"
    unknown.write_text("sector,min_cluster_size\nzz,5\n")
    with pytest.raises(ValueError, match="match no group"):
        cluster_files_by_column(
            points_path=files["points"], network_path=files["roads"], output_dir=tmp_path / "o",
            group_col="sector", config=CONFIG, group_params=read_group_params(unknown, "sector"),
        )


def test_group_with_far_points_is_retained(points, roads, tmp_path):
    pytest.importorskip("pyarrow")
    data = points.copy()
    data.loc[data.index[:3], "sector"] = "outside"
    data.loc[data.index[:3], "geometry"] = [Point(5000, 5000)] * 3
    paths = {"points": tmp_path / "p.parquet", "roads": tmp_path / "r.parquet"}
    data.to_parquet(paths["points"])
    roads.to_parquet(paths["roads"])
    summary = cluster_files_by_column(
        points_path=paths["points"], network_path=paths["roads"], output_dir=tmp_path / "out",
        group_col="sector", config=CONFIG,
    )
    row = summary.set_index("group").loc["outside"]
    assert row["n_points_input"] == 3 and row["n_points"] == 3
    assert len(gpd.read_parquet(tmp_path / "out" / "points" / "group_outside.parquet")) == 3


def test_group_parameter_file_saved_by_excel(tmp_path):
    path = tmp_path / "excel.csv"
    path.write_bytes("\ufeffsector,max_distance\r\na,750\r\n".encode("utf-8"))
    assert read_group_params(path, "sector") == {"a": {"max_distance": 750.0}}
