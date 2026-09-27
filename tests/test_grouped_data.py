"""Missing group values, declared group lists, overrides and grouped-run diagnostics."""

import json
import sys

import geopandas as gpd
import pandas as pd
import pytest

from net_hdbscan import HDBSCANConfig, NeighborPairLimitError, cluster_files_by_column, read_group_universe

CONFIG = HDBSCANConfig(max_distance=600, min_cluster_size=15)


def write_inputs(tmp_path, points, boundary, roads):
    pytest.importorskip("pyarrow")
    paths = {"points": tmp_path / "p.parquet", "boundary": tmp_path / "b.parquet", "roads": tmp_path / "r.parquet"}
    points.to_parquet(paths["points"])
    boundary.to_parquet(paths["boundary"])
    roads.to_parquet(paths["roads"])
    return paths


def with_missing_groups(points):
    data = points.copy()
    data.loc[0:4, "sector"] = None
    data.loc[5:9, "sector"] = ""
    data.loc[10:14, "sector"] = "   "  # whitespace only counts as blank
    return data


def run(paths, out_dir, **kwargs):
    return cluster_files_by_column(
        points_path=paths["points"], boundary_path=paths["boundary"], network_path=paths["roads"],
        output_dir=out_dir, group_col="sector", config=kwargs.pop("config", CONFIG), **kwargs,
    )


def test_missing_group_values_are_excluded_by_default(points, boundary, roads, tmp_path):
    paths = write_inputs(tmp_path, with_missing_groups(points), boundary, roads)
    summary = run(paths, tmp_path / "out")
    assert summary["group"].tolist() == ["a", "b"]
    on_disk = pd.read_csv(tmp_path / "out" / "summary.csv", dtype={"group": str})
    assert on_disk["n_missing_group_input"].eq(15).all() and on_disk["missing_group_policy"].eq("exclude").all()
    assert sorted(p.name for p in (tmp_path / "out" / "points").iterdir()) == ["group_a.parquet", "group_b.parquet"]
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["n_missing_group_input"] == 15


def test_missing_group_values_can_be_included_or_rejected(points, boundary, roads, tmp_path):
    paths = write_inputs(tmp_path, with_missing_groups(points), boundary, roads)
    summary = run(paths, tmp_path / "included", missing_group_policy="include").set_index("group")
    assert list(summary.index) == ["__blank__", "a", "b", "__null__"]
    assert summary.loc["__blank__", "n_points_input"] == 10 and summary.loc["__null__", "n_points_input"] == 5
    with pytest.raises(ValueError, match="missing/blank"):
        run(paths, tmp_path / "rejected", missing_group_policy="error")


def test_group_universe_adds_absent_groups_and_flags_undeclared_ones(points, boundary, roads, tmp_path):
    paths = write_inputs(tmp_path, points, boundary, roads)
    universe_file = tmp_path / "universe.csv"
    universe_file.write_text("sector\nb\nzz\n")
    summary = run(paths, tmp_path / "out", group_universe=read_group_universe(universe_file, "sector")).set_index("group")
    assert list(summary.index) == ["b", "zz", "a"]  # declared order first, then observed extras
    assert summary.loc["zz", "n_points_input"] == 0 and summary.loc["zz", "n_points"] == 0
    assert summary["declared_in_universe"].tolist() == [True, True, False]
    assert len(gpd.read_parquet(tmp_path / "out" / "points" / "group_zz.parquet")) == 0
    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["groups_outside_universe"] == ["a"]


@pytest.mark.parametrize(
    "text, message",
    [
        ('sector\nb\n""\n', "null/blank"),  # a quoted empty value (pandas skips truly empty lines)
        ("sector\nb\nb\n", "more than once"),
        ("other\nb\n", "must be"),
        ("sector,x\nb,1\n", "exactly one column"),
    ],
)
def test_group_universe_file_checks(tmp_path, text, message):
    path = tmp_path / "universe.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        read_group_universe(path, "sector")


def test_a_failing_group_is_named_and_keeps_its_error_type(points, boundary, roads, tmp_path):
    paths = write_inputs(tmp_path, points, boundary, roads)
    tiny = HDBSCANConfig(max_distance=3000, min_cluster_size=15, max_neighbor_pairs=500)
    with pytest.raises(ValueError) as caught:
        run(paths, tmp_path / "out", config=tiny)
    if sys.version_info >= (3, 11):
        assert isinstance(caught.value, NeighborPairLimitError)
        assert any("group 'a' failed" in note for note in caught.value.__notes__)
    else:
        assert "group 'a' failed" in str(caught.value)


def test_literal_reserved_group_names_do_not_become_missing(points, boundary, roads, tmp_path):
    data = points.copy()
    data.loc[0:4, "sector"] = "__null__"
    data.loc[5:9, "sector"] = "__blank__"
    data.loc[10:14, "sector"] = None
    paths = write_inputs(tmp_path, data, boundary, roads)
    summary = run(paths, tmp_path / "out", missing_group_policy="include").set_index("group")
    assert "literal:__null__" in summary.index
    assert "literal:__blank__" in summary.index
    assert "__null__" in summary.index
    assert summary.loc["literal:__null__", "n_points_input"] == 5
    assert summary.loc["literal:__blank__", "n_points_input"] == 5
    assert summary.loc["__null__", "n_points_input"] == 5
    names = {p.name for p in (tmp_path / "out" / "points").iterdir()}
    assert "group___null__.parquet" in names
    assert "group_literal%3A__null__.parquet" in names


def test_case_only_group_names_are_rejected_for_windows_safe_outputs(points, boundary, roads, tmp_path):
    data = points.copy()
    data.loc[data.index[: len(data) // 2], "sector"] = "A"
    data.loc[data.index[len(data) // 2 :], "sector"] = "a"
    paths = write_inputs(tmp_path, data, boundary, roads)
    with pytest.raises(ValueError, match="case-insensitive"):
        run(paths, tmp_path / "out")


def test_excluded_missing_groups_are_removed_before_snapping(monkeypatch, points, boundary, roads, tmp_path):
    import net_hdbscan.pipeline as pipeline

    data = with_missing_groups(points)
    paths = write_inputs(tmp_path, data, boundary, roads)
    original = pipeline._snap_inside
    seen = []

    def wrapped(network, inside, vertex_digits):
        seen.append(len(inside))
        return original(network, inside, vertex_digits)

    monkeypatch.setattr(pipeline, "_snap_inside", wrapped)
    run(paths, tmp_path / "out")
    assert seen == [len(data) - 15]


def test_group_parameter_reserved_tokens_can_be_escaped_as_literals(tmp_path):
    from net_hdbscan.config import BLANK_GROUP_KEY, NULL_GROUP_KEY, normalize_group_params, read_group_params

    path = tmp_path / "params.csv"
    path.write_text(
        "sector,max_distance\n"
        "__null__,1000\n"
        "__blank__,1100\n"
        "literal:__null__,1200\n"
        "literal:__blank__,1300\n",
        encoding="utf-8",
    )
    public = read_group_params(path, "sector")
    assert set(public) == {"__null__", "__blank__", "literal:__null__", "literal:__blank__"}
    got = normalize_group_params(public)
    assert got[NULL_GROUP_KEY]["max_distance"] == 1000
    assert got[BLANK_GROUP_KEY]["max_distance"] == 1100
    assert got["__null__"]["max_distance"] == 1200
    assert got["__blank__"]["max_distance"] == 1300
