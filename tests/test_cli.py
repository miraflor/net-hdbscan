import pandas as pd
from typer.testing import CliRunner

from net_hdbscan import __version__
from net_hdbscan.cli import app

runner = CliRunner()


def base_args(files, out_dir):
    return [
        "cluster",
        "--points", str(files["points"]),
        "--network", str(files["roads"]),
        "--output-dir", str(out_dir),
        "--max-distance", "600",
        "--min-cluster-size", "15",
    ]


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and __version__ in result.stdout


def test_single_run_prints_a_summary(files, tmp_path):
    result = runner.invoke(app, base_args(files, tmp_path / "out"))
    assert result.exit_code == 0, result.output
    assert "clusters" in result.stdout and "(all)" in result.stdout
    assert (tmp_path / "out" / "clustered_points.parquet").exists()
    assert (tmp_path / "out" / "cluster_hierarchy.parquet").exists()


def test_grouped_run_with_parameter_file(files, tmp_path):
    params = tmp_path / "params.csv"
    params.write_text("sector,min_cluster_size\nb,25\n")
    args = base_args(files, tmp_path / "out") + [
        "--group-col", "sector",
        "--group-params", str(params),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    summary = pd.read_csv(tmp_path / "out" / "summary.csv", dtype={"group": str}).set_index("group")
    assert summary.loc["b", "min_cluster_size"] == 25 and summary.loc["a", "min_cluster_size"] == 15


def test_package_has_no_polygon_output(files, tmp_path):
    result = runner.invoke(app, base_args(files, tmp_path / "out"))
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "out" / "cluster_polygons.parquet").exists()


def test_errors_are_reported(files, tmp_path):
    bad_method = runner.invoke(app, base_args(files, tmp_path / "a") + ["--cluster-selection-method", "best"])
    assert bad_method.exit_code != 0
    params_without_groups = runner.invoke(app, base_args(files, tmp_path / "b") + ["--group-params", "x.csv"])
    assert params_without_groups.exit_code != 0
    runner.invoke(app, base_args(files, tmp_path / "c"))
    again = runner.invoke(app, base_args(files, tmp_path / "c"))
    assert again.exit_code != 0 and isinstance(again.exception, FileExistsError)


def test_cli_reports_excluded_rows_and_undeclared_groups(files, tmp_path, points):
    data = points.copy()
    data.loc[0:2, "sector"] = None
    points_file = tmp_path / "points_missing.parquet"
    data.to_parquet(points_file)
    universe = tmp_path / "universe.csv"
    universe.write_text("sector\na\n")
    args = base_args(files, tmp_path / "out")
    args[args.index("--points") + 1] = str(points_file)
    result = runner.invoke(app, args + ["--group-col", "sector", "--group-universe", str(universe)])
    assert result.exit_code == 0, result.output
    assert "Excluded 3 input row(s)" in result.stdout
    assert "not in --group-universe" in result.stdout and "b" in result.stdout


def test_cli_warns_when_the_pair_limit_stops_the_search(files, tmp_path):
    args = base_args(files, tmp_path / "out")
    args[args.index("--max-distance") + 1] = "3000"
    result = runner.invoke(app, args + ["--distance-mode", "adaptive", "--max-neighbor-pairs", "3000"])
    assert result.exit_code == 0, result.output
    assert "WARNING" in result.stdout and "max_neighbor_pairs" in result.stdout


def test_cluster_help_has_no_boundary_options():
    result = runner.invoke(app, ["cluster", "--help"])
    assert result.exit_code == 0
    assert "--boundary" not in result.stdout
    assert "--boundary-layer" not in result.stdout
