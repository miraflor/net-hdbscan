"""Adaptive search radius and per-trial diagnostics."""

import numpy as np
import pandas as pd
import pytest

from conftest import irregular_network_data
from net_hdbscan import HDBSCANConfig, NeighborPairLimitError, cluster_files, cluster_geodataframes



def adaptive(**changes):
    settings = dict(max_distance=3000, min_cluster_size=15, distance_mode="adaptive")
    settings.update(changes)
    return HDBSCANConfig(**settings)


def test_fixed_mode_records_one_trial(points, boundary, roads):
    out = cluster_geodataframes(points, boundary, roads, HDBSCANConfig(max_distance=600, min_cluster_size=15))
    assert out.summary["distance_status"] == "fixed"
    assert out.trace[["radius", "outcome"]].values.tolist() == [[600.0, "ok"]]


def test_adaptive_stops_when_two_radii_agree_and_truncation_is_low(points, boundary, roads):
    out = cluster_geodataframes(points, boundary, roads, adaptive())
    summary, trace = out.summary, out.trace
    assert summary["distance_status"] == "converged" and summary["max_distance"] < 3000
    assert summary["share_core_truncated"] <= summary["core_truncation_tolerance"]
    assert trace["outcome"].eq("ok").all() and bool(trace["same_as_previous"].iloc[-1])
    assert len(trace) == summary["distance_steps_run"]
    # The result is exactly the fixed-mode result at the chosen radius.
    fixed = HDBSCANConfig(max_distance=summary["max_distance"], min_cluster_size=15)
    same = cluster_geodataframes(points, boundary, roads, fixed)
    pd.testing.assert_series_equal(out.points["cluster_id"], same.points["cluster_id"])


def test_adaptive_runs_to_the_ceiling_when_truncation_stays_high(points, boundary, roads):
    # At 600 m about 6% of core distances are still truncated, so a zero tolerance cannot be met.
    out = cluster_geodataframes(points, boundary, roads, adaptive(max_distance=600, core_truncation_tolerance=0.0))
    assert out.summary["distance_status"] == "ceiling_reached" and out.summary["share_core_truncated"] > 0
    assert np.allclose(out.trace["radius"], [600 / 1.5**3, 600 / 1.5**2, 600 / 1.5, 600])


def test_a_small_min_distance_still_reaches_the_end_of_the_ladder(points, boundary, roads):
    """A long geometric ladder must still reach its requested ceiling when needed."""
    config = adaptive(max_distance=2000, min_distance=2)
    out = cluster_geodataframes(points, boundary, roads, config)
    assert out.summary["distance_status"] in {"converged", "ceiling_reached"}
    assert len(out.trace) > config.distance_steps + 12
    # Two all-noise results at tiny radii agree, but truncation prevents a stop there.
    assert out.summary["n_clusters"] > 0


def test_pair_limit_keeps_the_last_result_and_does_not_retry_a_radius(points, boundary, roads):
    out = cluster_geodataframes(points, boundary, roads, adaptive(max_neighbor_pairs=3000))
    summary, trace = out.summary, out.trace
    assert summary["distance_status"] == "pair_cap_limited" and bool(summary["pair_limit_encountered"])
    assert trace["outcome"].eq("ok").sum() == summary["distance_steps_run"] >= 1
    assert not trace["radius"].round(6).duplicated().any()
    assert summary["max_distance"] == trace.loc[trace["outcome"] == "ok", "radius"].iloc[-1]
    assert (trace.loc[trace["outcome"] == "pair_limit", "radius"] > summary["max_distance"]).all()


def test_pair_limit_errors_where_no_result_is_allowed(points, boundary, roads):
    with pytest.raises(NeighborPairLimitError):  # an explicit first radius is never reduced
        cluster_geodataframes(points, boundary, roads, adaptive(max_neighbor_pairs=3000, min_distance=900))
    with pytest.raises(NeighborPairLimitError):  # fixed mode keeps the strict limit
        cluster_geodataframes(
            points, boundary, roads, HDBSCANConfig(max_distance=3000, min_cluster_size=15, max_neighbor_pairs=3000),
        )


def test_adaptive_results_do_not_depend_on_point_ids():
    points, boundary, roads = irregular_network_data(seed=0, n_points=400)
    config = HDBSCANConfig(max_distance=1500, min_cluster_size=15, min_samples=8, distance_mode="adaptive", max_neighbor_pairs=30_000)
    first = cluster_geodataframes(points, boundary, roads, config)
    renamed = points.copy()
    renamed["point_id"] = [f"r{k:04d}" for k in range(len(points), 0, -1)]
    second = cluster_geodataframes(renamed, boundary, roads, config)
    assert first.summary["max_distance"] == second.summary["max_distance"]
    assert first.summary["distance_status"] == second.summary["distance_status"]
    back = dict(zip(renamed["point_id"], points["point_id"]))
    a = first.points.set_index("point_id").sort_index()
    b = second.points.assign(point_id=second.points["point_id"].map(back)).set_index("point_id").sort_index()
    la, lb = pd.factorize(a["cluster_id"])[0], pd.factorize(b["cluster_id"])[0]
    assert len(set(zip(la, lb))) == len(set(la)) == len(set(lb)) and np.array_equal(la < 0, lb < 0)
    assert np.array_equal(a["membership"].to_numpy(), b["membership"].to_numpy())


def test_trace_file_is_written(files, tmp_path):
    cluster_files(
        points_path=files["points"], boundary_path=files["boundary"], network_path=files["roads"],
        output_dir=tmp_path / "out", config=adaptive(),
    )
    trace = pd.read_csv(tmp_path / "out" / "distance_trace.csv")
    assert list(trace.columns[:4]) == ["group", "trial", "radius", "outcome"] and len(trace) >= 2
