"""Small synthetic data shared by the tests."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import shapely

CRS = "EPSG:32651"


def grid_segments(n_lines: int = 21, step: float = 100.0) -> np.ndarray:
    """A noded square grid: one straight segment between each pair of adjacent crossings."""
    xs = np.arange(n_lines) * step
    segments = []
    for x in xs:
        for k in range(n_lines - 1):
            segments.append([[x, xs[k]], [x, xs[k + 1]]])
            segments.append([[xs[k], x], [xs[k + 1], x]])
    return shapely.linestrings(np.asarray(segments))


def make_points(seed: int = 3) -> gpd.GeoDataFrame:
    """Two dense blobs, a few scattered points, and some exact duplicates."""
    rng = np.random.default_rng(seed)
    blob_a = rng.normal([500, 500], 80, size=(120, 2))
    blob_b = rng.normal([1500, 1500], 80, size=(120, 2))
    scattered = rng.uniform(0, 2000, size=(40, 2))
    xy = np.vstack([blob_a, blob_b, scattered])
    xy = np.vstack([xy, xy[:10]])  # ten exact duplicates of blob A points
    xy = np.clip(xy, 1, 1999)
    n = len(xy)
    frame = gpd.GeoDataFrame(
        {
            "point_id": [f"q{i:04d}" for i in range(n)],
            "sector": np.where(np.arange(n) % 2 == 0, "a", "b"),
        },
        geometry=shapely.points(xy),
        crs=CRS,
    )
    return frame


@pytest.fixture
def roads() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=grid_segments(), crs=CRS)



@pytest.fixture
def points() -> gpd.GeoDataFrame:
    return make_points()


@pytest.fixture
def files(tmp_path, roads, points):
    """The fixtures written to GeoParquet files."""
    pytest.importorskip("pyarrow")
    paths = {
        "roads": tmp_path / "roads.parquet",
        "points": tmp_path / "points.parquet",
    }
    roads.to_parquet(paths["roads"])
    points.to_parquet(paths["points"])
    return paths


def irregular_network_data(seed: int = 0, n: int = 21, step: float = 150.0, jitter: float = 20.0, n_points: int = 700):
    """A jittered grid (segment lengths are not round numbers) with two blobs and scattered points.

    Returns ``(points, roads)``. On such a network, the last bits of
    a computed distance can depend on which end of a pair starts the search,
    so it exposes any dependence of results on observation order or IDs.
    """
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(n) * step, np.arange(n) * step)
    vertices = np.column_stack([gx.ravel(), gy.ravel()]) + rng.normal(0, jitter, size=(n * n, 2))
    index = np.arange(n * n).reshape(n, n)
    edges = [(index[r, c], index[r, c + 1]) for r in range(n) for c in range(n - 1)]
    edges += [(index[r, c], index[r + 1, c]) for r in range(n - 1) for c in range(n)]
    first, second = np.array(edges).T
    roads = gpd.GeoDataFrame(geometry=shapely.linestrings(np.stack([vertices[first], vertices[second]], axis=1)), crs=CRS)
    extent = (n - 1) * step
    xy = np.vstack(
        [
            rng.normal([0.3 * extent, 0.3 * extent], 0.07 * extent, size=(n_points * 4 // 10, 2)),
            rng.normal([0.7 * extent, 0.65 * extent], 0.1 * extent, size=(n_points * 4 // 10, 2)),
            rng.uniform(0, extent, size=(n_points * 2 // 10, 2)),
        ]
    )
    points = gpd.GeoDataFrame({"point_id": [f"q{i:04d}" for i in range(len(xy))]}, geometry=shapely.points(xy), crs=CRS)
    return points, roads
