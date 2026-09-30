"""Reading, validating, and writing vector data."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

PARQUET_SUFFIXES = {".parquet", ".geoparquet"}

# Columns that net-hdbscan adds to the point layer. Input points must not
# already contain them.
POINT_OUTPUT_COLUMNS = (
    "cluster_id",
    "is_noise",
    "membership",
    "core_distance",
    "snap_distance",
    "snapped_x",
    "snapped_y",
)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _require_pyarrow() -> None:
    """Parquet needs pyarrow, which is an optional extra of net-hdbscan."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "reading or writing (Geo)Parquet needs pyarrow; install it with: pip install \"net-hdbscan[files]\""
        ) from exc


def read_vector(path: str | Path, *, name: str = "input", layer: str | None = None) -> gpd.GeoDataFrame:
    """Read GeoParquet, GeoPackage, or any other format GDAL can open."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in PARQUET_SUFFIXES:
        _require_pyarrow()
        if layer is not None:
            raise ValueError(f"{name} layer may be specified only for non-Parquet inputs")
        return gpd.read_parquet(path)
    if suffix == ".gpkg":
        names = gpd.list_layers(path)["name"].astype(str).tolist()
        if layer is None:
            if len(names) != 1:
                choices = ", ".join(repr(x) for x in names)
                raise ValueError(f"{name} GeoPackage {path} has {len(names)} layers ({choices}); specify the layer")
            layer = names[0]
        elif layer not in names:
            choices = ", ".join(repr(x) for x in names)
            raise ValueError(f"{name} layer {layer!r} not found in {path}; available layers: {choices}")
        return gpd.read_file(path, layer=layer)
    if layer is not None:
        raise ValueError(f"{name} layer may be specified only for .gpkg inputs")
    return gpd.read_file(path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _check_geometries(frame: gpd.GeoDataFrame, name: str, kinds: list[str]) -> None:
    bad = frame.geometry.isna() | frame.geometry.is_empty
    if bad.any():
        raise ValueError(f"{name} contains {int(bad.sum())} null or empty geometries")
    allowed = frame.geometry.geom_type.isin(kinds)
    if not allowed.all():
        found = sorted(frame.loc[~allowed].geometry.geom_type.unique().tolist())
        raise ValueError(f"{name} must contain only {'/'.join(kinds)} geometries; found {found}")


def prepare_network(network: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Check the network layer: projected CRS, valid LineString/MultiLineString, finite coordinates."""
    if network.empty:
        raise ValueError("network is empty")
    if network.crs is None:
        raise ValueError("network has no CRS")
    if not network.crs.is_projected:
        raise ValueError("network must use a projected CRS (distances are measured in its units)")
    _check_geometries(network, "network", ["LineString", "MultiLineString"])
    if not network.geometry.is_valid.all():
        raise ValueError("network contains invalid geometry; repair it before analysis")
    coords = shapely.get_coordinates(network.geometry.to_numpy())
    if not np.isfinite(coords).all():
        raise ValueError("network contains NaN or infinite coordinates")
    return network



def prepare_points(points: gpd.GeoDataFrame, crs, point_id_col: str) -> gpd.GeoDataFrame:
    """Check the point layer and reproject it to ``crs``.

    Point IDs are checked per clustering unit (see ``canonical_order``),
    because in a grouped run the same ID may appear in different groups.
    """
    if points.crs is None:
        raise ValueError("points has no CRS")
    if point_id_col not in points.columns:
        raise ValueError(f"point id column {point_id_col!r} not found")
    if point_id_col == points.geometry.name:
        raise ValueError("point id column must not be the geometry column")
    collisions = sorted(set(POINT_OUTPUT_COLUMNS).intersection(points.columns))
    if collisions:
        raise ValueError(f"points already contains reserved output columns: {collisions}; rename them first")
    if len(points):
        _check_geometries(points, "points", ["Point"])
    work = points.to_crs(crs) if points.crs != crs else points.copy()
    work = work.reset_index(drop=True)
    coords = shapely.get_coordinates(work.geometry.to_numpy())
    if len(coords) and not np.isfinite(coords).all():
        raise ValueError("points contains NaN or infinite coordinates")
    return work



def canonical_order(point_ids) -> np.ndarray:
    """Row order sorted by ``str(point_id)`` after checking the IDs.

    IDs must be non-null, unique, unique after conversion to text, and of
    one Python type. Sorting by text makes the results independent of the
    input row order.
    """
    values = pd.Series(list(point_ids), dtype=object)
    if values.isna().any():
        raise ValueError("point_id values must not be null")
    if values.duplicated().any():
        raise ValueError("duplicate point_id values are not allowed within one clustering unit")
    keys = values.map(str)
    if keys.duplicated().any():
        raise ValueError("point_id values must be unique after conversion to text")
    kinds = {type(v.item() if isinstance(v, np.generic) else v) for v in values.tolist()}
    if len(kinds) > 1:
        raise ValueError(f"point_id values mix Python types {sorted(k.__name__ for k in kinds)}; use one type")
    return np.argsort(keys.to_numpy(dtype=object), kind="stable")


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, write, *, force: bool) -> Path:
    if path.exists() and not force:
        raise FileExistsError(f"output already exists: {path}; pass force=True or --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def write_geoparquet(path: str | Path, frame: gpd.GeoDataFrame, *, force: bool = False) -> Path:
    path = Path(path)
    if path.suffix.lower() not in PARQUET_SUFFIXES:
        raise ValueError("output must end in .parquet or .geoparquet")
    _require_pyarrow()
    return _atomic_write(path, lambda tmp: frame.to_parquet(tmp, index=False), force=force)


def write_parquet(path: str | Path, frame: pd.DataFrame, *, force: bool = False) -> Path:
    path = Path(path)
    _require_pyarrow()
    return _atomic_write(path, lambda tmp: frame.to_parquet(tmp, index=False), force=force)


def write_csv(path: str | Path, frame: pd.DataFrame, *, force: bool = False) -> Path:
    path = Path(path)
    return _atomic_write(path, lambda tmp: frame.to_csv(tmp, index=False, lineterminator="\n"), force=force)


def write_json(path: str | Path, data: dict, *, force: bool = False) -> Path:
    path = Path(path)

    def write(tmp: Path) -> None:
        tmp.write_text(json.dumps(data, indent=2, default=_json_default) + "\n", encoding="utf-8", newline="\n")

    return _atomic_write(path, write, force=force)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot write {type(value).__name__} to JSON")
