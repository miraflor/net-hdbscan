"""Validated WGS84 circular acquisition regions and exact line clipping.

These boundaries constrain *network acquisition*, not observation eligibility.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Polygon

_CIRCLE_SEGMENTS = 360
_MAX_AREA_KM2 = 10_000.0
_GEOD = Geod(ellps="WGS84")


class Circle(NamedTuple):
    longitude: float
    latitude: float
    radius_km: float


def circle_polygon(circle: Circle) -> Polygon:
    """Build a 360-segment geodesic disk in EPSG:4326, rejecting wrapped geometry."""
    lon, lat, radius_km = circle
    vertices = [
        _GEOD.fwd(lon, lat, float(bearing), radius_km * 1000)[:2]
        for bearing in range(_CIRCLE_SEGMENTS)
    ]
    # OSMnx/Overpass polygons cannot safely represent unwrapped dateline crossings.
    if any(abs(x - lon) > 180 for x, _ in vertices):
        raise ValueError("bounding circle crosses the antimeridian; use a smaller radius or another center")
    polygon = Polygon(vertices)
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError("bounding circle could not be represented as a valid WGS84 polygon")
    area_m2, _ = _GEOD.geometry_area_perimeter(polygon)
    area_km2 = abs(area_m2) / 1_000_000
    if area_km2 > _MAX_AREA_KM2:
        raise ValueError(
            f"circle area is approximately {area_km2:,.0f} km2; "
            f"public OSM requests are capped at {_MAX_AREA_KM2:,.0f} km2"
        )
    return polygon


def parse_circle(center: str | tuple[float, float], radius_km: float) -> Circle:
    """Parse EPSG:4326 ``longitude,latitude`` plus geodesic radius in km."""
    try:
        coords = (
            [float(part.strip()) for part in center.split(",")]
            if isinstance(center, str)
            else [float(part) for part in center]
        )
        radius = float(radius_km)
    except (ValueError, TypeError) as exc:
        raise ValueError("center must be longitude,latitude and radius-km must be numeric") from exc
    if len(coords) != 2 or not all(math.isfinite(v) for v in coords):
        raise ValueError("center must contain two finite longitude,latitude numbers")
    lon, lat = coords
    if not (-180 <= lon <= 180 and -90 < lat < 90):
        raise ValueError("center longitude/latitude is outside supported world bounds")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius-km must be positive and finite")
    # A circle encompassing either pole is not representable as a simple WGS84 polygon.
    pole = 90.0 if lat >= 0 else -90.0
    _, _, pole_distance_m = _GEOD.inv(lon, lat, lon, pole)
    if radius * 1000 >= pole_distance_m:
        raise ValueError("bounding circle reaches a pole; use a smaller radius")
    circle = Circle(lon, lat, radius)
    circle_polygon(circle)  # validate topology, antimeridian and public-service area cap
    return circle


def clip_lines_to_circle(lines: gpd.GeoDataFrame, polygon: Polygon) -> gpd.GeoDataFrame:
    """Clip projected edge geometry to the actual circle, keeping only line parts."""
    if lines.crs is None or not lines.crs.is_projected:
        raise ValueError("circle clipping requires a projected line-network CRS")
    boundary = gpd.GeoSeries([polygon], crs="EPSG:4326").to_crs(lines.crs).iloc[0]
    segments = []

    def collect(geom):
        if geom.is_empty:
            return
        if geom.geom_type == "LineString":
            if geom.length > 0:
                segments.append(geom)
        elif geom.geom_type in {"MultiLineString", "GeometryCollection"}:
            for part in geom.geoms:
                collect(part)

    for geom in lines.geometry:
        if geom is not None and not geom.is_empty:
            collect(geom.intersection(boundary))
    if not segments:
        raise ValueError("OSM returned no line geometry within the bounding circle")
    return gpd.GeoDataFrame(geometry=segments, crs=lines.crs)
