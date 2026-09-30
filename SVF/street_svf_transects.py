"""
Street-level Sky View Factor (SVF) from building footprints + heights,
OSM roads and a canopy height model, using the urban-canyon formula.

Method
------
Points are placed every --spacing metres along each road. From each point a
ray is cast perpendicular to the road on both sides (up to --max-dist m).
For each side the steepest obstacle angle is found over buildings and trees:

    beta_side = max over obstacles of atan(height / horizontal_distance)

    svf         = (cos(beta_left) + cos(beta_right)) / 2
    svf_classic = cos(atan(2 * H / W))        # textbook symmetric canyon,
                                              # H = mean wall height, W = d_L + d_R

For a point at the exact centre of a symmetric canyon both are identical
(self-test: H = W = 10 m -> 0.4472). The per-side form also handles
off-centre points and unequal walls (self-test: 0.4213).

Assumptions (canyon model): walls are long compared with street width, and
obstacles only matter perpendicular to the road. At junctions and open plots
the result is approximate - see the `flag` column.

Inputs
------
--buildings  GeoJSON/GPKG from open_buildings_vector_heights.py (height_mean_m)
--chm        canopy height GeoTIFF in METRES (optional but recommended)
Roads are downloaded from OpenStreetMap via Overpass for the buildings' extent.

Requirements
------------
    pip install geopandas shapely>=2 numpy requests rasterio pyproj

Example
-------
    python street_svf_transects.py --buildings cp_buildings.geojson \
        --chm canopy_pct_delhi_chm.tif --out cp_street_svf.gpkg
    python street_svf_transects.py --self-test
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

LOGGER = logging.getLogger("street_svf")

DEFAULT_CRS = "EPSG:32643"           # UTM zone 43N (metres) - covers Delhi
OVERPASS_ENDPOINTS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
# Ways that are not street canyons (or duplicate a road's sidewalk).
DEFAULT_EXCLUDE_HIGHWAY = {
    "footway", "cycleway", "path", "steps", "track", "bridleway", "corridor",
    "construction", "proposed", "platform", "raceway", "elevator", "via_ferrata",
}
MIN_OBSTACLE_DIST_M = 1.0            # ignore obstacles closer than this (noise)
INSIDE_BUILDING_TOL_M = 0.5
MAX_CANOPY_PIXEL_M = 5.0           # coarser canopy rasters cannot place street trees


# --------------------------------------------------------------------------
# Formulas
# --------------------------------------------------------------------------
def svf_per_side(beta_left: float, beta_right: float) -> float:
    """SVF of a canyon floor point from the two wall elevation angles (radians)."""
    return (math.cos(beta_left) + math.cos(beta_right)) / 2.0


def svf_classic(d_left: float, d_right: float, h_left: float, h_right: float) -> float:
    """Textbook symmetric-canyon SVF: cos(atan(2H/W))."""
    width = d_left + d_right
    height = (h_left + h_right) / 2.0
    return math.cos(math.atan(2.0 * height / width))


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_buildings(path: Path, height_field: str, fill_missing: str, crs: str):
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if height_field not in gdf.columns:
        raise ValueError(f"'{height_field}' not found in {path}. Columns: {list(gdf.columns)}")

    n_missing = int(gdf[height_field].isna().sum())
    if n_missing:
        if fill_missing == "drop":
            gdf = gdf[gdf[height_field].notna()]
            LOGGER.warning("Dropped %d buildings without height.", n_missing)
        else:
            fill = gdf[height_field].median() if fill_missing == "median" else float(fill_missing)
            gdf[height_field] = gdf[height_field].fillna(fill)
            LOGGER.warning("Filled %d missing heights with %.1f m (%s).", n_missing, fill,
                           fill_missing)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    LOGGER.info("Buildings: %d, height median %.1f m, max %.1f m.", len(gdf),
                gdf[height_field].median(), gdf[height_field].max())
    return gdf.to_crs(crs)


def _post_overpass(url: str, query: str, user_agent: str, timeout_s: int,
                   max_retries: int) -> dict:
    """POST one Overpass query; back off and retry on HTTP 429 (rate limit)."""
    import requests

    headers = {"User-Agent": user_agent}   # generic python-requests agents may be refused
    for attempt in range(max_retries + 1):
        response = requests.post(url, data={"data": query}, headers=headers, timeout=timeout_s)
        if response.status_code == 429 and attempt < max_retries:
            retry_after = response.headers.get("Retry-After", "")
            wait_s = int(retry_after) if retry_after.isdigit() else 15 * (attempt + 1)
            LOGGER.warning("%s rate-limited (429); waiting %d s before retry %d/%d.",
                           url, wait_s, attempt + 1, max_retries)
            time.sleep(wait_s)
            continue
        if not response.ok:
            # Overpass puts the reason in the (HTML) body - surface a short excerpt.
            excerpt = " ".join(response.text.split())[:200]
            raise requests.HTTPError(f"{response.status_code} {response.reason}: {excerpt}")
        return response.json()                 # HTML error pages raise ValueError here
    raise requests.HTTPError("429 Too Many Requests (retries exhausted)")


def fetch_osm_roads(bbox_wgs84: tuple[float, float, float, float], endpoints: list[str],
                    exclude: set[str], user_agent: str, timeout_s: int = 180,
                    max_retries: int = 2):
    """Download OSM highway ways for the bbox; return a GeoDataFrame in EPSG:4326."""
    import geopandas as gpd
    import requests
    from shapely.geometry import LineString

    west, south, east, north = bbox_wgs84
    query = f'[out:json][timeout:120];way["highway"]({south},{west},{north},{east});out geom;'

    last_error: Exception | None = None
    for url in endpoints:
        try:
            LOGGER.info("Requesting OSM roads from %s", url)
            payload = _post_overpass(url, query, user_agent, timeout_s, max_retries)
            break
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning("Overpass endpoint failed (%s): %s", url, str(exc)[:300])
            last_error = exc
    else:
        raise RuntimeError(
            f"All Overpass endpoints failed: {last_error}. Wait a few minutes and retry, "
            "or pass a local roads file with --roads (e.g. roads cut from a Geofabrik .osm.pbf)."
        )

    LOGGER.info("OSM data timestamp: %s (check it is recent enough).",
                payload.get("osm3s", {}).get("timestamp_osm_base", "unknown"))

    rows = []
    for el in payload.get("elements", []):
        tags = el.get("tags", {})
        highway = tags.get("highway")
        geom = el.get("geometry") or []
        if el.get("type") != "way" or highway in exclude or len(geom) < 2:
            continue
        rows.append({
            "osm_id": el["id"],
            "highway": highway,
            "name": tags.get("name"),
            "geometry": LineString([(p["lon"], p["lat"]) for p in geom]),
        })
    if not rows:
        raise RuntimeError("No usable roads returned for this area.")
    roads = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    LOGGER.info("Roads kept: %d ways (%s).", len(roads),
                ", ".join(f"{k}={v}" for k, v in roads["highway"].value_counts().head(6).items()))
    return roads


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
@dataclass
class Transect:
    point: "Point"                  # noqa: F821 - shapely Point
    normal: np.ndarray              # unit vector pointing to the LEFT of travel
    osm_id: int
    highway: str
    name: str | None
    dist_along_m: float


def iter_transects(roads_utm, spacing_m: float) -> Iterator[Transect]:
    """Yield evenly spaced points along each road with their left-hand normal."""
    for row in roads_utm.itertuples(index=False):
        line = row.geometry
        length = line.length
        if length < 1.0:
            continue
        for d in np.arange(spacing_m / 2.0, length, spacing_m):
            a = line.interpolate(max(d - 1.0, 0.0))
            b = line.interpolate(min(d + 1.0, length))
            tangent = np.array([b.x - a.x, b.y - a.y])
            norm = np.linalg.norm(tangent)
            if norm == 0:
                continue
            tx, ty = tangent / norm
            yield Transect(line.interpolate(d), np.array([-ty, tx]), row.osm_id,
                           row.highway, row.name, float(d))


class BuildingIndex:
    """Spatial index over building polygons for ray intersection."""

    def __init__(self, buildings_utm, height_field: str):
        from shapely import STRtree

        self.geoms = list(buildings_utm.geometry.values)
        self.heights = buildings_utm[height_field].to_numpy(dtype=float)
        self.tree = STRtree(self.geoms)

    def contains(self, point) -> bool:
        for idx in self.tree.query(point):
            if self.geoms[idx].distance(point) <= INSIDE_BUILDING_TOL_M:
                return True
        return False

    def side_angle(self, point, direction: np.ndarray, max_dist: float):
        """Max elevation angle along the ray: (beta_rad, distance_m, height_m) or None."""
        from shapely.geometry import LineString

        end = (point.x + direction[0] * max_dist, point.y + direction[1] * max_dist)
        ray = LineString([(point.x, point.y), end])
        best = None
        for idx in self.tree.query(ray):
            hit = ray.intersection(self.geoms[idx])
            if hit.is_empty:
                continue
            dist = max(point.distance(hit), MIN_OBSTACLE_DIST_M)
            beta = math.atan(self.heights[idx] / dist)
            if best is None or beta > best[0]:
                best = (beta, dist, float(self.heights[idx]))
        return best


class CanopySampler:
    """Sample canopy height (m) along rays from a GeoTIFF."""

    def __init__(self, path: Path, work_crs: str, step_m: float, min_tree_h: float,
                 allow_coarse: bool = False):
        import rasterio
        from pyproj import Transformer

        with rasterio.open(path) as ds:
            self.array = ds.read(1, masked=True).filled(0).astype(float)
            self.transform = ds.transform
            self.crs = ds.crs
            nodata = ds.nodata
        self.array[~np.isfinite(self.array)] = 0.0
        if nodata is not None:
            self.array[self.array == nodata] = 0.0
        self.to_raster = Transformer.from_crs(work_crs, self.crs, always_xy=True)
        self.step_m = step_m
        self.min_tree_h = min_tree_h
        self.pixel_m = self._pixel_size_m()

        valid = self.array[self.array > 0]
        LOGGER.info("Canopy raster %s: CRS %s, shape %s, pixel ~%.1f m, max %.1f, p95 %.1f.",
                    path.name, self.crs, self.array.shape, self.pixel_m,
                    float(self.array.max()),
                    float(np.percentile(valid, 95)) if valid.size else 0.0)
        if self.array.max() > 80:
            LOGGER.warning("Canopy max > 80 - values may be PERCENT cover, not metres. Check!")
        if self.pixel_m > MAX_CANOPY_PIXEL_M and not allow_coarse:
            raise ValueError(
                f"Canopy raster pixels are ~{self.pixel_m:.0f} m. Street trees cannot be "
                f"placed reliably at > {MAX_CANOPY_PIXEL_M:.0f} m (a whole road fits in 1-2 "
                "pixels, so mixed road/tree pixels act as walls a few metres away and SVF "
                "collapses). Use a ~1 m canopy height map (e.g. meta_canopy_export.py), run "
                "without --chm, or pass --allow-coarse-chm to override."
            )
        if self.pixel_m > MAX_CANOPY_PIXEL_M:
            LOGGER.warning("Coarse canopy (~%.0f m) allowed by override: tree side of SVF "
                           "is unreliable.", self.pixel_m)

    def _pixel_size_m(self) -> float:
        """Approximate pixel width in metres (handles geographic CRSs)."""
        res = abs(self.transform.a)
        if self.crs is not None and self.crs.is_geographic:
            rows, cols = self.array.shape
            lat_mid = (self.transform * (cols / 2, rows / 2))[1]
            return res * 111_320.0 * math.cos(math.radians(lat_mid))
        return res

    def _rowcol(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from rasterio.transform import rowcol

        rx, ry = self.to_raster.transform(xs, ys)
        rows, cols = rowcol(self.transform, rx, ry)
        return np.atleast_1d(np.asarray(rows)), np.atleast_1d(np.asarray(cols))

    def side_angle(self, point, direction: np.ndarray, max_dist: float):
        dists = np.arange(max(self.step_m, MIN_OBSTACLE_DIST_M), max_dist + 1e-9, self.step_m)
        xs = point.x + direction[0] * dists
        ys = point.y + direction[1] * dists
        rows, cols = self._rowcol(xs, ys)
        inside = (rows >= 0) & (rows < self.array.shape[0]) & (cols >= 0) & (cols < self.array.shape[1])

        # Exclude the pixel the road point itself sits in: with coarse rasters that
        # pixel mixes road and roadside canopy, and would otherwise be read as a
        # "tree wall" a couple of metres away on BOTH sides (SVF collapses to ~0).
        own_r, own_c = self._rowcol(np.array([point.x]), np.array([point.y]))
        inside &= ~((rows == own_r[0]) & (cols == own_c[0]))
        if not inside.any():
            return None
        heights = np.zeros_like(dists)
        heights[inside] = self.array[rows[inside], cols[inside]]
        heights[heights < self.min_tree_h] = 0.0
        if not heights.any():
            return None
        betas = np.arctan(heights / dists)
        i = int(np.argmax(betas))
        return float(betas[i]), float(dists[i]), float(heights[i])


# --------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------
def compute_side(point, direction, buildings: BuildingIndex, canopy: CanopySampler | None,
                 max_dist: float) -> dict:
    """Steepest obstacle on one side, from buildings and trees."""
    candidates = []
    b = buildings.side_angle(point, direction, max_dist)
    if b:
        candidates.append((*b, "building"))
    if canopy is not None:
        t = canopy.side_angle(point, direction, max_dist)
        if t:
            candidates.append((*t, "tree"))
    if not candidates:
        return {"beta": 0.0, "dist": None, "height": 0.0, "kind": "none", "bldg": b}
    beta, dist, height, kind = max(candidates, key=lambda c: c[0])
    return {"beta": beta, "dist": dist, "height": height, "kind": kind, "bldg": b}


def evaluate_transect(tr: Transect, buildings: BuildingIndex, canopy: CanopySampler | None,
                      max_dist: float, study_area_inner) -> dict | None:
    if buildings.contains(tr.point):
        return None                                           # road drawn through a building

    left = compute_side(tr.point, tr.normal, buildings, canopy, max_dist)
    right = compute_side(tr.point, -tr.normal, buildings, canopy, max_dist)

    flags = []
    if not study_area_inner.contains(tr.point):
        flags.append("edge")                                  # obstacles beyond data extent unseen
    if left["kind"] == "none" or right["kind"] == "none":
        flags.append("open_side")

    classic = None
    bl, br = left["bldg"], right["bldg"]
    if bl and br:                                             # classic needs a wall on both sides
        classic = svf_classic(bl[1], br[1], bl[2], br[2])

    return {
        "geometry": tr.point,
        "osm_id": tr.osm_id, "highway": tr.highway, "name": tr.name,
        "dist_along_m": round(tr.dist_along_m, 1),
        "d_left_m": left["dist"], "d_right_m": right["dist"],
        "h_left_m": left["height"], "h_right_m": right["height"],
        "obst_left": left["kind"], "obst_right": right["kind"],
        "beta_left_deg": round(math.degrees(left["beta"]), 2),
        "beta_right_deg": round(math.degrees(right["beta"]), 2),
        "width_m": (bl[1] + br[1]) if (bl and br) else None,
        "svf": round(svf_per_side(left["beta"], right["beta"]), 4),
        "svf_classic": round(classic, 4) if classic is not None else None,
        "flag": ",".join(flags) if flags else "ok",
    }


def run(args: argparse.Namespace) -> int:
    import geopandas as gpd

    buildings = load_buildings(args.buildings, args.height_field, args.fill_missing, args.crs)
    from shapely.geometry import box

    extent_utm = box(*buildings.total_bounds)
    bbox_wgs84 = tuple(float(v) for v in
                       gpd.GeoSeries([extent_utm], crs=args.crs).to_crs("EPSG:4326").total_bounds)
    LOGGER.info("Study extent (WGS84): %s", [round(v, 5) for v in bbox_wgs84])

    cache = args.out.with_name(f"{args.out.stem}_roads.gpkg")
    if args.roads:
        roads = gpd.read_file(args.roads)
        if "highway" in roads.columns:
            roads = roads[~roads["highway"].isin(DEFAULT_EXCLUDE_HIGHWAY)]
        for col, default in (("osm_id", -1), ("highway", "unknown"), ("name", None)):
            if col not in roads.columns:
                roads[col] = default
        LOGGER.info("Roads loaded from %s: %d features.", args.roads, len(roads))
    else:
        endpoints = ([args.overpass_url] if args.overpass_url else []) + \
            [u for u in OVERPASS_ENDPOINTS if u != args.overpass_url]
        roads = fetch_osm_roads(bbox_wgs84, endpoints, DEFAULT_EXCLUDE_HIGHWAY, args.user_agent)
        roads.to_file(cache, layer="roads", driver="GPKG")
        LOGGER.info("Roads cached to %s - reuse with --roads to avoid re-downloading.", cache)

    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    roads_utm = roads.to_crs(args.crs).clip(extent_utm).explode(index_parts=False)
    roads_utm = roads_utm[roads_utm.geometry.geom_type == "LineString"]

    index = BuildingIndex(buildings, args.height_field)
    canopy = CanopySampler(args.chm, args.crs, args.tree_step, args.min_tree_height,
                           args.allow_coarse_chm) \
        if args.chm else None
    if canopy is None:
        LOGGER.warning("No --chm given: trees ignored, SVF will be too high on leafy streets.")

    inner = extent_utm.buffer(-args.max_dist)
    records, skipped_inside = [], 0
    for i, tr in enumerate(iter_transects(roads_utm, args.spacing), start=1):
        rec = evaluate_transect(tr, index, canopy, args.max_dist, inner)
        if rec is None:
            skipped_inside += 1
        else:
            records.append(rec)
        if i % 2000 == 0:
            LOGGER.info("Processed %d transects...", i)

    if not records:
        LOGGER.error("No transects produced.")
        return 1

    out = gpd.GeoDataFrame(records, crs=args.crs)
    out.to_file(args.out, layer="street_svf", driver="GPKG")

    ok = out[out["flag"] == "ok"]
    LOGGER.info("Saved %s: %d points (%d ok, %d skipped inside buildings).",
                args.out, len(out), len(ok), skipped_inside)
    if len(ok):
        LOGGER.info("SVF (ok points): median %.3f, p10 %.3f, p90 %.3f. Obstacles: %s",
                    ok["svf"].median(), ok["svf"].quantile(0.1), ok["svf"].quantile(0.9),
                    dict(pd_counts(ok)))
    return 0


def pd_counts(df) -> dict:
    kinds = list(df["obst_left"]) + list(df["obst_right"])
    return {k: kinds.count(k) for k in sorted(set(kinds))}


# --------------------------------------------------------------------------
# Self-test on synthetic canyons (no network, no input files)
# --------------------------------------------------------------------------
def self_test() -> int:
    import geopandas as gpd
    from shapely.geometry import Point, box

    walls = gpd.GeoDataFrame(
        {"height_mean_m": [10.0, 10.0]},
        geometry=[box(-500, 5, 500, 15), box(-500, -15, 500, -5)],   # W = 10 m, H = 10 m
        crs=DEFAULT_CRS,
    )
    index = BuildingIndex(walls, "height_mean_m")
    normal = np.array([0.0, 1.0])

    def svf_at(y: float) -> float:
        p = Point(0, y)
        left = compute_side(p, normal, index, None, 100.0)
        right = compute_side(p, -normal, index, None, 100.0)
        return svf_per_side(left["beta"], right["beta"])

    cases = [
        ("centre, H=W=10 (per-side)", svf_at(0.0), 0.4472),
        ("centre, H=W=10 (classic)", svf_classic(5, 5, 10, 10), 0.4472),
        ("2.5 m from wall, H=W=10", svf_at(2.5), 0.4213),
        ("classic, H=10, W=40", svf_classic(20, 20, 10, 10), 0.8944),
    ]
    ok = True
    for label, got, expected in cases:
        passed = abs(got - expected) < 5e-4
        ok &= passed
        LOGGER.info("%-28s got %.4f expected %.4f  %s", label, got, expected,
                    "PASS" if passed else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Street-level SVF from buildings, OSM roads and CHM.")
    p.add_argument("--self-test", action="store_true", help="Run synthetic canyon checks and exit")
    p.add_argument("--buildings", type=Path, help="GeoJSON/GPKG with building heights")
    p.add_argument("--height-field", default="height_mean_m")
    p.add_argument("--fill-missing", default="median",
                   help="'median', 'drop', or a number of metres for buildings without height")
    p.add_argument("--chm", type=Path, help="Canopy height GeoTIFF in metres")
    p.add_argument("--allow-coarse-chm", action="store_true",
                   help=f"Accept canopy rasters coarser than {MAX_CANOPY_PIXEL_M:.0f} m (not recommended)")
    p.add_argument("--min-tree-height", type=float, default=2.0, help="Ignore canopy below this (m)")
    p.add_argument("--tree-step", type=float, default=2.0, help="Canopy sampling step (m)")
    p.add_argument("--spacing", type=float, default=20.0, help="Distance between transects (m)")
    p.add_argument("--max-dist", type=float, default=100.0, help="Ray length each side (m)")
    p.add_argument("--crs", default=DEFAULT_CRS, help="Metric CRS for calculations")
    p.add_argument("--overpass-url", default=None, help="Preferred Overpass endpoint")
    p.add_argument("--roads", type=Path, default=None,
                   help="Local roads file (GPKG/GeoJSON/SHP) instead of downloading from Overpass")
    p.add_argument("--user-agent", default="street-svf-transects/1.1 (urban heat research)",
                   help="HTTP User-Agent sent to Overpass; add contact info if requests are refused")
    p.add_argument("--out", type=Path, default=Path("street_svf.gpkg"))
    a = p.parse_args(argv)
    if not a.self_test and a.buildings is None:
        p.error("--buildings is required (or use --self-test)")
    return a


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    try:
        return run(args)
    except (RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())