"""
Every pixel as a polygon with its actual NDVI / NDWI / NDBI values.

For each composite period (default: monthly) Earth Engine computes, PER PIXEL,
statistics over all clear acquisitions in that period:

    <index>_mean, <index>_min, <index>_max, <index>_std    for ndvi, ndwi, ndbi
    n_clear                                                number of clear views
    class   1 water / 2 vegetation / 3 built-up / 0 other  (rule from the means;
            water uses NDWI here, thresholds are starting values - tune them)

Pixels are on an exact UTM 43N grid (EPSG:32643, default 30 m) and downloaded in
non-overlapping tiles that stay under the direct-download size limit. Each tile
is turned into square polygons LOCALLY and appended to a GeoPackage
(one layer per period). GeoPackage is used instead of GeoJSON because it handles
millions of features with a spatial index.

Scale check: ~2.9 million polygons per period for a 0.5 x 0.5 deg box at 30 m.

Reuses collections and indices from spectral_indices_export.py (same folder).

Requirements:  pip install earthengine-api requests rasterio geopandas shapely>=2 pyproj numpy

Examples
--------
    python pixel_polygons_export.py --project my-gee-project --bbox 76.84 28.40 77.35 28.88 \
        --start 2025-10-01 --end 2025-11-01 --out ./pixels_delhi
    # Convert a GeoTIFF you already have (bands as written by this script):
    python pixel_polygons_export.py --from-tif pixels_delhi/tiles/s2_2025-10_r0_c0.tif --out ./pixels_delhi
    python pixel_polygons_export.py --self-test
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("pixel_polygons")

UTM_CRS = "EPSG:32643"
NODATA = -9999.0
DIRECT_LIMIT_BYTES = 30 * 1024 * 1024
STATS = ("mean", "min", "max", "std")
DEFAULT_INDICES = ("ndvi", "ndwi", "ndbi")


def band_names(indices: tuple[str, ...]) -> list[str]:
    return [f"{i}_{s}" for i in indices for s in STATS] + ["n_clear"]


# --------------------------------------------------------------------------
# Pure helpers (tested offline)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    x0: float          # left (UTM m)
    y0: float          # top  (UTM m)
    nx: int
    ny: int


def plan_tiles(bbox_utm: tuple[float, float, float, float], scale: float,
               n_bands: int, limit_bytes: int = DIRECT_LIMIT_BYTES) -> list[Tile]:
    """Snap bbox to the pixel grid and split into non-overlapping tiles under the limit."""
    minx, miny, maxx, maxy = bbox_utm
    gx0 = math.floor(minx / scale) * scale
    gy0 = math.ceil(maxy / scale) * scale
    total_nx = math.ceil((maxx - gx0) / scale)
    total_ny = math.ceil((gy0 - miny) / scale)
    side = max(64, int(math.sqrt(limit_bytes / (n_bands * 4))) // 16 * 16)
    tiles = []
    for r, ty in enumerate(range(0, total_ny, side)):
        for c, tx in enumerate(range(0, total_nx, side)):
            tiles.append(Tile(r, c, gx0 + tx * scale, gy0 - ty * scale,
                              min(side, total_nx - tx), min(side, total_ny - ty)))
    return tiles


def pixel_polygons_from_arrays(data: np.ndarray, names: list[str], transform, crs,
                               keep_area=None):
    """data: (bands, rows, cols). Returns GeoDataFrame of square pixels with attributes."""
    import geopandas as gpd
    import shapely

    n_idx = names.index("n_clear")
    valid = (data[n_idx] != NODATA) & (data[n_idx] > 0)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return gpd.GeoDataFrame({n: [] for n in names}, geometry=[], crs=crs)
    a, e, xoff, yoff = transform.a, transform.e, transform.c, transform.f
    x0 = xoff + cols * a
    y0 = yoff + rows * e
    geoms = shapely.box(np.minimum(x0, x0 + a), np.minimum(y0, y0 + e),
                        np.maximum(x0, x0 + a), np.maximum(y0, y0 + e))
    attrs = {}
    for b, name in enumerate(names):
        vals = data[b][rows, cols].astype("float64")
        vals[vals == NODATA] = np.nan
        attrs[name] = np.round(vals, 4) if name != "n_clear" else vals
    gdf = gpd.GeoDataFrame(attrs, geometry=geoms, crs=crs)
    if keep_area is not None:                       # drop pixels centred outside the study box
        gdf = gdf[shapely.contains_xy(keep_area, x0 + a / 2, y0 + e / 2)]
    return gdf


def classify_means(gdf, wt: float, vt: float, bt: float):
    """Same rule as spectral_indices_export (NDWI used for water here)."""
    cls = np.zeros(len(gdf), dtype="int16")
    ndvi = gdf.get("ndvi_mean")
    ndwi = gdf.get("ndwi_mean")
    ndbi = gdf.get("ndbi_mean")
    if ndwi is not None:
        water = (ndwi > wt).to_numpy()
        cls[water] = 1
    else:
        water = np.zeros(len(gdf), bool)
    if ndvi is not None:
        veg = (ndvi > vt).to_numpy() & ~water
        cls[veg] = 2
        if ndbi is not None:
            built = (ndbi > bt).to_numpy() & (ndvi <= vt).to_numpy() & ~water
            cls[built] = 3
    gdf["class"] = cls
    gdf["class_name"] = np.select([cls == 1, cls == 2, cls == 3],
                                  ["water", "vegetation", "built_up"], "other")
    return gdf


# --------------------------------------------------------------------------
# Earth Engine
# --------------------------------------------------------------------------
def stats_image(col, indices: tuple[str, ...]):
    import ee
    from spectral_indices_export import add_indices

    idx = col.map(add_indices).select(list(indices))
    reducer = (ee.Reducer.mean().combine(ee.Reducer.min(), sharedInputs=True)
               .combine(ee.Reducer.max(), sharedInputs=True)
               .combine(ee.Reducer.stdDev(), sharedInputs=True))
    stats = idx.reduce(reducer)                    # bands like ndvi_mean, ndvi_min, ndvi_stdDev
    renamed = stats.select([f"{i}_{'stdDev' if s == 'std' else s}" for i in indices for s in STATS],
                           [f"{i}_{s}" for i in indices for s in STATS])
    n_clear = idx.select(indices[0]).count().rename("n_clear")
    return renamed.addBands(n_clear).toFloat().unmask(NODATA)


def download_tile(image, tile: Tile, scale: float, path: Path, retries: int = 4) -> None:
    import requests

    params = {"crs": UTM_CRS, "crs_transform": [scale, 0, tile.x0, 0, -scale, tile.y0],
              "dimensions": f"{tile.nx}x{tile.ny}", "format": "GEO_TIFF"}
    for attempt in range(1, retries + 1):
        try:
            url = image.getDownloadURL(params)
            resp = requests.get(url, timeout=900)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            return
        except Exception as exc:  # noqa: BLE001 - ee / network errors
            if attempt == retries:
                raise
            wait = 30 * attempt
            LOGGER.warning("Tile r%d c%d failed (%s); retry in %d s.", tile.row, tile.col,
                           str(exc)[:120], wait)
            time.sleep(wait)


# --------------------------------------------------------------------------
# Local conversion
# --------------------------------------------------------------------------
def tif_to_pixel_polygons(tif: Path, names: list[str] | None, keep_area=None):
    import rasterio

    with rasterio.open(tif) as ds:
        data = ds.read()
        desc = [d for d in ds.descriptions]
        names = names or ([d for d in desc] if all(desc) else None)
        if not names or len(names) != ds.count:
            raise ValueError(f"{tif.name}: cannot determine band names ({ds.count} bands); "
                             "pass --bands in the right order")
        if ds.nodata is not None and ds.nodata != NODATA:
            data = np.where(data == ds.nodata, NODATA, data)
        return pixel_polygons_from_arrays(data, names, ds.transform, ds.crs, keep_area)


def append_layer(gdf, gpkg: Path, layer: str, first: bool) -> None:
    gdf.to_file(gpkg, layer=layer, driver="GPKG", mode="w" if first else "a")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_gee(args) -> int:
    import ee
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import box

    from spectral_indices_export import (collection_for, init_earth_engine, month_windows,
                                         Config as IdxConfig)

    indices = tuple(args.indices)
    names = band_names(indices)
    init_earth_engine(args.project)

    west, south, east, north = args.bbox
    to_utm = Transformer.from_crs("EPSG:4326", UTM_CRS, always_xy=True)
    xs, ys = to_utm.transform([west, east, east, west], [south, south, north, north])
    bbox_utm = (min(xs), min(ys), max(xs), max(ys))
    keep_area = gpd.GeoSeries([box(*args.bbox)], crs="EPSG:4326").to_crs(UTM_CRS).iloc[0]
    tiles = plan_tiles(bbox_utm, args.scale, len(names))
    est = sum(t.nx * t.ny for t in tiles)
    LOGGER.info("Grid: %.0f m pixels, %d tiles, ~%.2f million pixels per period.",
                args.scale, len(tiles), est / 1e6)

    cfg = IdxConfig(bbox=tuple(args.bbox), start=args.start, end=args.end, sensors=(args.sensor,),
                    composite="monthly", water_threshold=0, veg_threshold=0.3, built_threshold=0,
                    cs_threshold=args.cs_threshold, max_scene_cloud=args.max_scene_cloud,
                    scale=args.scale, out=args.out, export_mode="download", vectors=False,
                    vector_scale=args.scale, min_polygon_m2=0, scene_stats=False)
    region = ee.Geometry.Rectangle(list(args.bbox))
    windows = month_windows(args.start, args.end) if args.composite == "monthly" \
        else [(f"{args.start}_{args.end}", args.start, args.end)]

    tile_dir = args.out / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    gpkg = args.out / f"pixels_{args.sensor}.gpkg"

    for label, s, e in windows:
        col = collection_for(args.sensor, cfg, region, s, e)
        n = col.size().getInfo()
        if n == 0:
            LOGGER.info("%s: no images, skipped.", label)
            continue
        image = stats_image(col, indices)
        layer = f"pixels_{label.replace('-', '_')}"
        total, first = 0, True
        for t in tiles:
            tif = tile_dir / f"{args.sensor}_{label}_r{t.row}_c{t.col}.tif"
            if not tif.exists():
                download_tile(image, t, args.scale, tif)
                _write_band_names(tif, names)
            gdf = tif_to_pixel_polygons(tif, names, keep_area)
            if len(gdf):
                gdf = classify_means(gdf, args.water_threshold, args.veg_threshold,
                                     args.built_threshold)
                append_layer(gdf, gpkg, layer, first)
                first = False
                total += len(gdf)
            LOGGER.info("%s tile r%d c%d: %d pixels (running total %d).",
                        label, t.row, t.col, len(gdf), total)
        LOGGER.info("%s: %d pixel polygons -> %s layer '%s' (%d images).",
                    label, total, gpkg.name, layer, n)
    return 0


def _write_band_names(path: Path, names: list[str]) -> None:
    import rasterio

    with rasterio.open(path, "r+") as ds:
        ds.nodata = NODATA
        if ds.count == len(names):
            for i, n in enumerate(names, start=1):
                ds.set_band_description(i, n)


def run_from_tif(args) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    gpkg = args.out / "pixels_from_tif.gpkg"
    first = True
    for tif in args.from_tif:
        gdf = tif_to_pixel_polygons(Path(tif), args.bands)
        gdf = classify_means(gdf, args.water_threshold, args.veg_threshold, args.built_threshold)
        layer = Path(tif).stem.replace("-", "_")
        append_layer(gdf, gpkg, layer, True)
        LOGGER.info("%s: %d pixel polygons -> %s layer '%s'", tif, len(gdf), gpkg.name, layer)
        first = False
    return 0


def self_test() -> int:
    from rasterio.transform import from_origin

    ok = True

    def check(label, cond):
        nonlocal ok
        ok &= bool(cond)
        LOGGER.info("%-50s %s", label, "PASS" if cond else "FAIL")

    names = band_names(DEFAULT_INDICES)
    check("13 bands (3 indices x 4 stats + n_clear)", len(names) == 13)
    tiles = plan_tiles((700000, 3140000, 750000, 3195000), 30, 13)
    covered = sum(t.nx * t.ny for t in tiles)
    expected = math.ceil(50000 / 30) * math.ceil(55000 / 30)
    check(f"tiles cover grid exactly ({len(tiles)} tiles)", covered == expected)
    check("each tile under size limit", all(t.nx * t.ny * 13 * 4 <= DIRECT_LIMIT_BYTES for t in tiles))
    check("tiles do not overlap (edges abut)",
          all(abs(tiles[i + 1].x0 - (tiles[i].x0 + tiles[i].nx * 30)) < 1e-6
              for i in range(len(tiles) - 1) if tiles[i + 1].row == tiles[i].row))

    # 2x2 raster: one pixel empty (n_clear=0) -> 3 polygons, exact squares
    data = np.full((13, 2, 2), 0.1, dtype="float32")
    data[names.index("ndvi_mean")] = [[0.6, 0.1], [0.05, 0.0]]
    data[names.index("ndwi_mean")] = [[-0.5, -0.2], [0.3, 0.0]]
    data[names.index("ndbi_mean")] = [[-0.3, 0.1], [-0.4, 0.0]]
    data[names.index("n_clear")] = [[5, 4], [3, 0]]
    tr = from_origin(700000, 3150000, 30, 30)
    g = pixel_polygons_from_arrays(data, names, tr, UTM_CRS)
    check("empty pixel dropped (3 of 4 kept)", len(g) == 3)
    check("pixel polygons are 30 x 30 m", np.allclose(g.area, 900))
    g = classify_means(g, 0.0, 0.3, 0.0)
    check("classes: vegetation, built-up, water",
          list(g["class_name"]) == ["vegetation", "built_up", "water"])
    check("values carried exactly", float(g.iloc[0]["ndvi_mean"]) == 0.6)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    p = argparse.ArgumentParser(description="Every pixel as a polygon with NDVI/NDWI/NDBI stats.")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--project")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--sensor", choices=["s2", "landsat"], default="s2")
    p.add_argument("--composite", choices=["monthly", "period"], default="monthly")
    p.add_argument("--indices", nargs="+", choices=["ndvi", "ndwi", "mndwi", "ndbi"],
                   default=list(DEFAULT_INDICES))
    p.add_argument("--scale", type=float, default=30.0, help="Pixel size in metres (10 = ~26M polygons)")
    p.add_argument("--water-threshold", type=float, default=0.0, help="NDWI mean > this = water")
    p.add_argument("--veg-threshold", type=float, default=0.3)
    p.add_argument("--built-threshold", type=float, default=0.0)
    p.add_argument("--cs-threshold", type=float, default=0.60)
    p.add_argument("--max-scene-cloud", type=float, default=60.0)
    p.add_argument("--from-tif", nargs="+", help="Convert existing GeoTIFF(s) instead of using GEE")
    p.add_argument("--bands", nargs="+", help="Band names for --from-tif if not stored in the file")
    p.add_argument("--out", type=Path, default=Path("./pixels_output"))
    a = p.parse_args(argv)

    if a.self_test:
        return self_test()
    try:
        if a.from_tif:
            return run_from_tif(a)
        if not (a.project and a.bbox and a.start and a.end):
            p.error("--project, --bbox, --start and --end are required (or use --from-tif)")
        a.out.mkdir(parents=True, exist_ok=True)
        return run_gee(a)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())