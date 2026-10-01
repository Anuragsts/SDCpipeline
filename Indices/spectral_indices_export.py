"""
NDVI, NDWI, MNDWI and NDBI from Sentinel-2 and/or Landsat 8/9 via Google
Earth Engine -> GeoTIFF composites, CSV statistics, class polygons.

Indices
-------
ndvi   (NIR - Red) / (NIR + Red)          vegetation
ndwi   (Green - NIR) / (Green + NIR)      open water (McFeeters 1996)
mndwi  (Green - SWIR1) / (Green + SWIR1)  water in built-up areas (Xu 2006) - used for water mask
ndbi   (SWIR1 - NIR) / (SWIR1 + NIR)      built-up / bare (Zha et al. 2003)
       (NDBI is exactly -1 x NDMI/Gao-NDWI, so NDMI is not exported separately)

Simple class map (per composite)
--------------------------------
1 water       mndwi > --water-threshold
2 vegetation  ndvi  > --veg-threshold              (and not water)
3 built-up    ndbi  > --built-threshold and ndvi <= --veg-threshold (and not water)
0 other
Thresholds are common STARTING values, not validated for any city - tune them
against imagery before using areas quantitatively.

Data (Earth Engine)
-------------------
Sentinel-2: COPERNICUS/S2_SR_HARMONIZED, B4/B3/B8 at 10 m, B11 at 20 m (x 1e-4);
            clouds masked with GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED (cs_cdf).
            MNDWI and NDBI use B11, so their effective resolution is 20 m.
Landsat:    LANDSAT/LC08/C02/T1_L2 + LANDSAT/LC09/C02/T1_L2, SR_B4/B3/B5/B6 at 30 m
            (x 2.75e-5 - 0.2); QA_PIXEL bits 0-5 masked.

Outputs (in --out)
------------------
rasters/ndvi/<sensor>_<label>_ndvi.tif     one single-band GeoTIFF per index, in its own folder:
rasters/ndwi/..., rasters/mndwi/..., rasters/ndbi/..., rasters/class/..., rasters/n_clear/...
                                           (choose with --rasters; large ones go to Drive
                                           folders spectral_indices_<index>)
stats_composites.csv          mean indices; water / vegetation / built-up km2; clear fraction
stats_scenes.csv              same per acquisition date (with --scene-stats)
vectors/water/<sensor>_<label>_water.geojson   one polygon file per class (with --vectors):
vectors/vegetation/..., vectors/built_up/...    each with class_name + area_m2

Large areas at 10 m exceed the direct-download size; the script estimates the
size and switches that raster to a Google Drive export automatically.

Requirements:  pip install earthengine-api requests rasterio

Example
-------
    python spectral_indices_export.py --project my-gee-project --bbox 76.84 28.40 77.35 28.88 \
        --start 2025-10-01 --end 2026-03-31 --sensor s2 --composite monthly \
        --scene-stats --vectors --out ./indices_delhi
    python spectral_indices_export.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

LOGGER = logging.getLogger("spectral_indices")

S2_ID = "COPERNICUS/S2_SR_HARMONIZED"
CS_PLUS_ID = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
L8_ID, L9_ID = "LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2"
INDICES = ("ndvi", "ndwi", "mndwi", "ndbi")
RASTER_BANDS = list(INDICES) + ["class", "n_clear"]
CLASS_NAMES = {1: "water", 2: "vegetation", 3: "built_up"}
NATIVE_SCALE = {"s2": 10, "landsat": 30}
DIRECT_LIMIT_BYTES = 30 * 1024 * 1024        # conservative GEE direct-download size


@dataclass(frozen=True)
class Config:
    bbox: tuple[float, float, float, float]
    start: str
    end: str
    sensors: tuple[str, ...]
    composite: str
    water_threshold: float
    veg_threshold: float
    built_threshold: float
    cs_threshold: float
    max_scene_cloud: float
    scale: float | None
    out: Path
    export_mode: str
    vectors: bool
    vector_scale: float
    min_polygon_m2: float
    scene_stats: bool
    raster_bands: tuple[str, ...] = tuple(RASTER_BANDS)


# --------------------------------------------------------------------------
# Pure helpers (tested offline)
# --------------------------------------------------------------------------
def month_windows(start: str, end: str) -> list[tuple[str, str, str]]:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    out, cur = [], date(s.year, s.month, 1)
    while cur < e:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        out.append((f"{cur:%Y-%m}", max(cur, s).isoformat(), min(nxt, e).isoformat()))
        cur = nxt
    return out


def estimate_bytes(bbox, scale_m: float, n_bands: int, bytes_per_px: int = 4) -> int:
    west, south, east, north = bbox
    lat = math.radians((south + north) / 2)
    width_m = (east - west) * 111_320 * math.cos(lat)
    height_m = (north - south) * 110_574
    return int((width_m / scale_m) * (height_m / scale_m) * n_bands * bytes_per_px)


def nd(a: float, b: float) -> float:
    return (a - b) / (a + b)


def classify(ndvi: float, mndwi: float, ndbi: float, wt: float, vt: float, bt: float) -> int:
    """Pure-Python mirror of the Earth Engine class rule (used by the self-test)."""
    if mndwi > wt:
        return 1
    if ndvi > vt:
        return 2
    if ndbi > bt:
        return 3
    return 0


# --------------------------------------------------------------------------
# Earth Engine: collections and indices
# --------------------------------------------------------------------------
def init_earth_engine(project: str) -> None:
    import ee

    try:
        ee.Initialize(project=project)
    except Exception:  # noqa: BLE001
        ee.Authenticate()
        ee.Initialize(project=project)
    LOGGER.info("Earth Engine initialised (project=%s).", project)


def s2_collection(region, start: str, end: str, cs_threshold: float, max_cloud: float):
    import ee

    cs = ee.ImageCollection(CS_PLUS_ID)

    def prep(img):
        clear = img.select("cs_cdf").gte(cs_threshold)
        bands = (img.select(["B4", "B3", "B8", "B11"], ["red", "green", "nir", "swir1"])
                 .multiply(1e-4))
        return bands.updateMask(clear).copyProperties(img, ["system:time_start"])

    return (ee.ImageCollection(S2_ID)
            .filterBounds(region).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
            .linkCollection(cs, ["cs_cdf"])
            .map(lambda i: ee.Image(prep(i))))


def landsat_collection(region, start: str, end: str, max_cloud: float):
    import ee

    def prep(img):
        clear = img.select("QA_PIXEL").bitwiseAnd(0b111111).eq(0)   # fill/dilated/cirrus/cloud/shadow/snow
        bands = (img.select(["SR_B4", "SR_B3", "SR_B5", "SR_B6"], ["red", "green", "nir", "swir1"])
                 .multiply(2.75e-05).add(-0.2))
        return bands.updateMask(clear).copyProperties(img, ["system:time_start"])

    col = ee.ImageCollection(L8_ID).merge(ee.ImageCollection(L9_ID))
    return (col.filterBounds(region).filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
            .map(lambda i: ee.Image(prep(i))))


def add_indices(img):
    import ee

    img = ee.Image(img)
    return ee.Image.cat([
        img.normalizedDifference(["nir", "red"]).rename("ndvi"),
        img.normalizedDifference(["green", "nir"]).rename("ndwi"),
        img.normalizedDifference(["green", "swir1"]).rename("mndwi"),
        img.normalizedDifference(["swir1", "nir"]).rename("ndbi"),
    ])


def class_band(med, cfg: Config):
    water = med.select("mndwi").gt(cfg.water_threshold)
    veg = med.select("ndvi").gt(cfg.veg_threshold).And(water.Not())
    built = (med.select("ndbi").gt(cfg.built_threshold)
             .And(med.select("ndvi").lte(cfg.veg_threshold)).And(water.Not()))
    return (water.multiply(1).add(veg.multiply(2)).add(built.multiply(3))
            .rename("class").updateMask(med.select("ndvi").mask()))


def composite(col, cfg: Config, region):
    idx = col.map(add_indices)
    med = idx.median()
    n_clear = idx.select("ndvi").count().rename("n_clear")
    return (med.addBands([class_band(med, cfg).toFloat(), n_clear.toFloat()])
            .select(RASTER_BANDS).toFloat().clip(region))


def collection_for(sensor: str, cfg: Config, region, start: str, end: str):
    if sensor == "s2":
        return s2_collection(region, start, end, cfg.cs_threshold, cfg.max_scene_cloud)
    return landsat_collection(region, start, end, cfg.max_scene_cloud)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def _r(v, nd_: int = 4):
    return round(v, nd_) if isinstance(v, (int, float)) else None


def area_stats(image, region, scale: float) -> dict:
    import ee

    cls = image.select("class")
    px = ee.Image.pixelArea()
    areas = ee.Image.cat([
        cls.eq(1).multiply(px).rename("water_m2"),
        cls.eq(2).multiply(px).rename("veg_m2"),
        cls.eq(3).multiply(px).rename("built_m2"),
        cls.mask().multiply(px).rename("valid_m2"),
        px.rename("total_m2"),
    ])
    means = image.select(list(INDICES) + ["n_clear"]).reduceRegion(ee.Reducer.mean(), region, scale,
                                                                   maxPixels=1e10, tileScale=4)
    sums = areas.reduceRegion(ee.Reducer.sum(), region, scale, maxPixels=1e10, tileScale=4)
    res = ee.Dictionary(means).combine(sums).getInfo()
    total = res.get("total_m2") or 0
    return {
        **{f"{k}_mean": _r(res.get(k)) for k in INDICES},
        "water_km2": _r((res.get("water_m2") or 0) / 1e6),
        "vegetation_km2": _r((res.get("veg_m2") or 0) / 1e6),
        "built_up_km2": _r((res.get("built_m2") or 0) / 1e6),
        "clear_fraction": _r((res.get("valid_m2") or 0) / total, 3) if total else None,
        "mean_clear_views": _r(res.get("n_clear"), 1),
    }


def retry(fn, what: str, retries: int = 4):
    """Retry Earth Engine AND network failures (DNS, dropped connection) with backoff."""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - ee, requests, google-auth and socket errors
            msg = str(exc)
            network = any(t in msg for t in ("NameResolution", "getaddrinfo", "Connection aborted",
                                              "RemoteDisconnected", "Max retries exceeded",
                                              "timed out", "oauth2.googleapis.com"))
            if attempt == retries:
                if network:
                    raise ConnectionError(f"{what}: network unavailable ({msg[:120]}). Check your "
                                          "internet connection, then re-run - finished work is kept.") from exc
                raise
            wait = 30 * attempt if network else 10 * attempt
            LOGGER.warning("%s failed (%s: %s); retry %d/%d in %d s.", what,
                           "network" if network else "Earth Engine", msg[:120], attempt,
                           retries - 1, wait)
            time.sleep(wait)


# --------------------------------------------------------------------------
# Exports
# --------------------------------------------------------------------------
def _write_band_names(path: Path, names: list[str]) -> None:
    try:
        import rasterio
    except ImportError:
        LOGGER.warning("rasterio not installed - band names not written.")
        return
    with rasterio.open(path, "r+") as ds:
        if ds.count == len(names):
            for i, n in enumerate(names, start=1):
                ds.set_band_description(i, n)


def _download_or_drive(image, stem: str, folder: str, cfg: Config, region, scale: float,
                       band_name: str) -> None:
    """Export one single-band image: direct download if small enough, else Google Drive."""
    import ee
    import requests

    size = estimate_bytes(cfg.bbox, scale, 1)
    to_drive = cfg.export_mode == "drive" or (cfg.export_mode == "auto" and size > DIRECT_LIMIT_BYTES)
    if to_drive:
        task = ee.batch.Export.image.toDrive(image=image, description=stem[:100],
                                             folder=f"spectral_indices_{band_name}",
                                             fileNamePrefix=stem, region=region, scale=scale,
                                             crs="EPSG:4326", maxPixels=1e13)
        task.start()
        LOGGER.info("  %s: ~%.0f MB -> Drive folder 'spectral_indices_%s' (task %s)",
                    stem, size / 1e6, band_name, task.id)
        return
    url = image.getDownloadURL({"region": region, "scale": scale, "crs": "EPSG:4326",
                                "format": "GEO_TIFF"})
    path = cfg.out / "rasters" / folder / f"{stem}.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=900)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    _write_band_names(path, [band_name])
    LOGGER.info("  saved %s", path)


def export_raster(image, name: str, cfg: Config, region, scale: float) -> None:
    """One GeoTIFF per index/band: rasters/<band>/<sensor>_<label>_<band>.tif"""
    for band in cfg.raster_bands:
        _download_or_drive(image.select(band), f"{name}_{band}", band, cfg, region, scale, band)


def export_class_polygons(image, name: str, cfg: Config, region) -> None:
    """One vector file per class: vectors/<class>/<sensor>_<label>_<class>.geojson"""
    import ee
    import requests

    cls = image.select("class").toInt()
    vectors = cls.updateMask(cls.gt(0)).reduceToVectors(
        geometry=region, scale=cfg.vector_scale, geometryType="polygon", eightConnected=True,
        labelProperty="class", maxPixels=1e10, bestEffort=True, tileScale=4)
    vectors = (vectors.map(lambda f: f.set("area_m2", f.geometry().area(1)))
               .filter(ee.Filter.gte("area_m2", cfg.min_polygon_m2)))

    for code, cname in CLASS_NAMES.items():
        subset = vectors.filter(ee.Filter.eq("class", code)).map(
            lambda f, n=cname: f.set("class_name", n))
        stem = f"{name}_{cname}"
        path = cfg.out / "vectors" / cname / f"{stem}.geojson"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            url = subset.getDownloadURL(filetype="geojson", filename=stem)
            resp = requests.get(url, timeout=900)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            LOGGER.info("  saved %s", path)
        except Exception as exc:  # noqa: BLE001 - usually too large for direct download
            LOGGER.info("  %s: direct download failed (%s) -> Drive.", stem, str(exc)[:100])
            task = ee.batch.Export.table.toDrive(collection=subset, description=stem[:100],
                                                 folder=f"spectral_indices_{cname}",
                                                 fileNamePrefix=stem, fileFormat="GeoJSON")
            task.start()
            LOGGER.info("  %s -> Drive folder 'spectral_indices_%s' (task %s)", stem, cname, task.id)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
STAT_FIELDS = [f"{k}_mean" for k in INDICES] + ["water_km2", "vegetation_km2",
                                                "built_up_km2", "clear_fraction", "mean_clear_views"]


def scene_stats(sensor: str, cfg: Config, region, scale: float, writer, done: set,
                flush) -> None:
    import ee

    col = collection_for(sensor, cfg, region, cfg.start, cfg.end)
    stamps = retry(lambda: col.aggregate_array("system:time_start").getInfo(), f"{sensor} dates")
    days = sorted({time.strftime("%Y-%m-%d", time.gmtime(t / 1000)) for t in stamps})
    todo = [d for d in days if (sensor, d) not in done]
    LOGGER.info("%s: %d acquisition dates (%d already done, %d to do).", sensor, len(days),
                len(days) - len(todo), len(todo))
    for i, d in enumerate(todo, start=1):
        img = composite(col.filterDate(d, ee.Date(d).advance(1, "day")), cfg, region)
        stats = retry(lambda: area_stats(img, region, scale), f"{sensor} {d}")
        writer.writerow({"sensor": sensor, "date": d, **stats})
        flush()                                     # keep finished dates if interrupted
        LOGGER.info("%s %s done (%d/%d).", sensor, d, i, len(todo))


def _done_scenes(path: Path) -> set:
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        return {(r["sensor"], r["date"]) for r in csv.DictReader(fh)}


def run(args) -> int:
    import ee

    cfg = Config(
        bbox=tuple(args.bbox), start=args.start, end=args.end,
        sensors=("s2", "landsat") if args.sensor == "both" else (args.sensor,),
        composite=args.composite, water_threshold=args.water_threshold,
        veg_threshold=args.veg_threshold, built_threshold=args.built_threshold,
        cs_threshold=args.cs_threshold, max_scene_cloud=args.max_scene_cloud,
        scale=args.scale, out=args.out, export_mode=args.export_mode, vectors=args.vectors,
        vector_scale=args.vector_scale, min_polygon_m2=args.min_polygon_area,
        scene_stats=args.scene_stats, raster_bands=tuple(args.rasters),
    )
    cfg.out.mkdir(parents=True, exist_ok=True)
    init_earth_engine(args.project)
    region = ee.Geometry.Rectangle(list(cfg.bbox))
    windows = month_windows(cfg.start, cfg.end) if cfg.composite == "monthly" \
        else [(f"{cfg.start}_{cfg.end}", cfg.start, cfg.end)]

    if args.skip_composites:
        LOGGER.info("--skip-composites: composites, rasters and vectors not re-run.")
        windows = []
    comp_path = cfg.out / "stats_composites.csv"
    with open(comp_path, "a" if args.skip_composites else "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sensor", "label", "start", "end", "n_images"] + STAT_FIELDS)
        if not args.skip_composites:
            writer.writeheader()
        for sensor in cfg.sensors:
            scale = cfg.scale or NATIVE_SCALE[sensor]
            for label, s, e in windows:
                col = collection_for(sensor, cfg, region, s, e)
                n = retry(lambda: col.size().getInfo(), f"{sensor} {label} size")
                if n == 0:
                    LOGGER.info("%s %s: no images, skipped.", sensor, label)
                    continue
                img = composite(col, cfg, region)
                stats = retry(lambda: area_stats(img, region, scale), f"{sensor} {label}")
                writer.writerow({"sensor": sensor, "label": label, "start": s, "end": e,
                                 "n_images": n, **stats})
                fh.flush()
                LOGGER.info("%s %s: %d imgs | NDVI %.2f | water %.2f, veg %.2f, built %.2f km2 | "
                            "clear views/pixel %.1f", sensor, label, n, stats["ndvi_mean"] or float("nan"),
                            stats["water_km2"] or 0, stats["vegetation_km2"] or 0,
                            stats["built_up_km2"] or 0, stats["mean_clear_views"] or 0)
                name = f"{sensor}_{label}"
                export_raster(img, name, cfg, region, scale)
                if cfg.vectors:
                    export_class_polygons(img, name, cfg, region)

    if cfg.scene_stats:
        scene_path = cfg.out / "stats_scenes.csv"
        done = _done_scenes(scene_path)
        fields = ["sensor", "date"] + STAT_FIELDS
        if done:
            with open(scene_path, newline="", encoding="utf-8") as fh:
                old_fields = next(csv.reader(fh))
            if old_fields != fields:                      # older file without new columns
                LOGGER.info("stats_scenes.csv has an older column layout - starting it fresh.")
                scene_path.unlink()
                done = set()
        new_file = not scene_path.exists()
        with open(scene_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if new_file:
                writer.writeheader()
            for sensor in cfg.sensors:
                scene_stats(sensor, cfg, region, cfg.scale or NATIVE_SCALE[sensor], writer, done,
                            fh.flush)
    LOGGER.info("Done. Outputs in %s", cfg.out)
    return 0


def self_test() -> int:
    ok = True

    def check(label, cond):
        nonlocal ok
        ok &= bool(cond)
        LOGGER.info("%-46s %s", label, "PASS" if cond else "FAIL")

    # Typical surface reflectances (red, green, nir, swir1) - illustrative values.
    water = dict(red=0.04, green=0.07, nir=0.03, swir1=0.02)
    trees = dict(red=0.04, green=0.08, nir=0.40, swir1=0.18)
    roofs = dict(red=0.20, green=0.18, nir=0.25, swir1=0.32)

    def idx(s):
        return (nd(s["nir"], s["red"]), nd(s["green"], s["nir"]),
                nd(s["green"], s["swir1"]), nd(s["swir1"], s["nir"]))

    for label, s, expected in (("water pixel -> class 1", water, 1),
                               ("vegetation pixel -> class 2", trees, 2),
                               ("built-up pixel -> class 3", roofs, 3)):
        ndvi, _, mndwi, ndbi = idx(s)
        check(label, classify(ndvi, mndwi, ndbi, 0.0, 0.3, 0.0) == expected)
    check("NDBI == -NDMI (same bands swapped)",
          abs(nd(roofs["swir1"], roofs["nir"]) + nd(roofs["nir"], roofs["swir1"])) < 1e-12)
    check("monthly windows Oct-15..Jan-10 -> 4", len(month_windows("2025-10-15", "2026-01-10")) == 4)
    check("Delhi @10 m x6 bands exceeds direct limit",
          estimate_bytes((76.84, 28.40, 77.35, 28.88), 10, 6) > DIRECT_LIMIT_BYTES)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    p = argparse.ArgumentParser(description="NDVI / NDWI / MNDWI / NDBI via Google Earth Engine.")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--project")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p.add_argument("--start", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", help="YYYY-MM-DD (exclusive)")
    p.add_argument("--sensor", choices=["s2", "landsat", "both"], default="s2")
    p.add_argument("--composite", choices=["monthly", "period"], default="monthly")
    p.add_argument("--water-threshold", type=float, default=0.0, help="MNDWI above this = water")
    p.add_argument("--veg-threshold", type=float, default=0.3, help="NDVI above this = vegetation")
    p.add_argument("--built-threshold", type=float, default=0.0, help="NDBI above this = built-up")
    p.add_argument("--cs-threshold", type=float, default=0.60,
                   help="Sentinel-2 Cloud Score+ cs_cdf threshold (catalogue: 0.50-0.65)")
    p.add_argument("--max-scene-cloud", type=float, default=60.0)
    p.add_argument("--scale", type=float, help="Output pixel size m (default 10 S2 / 30 Landsat)")
    p.add_argument("--export-mode", choices=["auto", "download", "drive"], default="auto")
    p.add_argument("--rasters", nargs="+", choices=RASTER_BANDS, default=RASTER_BANDS,
                   help="Which rasters to export, one file each (default: all)")
    p.add_argument("--vectors", action="store_true",
                   help="Export water / vegetation / built-up polygons, one file per class")
    p.add_argument("--vector-scale", type=float, default=30.0)
    p.add_argument("--min-polygon-area", type=float, default=900.0, help="m2")
    p.add_argument("--scene-stats", action="store_true")
    p.add_argument("--skip-composites", action="store_true",
                   help="Only run per-date stats (e.g. to resume after an interruption)")
    p.add_argument("--out", type=Path, default=Path("./indices_output"))
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.project and a.bbox and a.start and a.end):
        p.error("--project, --bbox, --start and --end are required")
    try:
        return run(a)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())