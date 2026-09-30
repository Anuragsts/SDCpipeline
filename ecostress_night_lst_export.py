"""
Export ECOSTRESS night-time Land Surface Temperature (70 m) from Google
Earth Engine as GeoTIFFs (degrees Celsius) for use in QGIS.

Why ECOSTRESS: it flies on the ISS, whose orbit is not sun-synchronous, so
overpass times vary and include night passes, at ~70 m resolution
(vs ~1 km for MODIS/VIIRS). Landsat 8/9 pass at ~10:30 AM only.

Pipeline
--------
1. Load NASA/ECOSTRESS/L2T_LSTE/V2 (tiled), filter by date and area.
2. Tag each tile with its local hour (IST) and keep night passes only.
3. Group tiles from the same overpass and mosaic them into one image.
4. Mask: cloud band, QC mandatory bits, optional LST_err threshold.
5. Convert K -> degrees C, clip, sanity-check value range.
6. Save one float32 GeoTIFF per night pass (NoData = -9999).

Requirements
------------
    pip install earthengine-api requests rasterio

VERIFY BEFORE PRODUCTION USE
----------------------------
- QC bit meaning (bits 0-1 == 00 treated as best quality) against the
  ECOSTRESS L2 LSTE user guide.
- NASA alert (June 2026): LST_err missing for 2025-12-16 .. 2026-06-10.
  The script detects this and skips the error filter for those passes.
- Check the logged min/max: LST is expected in Kelvin; if values look
  already scaled or wrong, stop and check the catalogue page.

Example
-------
    python ecostress_night_lst_export.py --project my-gee-project \
        --bbox 76.84 28.40 77.35 28.88 --start 2026-06-01 --end 2026-09-27 \
        --out-dir ./ecostress_delhi
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import ee
import requests

LOGGER = logging.getLogger("ecostress_night_lst")

COLLECTION_ID = "NASA/ECOSTRESS/L2T_LSTE/V2"
LST_BAND, ERR_BAND, QC_BAND, CLOUD_BAND = "LST", "LST_err", "QC", "cloud"
OUTPUT_BAND = "LST_Night_C"
KELVIN_OFFSET = 273.15
TIMEZONE = "Asia/Kolkata"
PASS_GAP_MS = 15 * 60 * 1000          # tiles within 15 min = same overpass
PLAUSIBLE_C = (-10.0, 60.0)           # sanity range for night LST in India


@dataclass(frozen=True)
class ExportConfig:
    project: str
    bbox: tuple[float, float, float, float]   # west, south, east, north
    start: str                                # inclusive YYYY-MM-DD
    end: str                                  # exclusive YYYY-MM-DD
    night_start_hour: int = 20                # local hour, inclusive
    night_end_hour: int = 5                   # local hour, exclusive
    qc_mode: str = "any"                      # "good" | "any" (cloud mask always applied)
    max_err_k: float | None = None            # e.g. 1.5; None = no error filter
    crs: str = "EPSG:4326"
    scale_m: float = 70.0
    nodata: float = -9999.0
    out_dir: Path = Path("./ecostress_output")
    skip_empty: bool = True


# --------------------------------------------------------------------------
# Earth Engine setup
# --------------------------------------------------------------------------
def init_earth_engine(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception:  # noqa: BLE001 - ee raises generic exceptions when unauthenticated
        LOGGER.info("Earth Engine not authenticated; starting authentication flow.")
        ee.Authenticate()
        ee.Initialize(project=project)
    LOGGER.info("Earth Engine initialised (project=%s).", project)


# --------------------------------------------------------------------------
# Collection building and night filtering
# --------------------------------------------------------------------------
def tag_local_hour(image: ee.Image) -> ee.Image:
    """Attach the acquisition hour in local time (IST) as a property."""
    hour = ee.Date(image.get("system:time_start")).get("hour", TIMEZONE)
    return image.set("local_hour", hour)


def build_night_collection(cfg: ExportConfig, region: ee.Geometry) -> ee.ImageCollection:
    """Tiles over the area, within the dates, acquired during local night."""
    night = ee.Filter.Or(
        ee.Filter.gte("local_hour", cfg.night_start_hour),
        ee.Filter.lt("local_hour", cfg.night_end_hour),
    )
    return (
        ee.ImageCollection(COLLECTION_ID)
        .filterDate(cfg.start, cfg.end)
        .filterBounds(region)
        .map(tag_local_hour)
        .filter(night)
    )


def group_into_passes(timestamps_ms: list[int]) -> list[tuple[int, int]]:
    """Cluster tile timestamps into overpasses; return (first_ms, last_ms) per pass."""
    passes: list[tuple[int, int]] = []
    for t in sorted(timestamps_ms):
        if passes and t - passes[-1][1] <= PASS_GAP_MS:
            passes[-1] = (passes[-1][0], t)
        else:
            passes.append((t, t))
    return passes


# --------------------------------------------------------------------------
# Per-pass processing
# --------------------------------------------------------------------------
def to_celsius(tiles: ee.ImageCollection, cfg: ExportConfig, region: ee.Geometry,
               has_err_band: bool) -> ee.Image:
    """Mosaic one pass, apply masks, convert to degrees Celsius, clip."""
    mosaic = tiles.mosaic()

    mask = mosaic.select(CLOUD_BAND).eq(0)                      # clear sky only
    if cfg.qc_mode == "good":
        mask = mask.And(mosaic.select(QC_BAND).bitwiseAnd(0b11).eq(0))
    if cfg.max_err_k is not None and has_err_band:
        mask = mask.And(mosaic.select(ERR_BAND).lte(cfg.max_err_k))

    return (
        mosaic.select(LST_BAND)
        .subtract(KELVIN_OFFSET)
        .updateMask(mask)
        .rename(OUTPUT_BAND)
        .toFloat()
        .clip(region)
    )


def summarise(image: ee.Image, region: ee.Geometry, scale_m: float) -> dict[str, float | None]:
    """Valid pixel count plus min/max in one server call."""
    reducer = ee.Reducer.count().combine(ee.Reducer.minMax(), sharedInputs=True)
    stats = image.reduceRegion(reducer=reducer, geometry=region,
                               scale=scale_m, maxPixels=1e10).getInfo()
    # Key names are "<band>_count", "<band>_min", "<band>_max"; match by suffix
    # so the code does not depend on the exact prefix format.
    pick = lambda suffix: next((v for k, v in stats.items() if k.endswith(suffix)), None)  # noqa: E731
    return {"count": pick("_count") or 0, "min": pick("_min"), "max": pick("_max")}


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def download_geotiff(image: ee.Image, path: Path, cfg: ExportConfig, region: ee.Geometry,
                     retries: int = 3, timeout_s: int = 180) -> None:
    url = image.unmask(cfg.nodata).getDownloadURL(
        {"region": region, "scale": cfg.scale_m, "crs": cfg.crs, "format": "GEO_TIFF"}
    )
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout_s)
            response.raise_for_status()
            path.write_bytes(response.content)
            return
        except requests.RequestException as exc:
            LOGGER.warning("Download attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)


def set_nodata_tag(path: Path, nodata: float) -> None:
    try:
        import rasterio
    except ImportError:
        LOGGER.warning("rasterio not installed; set NoData=%s manually in QGIS.", nodata)
        return
    with rasterio.open(path, "r+") as dataset:
        dataset.nodata = nodata


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run(cfg: ExportConfig) -> None:
    init_earth_engine(cfg.project)
    region = ee.Geometry.Rectangle(list(cfg.bbox))
    night_tiles = build_night_collection(cfg, region)

    timestamps = night_tiles.aggregate_array("system:time_start").getInfo()
    passes = group_into_passes(timestamps)
    LOGGER.info("Found %d night tiles in %d overpasses (%02d:00-%02d:00 IST).",
                len(timestamps), len(passes), cfg.night_start_hour, cfg.night_end_hour)
    if not passes:
        LOGGER.warning("No night passes found. Widen dates or night hours.")
        return

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    saved = skipped = 0

    for first_ms, last_ms in passes:
        tiles = night_tiles.filterDate(ee.Date(first_ms - 60_000), ee.Date(last_ms + 60_000))
        label = ee.Date(first_ms).format("YYYY-MM-dd_HHmm", TIMEZONE).getInfo()

        band_names = tiles.first().bandNames().getInfo()
        has_err = ERR_BAND in band_names
        if cfg.max_err_k is not None and not has_err:
            LOGGER.warning("%s IST: LST_err band missing (known NASA issue); error filter skipped.",
                           label)

        lst = to_celsius(tiles, cfg, region, has_err)
        stats = summarise(lst, region, cfg.scale_m)

        if stats["count"] == 0 and cfg.skip_empty:
            LOGGER.info("%s IST: 0 valid pixels (cloud/outside swath), skipped.", label)
            skipped += 1
            continue

        LOGGER.info("%s IST: %d valid pixels, min %.1f C, max %.1f C.", label,
                    stats["count"], stats["min"] or float("nan"), stats["max"] or float("nan"))
        lo, hi = PLAUSIBLE_C
        if stats["min"] is not None and (stats["min"] < lo or stats["max"] > hi):
            LOGGER.warning("%s IST: values outside %s C - check units/masking before use.",
                           label, PLAUSIBLE_C)

        path = cfg.out_dir / f"ECOSTRESS_{OUTPUT_BAND}_{label}IST.tif"
        download_geotiff(lst, path, cfg, region)
        set_nodata_tag(path, cfg.nodata)
        LOGGER.info("%s IST: saved %s", label, path)
        saved += 1

    LOGGER.info("Done. Saved: %d, skipped: %d.", saved, skipped)


def parse_args(argv: list[str] | None = None) -> ExportConfig:
    p = argparse.ArgumentParser(description="Export ECOSTRESS night LST GeoTIFFs from GEE.")
    p.add_argument("--project", required=True)
    p.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (exclusive)")
    p.add_argument("--night-start", type=int, default=20, help="Local hour night begins (0-23)")
    p.add_argument("--night-end", type=int, default=5, help="Local hour night ends (0-23)")
    p.add_argument("--qc-mode", choices=["good", "any"], default="any",
                   help="'good' = QC bits 0-1 == 00; 'any' = cloud mask only")
    p.add_argument("--max-err", type=float, default=None, help="Max LST_err in K, e.g. 1.5")
    p.add_argument("--crs", default="EPSG:4326")
    p.add_argument("--scale", type=float, default=70.0, help="Output pixel size in metres")
    p.add_argument("--out-dir", type=Path, default=Path("./ecostress_output"))
    p.add_argument("--keep-empty", action="store_true")
    a = p.parse_args(argv)

    west, south, east, north = a.bbox
    if not (west < east and south < north):
        p.error("bbox must be WEST SOUTH EAST NORTH with west<east and south<north")
    if not (0 <= a.night_start <= 23 and 0 <= a.night_end <= 23):
        p.error("night hours must be 0-23")

    return ExportConfig(
        project=a.project, bbox=(west, south, east, north), start=a.start, end=a.end,
        night_start_hour=a.night_start, night_end_hour=a.night_end, qc_mode=a.qc_mode,
        max_err_k=a.max_err, crs=a.crs, scale_m=a.scale, out_dir=a.out_dir,
        skip_empty=not a.keep_empty,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    LOGGER.info("Starting ECOSTRESS night LST export.")
    try:
        run(parse_args(argv))
    except ee.EEException as exc:
        LOGGER.error("Earth Engine error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())