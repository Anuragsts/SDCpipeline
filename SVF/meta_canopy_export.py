"""
Export the Meta/WRI 1 m global canopy height map (metres) for a bounding box
from Google Earth Engine as a GeoTIFF - input for street_svf_transects.py --chm.

Source: projects/sat-io/open-datasets/facebook/meta-canopy-height
        (awesome-gee-community-catalog; 1 m, metres, imagery mostly 2018-2020,
        reported MAE ~2.8 m; CC BY 4.0)

Requirements:  pip install earthengine-api requests

Notes
-----
- Output is in UTM 43N (EPSG:32643) at 1 m so distances are in metres.
- 1 m pixels add up fast: ~2.5 M pixels per 1.6 x 1.6 km. Direct download has
  a per-request size limit; for larger areas use --mode drive.

Example
-------
    python meta_canopy_export.py --project my-gee-project \
        --bbox 77.209 28.624 77.226 28.639 --out cp_canopy_1m.tif
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import ee
import requests

LOGGER = logging.getLogger("meta_canopy")
COLLECTION_ID = "projects/sat-io/open-datasets/facebook/meta-canopy-height"
OUT_CRS = "EPSG:32643"
SCALE_M = 1


def init_earth_engine(project: str) -> None:
    try:
        ee.Initialize(project=project)
    except Exception:  # noqa: BLE001
        ee.Authenticate()
        ee.Initialize(project=project)


def build_image(region: ee.Geometry) -> ee.Image:
    col = ee.ImageCollection(COLLECTION_ID).filterBounds(region)
    n = col.size().getInfo()
    if n == 0:
        raise RuntimeError("No Meta canopy tiles over this area.")
    first = col.first()
    band = first.bandNames().getInfo()[0]
    LOGGER.info("Tiles over area: %d, band used: %s", n, band)
    return col.mosaic().select([band]).rename("canopy_height_m").toFloat().clip(region)


def log_stats(image: ee.Image, region: ee.Geometry) -> None:
    reducer = ee.Reducer.minMax().combine(ee.Reducer.percentile([50, 95]), sharedInputs=True)
    stats = image.reduceRegion(reducer, region, scale=5, maxPixels=1e10).getInfo()
    LOGGER.info("Canopy height stats (5 m sample): %s", stats)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    p = argparse.ArgumentParser(description="Export Meta/WRI 1 m canopy height from GEE.")
    p.add_argument("--project", required=True)
    p.add_argument("--bbox", nargs=4, type=float, required=True,
                   metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p.add_argument("--out", type=Path, default=Path("canopy_height_1m.tif"))
    p.add_argument("--mode", choices=["download", "drive"], default="download")
    p.add_argument("--drive-folder", default="meta_canopy")
    a = p.parse_args(argv)

    try:
        init_earth_engine(a.project)
        region = ee.Geometry.Rectangle(a.bbox)
        image = build_image(region)
        log_stats(image, region)

        if a.mode == "drive":
            task = ee.batch.Export.image.toDrive(
                image=image.unmask(0), description=a.out.stem[:100], folder=a.drive_folder,
                fileNamePrefix=a.out.stem, region=region, scale=SCALE_M, crs=OUT_CRS,
                maxPixels=1e13, fileFormat="GeoTIFF",
            )
            task.start()
            LOGGER.info("Drive export started (task %s).", task.id)
            return 0

        url = image.unmask(0).getDownloadURL(
            {"region": region, "scale": SCALE_M, "crs": OUT_CRS, "format": "GEO_TIFF"}
        )
        response = requests.get(url, timeout=600)
        response.raise_for_status()
        a.out.write_bytes(response.content)
        LOGGER.info("Saved %s", a.out)
    except ee.EEException as exc:
        LOGGER.error("Earth Engine error: %s (area may be too large for direct download; "
                     "try --mode drive)", exc)
        return 1
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())