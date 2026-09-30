"""
Road (carriageway) width from Overture Maps transportation segments, with a
road-class fallback, plus a coverage report.

Overture segment property used: `width_rules` - "edge-to-edge width of the
feature modeled by this segment, in meters" (optional). The exact internal
structure of width_rules is not shown in the schema page, so this script
parses it defensively (expects items with a numeric `value` and an optional
`between` = [start, end] fraction of the segment) and LOGS RAW EXAMPLES so you
can confirm the structure on first run.

Output (GeoPackage, layer "road_width"):
    id, class, subclass, name, length_m,
    width_overture_m   width from width_rules (length-weighted if scoped)
    width_est_m        width_overture_m if present, else class default
    width_source       "overture" | "class_default" | "none"
    n_width_rules
And a CSV coverage report per road class.

IMPORTANT: CLASS_DEFAULT_WIDTH_M values are ROUGH PLACEHOLDERS chosen for
illustration, not an official standard. Replace them (e.g. with IRC urban road
guidance or local measurements) before using width_est_m for analysis, or pass
your own table with --defaults-csv (columns: class,width_m).

Requirements:  pip install overturemaps geopandas pyarrow shapely

Example
-------
    python overture_road_width.py --bbox 77.209 28.624 77.226 28.639 --out cp_road_width.gpkg
    python overture_road_width.py --segments cp_segments.parquet --out cp_road_width.gpkg
    python overture_road_width.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger("overture_width")
METRIC_CRS = "EPSG:32643"          # UTM 43N, for lengths in metres (Delhi)

# ROUGH PLACEHOLDERS (metres, carriageway) - replace before real use.
CLASS_DEFAULT_WIDTH_M: dict[str, float] = {
    "motorway": 21.0, "trunk": 14.0, "primary": 14.0, "secondary": 10.5,
    "tertiary": 7.5, "residential": 5.5, "unclassified": 5.5, "living_street": 4.0,
    "service": 3.5, "pedestrian": 4.0, "footway": 2.0, "cycleway": 2.0,
    "steps": 2.0, "track": 3.0, "bridleway": 2.0,
}


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download_segments(bbox: tuple[float, float, float, float], out_path: Path) -> Path:
    """Download Overture transportation segments for bbox via the overturemaps CLI."""
    exe = shutil.which("overturemaps")
    if exe is None:
        raise RuntimeError("overturemaps CLI not found. Install with: pip install overturemaps")
    west, south, east, north = bbox
    cmd = [exe, "download", f"--bbox={west},{south},{east},{north}",
           "-f", "geoparquet", "--type=segment", "-o", str(out_path)]
    LOGGER.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"overturemaps download failed:\n{result.stderr[-1500:]}")
    LOGGER.info("Downloaded segments to %s", out_path)
    return out_path


def read_segments(path: Path):
    import geopandas as gpd

    try:
        gdf = gpd.read_parquet(path)
    except ValueError:
        # Plain Parquet without GeoParquet metadata: decode WKB geometry.
        import pandas as pd
        import shapely

        df = pd.read_parquet(path)
        gdf = gpd.GeoDataFrame(df, geometry=shapely.from_wkb(df["geometry"]), crs="EPSG:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    LOGGER.info("Segments read: %d (columns: %s)", len(gdf), ", ".join(map(str, gdf.columns)))
    return gdf


# --------------------------------------------------------------------------
# Parsing width_rules
# --------------------------------------------------------------------------
def _as_list(obj) -> list:
    """Normalise Parquet/Arrow list values (numpy arrays, tuples, None) to a list."""
    if obj is None:
        return []
    if isinstance(obj, float) and math.isnan(obj):
        return []
    if isinstance(obj, dict):
        return [obj]
    try:
        return list(obj)
    except TypeError:
        return []


def parse_width_rules(rules) -> tuple[float | None, int]:
    """Return (width_m, n_rules). Scoped rules are weighted by the fraction they cover."""
    items = _as_list(rules)
    total_w, total_frac, n = 0.0, 0.0, 0
    for item in items:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or value <= 0:
            continue
        between = _as_list(item.get("between"))
        if len(between) == 2 and all(b is not None for b in between):
            frac = max(float(between[1]) - float(between[0]), 0.0)
        else:
            frac = 1.0                                   # applies to the whole segment
        if frac == 0:
            continue
        total_w += value * frac
        total_frac += frac
        n += 1
    if total_frac == 0:
        return None, n
    return round(total_w / total_frac, 2), n


def primary_name(names) -> str | None:
    if isinstance(names, dict):
        return names.get("primary")
    return None


# --------------------------------------------------------------------------
# Processing
# --------------------------------------------------------------------------
def load_defaults(path: Path | None) -> dict[str, float]:
    if path is None:
        LOGGER.warning("Using built-in PLACEHOLDER class widths - replace with real values "
                       "(--defaults-csv) before analysis.")
        return dict(CLASS_DEFAULT_WIDTH_M)
    table: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            table[row["class"].strip()] = float(row["width_m"])
    LOGGER.info("Loaded %d class widths from %s", len(table), path)
    return table


def estimate_widths(segments, defaults: dict[str, float]):
    gdf = segments.copy()
    if "subtype" in gdf.columns:
        gdf = gdf[gdf["subtype"] == "road"].copy()
    if gdf.empty:
        raise RuntimeError("No road segments in input.")

    if "width_rules" not in gdf.columns:
        LOGGER.warning("No 'width_rules' column in this data - all widths will be class defaults.")
        gdf["width_rules"] = None

    examples = [r for r in gdf["width_rules"] if len(_as_list(r))][:3]
    LOGGER.info("Raw width_rules examples (confirm structure): %s",
                examples if examples else "none found")

    parsed = gdf["width_rules"].map(parse_width_rules)
    gdf["width_overture_m"] = parsed.map(lambda t: t[0])
    gdf["n_width_rules"] = parsed.map(lambda t: t[1])

    road_class = gdf["class"] if "class" in gdf.columns else "unknown"
    gdf["class"] = road_class
    gdf["class_default_m"] = gdf["class"].map(defaults)
    gdf["width_est_m"] = gdf["width_overture_m"].fillna(gdf["class_default_m"])
    gdf["width_source"] = "none"
    gdf.loc[gdf["class_default_m"].notna(), "width_source"] = "class_default"
    gdf.loc[gdf["width_overture_m"].notna(), "width_source"] = "overture"

    gdf["name"] = gdf["names"].map(primary_name) if "names" in gdf.columns else None
    gdf["length_m"] = gdf.to_crs(METRIC_CRS).length.round(1)

    keep = ["id", "class", "subclass", "name", "length_m", "width_overture_m",
            "class_default_m", "width_est_m", "width_source", "n_width_rules", "geometry"]
    return gdf[[c for c in keep if c in gdf.columns]]


def coverage_report(roads, csv_path: Path) -> None:
    rows = []
    for cls, grp in roads.groupby("class", dropna=False):
        has = grp["width_overture_m"].notna()
        rows.append({
            "class": cls,
            "segments": len(grp),
            "segments_with_width": int(has.sum()),
            "pct_segments_with_width": round(100 * has.mean(), 1),
            "length_km": round(grp["length_m"].sum() / 1000, 2),
            "pct_length_with_width": round(100 * grp.loc[has, "length_m"].sum()
                                           / max(grp["length_m"].sum(), 1e-9), 1),
            "median_overture_width_m": grp.loc[has, "width_overture_m"].median() if has.any() else None,
        })
    rows.sort(key=lambda r: -r["length_km"])
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total = roads["width_overture_m"].notna()
    LOGGER.info("OVERALL: %d/%d segments (%.1f%%) and %.1f%% of road length have an Overture width.",
                int(total.sum()), len(roads), 100 * total.mean(),
                100 * roads.loc[total, "length_m"].sum() / max(roads["length_m"].sum(), 1e-9))
    for r in rows[:10]:
        LOGGER.info("  %-14s %5d segs  %6.2f km  width on %5.1f%% of length  median %s m",
                    r["class"], r["segments"], r["length_km"], r["pct_length_with_width"],
                    r["median_overture_width_m"])
    LOGGER.info("Coverage report saved to %s", csv_path)


# --------------------------------------------------------------------------
# Self-test (offline)
# --------------------------------------------------------------------------
def self_test() -> int:
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import LineString

    cases = [
        ("whole-segment rule", [{"value": 7.0, "between": None}], 7.0),
        ("two scoped rules", [{"value": 10.0, "between": [0.0, 0.5]},
                              {"value": 6.0, "between": [0.5, 1.0]}], 8.0),
        ("numpy array input", np.array([{"value": 5.0}], dtype=object), 5.0),
        ("empty / missing", None, None),
        ("bad value skipped", [{"value": "abc"}, {"value": 4.0}], 4.0),
    ]
    ok = True
    for label, rules, expected in cases:
        got, _ = parse_width_rules(rules)
        passed = got == expected
        ok &= passed
        LOGGER.info("%-20s got %s expected %s  %s", label, got, expected, "PASS" if passed else "FAIL")

    seg = gpd.GeoDataFrame({
        "id": ["a", "b", "c"], "subtype": ["road", "road", "road"],
        "class": ["primary", "residential", "mystery"],
        "width_rules": [[{"value": 12.0}], None, None],
        "names": [{"primary": "Test Rd"}, None, None],
    }, geometry=[LineString([(77.2, 28.6), (77.201, 28.6)])] * 3, crs="EPSG:4326")
    out = estimate_widths(seg, dict(CLASS_DEFAULT_WIDTH_M))
    expected_src = ["overture", "class_default", "none"]
    passed = list(out["width_source"]) == expected_src and out["width_est_m"].iloc[0] == 12.0
    ok &= passed
    LOGGER.info("%-20s got %s  %s", "fallback logic", list(out["width_source"]),
                "PASS" if passed else "FAIL")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    p = argparse.ArgumentParser(description="Road width from Overture segments + class fallback.")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p.add_argument("--segments", type=Path, help="Existing Overture segments GeoParquet (skip download)")
    p.add_argument("--defaults-csv", type=Path, help="CSV with columns class,width_m")
    p.add_argument("--out", type=Path, default=Path("road_width.gpkg"))
    a = p.parse_args(argv)

    if a.self_test:
        return self_test()
    if a.segments is None and a.bbox is None:
        p.error("give --bbox (to download) or --segments (existing file)")

    try:
        seg_path = a.segments or download_segments(tuple(a.bbox),
                                                   a.out.with_name(f"{a.out.stem}_segments.parquet"))
        roads = estimate_widths(read_segments(seg_path), load_defaults(a.defaults_csv))
        roads.to_file(a.out, layer="road_width", driver="GPKG")
        LOGGER.info("Saved %s (%d road segments).", a.out, len(roads))
        coverage_report(roads, a.out.with_name(f"{a.out.stem}_coverage.csv"))
    except (RuntimeError, ValueError, KeyError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())