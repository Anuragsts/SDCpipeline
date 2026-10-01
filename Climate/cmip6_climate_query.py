"""
Query past and future climate (CMIP6) for an area from Google Earth Engine:
time-series CSVs (per model + ensemble summary) and period GeoTIFF maps.

Data
----
NASA/GDDP-CMIP6 (NEX-GDDP-CMIP6): statistically downscaled and bias-corrected
CMIP6 at ~0.25 deg (~27.8 km), DAILY, 1950-2100, 34 models.
Scenarios: historical (1950-2014), ssp245 and ssp585 (2015-2100).
Bands used: tas, tasmax, tasmin (K), pr (kg m-2 s-1), hurs (%), sfcWind (m s-1),
rsds, rlds (W m-2). huss (specific humidity) is deliberately not used.

Indices (per year or per month, per model)
------------------------------------------
means    tas_mean_c, tasmax_mean_c, tasmin_mean_c, pr_total_mm, pr_mean_mmday
heat     txx_c (hottest day), tnx_c (warmest night), hot_days (Tmax > --hot-day),
         warm_nights (Tmin > --warm-night)
rain     rx1day_mm (wettest day), heavy_rain_days (pr > --heavy-rain), wet_days (pr >= 1 mm)
humidity hurs_mean_pct (mean relative humidity)
wind     sfcwind_mean_ms, sfcwind_min_ms (mean / calmest-day wind speed)
radiation rsds_mean_wm2 (incoming sunlight), rlds_mean_wm2 (incoming longwave)
humid    twb_max_c, twb_mean_c: wet-bulb temperature from daily-mean tas + hurs using
         Stull (2011). Daily-mean inputs UNDERESTIMATE peak afternoon wet-bulb.

Outputs
-------
<out>/timeseries/ts_<scenario>_<model>.csv    area-weighted mean per period unit
<out>/timeseries_ensemble.csv                  mean / p10 / p50 / p90 / n_models
<out>/maps/<scenario>_<start>-<end>.tif        ensemble mean + p10/p50/p90 per index
<out>/maps/change_<scenario>_<start>-<end>.tif ensemble-mean change vs baseline

Time series are resumable: existing per-model CSVs are skipped (use --overwrite).

VERIFY / KNOWN LIMITS
---------------------
- ~27.8 km pixels: a city-sized box covers only a few cells. Area means are
  area-weighted by Earth Engine; for maps use a larger --map-bbox.
- Some models lack some variables (e.g. hurs); those indices are left empty for
  that model, and maps only use indices available in every selected model.
- The dataset notes that some days are interpolated (duplicated) for models
  that lack daily output - check the `interpolated` property if that matters.
- All 34 models x 3 scenarios x 151 years is a heavy job (many GEE requests).
  Start with --models or a short --years range to check results.

Requirements:  pip install earthengine-api requests pandas numpy

Examples
--------
    # Annual indices, all models, both SSPs + historical, Delhi box
    python cmip6_climate_query.py --project my-gee-project \
        --bbox 76.84 28.40 77.35 28.88 --mode timeseries --out ./cmip6_delhi

    # Quick test: 2 models, 2015-2030, monthly, heat + rain only
    python cmip6_climate_query.py --project my-gee-project --bbox 76.84 28.40 77.35 28.88 \
        --models ACCESS-CM2 MIROC6 --years 2015 2030 --freq monthly \
        --groups heat rain --out ./cmip6_test

    # Period maps over NCR (baseline 1985-2014 vs 2041-2060 and 2081-2100)
    python cmip6_climate_query.py --project my-gee-project --bbox 76.84 28.40 77.35 28.88 \
        --map-bbox 75.5 27.5 78.5 29.5 --mode maps --periods 2041-2060 2081-2100 --out ./cmip6_delhi

    python cmip6_climate_query.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

LOGGER = logging.getLogger("cmip6_query")

COLLECTION_ID = "NASA/GDDP-CMIP6"
NATIVE_SCALE_M = 27_830
KELVIN = 273.15
SECONDS_PER_DAY = 86_400
SCENARIO_YEARS = {"historical": (1950, 2014), "ssp245": (2015, 2100), "ssp585": (2015, 2100)}
GROUP_BANDS = {                                   # source bands each index group needs
    "means": {"tas", "tasmax", "tasmin", "pr"},
    "heat": {"tasmax", "tasmin"},
    "rain": {"pr"},
    "humid": {"tas", "hurs"},
    "humidity": {"hurs"},
    "wind": {"sfcWind"},
    "radiation": {"rsds", "rlds"},
}


@dataclass(frozen=True)
class Thresholds:
    hot_day_c: float = 40.0
    warm_night_c: float = 30.0
    heavy_rain_mm: float = 50.0
    wet_day_mm: float = 1.0


@dataclass
class Query:
    bbox: tuple[float, float, float, float]
    scenarios: list[str]
    models: list[str] | None
    groups: list[str]
    years: tuple[int, int] | None
    freq: str
    thresholds: Thresholds
    out: Path
    overwrite: bool = False
    chunk_years: int = 10
    map_bbox: tuple[float, float, float, float] | None = None
    baseline: tuple[int, int] = (1985, 2014)
    periods: list[tuple[int, int]] = field(default_factory=lambda: [(2041, 2060), (2081, 2100)])
    map_mode: str = "download"


# --------------------------------------------------------------------------
# Pure helpers (tested offline)
# --------------------------------------------------------------------------
def stull_wetbulb(t_c: float, rh: float) -> float:
    """Stull (2011) wet-bulb temperature (deg C) from air temperature and RH (%)."""
    return (t_c * math.atan(0.151977 * math.sqrt(rh + 8.313659))
            + math.atan(t_c + rh) - math.atan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * math.atan(0.023101 * rh) - 4.686035)


def period_units(start_year: int, end_year: int, freq: str) -> list[tuple[str, str, str]]:
    """(label, start_date, end_date_exclusive) for each year or month."""
    units = []
    for y in range(start_year, end_year + 1):
        if freq == "annual":
            units.append((f"{y}", f"{y}-01-01", f"{y + 1}-01-01"))
        else:
            for m in range(1, 13):
                ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
                units.append((f"{y}-{m:02d}", f"{y}-{m:02d}-01", f"{ny}-{nm:02d}-01"))
    return units


def year_chunks(start: int, end: int, size: int) -> list[tuple[int, int]]:
    return [(y, min(y + size - 1, end)) for y in range(start, end + 1, size)]


def clip_years(scenario: str, years: tuple[int, int] | None) -> tuple[int, int] | None:
    lo, hi = SCENARIO_YEARS[scenario]
    if years:
        lo, hi = max(lo, years[0]), min(hi, years[1])
    return (lo, hi) if lo <= hi else None


def parse_period(text: str) -> tuple[int, int]:
    a, b = text.split("-")
    return int(a), int(b)


# --------------------------------------------------------------------------
# Earth Engine
# --------------------------------------------------------------------------
def init_earth_engine(project: str) -> None:
    import ee

    try:
        ee.Initialize(project=project)
    except Exception:  # noqa: BLE001
        ee.Authenticate()
        ee.Initialize(project=project)
    LOGGER.info("Earth Engine initialised (project=%s).", project)


def get_info_retry(obj, what: str, retries: int = 3):
    import ee

    for attempt in range(1, retries + 1):
        try:
            return obj.getInfo()
        except ee.EEException as exc:
            if attempt == retries:
                raise
            wait = 10 * attempt
            LOGGER.warning("%s failed (%s); retry %d/%d in %d s.", what, str(exc)[:150],
                           attempt, retries - 1, wait)
            time.sleep(wait)


def list_models() -> list[str]:
    import ee

    col = ee.ImageCollection(COLLECTION_ID)
    one_day = ee.ImageCollection(col.filterDate("2014-01-01", "2014-01-02")).merge(
        col.filterDate("2015-01-01", "2015-01-02"))
    return sorted(get_info_retry(one_day.aggregate_array("model").distinct(), "model list"))


def daily_collection(model: str, scenario: str):
    import ee

    return (ee.ImageCollection(COLLECTION_ID)
            .filter(ee.Filter.eq("model", model))
            .filter(ee.Filter.eq("scenario", scenario))
            .filter(ee.Filter.neq("grid_label", "gr2")))       # GFDL-CM4 has a 2nd grid


def available_bands(model: str, scenario: str, year: int) -> set[str]:
    first = daily_collection(model, scenario).filterDate(f"{year}-01-01", f"{year + 1}-01-01").first()
    try:
        return set(get_info_retry(first.bandNames(), f"bands {model}/{scenario}") or [])
    except Exception:  # noqa: BLE001 - empty collection -> first() is null
        return set()


def usable_groups(groups: list[str], bands: set[str]) -> list[str]:
    return [g for g in groups if GROUP_BANDS[g] <= bands]


def build_indices(daily, groups: list[str], th: Thresholds):
    """Aggregate a daily ImageCollection (one period unit) into one multi-band index image."""
    import ee

    parts = []
    if "means" in groups:
        parts += [
            daily.select("tas").mean().subtract(KELVIN).rename("tas_mean_c"),
            daily.select("tasmax").mean().subtract(KELVIN).rename("tasmax_mean_c"),
            daily.select("tasmin").mean().subtract(KELVIN).rename("tasmin_mean_c"),
            daily.select("pr").sum().multiply(SECONDS_PER_DAY).rename("pr_total_mm"),
            daily.select("pr").mean().multiply(SECONDS_PER_DAY).rename("pr_mean_mmday"),
        ]
    if "heat" in groups:
        tmax_k, tmin_k = th.hot_day_c + KELVIN, th.warm_night_c + KELVIN
        parts += [
            daily.select("tasmax").max().subtract(KELVIN).rename("txx_c"),
            daily.select("tasmin").max().subtract(KELVIN).rename("tnx_c"),
            daily.select("tasmax").map(lambda i: i.gt(tmax_k)).sum().rename("hot_days"),
            daily.select("tasmin").map(lambda i: i.gt(tmin_k)).sum().rename("warm_nights"),
        ]
    if "rain" in groups:
        heavy = th.heavy_rain_mm / SECONDS_PER_DAY
        wet = th.wet_day_mm / SECONDS_PER_DAY
        parts += [
            daily.select("pr").max().multiply(SECONDS_PER_DAY).rename("rx1day_mm"),
            daily.select("pr").map(lambda i: i.gt(heavy)).sum().rename("heavy_rain_days"),
            daily.select("pr").map(lambda i: i.gte(wet)).sum().rename("wet_days"),
        ]
    if "humid" in groups:
        def wetbulb(img):
            t = img.select("tas").subtract(KELVIN)
            rh = img.select("hurs").clamp(5, 99)               # Stull valid ~5-99 % RH
            tw = (t.multiply(rh.add(8.313659).sqrt().multiply(0.151977).atan())
                  .add(t.add(rh).atan())
                  .subtract(rh.subtract(1.676331).atan())
                  .add(rh.pow(1.5).multiply(rh.multiply(0.023101).atan()).multiply(0.00391838))
                  .subtract(4.686035))
            return tw.rename("twb")
        twb = daily.map(wetbulb)
        parts += [twb.max().rename("twb_max_c"), twb.mean().rename("twb_mean_c")]
    if "humidity" in groups:
        parts += [daily.select("hurs").mean().rename("hurs_mean_pct")]
    if "wind" in groups:
        parts += [daily.select("sfcWind").mean().rename("sfcwind_mean_ms"),
                  daily.select("sfcWind").min().rename("sfcwind_min_ms")]
    if "radiation" in groups:
        parts += [daily.select("rsds").mean().rename("rsds_mean_wm2"),
                  daily.select("rlds").mean().rename("rlds_mean_wm2")]
    return ee.Image.cat(parts).toFloat()


# --------------------------------------------------------------------------
# Time series
# --------------------------------------------------------------------------
def timeseries_for(model: str, scenario: str, q: Query, region) -> Path | None:
    import ee

    span = clip_years(scenario, q.years)
    if span is None:
        return None
    out_csv = q.out / "timeseries" / f"ts_{scenario}_{model}.csv"
    if out_csv.exists() and not q.overwrite:
        LOGGER.info("Skip %s/%s (exists).", model, scenario)
        return out_csv

    bands = available_bands(model, scenario, span[0])
    groups = usable_groups(q.groups, bands)
    if not groups:
        LOGGER.warning("%s/%s: no data or none of the requested variables; skipped.", model, scenario)
        return None
    missing = sorted(set(q.groups) - set(groups))
    if missing:
        LOGGER.warning("%s/%s: groups %s unavailable (bands: %s).", model, scenario, missing, sorted(bands))

    base = daily_collection(model, scenario)
    rows: list[dict] = []
    for c0, c1 in year_chunks(span[0], span[1], q.chunk_years):
        feats = []
        for label, d0, d1 in period_units(c0, c1, q.freq):
            idx = build_indices(base.filterDate(d0, d1), groups, q.thresholds)
            stats = idx.reduceRegion(reducer=ee.Reducer.mean(), geometry=region,
                                     scale=NATIVE_SCALE_M, maxPixels=1e9)
            feats.append(ee.Feature(None, stats).set("period", label))
        info = get_info_retry(ee.FeatureCollection(feats), f"{model}/{scenario} {c0}-{c1}")
        for f in info["features"]:
            props = f["properties"]
            rows.append({"model": model, "scenario": scenario, **props})
        LOGGER.info("%s/%s %d-%d done.", model, scenario, c0, c1)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "scenario", "period"] + sorted(
        {k for r in rows for k in r} - {"model", "scenario", "period"})
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["period"]))
    return out_csv


def ensemble_summary(q: Query) -> Path | None:
    import pandas as pd

    files = sorted((q.out / "timeseries").glob("ts_*.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    value_cols = [c for c in df.columns if c not in ("model", "scenario", "period")]
    long = df.melt(id_vars=["model", "scenario", "period"], value_vars=value_cols,
                   var_name="index", value_name="value").dropna(subset=["value"])
    summary = (long.groupby(["scenario", "period", "index"])["value"]
               .agg(mean="mean", p10=lambda s: s.quantile(0.10), p50="median",
                    p90=lambda s: s.quantile(0.90), n_models="count")
               .round(3).reset_index())
    path = q.out / "timeseries_ensemble.csv"
    summary.to_csv(path, index=False)
    LOGGER.info("Ensemble summary: %s (%d rows from %d model files).", path, len(summary), len(files))
    return path


# --------------------------------------------------------------------------
# Maps
# --------------------------------------------------------------------------
def climatology(model: str, scenario: str, period: tuple[int, int], groups: list[str],
                th: Thresholds):
    """Mean over years of the annual indices (e.g. mean annual hot days) for one model."""
    import ee

    base = daily_collection(model, scenario)
    annual = [build_indices(base.filterDate(f"{y}-01-01", f"{y + 1}-01-01"), groups, th)
              for y in range(period[0], period[1] + 1)]
    return ee.ImageCollection(annual).mean()


def export_map(image, name: str, q: Query, region) -> None:
    import ee
    import requests

    if q.map_mode == "drive":
        task = ee.batch.Export.image.toDrive(image=image, description=name[:100],
                                             folder="cmip6_maps", fileNamePrefix=name,
                                             region=region, scale=NATIVE_SCALE_M,
                                             crs="EPSG:4326", maxPixels=1e10)
        task.start()
        LOGGER.info("Drive export started: %s (task %s)", name, task.id)
        return
    url = image.getDownloadURL({"region": region, "scale": NATIVE_SCALE_M,
                                "crs": "EPSG:4326", "format": "GEO_TIFF"})
    path = q.out / "maps" / f"{name}.tif"
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=900)
    response.raise_for_status()
    path.write_bytes(response.content)
    LOGGER.info("Saved %s", path)


def period_maps(models: list[str], q: Query) -> None:
    import ee

    region = ee.Geometry.Rectangle(list(q.map_bbox or q.bbox))
    # Only indices every selected model can provide (bands must match to stack).
    common: set[str] | None = None
    for m in models:
        b = available_bands(m, "historical", q.baseline[0]) & available_bands(
            m, q.scenarios[-1] if q.scenarios[-1] != "historical" else "ssp245", 2050)
        common = b if common is None else common & b
    groups = usable_groups(q.groups, common or set())
    if not groups:
        LOGGER.error("No index group available in all selected models; maps skipped.")
        return
    LOGGER.info("Map index groups (common to all %d models): %s", len(models), groups)

    stats = (ee.Reducer.mean()
             .combine(ee.Reducer.percentile([10, 50, 90]), sharedInputs=True))

    baseline = ee.ImageCollection(
        [climatology(m, "historical", q.baseline, groups, q.thresholds) for m in models])
    base_mean = baseline.mean()
    export_map(baseline.reduce(stats), f"historical_{q.baseline[0]}-{q.baseline[1]}", q, region)

    for scenario in [s for s in q.scenarios if s != "historical"]:
        for period in q.periods:
            fut = ee.ImageCollection(
                [climatology(m, scenario, period, groups, q.thresholds) for m in models])
            tag = f"{scenario}_{period[0]}-{period[1]}"
            export_map(fut.reduce(stats), tag, q, region)
            export_map(fut.mean().subtract(base_mean), f"change_{tag}", q, region)


# --------------------------------------------------------------------------
# Orchestration / CLI
# --------------------------------------------------------------------------
def run(args) -> int:
    import ee

    q = Query(
        bbox=tuple(args.bbox), scenarios=args.scenarios, models=args.models,
        groups=args.groups, years=tuple(args.years) if args.years else None, freq=args.freq,
        thresholds=Thresholds(args.hot_day, args.warm_night, args.heavy_rain),
        out=args.out, overwrite=args.overwrite, chunk_years=args.chunk_years,
        map_bbox=tuple(args.map_bbox) if args.map_bbox else None,
        baseline=parse_period(args.baseline), periods=[parse_period(p) for p in args.periods],
        map_mode=args.map_mode,
    )
    q.out.mkdir(parents=True, exist_ok=True)
    init_earth_engine(args.project)

    all_models = list_models()
    models = all_models if not q.models else [m for m in q.models if m in all_models]
    unknown = sorted(set(q.models or []) - set(all_models))
    if unknown:
        LOGGER.warning("Unknown model names ignored: %s", unknown)
    LOGGER.info("Models (%d): %s", len(models), ", ".join(models))
    if not models:
        return 1

    if args.mode in ("timeseries", "both"):
        region = ee.Geometry.Rectangle(list(q.bbox))
        failures = []
        for scenario in q.scenarios:
            for model in models:
                try:
                    timeseries_for(model, scenario, q, region)
                except ee.EEException as exc:
                    LOGGER.error("%s/%s failed: %s", model, scenario, str(exc)[:200])
                    failures.append((model, scenario, str(exc)[:200]))
        ensemble_summary(q)
        if failures:
            with open(q.out / "failures.csv", "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerows([("model", "scenario", "error"), *failures])
            LOGGER.warning("%d model/scenario runs failed - rerun to retry (finished ones are "
                           "skipped). See failures.csv.", len(failures))

    if args.mode in ("maps", "both"):
        period_maps(models, q)
    return 0


def self_test() -> int:
    ok = True

    def check(label, got, expected, tol=0.05):
        nonlocal ok
        passed = abs(got - expected) <= tol
        ok &= passed
        LOGGER.info("%-38s got %.3f expected %.3f  %s", label, got, expected,
                    "PASS" if passed else "FAIL")

    check("Stull wet-bulb T=20C RH=50% (paper ~13.7)", stull_wetbulb(20, 50), 13.7, 0.1)
    check("wet-bulb at RH~99% ~ air temp (T=30)", stull_wetbulb(30, 99), 30.0, 0.6)
    check("pr 1e-4 kg/m2/s -> mm/day", 1e-4 * SECONDS_PER_DAY, 8.64, 1e-9)
    check("tas 300 K -> C", 300 - KELVIN, 26.85, 1e-9)
    check("monthly units 2015-2016", len(period_units(2015, 2016, "monthly")), 24, 0)
    check("annual chunks 1950-2014 by 10", len(year_chunks(1950, 2014, 10)), 7, 0)
    check("historical clipped to 2000-2030 ends 2014",
          clip_years("historical", (2000, 2030))[1], 2014, 0)
    ok &= clip_years("ssp245", (1990, 2000)) is None
    LOGGER.info("ssp245 clipped to 1990-2000 is empty           %s",
                "PASS" if clip_years("ssp245", (1990, 2000)) is None else "FAIL")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    p = argparse.ArgumentParser(description="CMIP6 (NEX-GDDP) climate query: CSV + GeoTIFF.")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--project")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p.add_argument("--map-bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                   help="Larger area for maps (default: --bbox)")
    p.add_argument("--mode", choices=["timeseries", "maps", "both"], default="timeseries")
    p.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIO_YEARS),
                   default=["historical", "ssp245", "ssp585"])
    p.add_argument("--models", nargs="+", help="Model names (default: all)")
    p.add_argument("--groups", nargs="+", choices=sorted(GROUP_BANDS),
                   default=["means", "heat", "rain", "humid", "humidity", "wind", "radiation"])
    p.add_argument("--years", nargs=2, type=int, metavar=("START", "END"),
                   help="Restrict years (default: full range per scenario)")
    p.add_argument("--freq", choices=["annual", "monthly"], default="annual")
    p.add_argument("--hot-day", type=float, default=40.0, help="Tmax threshold, deg C")
    p.add_argument("--warm-night", type=float, default=30.0, help="Tmin threshold, deg C")
    p.add_argument("--heavy-rain", type=float, default=50.0, help="Daily rain threshold, mm")
    p.add_argument("--chunk-years", type=int, default=10, help="Years per GEE request")
    p.add_argument("--baseline", default="1985-2014", help="Baseline period for maps")
    p.add_argument("--periods", nargs="+", default=["2041-2060", "2081-2100"],
                   help="Future periods for maps, e.g. 2021-2040 2041-2060")
    p.add_argument("--map-mode", choices=["download", "drive"], default="download")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--out", type=Path, default=Path("./cmip6_output"))
    a = p.parse_args(argv)

    if a.self_test:
        return self_test()
    if not a.project or not a.bbox:
        p.error("--project and --bbox are required")
    try:
        return run(a)
    except Exception as exc:  # noqa: BLE001 - top-level: report cleanly
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())