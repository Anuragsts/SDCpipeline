"""
Collect official IMD heat-wave alerts from the public CAP (Common Alerting
Protocol) archive and turn them into a list of heat-wave dates for an area.

Source
------
Public S3 archive of IMD CAP alerts used by WMO Alert Hub:
    https://cap-sources.s3.amazonaws.com/in-imd-en/<YYYY-MM-DD-HH-MM-SS>.xml
One XML per alert, from Aug 2019. Each alert has event, severity, onset,
expires, areaDesc (region names), description and a polygon.

NOTE: these are IMD-issued ALERTS for the day ("very likely"), not the
observed confirmation in IMD's heat bulletin. The archive may also have gaps
(days with no file). Treat the result as "IMD heat-wave alert days".

Outputs (in --out)
------------------
heat_alerts_all.csv      every heat-related alert in the date range (all India)
heat_alerts_area.csv     alerts matching your area (keyword and/or polygon)
heatwave_days.csv        one row per IST date with a matching alert (max severity)
heat_alerts_area.gpkg    alert polygons (vector) for QGIS
cache/                   downloaded XMLs (re-runs do not re-download)

Requirements:  pip install requests geopandas shapely

Examples
--------
    python imd_cap_heatwave_archive.py --start 2019-08-01 --end 2026-09-30 \
        --keywords Delhi --bbox 76.84 28.40 77.35 28.88 --out ./imd_heat
    python imd_cap_heatwave_archive.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

LOGGER = logging.getLogger("imd_cap")

BUCKET_URL = "https://cap-sources.s3.amazonaws.com"
PREFIX = "in-imd-en/"
KEY_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.xml$")
DEFAULT_EVENT_PATTERN = r"heat\s*wave|warm\s*night|hot\s*and\s*humid|hot\s*&\s*humid"
SEVERITY_RANK = {"Unknown": 0, "Minor": 1, "Moderate": 2, "Severe": 3, "Extreme": 4}
USER_AGENT = "imd-cap-heatwave-archive/1.0 (urban heat research)"


@dataclass
class CapAlert:
    key: str
    identifier: str | None = None
    sent: str | None = None
    status: str | None = None
    msg_type: str | None = None
    event: str | None = None
    severity: str | None = None
    urgency: str | None = None
    certainty: str | None = None
    onset: str | None = None
    expires: str | None = None
    headline: str | None = None
    description: str | None = None
    area_desc: str | None = None
    polygons: list[list[tuple[float, float]]] = field(default_factory=list)  # (lon, lat)


# --------------------------------------------------------------------------
# Parsing (namespace-agnostic, tested offline)
# --------------------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_text(elem, name: str) -> str | None:
    for child in elem.iter():
        if _local(child.tag) == name and child.text:
            return child.text.strip()
    return None


def parse_polygon(text: str) -> list[tuple[float, float]]:
    """CAP polygon 'lat,lon lat,lon ...' -> [(lon, lat), ...]."""
    points = []
    for pair in text.split():
        try:
            lat, lon = (float(v) for v in pair.split(",")[:2])
            points.append((lon, lat))
        except ValueError:
            continue
    return points


def parse_cap(xml_bytes: bytes, key: str) -> list[CapAlert]:
    """One CapAlert per <info> block (an alert can carry several)."""
    root = ET.fromstring(xml_bytes)
    header = {n: _find_text(root, n) for n in ("identifier", "sent", "status", "msgType")}
    infos = [e for e in root.iter() if _local(e.tag) == "info"] or [root]
    alerts = []
    for info in infos:
        a = CapAlert(key=key, identifier=header["identifier"], sent=header["sent"],
                     status=header["status"], msg_type=header["msgType"])
        for attr, name in (("event", "event"), ("severity", "severity"), ("urgency", "urgency"),
                           ("certainty", "certainty"), ("onset", "onset"),
                           ("expires", "expires"), ("headline", "headline"),
                           ("description", "description")):
            setattr(a, attr, _find_text(info, name))
        areas = [e for e in info.iter() if _local(e.tag) == "area"]
        a.area_desc = "; ".join(filter(None, (_find_text(ar, "areaDesc") for ar in areas))) or None
        for ar in areas:
            for poly in (e for e in ar.iter() if _local(e.tag) == "polygon"):
                pts = parse_polygon(poly.text or "")
                if len(pts) >= 3:
                    a.polygons.append(pts)
        alerts.append(a)
    return alerts


def is_heat_alert(a: CapAlert, pattern: re.Pattern) -> bool:
    text = " ".join(filter(None, (a.event, a.headline)))
    return bool(pattern.search(text))


def matches_area(a: CapAlert, keywords: list[str], bbox_geom) -> tuple[bool, str]:
    """Match by region keyword (areaDesc/description) and/or polygon overlap."""
    reasons = []
    text = " ".join(filter(None, (a.area_desc, a.description))).lower()
    if keywords and any(k.lower() in text for k in keywords):
        reasons.append("keyword")
    if bbox_geom is not None and a.polygons:
        from shapely.geometry import Polygon

        if any(Polygon(p).buffer(0).intersects(bbox_geom) for p in a.polygons):
            reasons.append("polygon")
    return bool(reasons), "+".join(reasons)


def _parse_dt(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def alert_days(a: CapAlert) -> list[date]:
    """IST calendar dates covered by onset..expires (expires exclusive at midnight)."""
    onset = _parse_dt(a.onset) or _parse_dt(a.sent)
    if onset is None:
        return []
    expires = _parse_dt(a.expires)
    first = onset.date()
    if expires is None or expires <= onset:
        return [first]
    last = (expires - timedelta(seconds=1)).date()
    n = (last - first).days
    return [first + timedelta(days=i) for i in range(max(n, 0) + 1)][:10]   # sanity cap


# --------------------------------------------------------------------------
# Network (S3 listing + fetch with cache)
# --------------------------------------------------------------------------
def key_datetime(key: str) -> datetime | None:
    m = KEY_DATE.search(key)
    return datetime(*map(int, m.groups())) if m else None


def list_keys(session, start: date, end: date) -> list[str]:
    """List archive keys between start and end (inclusive), month by month."""
    keys: list[str] = []
    month = date(start.year, start.month, 1)
    while month <= end:
        prefix = f"{PREFIX}{month:%Y-%m}"
        token = None
        while True:
            params = {"list-type": "2", "prefix": prefix}
            if token:
                params["continuation-token"] = token
            resp = session.get(BUCKET_URL, params=params, timeout=60)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            for c in root.iter():
                if _local(c.tag) == "Key" and c.text and c.text.endswith(".xml"):
                    kd = key_datetime(c.text)
                    if kd and start <= kd.date() <= end:
                        keys.append(c.text)
            truncated = (_find_text(root, "IsTruncated") or "false").lower() == "true"
            token = _find_text(root, "NextContinuationToken")
            if not truncated or not token:
                break
        month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
    LOGGER.info("Archive files in range: %d", len(keys))
    return sorted(keys)


def fetch_cap(session, key: str, cache_dir: Path, delay_s: float, retries: int = 3) -> bytes:
    path = cache_dir / Path(key).name
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(f"{BUCKET_URL}/{key}", timeout=60)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            time.sleep(delay_s)                       # be polite to the public archive
            return resp.content
        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                raise
            LOGGER.warning("Fetch %s failed (%s); retry %d.", key, exc, attempt)
            time.sleep(5 * attempt)
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
ALERT_FIELDS = ["key", "identifier", "sent", "status", "msg_type", "event", "severity",
                "urgency", "certainty", "onset", "expires", "headline", "area_desc",
                "description"]


def write_alerts_csv(path: Path, rows: list[tuple[CapAlert, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(ALERT_FIELDS + ["match", "ist_days"])
        for a, match in rows:
            w.writerow([getattr(a, f) for f in ALERT_FIELDS] +
                       [match, " ".join(d.isoformat() for d in alert_days(a))])


def write_days_csv(path: Path, rows: list[tuple[CapAlert, str]]) -> int:
    days: dict[date, dict] = {}
    for a, _ in rows:
        if (a.status or "Actual") != "Actual" or (a.msg_type or "Alert") == "Cancel":
            continue
        for d in alert_days(a):
            rec = days.setdefault(d, {"max_severity": "Unknown", "events": set(), "n_alerts": 0})
            rec["n_alerts"] += 1
            rec["events"].add(a.event or "")
            if SEVERITY_RANK.get(a.severity or "Unknown", 0) > SEVERITY_RANK[rec["max_severity"]]:
                rec["max_severity"] = a.severity
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "max_severity", "n_alerts", "events"])
        for d in sorted(days):
            r = days[d]
            w.writerow([d.isoformat(), r["max_severity"], r["n_alerts"], " | ".join(sorted(r["events"]))])
    return len(days)


def write_polygons(path: Path, rows: list[tuple[CapAlert, str]]) -> int:
    import geopandas as gpd
    from shapely.geometry import MultiPolygon, Polygon

    recs = []
    for a, match in rows:
        polys = [Polygon(p).buffer(0) for p in a.polygons]
        polys = [p for p in polys if not p.is_empty]
        if not polys:
            continue
        geom = polys[0] if len(polys) == 1 else MultiPolygon(
            [g for p in polys for g in (p.geoms if hasattr(p, "geoms") else [p])])
        days = alert_days(a)
        recs.append({"key": a.key, "event": a.event, "severity": a.severity,
                     "onset": a.onset, "expires": a.expires,
                     "first_day": days[0].isoformat() if days else None,
                     "area_desc": a.area_desc, "match": match, "geometry": geom})
    if recs:
        gpd.GeoDataFrame(recs, crs="EPSG:4326").to_file(path, layer="heat_alerts", driver="GPKG")
    return len(recs)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run(args) -> int:
    import requests

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    out = args.out
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(args.event_pattern, re.IGNORECASE)

    bbox_geom = None
    if args.bbox:
        from shapely.geometry import box

        bbox_geom = box(*args.bbox)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    keys = list_keys(session, start, end)

    heat_all, heat_area, failed = [], [], 0
    for i, key in enumerate(keys, start=1):
        try:
            alerts = parse_cap(fetch_cap(session, key, cache, args.delay), key)
        except Exception as exc:  # noqa: BLE001 - skip bad files, keep going
            LOGGER.warning("Skipping %s: %s", key, exc)
            failed += 1
            continue
        for a in alerts:
            if not is_heat_alert(a, pattern):
                continue
            ok, how = matches_area(a, args.keywords, bbox_geom)
            heat_all.append((a, how))
            if ok:
                heat_area.append((a, how))
        if i % 200 == 0:
            LOGGER.info("Processed %d/%d files (%d heat alerts, %d for area).",
                        i, len(keys), len(heat_all), len(heat_area))

    write_alerts_csv(out / "heat_alerts_all.csv", heat_all)
    write_alerts_csv(out / "heat_alerts_area.csv", heat_area)
    n_days = write_days_csv(out / "heatwave_days.csv", heat_area)
    n_poly = write_polygons(out / "heat_alerts_area.gpkg", heat_area)
    LOGGER.info("Done: %d files, %d failed | heat alerts: %d all-India, %d for area | "
                "%d heat-wave alert days | %d polygons.", len(keys), failed, len(heat_all),
                len(heat_area), n_days, n_poly)
    return 0


def self_test() -> int:
    sample = b"""<?xml version="1.0" encoding="UTF-8"?>
<cap:alert xmlns:cap="urn:oasis:names:tc:emergency:cap:1.2">
 <cap:identifier>urn:oid:2.49.0.1.356.0.2024.5.28.8.30.31</cap:identifier>
 <cap:sent>2024-05-28T14:00:31+05:30</cap:sent><cap:status>Actual</cap:status>
 <cap:msgType>Alert</cap:msgType>
 <cap:info><cap:event>Heat wave to severe heat wave</cap:event>
  <cap:severity>Severe</cap:severity><cap:urgency>Expected</cap:urgency>
  <cap:certainty>Likely</cap:certainty>
  <cap:onset>2024-05-28T03:00:00+05:30</cap:onset><cap:expires>2024-05-29T00:00:00+05:30</cap:expires>
  <cap:headline>Heat wave to severe heat wave</cap:headline>
  <cap:description>Heat wave conditions very likely in many parts of Delhi.</cap:description>
  <cap:area><cap:areaDesc>Punjab Haryana Chandigarh Delhi Rajasthan UP MP</cap:areaDesc>
   <cap:polygon>28.0,76.0 29.5,76.0 29.5,78.0 28.0,78.0 28.0,76.0</cap:polygon></cap:area>
 </cap:info></cap:alert>"""
    from shapely.geometry import box

    ok = True

    def check(label, cond):
        nonlocal ok
        ok &= bool(cond)
        LOGGER.info("%-40s %s", label, "PASS" if cond else "FAIL")

    alerts = parse_cap(sample, "in-imd-en/2024-05-28-08-30-31.xml")
    a = alerts[0]
    check("parsed event", a.event == "Heat wave to severe heat wave")
    check("parsed polygon (lon,lat order)", a.polygons and a.polygons[0][0] == (76.0, 28.0))
    check("heat pattern matches", is_heat_alert(a, re.compile(DEFAULT_EVENT_PATTERN, re.I)))
    check("rain alert not matched", not re.compile(DEFAULT_EVENT_PATTERN, re.I).search("Heavy rainfall"))
    m, how = matches_area(a, ["Delhi"], box(76.84, 28.40, 77.35, 28.88))
    check("area match keyword+polygon", m and how == "keyword+polygon")
    m2, _ = matches_area(a, ["Kerala"], box(80, 10, 81, 11))
    check("no match elsewhere", not m2)
    check("IST day = 2024-05-28 only", alert_days(a) == [date(2024, 5, 28)])
    check("key date parsed", key_datetime(a.key) == datetime(2024, 5, 28, 8, 30, 31))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        force=True)
    p = argparse.ArgumentParser(description="IMD heat-wave alert days from the public CAP archive.")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--start", default="2019-08-01", help="YYYY-MM-DD (archive starts Aug 2019)")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--keywords", nargs="*", default=["Delhi"],
                   help="Region names to match in areaDesc/description")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                   help="Also match alerts whose polygon overlaps this box")
    p.add_argument("--event-pattern", default=DEFAULT_EVENT_PATTERN,
                   help="Regex for heat-related events")
    p.add_argument("--delay", type=float, default=0.3, help="Seconds between downloads")
    p.add_argument("--out", type=Path, default=Path("./imd_heat"))
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    try:
        return run(a)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())