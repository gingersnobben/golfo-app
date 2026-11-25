#!/usr/bin/env python3
"""Bygger Golfos banfiler från OpenStreetMap.

Indata: en region exporterad med osmium till GeoJSON-sekvens (se course-data.yml).
Utdata (under --out):
  courses/<id>.json   en fil per bana: gräns, hål, greener, bunkrar, fairways, vatten
  tiles/<lat>_<lon>.json   index per ruta på 1 × 1 grad: vilka banor som ligger där
  meta.json           när datan byggdes och källhänvisning

Användning:
  build_course_data.py region --input sweden.geojsonseq --out data --manifest manifest.jsonl
  build_course_data.py tiles  --out data --manifest manifest.jsonl --regions "Sweden, Spain"

Data © OpenStreetMap contributors, tillgänglig under Open Database License (ODbL).
"""

import argparse
import datetime
import json
import math
import os
import sys

from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

# Ungefär 300 m i grader: vatten så här nära banan följer med i banans fil.
WATER_MARGIN = 0.003
# Hål utan inritad banyta klustras ihop om de ligger så här nära varandra (ca 800 m).
CLUSTER_DISTANCE = 0.008
# Förenkling av ytor, i grader (ca 1 m) – mindre filer, ingen märkbar skillnad på kartan.
SIMPLIFY = 0.00001
DIGITS = 6


# ---------------------------------------------------------------- läsning

def read_features(path):
    """GeoJSON-sekvens från osmium (rader, ev. med RS-tecken först)."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().lstrip("\x1e").strip()
            if not line:
                continue
            try:
                feature = json.loads(line)
            except json.JSONDecodeError:
                continue
            if feature.get("type") == "Feature" and feature.get("geometry"):
                yield feature


def osm_id(feature):
    props = feature.get("properties") or {}
    kind = {"node": "n", "way": "w", "relation": "r", "area": "a"}.get(props.get("@type"), "x")
    return f"{kind}{props.get('@id', 0)}"


def tags(feature):
    return feature.get("properties") or {}


def geometry(feature):
    try:
        g = shape(feature["geometry"])
        return g if g.is_valid else g.buffer(0)
    except Exception:
        return None


def is_area(g):
    return isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty


def polygons(g):
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    if hasattr(g, "geoms"):
        return [p for p in g.geoms if isinstance(p, Polygon)]
    return []


# ---------------------------------------------------------------- utdataformat

def latlon(x, y):
    # GeoJSON är (lon, lat); Golfo använder [lat, lon].
    return [round(y, DIGITS), round(x, DIGITS)]


def ring(poly, simplify=SIMPLIFY):
    p = poly.simplify(simplify, preserve_topology=True) if simplify else poly
    if p.is_empty or not isinstance(p, Polygon):
        return None
    return [latlon(x, y) for x, y in p.exterior.coords]


def rings(g, simplify=SIMPLIFY):
    return [r for r in (ring(p, simplify) for p in polygons(g)) if r and len(r) >= 4]


def line(g):
    if isinstance(g, LineString):
        coords = list(g.coords)
    elif isinstance(g, Polygon):
        coords = list(g.exterior.coords)
    else:
        return None
    return [latlon(x, y) for x, y in coords] if len(coords) >= 2 else None


def hole_ref(t):
    digits = "".join(ch for ch in str(t.get("ref", "")) if ch.isdigit())
    return int(digits) if digits else None


def hole_course(t):
    """Vilken bana hålet hör till på en anläggning med flera banor (t.ex. course=TC på
    Vasatorp). Ingen fast standard i OpenStreetMap – de vanligaste taggarna prövas."""
    for key in ("course", "golf:course", "golf:course:name", "course_name"):
        value = str(t.get(key, "")).strip()
        if value:
            return value[:40]
    return None


def hole_handicap(t):
    """Hålets index (1–18), från handicap= eller stroke_index=."""
    for key in ("handicap", "stroke_index", "golf:handicap"):
        digits = "".join(ch for ch in str(t.get(key, "")) if ch.isdigit())
        if digits and 1 <= int(digits) <= 36:
            return int(digits)
    return None


def hole_par(t):
    digits = "".join(ch for ch in str(t.get("par", "")) if ch.isdigit())
    if not digits:
        return None
    par = int(digits)
    return par if 3 <= par <= 6 else None


def water_kind(t):
    golf = t.get("golf")
    if golf == "lateral_water_hazard":
        return "penaltyRed"
    if golf == "water_hazard":
        return "penaltyYellow"
    if golf == "penalty_area":
        return "penaltyYellow" if t.get("penalty") == "yellow" else "penaltyRed"
    return "water"


def is_water(t):
    return t.get("golf") in ("water_hazard", "lateral_water_hazard", "penalty_area") or t.get("natural") == "water"


# ---------------------------------------------------------------- en region

class Course:
    def __init__(self, cid, name, g):
        self.id = cid
        self.name = name
        self.geometry = g
        self.area = g.area
        self.holes = []
        self.greens = []
        self.bunkers = []
        self.fairways = []
        self.water = []

    def to_json(self):
        minx, miny, maxx, maxy = self.geometry.bounds
        c = self.geometry.representative_point()
        return {
            "v": 1,
            "id": self.id,
            "name": self.name,
            "center": latlon(c.x, c.y),
            "bbox": [round(miny, DIGITS), round(minx, DIGITS), round(maxy, DIGITS), round(maxx, DIGITS)],
            "outline": rings(self.geometry),
            "holes": self.holes,
            "greens": self.greens,
            "bunkers": self.bunkers,
            "fairways": self.fairways,
            "water": self.water,
        }


def build_region(path, out_dir, manifest):
    # 1. Banornas ytor.
    courses = []
    for f in read_features(path):
        t = tags(f)
        if t.get("leisure") != "golf_course":
            continue
        g = geometry(f)
        if g is None or not is_area(g):
            continue
        courses.append(Course(osm_id(f), t.get("name"), g))
    print(f"  {len(courses)} course areas", flush=True)

    def smallest_containing(tree, items, point):
        best = None
        for i in tree.query(point):
            c = items[i]
            if c.geometry.contains(point) and (best is None or c.area < best.area):
                best = c
        return best

    # 2. Hålen. Hål utanför alla banytor klustras till egna "banor" (utan namn).
    tree = STRtree([c.geometry for c in courses]) if courses else None
    loose = []
    for f in read_features(path):
        t = tags(f)
        if t.get("golf") != "hole":
            continue
        ref = hole_ref(t)
        g = geometry(f)
        coords = line(g) if g is not None else None
        if ref is None or not coords:
            continue
        hole = {"ref": ref, "par": hole_par(t), "line": coords}
        # Bannamn och index bara när de finns, så att filerna inte växer i onödan.
        if hole_course(t):
            hole["course"] = hole_course(t)
        if hole_handicap(t):
            hole["si"] = hole_handicap(t)
        mid = g.interpolate(0.5, normalized=True) if isinstance(g, LineString) else g.representative_point()
        course = smallest_containing(tree, courses, mid) if tree is not None else None
        if course:
            course.holes.append(hole)
        else:
            loose.append((osm_id(f), hole, g))

    clusters = cluster_holes(loose)
    for members in clusters:
        geoms = [g for _, _, g in members]
        outline = unary_union(geoms).convex_hull.buffer(0.001)
        cid = "h" + min(m[0] for m in members).lstrip("w")
        course = Course(cid, None, outline)
        course.holes = [h for _, h, _ in members]
        courses.append(course)
    print(f"  {len(clusters)} courses from loose holes", flush=True)
    if not courses:
        return 0

    # 3. Greener, bunkrar, fairways och vatten.
    tree = STRtree([c.geometry for c in courses])
    margins = [c.geometry.envelope.buffer(WATER_MARGIN, join_style=2) for c in courses]
    margin_tree = STRtree(margins)
    for f in read_features(path):
        t = tags(f)
        golf = t.get("golf")
        if golf in ("green", "bunker", "fairway"):
            g = geometry(f)
            if g is None or not is_area(g):
                continue
            course = smallest_containing(tree, courses, g.representative_point())
            if course is None:
                continue
            target = {"green": course.greens, "bunker": course.bunkers, "fairway": course.fairways}[golf]
            target.extend(rings(g, simplify=0.000003))
        elif is_water(t):
            g = geometry(f)
            if g is None or not is_area(g):
                continue
            kind = water_kind(t)
            for i in margin_tree.query(g):
                clipped = g.intersection(margins[i])
                for r in rings(clipped):
                    courses[i].water.append({"kind": kind, "ring": r})

    # 4. Skriv en fil per bana med hål.
    os.makedirs(os.path.join(out_dir, "courses"), exist_ok=True)
    written = 0
    for c in courses:
        if not c.holes:
            continue
        c.holes.sort(key=lambda h: h["ref"])
        data = c.to_json()
        with open(os.path.join(out_dir, "courses", f"{c.id}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
        manifest.write(json.dumps({"id": c.id, "name": c.name, "center": data["center"],
                                   "bbox": data["bbox"], "holes": len(c.holes)}) + "\n")
        written += 1
    print(f"  {written} course files written", flush=True)
    return written


def cluster_holes(loose):
    """Grupperar lösa hål som ligger nära varandra (minst 3 hål per grupp)."""
    n = len(loose)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if n:
        geoms = [g for _, _, g in loose]
        tree = STRtree(geoms)
        for i, g in enumerate(geoms):
            for j in tree.query(g.buffer(CLUSTER_DISTANCE)):
                j = int(j)
                if j != i and g.distance(geoms[j]) <= CLUSTER_DISTANCE:
                    a, b = find(i), find(j)
                    if a != b:
                        parent[a] = b
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(loose[i])
    return [g for g in groups.values() if len(g) >= 3]


# ---------------------------------------------------------------- index

def build_tiles(out_dir, manifest_path, regions):
    tiles = {}
    count = 0
    seen = set()
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            # Regionerna överlappar lite vid gränserna (t.ex. delstater): samma bana bara en gång.
            if entry["id"] in seen:
                continue
            seen.add(entry["id"])
            s, w, n, e = entry["bbox"]
            count += 1
            for lat in range(math.floor(s), math.floor(n) + 1):
                for lon in range(math.floor(w), math.floor(e) + 1):
                    tiles.setdefault(f"{lat}_{lon}", []).append(entry)
    tile_dir = os.path.join(out_dir, "tiles")
    os.makedirs(tile_dir, exist_ok=True)
    for key, entries in tiles.items():
        with open(os.path.join(tile_dir, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump({"v": 1, "courses": entries}, f, separators=(",", ":"), ensure_ascii=False)
    meta = {
        "v": 1,
        "updated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "regions": [r.strip() for r in regions.split(",") if r.strip()],
        "courses": count,
        "tiles": len(tiles),
        "attribution": "© OpenStreetMap contributors",
        "license": "Open Database License (ODbL) 1.0 – https://opendatacommons.org/licenses/odbl/",
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print(f"{count} courses in {len(tiles)} tiles", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("region")
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--manifest", required=True)
    t = sub.add_parser("tiles")
    t.add_argument("--out", required=True)
    t.add_argument("--manifest", required=True)
    t.add_argument("--regions", default="")
    args = parser.parse_args()

    if args.cmd == "region":
        with open(args.manifest, "a", encoding="utf-8") as manifest:
            build_region(args.input, args.out, manifest)
    else:
        build_tiles(args.out, args.manifest, args.regions)


if __name__ == "__main__":
    sys.exit(main())
