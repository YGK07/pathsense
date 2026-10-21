"""Named campus places for the route planner's search box (from OpenStreetMap, plus a few local names).

    python -c "from pathsense.places import fetch_places; fetch_places()"     # refresh data/osm_places.json
"""
import json, os, re, urllib.parse, urllib.request
from .osm import ENDPOINTS

# OpenStreetMap misses some places everyone uses; positions from our recorded walks.
EXTRA = [dict(name="Fitty gym", lat=12.97322, lon=79.16440, kind="building"),
         dict(name="Men's hostel entrance", lat=12.96967, lon=79.15822, kind="gate")]
# what people actually type -> the OSM name
ALIASES = {"SJT": "Silver Jubilee Tower", "TT": "Technology Tower", "MGB": "Mahatma Gandhi Block (MGB)",
           "GDN": "G.D Naidu Block", "SMV": "Sir M Visvesvaraya Block", "PRP": "Perl Research Park",
           "Library": "EV Periyar Library", "MH": "Mens Hostel A Block"}


def _kind(tags):
    if tags.get("highway"):
        return "road"
    if tags.get("amenity") in ("fast_food", "cafe", "restaurant", "food_court") or "food" in tags.get("name", "").lower():
        return "food"
    if tags.get("leisure") or tags.get("sport"):
        return "sport"
    if tags.get("building") or tags.get("amenity"):
        return "building"
    return "place"


def build_places(raw, bbox, extra=EXTRA):
    """Overpass JSON (elements with `center` or lat/lon) -> [{name, lat, lon, kind, alias?}] inside bbox
    (south, west, north, east), one entry per name (first wins), sorted by name."""
    s, w, n, e = bbox
    seen, out = set(), []
    for el in raw.get("elements", []):
        tags = el.get("tags", {}); name = (tags.get("name") or "").strip()
        lat = el.get("lat", el.get("center", {}).get("lat")); lon = el.get("lon", el.get("center", {}).get("lon"))
        if not name or lat is None or not (s <= lat <= n and w <= lon <= e) or name.lower() in seen:
            continue
        if re.search(r"chennai", name, re.I):
            continue
        seen.add(name.lower()); out.append(dict(name=name, lat=round(lat, 6), lon=round(lon, 6), kind=_kind(tags)))
    for x in extra:
        if x["name"].lower() not in seen:
            seen.add(x["name"].lower()); out.append(dict(x))
    by_name = {p["name"]: p for p in out}
    for short, full in ALIASES.items():
        if full in by_name and short.lower() not in seen:
            out.append(dict(by_name[full], name=f"{short} ({full})", alias=short))
    return sorted(out, key=lambda p: p["name"].lower())


def fetch_places(out="data/osm_places.json", bbox=(12.9660, 79.1500, 12.9775, 79.1700), timeout=120):
    q = f'[out:json][timeout:{timeout}];nwr["name"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out center tags;'
    last = None
    for url in ENDPOINTS:
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "PathSense-student-project/0.3"})
            raw = json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode()); break
        except Exception as e:
            last = e
    else:
        raise RuntimeError(f"Overpass download failed: {last}")
    places = build_places(raw, bbox)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(places, open(out, "w"), indent=0)
    return places
