"""Real footpath/road graph from OpenStreetMap (roadmap task D3). Standard library + networkx only.

    python scripts/fetch_osm.py                 # whole VIT Vellore campus -> data/osm_vit.json (cached)
    python scripts/run_pipeline.py --osm data/osm_vit.json

Why this exists: the original demo drew a *fake* jittered grid over the campus coordinates, so "roads" cut through
buildings and the lake. Real OSM ways follow real paths, so that cannot happen any more.
"""
import json, math, os, urllib.parse, urllib.request
from collections import Counter
import networkx as nx
from .config import LAT0, LON0

ENDPOINTS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
             "https://overpass.private.coffee/api/interpreter"]
HIGHWAYS = ("footway|path|pedestrian|steps|service|residential|living_street|unclassified|tertiary|secondary|"
            "primary|track|cycleway|road|tertiary_link|secondary_link|primary_link")
# the campus polygon as mapped in OSM (name match is case-insensitive); used to cover the WHOLE campus
CAMPUS_NAME = "Vellore Institute of Technology"


def _query(kind, lat, lon, radius, bbox):
    hw = f'["highway"~"^({HIGHWAYS})$"]'
    if kind == "bbox":
        s, w, n, e = bbox
        sel = f"way{hw}({s},{w},{n},{e});"
    elif kind == "area":    # every walkable way inside the campus boundary (+ nothing outside it)
        return (f'[out:json][timeout:120];(way["amenity"="university"]["name"~"{CAMPUS_NAME}",i];'
                f'relation["amenity"="university"]["name"~"{CAMPUS_NAME}",i];);map_to_area->.c;'
                f'way{hw}(area.c);(._;>;);out body;')
    else:
        sel = f"way{hw}(around:{radius},{lat},{lon});"
    return f"[out:json][timeout:120];({sel});(._;>;);out body;"


def _post(q, timeout):
    last = None
    for url in ENDPOINTS:
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(),
                                         headers={"User-Agent": "PathSense-student-project/0.2"})
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read().decode())
        except Exception as e:                      # try the next mirror
            last = e
    raise RuntimeError(f"Overpass download failed on all mirrors: {last}")


def fetch(mode="area", lat=LAT0, lon=LON0, radius=1500, bbox=None, cache=None, timeout=180):
    """mode: 'area' (campus boundary from OSM; falls back to radius if the boundary isn't found),
    'radius' (circle around lat/lon) or 'bbox' (south,west,north,east). Result cached as JSON."""
    if cache and os.path.exists(cache):
        return json.load(open(cache))
    data = None
    if mode == "area":
        data = _post(_query("area", lat, lon, radius, bbox), timeout)
        if not any(e["type"] == "way" for e in data.get("elements", [])):
            print("campus boundary not found in OSM -> falling back to radius search")
            data = None
    if data is None:
        data = _post(_query("bbox" if mode == "bbox" else "radius", lat, lon, radius, bbox), timeout)
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        json.dump(data, open(cache, "w"))
    return data


def _m(lat1, lon1, lat2, lon2):
    return math.hypot((lon2 - lon1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2)), (lat2 - lat1) * 111320)


def _walkable(tags):
    if tags.get("foot") == "no" or tags.get("indoor") or tags.get("area") == "yes":
        return False
    return tags.get("access") not in ("private", "no") or tags.get("foot") in ("yes", "designated", "permissive")


def build_graph(osm, lat0=None, lon0=None, min_len=2.0):
    """Overpass JSON -> undirected nx.Graph, nodes 0..n-1 with (x,y metres, lat, lon). Edges are junction-to-junction
    segments with: length (m), geom ([[lat,lon],...] following the real path), geom_from (node id geom starts at),
    steps (bool), osm_surface, name. Only the largest connected component is kept (islands can't be routed to)."""
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in osm["elements"] if e["type"] == "node"}
    ways = [w for w in osm["elements"] if w["type"] == "way" and w.get("nodes") and _walkable(w.get("tags", {}))]
    if not ways:
        raise ValueError("no walkable ways found")
    if lat0 is None:                                # centre the local metric frame on the data itself
        lat0 = sum(v[0] for v in nodes.values()) / len(nodes); lon0 = sum(v[1] for v in nodes.values()) / len(nodes)
    cnt = Counter(n for w in ways for n in set(w["nodes"]))
    raw = nx.Graph()
    for w in ways:
        ns = [n for n in w["nodes"] if n in nodes]
        if len(ns) < 2:
            continue
        cut = [0] + [i for i in range(1, len(ns) - 1) if cnt[ns[i]] > 1] + [len(ns) - 1]
        tags = w.get("tags", {})
        for a, b in zip(cut[:-1], cut[1:]):
            seg = ns[a:b + 1]; u, v = seg[0], seg[-1]
            geom = [list(nodes[n]) for n in seg]
            L = sum(_m(*geom[i], *geom[i + 1]) for i in range(len(geom) - 1))
            if u == v or L < min_len:
                continue
            if raw.has_edge(u, v) and raw[u][v]["length"] <= L:
                continue
            raw.add_edge(u, v, length=L, geom=geom, geom_from=u, steps=tags.get("highway") == "steps",
                         osm_surface=tags.get("surface"), name=tags.get("name"))
    comp = max(nx.connected_components(raw), key=len)
    raw = raw.subgraph(comp).copy()
    ids = {n: i for i, n in enumerate(raw.nodes)}
    G = nx.Graph()
    for n, i in ids.items():
        lat, lon = nodes[n]
        G.add_node(i, lat=lat, lon=lon, x=(lon - lon0) * 111320 * math.cos(math.radians(lat0)), y=(lat - lat0) * 111320)
    for u, v, d in raw.edges(data=True):
        G.add_edge(ids[u], ids[v], **{**d, "geom_from": ids[d["geom_from"]]})
    G.graph.update(lat0=lat0, lon0=lon0)
    return G


def load_graph(path, **kw):
    return build_graph(json.load(open(path)), **kw)


def edge_coords(G, path):
    """Lat/lon polyline for a node path that FOLLOWS the real edge geometry (not straight lines between junctions)."""
    out = []
    for u, v in zip(path[:-1], path[1:]):
        d = G[u][v]
        g = d.get("geom")
        if not g:
            g = [[G.nodes[u]["lat"], G.nodes[u]["lon"]], [G.nodes[v]["lat"], G.nodes[v]["lon"]]]
        elif d.get("geom_from", u) != u:
            g = g[::-1]
        out.extend(g if not out else g[1:])
    return out
