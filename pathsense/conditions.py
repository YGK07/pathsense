"""What is really known about each path segment, and where it came from. Used by the website instead of the
simulated surface values. Nothing here is simulated.

Surface, strongest source first:
  1. team edits on the site ("Path conditions")        data/path_conditions.json
  2. a visual check of satellite imagery               data/satellite_review.json
  3. the OpenStreetMap surface / smoothness tags
  4. the OSM road type: campus roads (service, residential) are taken as paved; tracks as unpaved
  otherwise the surface is "not checked yet".
Kerbs are the kerbs walkers marked with the Annotation button (press positions from data/raw walks, grouped within
8 m). Jolts the detector found on 2+ walks with no button press nearby are only "possible kerbs". Steps come from
OSM highway=steps and from team edits; stairs reported on the site or on walks are added later by
routing.mark_reported.
"""
import json, math, os
import numpy as np

SURFACES = {   # label, roughness (0-1 field scale), poor = per-metre penalty share
    "smooth":  dict(label="Smooth paving", rough=0.15, poor=0.0),
    "paved":   dict(label="Paved", rough=0.3, poor=0.0),
    "broken":  dict(label="Broken or patchy paving", rough=0.6, poor=0.4),
    "unpaved": dict(label="Unpaved (dirt, gravel or grass)", rough=0.8, poor=1.0),
    "unknown": dict(label="Not checked yet", rough=0.35, poor=0.0),
}
OSM_SURFACE = {"asphalt": "smooth", "concrete": "smooth", "paved": "paved", "paving_stones": "paved", "concrete:plates": "paved",
               "metal": "paved", "wood": "paved", "sett": "broken", "cobblestone": "broken", "unhewn_cobblestone": "broken",
               "compacted": "unpaved", "fine_gravel": "unpaved", "gravel": "unpaved", "pebblestone": "unpaved",
               "unpaved": "unpaved", "ground": "unpaved", "dirt": "unpaved", "earth": "unpaved", "mud": "unpaved",
               "grass": "unpaved", "sand": "unpaved"}
BAD_SMOOTHNESS = {"bad": "broken", "very_bad": "broken", "horrible": "unpaved", "very_horrible": "unpaved", "impassable": "unpaved"}
ROADS = ("service", "residential", "unclassified", "tertiary", "secondary", "primary", "living_street", "road")
SOURCE_LABEL = {"team": "checked by the team", "satellite": "satellite imagery check", "osm": "OpenStreetMap tag",
                "road": "campus road, taken as paved", "track": "OpenStreetMap track, taken as unpaved"}
KERB_GROUP_M, KERB_SNAP_M, MAYBE_CLEAR_M = 8.0, 12.0, 10.0


def _m(a, b):
    return math.hypot((a[1] - b[1]) * 111320 * math.cos(math.radians(a[0])), (a[0] - b[0]) * 111320)


def load(path, default):
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else default


def kerb_spots(walks):
    """Button-pressed kerbs from all walks, grouped: a kerb pressed on several walks (or up on one, down on another)
    is one spot. -> [{lat, lon, presses, walks:[ids]}]"""
    spots = []
    for w in walks:
        for p in w.get("labelled", []):
            for s in spots:
                if _m((s["lat"], s["lon"]), (p["lat"], p["lon"])) <= KERB_GROUP_M:
                    n = s["presses"]; s["lat"] = (s["lat"] * n + p["lat"]) / (n + 1); s["lon"] = (s["lon"] * n + p["lon"]) / (n + 1)
                    s["presses"] += 1; s["walks"].add(w["id"]); break
            else:
                spots.append(dict(lat=p["lat"], lon=p["lon"], presses=1, walks={w["id"]}))
    return [dict(s, walks=sorted(s["walks"])) for s in spots]


class Segments:
    """Nearest-segment lookup on (lat, lon) that can prefer footpaths over roads."""

    def __init__(self, G):
        lat0 = float(np.mean([d["lat"] for _, d in G.nodes(data=True)])); self.c = math.cos(math.radians(lat0))
        A, B, E = [], [], []
        for u, v, d in G.edges(data=True):
            g = d.get("geom") or [[G.nodes[u]["lat"], G.nodes[u]["lon"]], [G.nodes[v]["lat"], G.nodes[v]["lon"]]]
            for p, q in zip(g[:-1], g[1:]):
                A.append(self.xy(*p)); B.append(self.xy(*q)); E.append(d["idx"])
        self.A, self.B, self.E = np.array(A), np.array(B), np.array(E)

    def xy(self, lat, lon):
        return (lon * 111320 * self.c, lat * 111320)

    def dists(self, lat, lon):
        """distance (m) from the point to every segment idx -> {idx: d}"""
        P = np.array(self.xy(lat, lon)); AB = self.B - self.A
        t = np.clip(((P - self.A) * AB).sum(1) / ((AB ** 2).sum(1) + 1e-9), 0, 1)
        d = np.linalg.norm(P - (self.A + t[:, None] * AB), axis=1)
        out = {}
        for e, x in zip(self.E, d):
            if x < out.get(int(e), 1e9):
                out[int(e)] = float(x)
        return out

    def nearest(self, lat, lon, max_d, prefer=None):
        D = self.dists(lat, lon)
        if prefer:
            near = [(x, e) for e, x in D.items() if x <= max_d and e in prefer]
            if near:
                return min(near)[1]
        e, x = min(D.items(), key=lambda kv: kv[1])
        return e if x <= max_d else None


def build(G, real, team=None, satellite=None):
    """-> ({idx: condition dict}, button-pressed kerb spots, possible-kerb spots). G: the OSM path graph (osm.build_graph attrs); real: outputs/real_walks.json;
    team / satellite: {str(idx): {...}} from the two data files."""
    team, satellite = team or {}, satellite or {}
    seg = Segments(G)
    foot = {d["idx"] for *_, d in G.edges(data=True) if d.get("osm_highway") not in ROADS}
    cond = {}
    for *_, d in G.edges(data=True):
        i = d["idx"]; hw = d.get("osm_highway")
        surf, src = "unknown", None
        if hw in ROADS:
            surf, src = "paved", "road"
        elif hw == "track":
            surf, src = "unpaved", "track"
        if d.get("osm_surface") in OSM_SURFACE:
            surf, src = OSM_SURFACE[d["osm_surface"]], "osm"
        if d.get("osm_smoothness") in BAD_SMOOTHNESS:
            surf, src = BAD_SMOOTHNESS[d["osm_smoothness"]], "osm"
        sat = satellite.get(str(i))
        if sat and sat.get("surface") in SURFACES and sat["surface"] != "unknown":
            surf, src = sat["surface"], "satellite"
        t = team.get(str(i), {})
        if t.get("surface") in SURFACES and t["surface"] != "unknown":
            surf, src = t["surface"], "team"
        grad = None
        from .osm import incline_deg
        g = incline_deg(d.get("osm_incline"))
        if g is not None:
            grad = abs(g)
        if t.get("slope") == "steep":
            grad = max(grad or 0, 8.0)
        elif t.get("slope") == "gentle":
            grad = max(grad or 0, 4.0)
        elif t.get("slope") == "flat":
            grad = 0.0
        steps = int(bool(d.get("steps")))
        if t.get("steps") == "yes":
            steps = max(steps, 1)
        elif t.get("steps") == "no":
            steps = 0
        cond[i] = dict(surface=surf, surface_src=src, grad=grad, steps=steps, kerbs=int(d.get("osm_kerbs", 0)),
                       kerb_walks=[], maybe=0, ramp=t.get("kerb") == "ramp", no_kerb=t.get("kerb") == "none",
                       team_kerb=t.get("kerb") == "kerb",
                       narrow=bool(t.get("narrow")), no_foot=t.get("foot") == "no", note=t.get("note", ""), photos=t.get("photos", []),
                       team=bool(t), team_by=t.get("contributor", ""), team_at=t.get("updated", ""),
                       satellite_note=(sat or {}).get("note", ""), highway=hw, name=d.get("name"))
    spots = kerb_spots(real.get("walks", []))
    for s in spots:
        e = seg.nearest(s["lat"], s["lon"], KERB_SNAP_M, prefer=foot)    # a kerb is the footpath's, not the road's
        if e is not None:
            cond[e]["kerbs"] += 1; cond[e]["kerb_walks"] = sorted(set(cond[e]["kerb_walks"]) | set(s["walks"]))
            s["edge"] = e
    maybes = []
    for c in real.get("clusters", []):
        if not c.get("confirmed") or any(_m((c["lat"], c["lon"]), (s["lat"], s["lon"])) <= MAYBE_CLEAR_M for s in spots):
            continue
        e = seg.nearest(c["lat"], c["lon"], KERB_SNAP_M)
        if e is not None:
            cond[e]["maybe"] += 1
            maybes.append(dict(lat=c["lat"], lon=c["lon"], walks=c["n_walks"], edge=e))
    for c in cond.values():
        if c["ramp"] or c["no_kerb"]:                    # the team looked: no kerb to climb here
            c["kerbs"] = 0; c["maybe"] = 0
        elif c["team_kerb"]:
            c["kerbs"] = max(c["kerbs"], 1)
    clear = {i for i, c in cond.items() if c["ramp"] or c["no_kerb"]}
    spots = [s for s in spots if s.get("edge") not in clear]
    maybes = [x for x in maybes if x["edge"] not in clear]
    return cond, spots, maybes


def attrs(c):
    """routing attributes (the edge['est'] dict) from a condition"""
    S = SURFACES[c["surface"]]
    grad = c["grad"] or 0.0
    a = dict(grad=grad, rough=S["rough"], kerbs=c["kerbs"], stairs=c["steps"], poor=S["poor"], maybe=c["maybe"],
             unknown=int(c["surface"] == "unknown"), narrow=int(c["narrow"]))
    if c.get("no_foot"):
        a["closed"] = 1                                  # nobody should walk or roll here (cars only)
    if c["kerbs"] or c["steps"] or c["surface"] == "unpaved" or grad >= 7 or c.get("no_foot"):
        a["cls"] = 2
    elif c["surface"] == "broken" or c["maybe"] or grad >= 4 or c["narrow"]:
        a["cls"] = 1
    else:
        a["cls"] = 0
    return a


def describe(c):
    """short human lines for a tooltip / editor"""
    out = [f"{SURFACES[c['surface']]['label']}" + (f" ({SOURCE_LABEL[c['surface_src']]})" if c["surface_src"] else "")]
    if c["kerbs"]:
        out.append(f"{c['kerbs']} kerb{'s' if c['kerbs'] > 1 else ''}" +
                   (f", marked on {len(c['kerb_walks'])} walk{'s' if len(c['kerb_walks']) > 1 else ''}" if c["kerb_walks"] else " (team)"))
    if c["ramp"]:
        out.append("kerbs have ramps (team)")
    if c["maybe"]:
        out.append(f"{c['maybe']} possible kerb{'s' if c['maybe'] > 1 else ''} (bumps found by the phone, not marked)")
    if c["steps"]:
        out.append("steps")
    if c["grad"]:
        out.append(f"slope about {c['grad']:.0f}°")
    if c["narrow"]:
        out.append("narrow")
    if c.get("no_foot"):
        out.append("not for walking (vehicles only)")
    return out
