"""Demo server. Run `python scripts/run_pipeline.py` first (creates outputs/map.json), then:
    python app/server.py   ->  http://127.0.0.1:5000

Real walks: `python scripts/process_walks.py` builds outputs/real_walks.json from data/raw/; the page shows it as the
"Real walks" layer. POST /api/upload adds a Sensor Logger .zip (stored under data/raw/uploads/) and rebuilds the layer.
"""
import json, os, re, sys, time, zipfile
import networkx as nx
from flask import Flask, jsonify, request, send_from_directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import copy, csv
from pathsense.routing import route, path_stats, mark_real_kerbs, mark_reported
from pathsense.config import PROFILES
from pathsense.osm import edge_coords
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from process_walks import process, find_walks, load_contributors

ROOT = os.path.join(os.path.dirname(__file__), "..")
app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024          # a 5-minute walk zip is ~3 MB
MAP_PATH = os.path.join(ROOT, "outputs", "map.json")
if not os.path.exists(MAP_PATH):
    sys.exit("outputs/map.json not found: run  python scripts/run_pipeline.py --osm data/osm_vit.json  first")
MAP = json.load(open(MAP_PATH))
RAW = os.path.join(ROOT, "data", "raw")
REAL_PATH = os.path.join(ROOT, "outputs", "real_walks.json")
OSM_PATH = os.path.join(ROOT, "data", "osm_vit.json")


def build_graph(m):
    G = nx.Graph()
    for n in m["nodes"]:
        G.add_node(n["id"], x=n["x"], y=n["y"], lat=n["lat"], lon=n["lon"])
    for e in m["edges"]:
        t = e["true"]
        G.add_edge(e["u"], e["v"], idx=e["idx"], length=e["length"], est=e["est"], observed=e["observed"],
                   grad=t["grad"], rough=t["rough"], kerbs=t["kerbs"], cls=t["cls"],
                   geom=e.get("geom"), geom_from=e.get("geom_from"))
    return G


# Fixed demo trip for the recorded demo (lat, lon; snapped to the nearest junction). Chosen because the shortest path
# follows the K/L/M Block footpath walked on 15 Oct 2026, which has 7 kerb spots confirmed by 2+ real walks.
DEMO = dict(start=(12.97305, 79.16405), end=(12.97100, 79.15843), profile="manual_wheelchair",
            label="Fitty gym (hostel side) to the Souvenier Subway (academic side)")


FEATURES_PATH = os.path.join(ROOT, "data", "ground_truth", "features.csv")


def reports():
    """Barriers walkers reported in the ground-truth sheet that have a position (lat/lon)."""
    if not os.path.exists(FEATURES_PATH):
        return []
    out = []
    for r in csv.DictReader(open(FEATURES_PATH, encoding="utf-8")):
        try:
            out.append(dict(r, lat=float(r["lat"]), lon=float(r["lon"])))
        except (TypeError, ValueError):
            continue
    return out


def rebuild():
    """Routing graph = simulated/OSM map + kerb spots confirmed by real walks + barriers reported by walkers.
    Each edge also records how many contributors walked it, so the page can say how much of a route is backed by data."""
    global G, REAL_EDGES, REPORTED_EDGES
    G = build_graph(MAP)
    real = json.load(open(REAL_PATH)) if os.path.exists(REAL_PATH) else {}
    REAL_EDGES = {}
    for i in mark_real_kerbs(G, real.get("clusters", [])):
        REAL_EDGES[i] = REAL_EDGES.get(i, 0) + 1
    REPORTED_EDGES = set(mark_reported(G, reports()))
    walked = {int(k): len(v) for k, v in real.get("edge_walks", {}).items()}
    for *_, d in G.edges(data=True):
        d["walks"] = walked.get(d["idx"], 0)


rebuild()


def coords(path):          # follows the real path geometry when the map came from OSM
    return edge_coords(G, path)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/map")
def api_map():
    m = copy.deepcopy(MAP)                      # show real confirmed kerbs on the map too
    by_idx = {d["idx"]: d for *_, d in G.edges(data=True)}
    for e in m["edges"]:
        d = by_idx[e["idx"]]
        if e["idx"] in REAL_EDGES or e["idx"] in REPORTED_EDGES:
            e["est"] = dict(d["est"])
        e["real_kerbs"] = d.get("real_kerbs", 0); e["reported"] = d.get("reported", []); e["walks"] = d["walks"]
    m["reports"] = [dict(lat=r["lat"], lon=r["lon"], type=r["type"], landmark=r["landmark"]) for r in reports()]
    return jsonify(m)


def nearest_node(lat, lon):
    return min(G.nodes, key=lambda n: (G.nodes[n]["lat"] - lat) ** 2 + (G.nodes[n]["lon"] - lon) ** 2)


@app.get("/api/demo")
def api_demo():
    return jsonify(src=nearest_node(*DEMO["start"]), dst=nearest_node(*DEMO["end"]), profile=DEMO["profile"], label=DEMO["label"])


@app.get("/api/real")
def api_real():
    if not os.path.exists(REAL_PATH):
        return jsonify(real=False, walks=[], clusters=[], summary={})
    return send_from_directory(os.path.dirname(REAL_PATH), os.path.basename(REAL_PATH), max_age=0)


@app.post("/api/upload")
def api_upload():
    """Add one Sensor Logger export (.zip) to the real-walks layer. The zip is read in place, never extracted."""
    f = request.files.get("file")
    contributor = request.form.get("contributor", "").strip().lower()
    carry = request.form.get("carry", "hand")
    if not f or not f.filename.lower().endswith(".zip"):
        return jsonify(error="choose the .zip that Sensor Logger exported"), 400
    if not re.fullmatch(r"c\d{2,3}", contributor):
        return jsonify(error="contributor id must look like c01 (no names, please)"), 400
    if carry not in ("hand", "pocket"):
        return jsonify(error="carry must be hand or pocket"), 400
    os.makedirs(os.path.join(RAW, "uploads"), exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(f.filename)[0])[:60]
    dest = os.path.join(RAW, "uploads", f"{time.strftime('%Y%m%d-%H%M%S')}_{contributor}_{stem}.zip")
    f.save(dest)
    try:
        names = {os.path.basename(n) for n in zipfile.ZipFile(dest).namelist()}
        missing = {"Accelerometer.csv", "Gravity.csv", "Location.csv"} - names
        if missing:
            raise ValueError("missing " + ", ".join(sorted(missing)) + " (switch those sensors on in Sensor Logger)")
        json.dump(dict(contributor=contributor, carry=carry), open(dest + ".json", "w"))
        summary = process(find_walks(RAW), OSM_PATH, REAL_PATH, load_contributors(os.path.join(RAW, "contributors.csv")))
        rebuild()
    except Exception as e:                           # bad zip: remove it so it doesn't break the next rebuild
        for p in (dest, dest + ".json"):
            if os.path.exists(p):
                os.remove(p)
        return jsonify(error=f"could not read this recording: {e}"), 400
    return jsonify(ok=True, summary=summary)


@app.get("/api/route")
def api_route():
    profile = request.args.get("profile", "manual_wheelchair")
    src, dst = int(request.args["src"]), int(request.args["dst"])
    if profile not in PROFILES or src == dst:
        return jsonify(error="bad request"), 400
    ps = route(G, src, dst, profile, "est")          # PathSense: uses the inferred map
    sh = route(G, src, dst, "shortest", "true")       # distance-only baseline
    # stats are measured against ground truth (simulation) so the demo shows the real benefit
    return jsonify(pathsense=dict(coords=coords(ps), **path_stats(G, ps, profile, "true")),
                   shortest=dict(coords=coords(sh), **path_stats(G, sh, profile, "true")))


if __name__ == "__main__":
    app.run(debug=True)
