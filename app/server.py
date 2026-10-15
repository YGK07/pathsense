"""Demo server. Run `python scripts/run_pipeline.py` first (creates outputs/map.json), then:
    python app/server.py   ->  http://127.0.0.1:5000

Real walks: `python scripts/process_walks.py` builds outputs/real_walks.json from data/raw/; the page shows it as the
"Real walks" layer. POST /api/upload adds a Sensor Logger .zip (stored under data/raw/uploads/) and rebuilds the layer.
"""
import json, os, re, sys, time, zipfile
import networkx as nx
from flask import Flask, jsonify, request, send_from_directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsense.routing import route, path_stats
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


G = build_graph(MAP)


def coords(path):          # follows the real path geometry when the map came from OSM
    return edge_coords(G, path)


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/map")
def api_map():
    return jsonify(MAP)


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
