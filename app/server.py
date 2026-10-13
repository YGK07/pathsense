"""Demo server. Run `python scripts/run_pipeline.py` first (creates outputs/map.json), then:
    python app/server.py   ->  http://127.0.0.1:5000
"""
import json, os, sys
import networkx as nx
from flask import Flask, jsonify, request, send_from_directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsense.routing import route, path_stats
from pathsense.config import PROFILES

ROOT = os.path.join(os.path.dirname(__file__), "..")
app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "static"))
MAP = json.load(open(os.path.join(ROOT, "outputs", "map.json")))


def build_graph(m):
    G = nx.Graph()
    for n in m["nodes"]:
        G.add_node(n["id"], x=n["x"], y=n["y"], lat=n["lat"], lon=n["lon"])
    for e in m["edges"]:
        t = e["true"]
        G.add_edge(e["u"], e["v"], idx=e["idx"], length=e["length"], est=e["est"], observed=e["observed"],
                   grad=t["grad"], rough=t["rough"], kerbs=t["kerbs"], cls=t["cls"])
    return G


G = build_graph(MAP)


def coords(path):
    return [[G.nodes[n]["lat"], G.nodes[n]["lon"]] for n in path]


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/map")
def api_map():
    return jsonify(MAP)


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
