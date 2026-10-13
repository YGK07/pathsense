"""Download the real walkable network of the VIT Vellore campus from OpenStreetMap and cache it (needs internet).

    python scripts/fetch_osm.py                         # whole campus, using the campus boundary mapped in OSM
    python scripts/fetch_osm.py --mode radius --radius 1800        # fallback: circle around config.LAT0/LON0
    python scripts/fetch_osm.py --mode bbox --bbox 12.955,79.140,12.985,79.175   # south,west,north,east

Then:  python scripts/run_pipeline.py --osm data/osm_vit.json && python app/server.py
Check the printed bounding box against the real campus; if part is missing, widen with --mode radius/bbox.
Map data (c) OpenStreetMap contributors, ODbL.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsense.osm import fetch, build_graph
from pathsense.config import LAT0, LON0

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["area", "radius", "bbox"], default="area")
ap.add_argument("--lat", type=float, default=LAT0); ap.add_argument("--lon", type=float, default=LON0)
ap.add_argument("--radius", type=int, default=1500); ap.add_argument("--bbox", type=lambda s: [float(x) for x in s.split(",")])
ap.add_argument("--out", default="data/osm_vit.json")
a = ap.parse_args()
if a.mode == "bbox" and not a.bbox:
    ap.error("--bbox south,west,north,east is required with --mode bbox")
if os.path.exists(a.out):
    os.remove(a.out)
data = fetch(a.mode, a.lat, a.lon, a.radius, a.bbox, cache=a.out)
G = build_graph(data)
la = [d["lat"] for _, d in G.nodes(data=True)]; lo = [d["lon"] for _, d in G.nodes(data=True)]
km = sum(d["length"] for *_, d in G.edges(data=True)) / 1000
print(f"saved {a.out}: {G.number_of_nodes()} junctions, {G.number_of_edges()} segments, {km:.1f} km of paths")
print(f"coverage  lat {min(la):.4f}..{max(la):.4f}   lon {min(lo):.4f}..{max(lo):.4f}")
