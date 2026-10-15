"""Turn real Sensor Logger walks into the demo's "Real walks" layer: outputs/real_walks.json.

    python scripts/process_walks.py                      # every export folder / .zip under data/raw/
    python scripts/process_walks.py --contributors data/raw/contributors.csv

contributors.csv (git-ignored, lives next to the raw data): `device,contributor` rows, e.g. `iPhone 15,c01`.
An upload from the demo stores `<zip>.json` beside the zip with its contributor id and carrying mode.

Everything here is REAL data, but the kerb detector is a simple threshold (see detect_events): report its
recall/precision against the annotated presses, not as a finished classifier.
"""
import argparse, csv, glob, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsense.osm import load_graph
from pathsense.io import (load_recording, align_annotations, kerb_presses, detect_events, position, walking_window,
                          edge_index, match_points)

LAG = (-1.5, 0.2)          # a press comes 0.4-0.9 s after the step: a detection 0-1.5 s before it counts as found
CLUSTER_M = 15.0


def find_walks(root):
    out = []
    for p in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if p.lower().endswith(".zip") or (os.path.isdir(p) and os.path.exists(os.path.join(p, "Accelerometer.csv"))):
            out.append(p)
    return out


def _m(a, b):
    return math.hypot((a[1] - b[1]) * 111320 * math.cos(math.radians(a[0])), (a[0] - b[0]) * 111320)


def walk_summary(path, ix, contributors):
    r = load_recording(path)
    side = {}
    if os.path.exists(path + ".json"):
        side = json.load(open(path + ".json"))
    a, b = walking_window(r)
    good = r.gps[(r.gps[:, 3] <= 30) & (r.gps[:, 0] >= a) & (r.gps[:, 0] <= b)]
    e, dist = match_points(ix, good[:, 1], good[:, 2])
    length = sum(_m(good[i, 1:3], good[i + 1, 1:3]) for i in range(len(good) - 1))
    off, conf = align_annotations(r)
    ev = detect_events(r)
    for x in ev:
        x["lat"], x["lon"] = (float(v) for v in position(r, x["t"]))
    lab = []
    for t, l in kerb_presses(r, off):
        la, lo = position(r, t)
        lab.append(dict(t=t, label=l, lat=float(la), lon=float(lo),
                        detected=any(LAG[0] <= x["t"] - t <= LAG[1] for x in ev)))
    for x in ev:
        x["matches_press"] = any(LAG[0] <= x["t"] - p["t"] <= LAG[1] for p in lab) if lab else None
    keep = np.r_[0:len(good):max(1, len(good) // 400)]                         # <= ~400 points per track
    return dict(id=r.name, contributor=side.get("contributor") or contributors.get(r.device, "c??"),
                device=r.device, platform=r.platform, carry=side.get("carry", "hand"),
                duration_s=round(b - a, 1), length_m=round(length), gps_median_acc_m=round(float(np.median(good[:, 3])), 1),
                off_map_frac=round(float((e < 0).mean()), 3), sync_ok=off is not None, sync_conf=round(conf, 2),
                track=[[round(float(la), 6), round(float(lo), 6)] for la, lo in good[keep, 1:3]],
                edges=sorted(int(x) for x in set(e.tolist()) if x >= 0), events=ev, labelled=lab)


def clusters(walks):
    """Group jolts by place: each group is a 15 m circle around its centre (no chaining along a path), strongest first."""
    pts = sorted(((x["lat"], x["lon"], w["id"], x["strength"]) for w in walks for x in w["events"]), key=lambda p: -p[3])
    cl = []
    for p in pts:
        best = min(cl, key=lambda c: _m(p, c["c"]), default=None)
        if best is not None and _m(p, best["c"]) < CLUSTER_M:
            best["pts"].append(p)
        else:
            cl.append(dict(c=p, pts=[p]))
    out = [dict(lat=round(float(np.mean([q[0] for q in c["pts"]])), 6), lon=round(float(np.mean([q[1] for q in c["pts"]])), 6),
                n_events=len(c["pts"]), walks=sorted(set(q[2] for q in c["pts"])),
                spread_m=round(max(_m(q, c["c"]) for q in c["pts"]), 1)) for c in cl]
    for c in out:
        c["n_walks"] = len(c["walks"]); c["confirmed"] = c["n_walks"] >= 2
    return sorted(out, key=lambda c: -c["n_walks"])


def process(paths, osm, out, contributors):
    G = load_graph(osm); ix = edge_index(G)
    walks = [walk_summary(p, ix, contributors) for p in paths]
    cov = {}
    for w in walks:
        for e in w["edges"]:
            cov.setdefault(e, set()).add(w["contributor"])
    lab = [p for w in walks for p in w["labelled"]]
    graded = [x for w in walks if w["labelled"] for x in w["events"]]
    S = dict(n_walks=len(walks), n_contributors=len({w["contributor"] for w in walks}),
             n_devices=len({w["device"] for w in walks}), total_km=round(sum(w["length_m"] for w in walks) / 1000, 2),
             edges_walked=len(cov), labelled_up=sum(p["label"] == "up" for p in lab),
             labelled_down=sum(p["label"] == "down" for p in lab),
             found_up=sum(p["detected"] for p in lab if p["label"] == "up"),
             found_down=sum(p["detected"] for p in lab if p["label"] == "down"),
             detections_in_labelled_walks=len(graded), detections_matching_press=sum(bool(x["matches_press"]) for x in graded))
    C = clusters(walks)
    S["clusters"] = len(C); S["confirmed_clusters"] = sum(c["confirmed"] for c in C)
    json.dump(dict(real=True, summary=S, walks=walks, clusters=C,
                   edge_walks={str(k): sorted(v) for k, v in cov.items()}), open(out, "w"))
    return S


def load_contributors(path):
    if path and os.path.exists(path):
        return {r["device"]: r["contributor"] for r in csv.DictReader(open(path))}
    return {}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--osm", default="data/osm_vit.json")
    ap.add_argument("--out", default="outputs/real_walks.json"); ap.add_argument("--contributors", default="data/raw/contributors.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print(json.dumps(process(find_walks(a.raw), a.osm, a.out, load_contributors(a.contributors)), indent=2))
