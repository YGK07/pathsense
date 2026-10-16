"""Functionality 6: profile-conditioned accessibility-cost routing (A* over the segment graph).

cost(e) = d(e) + w_g*grad_deg + w_r*roughness + w_k*kerbs   (weights per mobility profile, see config.PROFILES)
"""
import numpy as np, networkx as nx
from .config import PROFILES


def edge_cost(d, profile, attrs):
    p = PROFILES[profile] if isinstance(profile, str) else profile
    return d["length"] + p["w_g"] * attrs["grad"] + p["w_r"] * attrs["rough"] + p["w_k"] * attrs["kerbs"]


def route(G, src, dst, profile, attr_key="est"):
    """attr_key: 'est' uses estimated attrs stored in edge['est']; 'true' uses ground truth (edge itself)."""
    def w(u, v, d):
        a = d[attr_key] if attr_key != "true" else d
        return edge_cost(d, profile, a)
    h = lambda a, b: float(np.hypot(G.nodes[a]["x"] - G.nodes[b]["x"], G.nodes[a]["y"] - G.nodes[b]["y"]))
    return nx.astar_path(G, src, dst, heuristic=h, weight=w)


def path_stats(G, path, profile="shortest", attr_key="true"):
    L = rough = kerbs = blocked = caution = real = reported = walked = 0
    cost = 0.0
    for u, v in zip(path[:-1], path[1:]):
        d = G[u][v]; a = d[attr_key] if attr_key != "true" else d
        L += d["length"]; kerbs += a["kerbs"]; blocked += int(a["cls"] == 2); caution += int(a["cls"] == 1)
        rough += a["rough"] * d["length"]; cost += edge_cost(d, profile, a)
        real += int(d.get("real_kerbs", 0)); reported += len(d.get("reported", []))
        walked += d["length"] if d.get("walks", 0) > 0 else 0
    return dict(length=L, kerbs=int(kerbs), blocked=blocked, caution=caution, mean_rough=rough / max(L, 1), cost=cost,
                real_kerbs=real, reported=reported, walked_frac=walked / max(L, 1e-9))


def mark_real_kerbs(G, spots, max_dist=15.0):
    """Put kerb spots confirmed by real walks (process_walks clusters with confirmed=True) on their nearest segment:
    edge['real_kerbs'] counts them and edge['est'] gets at least that many kerbs, so routing avoids them.
    Spots farther than max_dist (m) from any mapped path are skipped. Returns the edge idx values marked."""
    from .io import edge_index, match_points
    spots = [s for s in spots if s.get("confirmed")]
    if not spots:
        return []
    e, _ = match_points(edge_index(G), [s["lat"] for s in spots], [s["lon"] for s in spots], max_dist=max_dist, smooth=1)
    by_idx = {d["idx"]: d for *_, d in G.edges(data=True)}
    marked = []
    for i in e:
        if i < 0:
            continue
        d = by_idx[int(i)]
        d["real_kerbs"] = d.get("real_kerbs", 0) + 1
        d["est"] = {**d["est"], "kerbs": max(d["est"]["kerbs"], d["real_kerbs"]), "cls": 2}
        marked.append(int(i))
    return marked


STEP_KERBS = 3        # a reported flight of steps costs like 3 kerbs: never worth it for a wheelchair profile


def mark_reported(G, reports, max_dist=15.0):
    """Barriers reported by walkers in the ground-truth sheet (data/ground_truth/features.csv rows with lat/lon):
    type 'steps' or 'kerb'. The nearest segment becomes blocked in edge['est'] and lists the report in
    edge['reported']. Rows without a position are skipped. Returns the edge idx values marked."""
    from .io import edge_index, match_points
    rows = []
    for r in reports:
        try:
            rows.append((float(r["lat"]), float(r["lon"]), r))
        except (TypeError, ValueError, KeyError):
            continue
    if not rows:
        return []
    e, _ = match_points(edge_index(G), [x[0] for x in rows], [x[1] for x in rows], max_dist=max_dist, smooth=1)
    by_idx = {d["idx"]: d for *_, d in G.edges(data=True)}
    marked = []
    for i, (_, _, r) in zip(e, rows):
        if i < 0:
            continue
        d = by_idx[int(i)]
        k = STEP_KERBS if r.get("type") == "steps" else 1
        d["est"] = {**d["est"], "kerbs": d["est"]["kerbs"] + k, "cls": 2}
        d.setdefault("reported", []).append(f"{r.get('type', 'barrier')}: {r.get('landmark', '')}".strip())
        marked.append(int(i))
    return marked
