"""Functionality 6: profile-conditioned accessibility-cost routing (A* over the segment graph).

cost(e) = d(e) + w_g*grad_deg + w_r*roughness + w_k*kerbs + w_s*stairs (+ CLOSED_COST if reported closed)
          + d(e)*(w_p*poor_surface + w_u*unchecked) + w_m*possible_kerbs + w_n*narrow   (real-condition terms)
(weights per mobility profile, see config.PROFILES)
"""
import numpy as np, networkx as nx
from .config import PROFILES, CLOSED_COST


def edge_cost(d, profile, attrs):
    p = PROFILES[profile] if isinstance(profile, str) else profile
    return (d["length"] + p["w_g"] * attrs["grad"] + p["w_r"] * attrs["rough"] + p["w_k"] * attrs["kerbs"]
            + p.get("w_s", 0.0) * attrs.get("stairs", 0) + CLOSED_COST * attrs.get("closed", 0)
            + d["length"] * (p.get("w_p", 0.0) * attrs.get("poor", 0) + p.get("w_u", 0.0) * attrs.get("unknown", 0))
            + p.get("w_m", 0.0) * attrs.get("maybe", 0) + p.get("w_n", 0.0) * attrs.get("narrow", 0))


def route(G, src, dst, profile, attr_key="est"):
    """attr_key: 'est' uses estimated attrs stored in edge['est']; 'true' uses ground truth (edge itself)."""
    def w(u, v, d):
        a = d[attr_key] if attr_key != "true" else d
        return edge_cost(d, profile, a)
    h = lambda a, b: float(np.hypot(G.nodes[a]["x"] - G.nodes[b]["x"], G.nodes[a]["y"] - G.nodes[b]["y"]))
    return nx.astar_path(G, src, dst, heuristic=h, weight=w)


def path_stats(G, path, profile="shortest", attr_key="true"):
    L = rough = kerbs = blocked = caution = real = reported = walked = stairs = closed = unpaved = unknown = maybe = 0
    cost = 0.0
    for u, v in zip(path[:-1], path[1:]):
        d = G[u][v]; a = d[attr_key] if attr_key != "true" else d
        L += d["length"]; kerbs += a["kerbs"]; blocked += int(a["cls"] == 2); caution += int(a["cls"] == 1)
        rough += a["rough"] * d["length"]; cost += edge_cost(d, profile, a)
        real += int(d.get("real_kerbs", 0)); reported += len(d.get("reported", []))
        stairs += int(a.get("stairs", 0)); closed += int(a.get("closed", 0))
        walked += d["length"] if d.get("walks", 0) > 0 else 0
        unpaved += d["length"] * (a.get("poor", 0) >= 1); unknown += d["length"] * a.get("unknown", 0); maybe += a.get("maybe", 0)
    return dict(length=L, kerbs=int(kerbs), blocked=blocked, caution=caution, mean_rough=rough / max(L, 1), cost=cost,
                real_kerbs=real, reported=reported, walked_frac=walked / max(L, 1e-9), stairs=stairs, closed=closed,
                unpaved_m=unpaved, unknown_m=unknown, maybe_kerbs=maybe)


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


REPORT_TYPES = {"stairs": "Stairs", "steps": "Stairs", "kerb": "Kerb", "broken": "Broken or rough path",
                "rough": "Broken or rough path", "steep": "Steep slope", "blocked": "Path blocked or closed",
                "closed": "Path blocked or closed"}


def mark_reported(G, reports, max_dist=15.0):
    """Barriers walkers reported, with a position (lat/lon): from the ground-truth sheet, the site's "Report a
    problem" form or stairs double-pressed on a walk. The nearest segment's edge['est'] changes by type -
    stairs/steps: +1 flight of stairs; kerb: +1 kerb; broken/rough: roughness >= 0.85; steep: slope >= 8 deg;
    blocked/closed: closed to everyone - and the report is listed in edge['reported']. Rows without a position or
    farther than max_dist from any path are skipped. Returns the edge idx values marked."""
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
        d = by_idx[int(i)]; a = dict(d["est"]); t = (r.get("type") or "").lower()
        if t in ("stairs", "steps"):
            a["stairs"] = a.get("stairs", 0) + 1
        elif t == "kerb":
            a["kerbs"] = a["kerbs"] + 1
        elif t in ("broken", "rough"):
            a["rough"] = max(a["rough"], 0.85)
        elif t == "steep":
            a["grad"] = max(a["grad"], 8.0)
        elif t in ("blocked", "closed"):
            a["closed"] = 1
        else:
            continue
        a["cls"] = 2
        d["est"] = a
        d.setdefault("reported", []).append(f"{r.get('type', 'barrier')}: {r.get('landmark') or r.get('note') or ''}".strip().rstrip(":"))
        marked.append(int(i))
    return marked
