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
    L = rough = kerbs = blocked = caution = 0
    cost = 0.0
    for u, v in zip(path[:-1], path[1:]):
        d = G[u][v]; a = d[attr_key] if attr_key != "true" else d
        L += d["length"]; kerbs += a["kerbs"]; blocked += int(a["cls"] == 2); caution += int(a["cls"] == 1)
        rough += a["rough"] * d["length"]; cost += edge_cost(d, profile, a)
    return dict(length=L, kerbs=int(kerbs), blocked=blocked, caution=caution, mean_rough=rough / max(L, 1), cost=cost)
