"""Tiny fake Overpass response (same JSON shape the real API returns) so OSM code is testable offline."""
import numpy as np
from pathsense.config import LAT0, LON0


def fake_overpass(nx_=14, ny_=10, step_m=55.0, seed=0):
    rng = np.random.default_rng(seed)
    dlat, dlon = step_m / 111320.0, step_m / (111320.0 * np.cos(np.deg2rad(LAT0)))
    els, nid = [], {}
    def node(i, j):
        if (i, j) not in nid:
            k = len(nid) + 1; nid[(i, j)] = k
            els.append(dict(type="node", id=k, lat=LAT0 + (j - ny_ / 2) * dlat + rng.normal(0, dlat * .1),
                            lon=LON0 + (i - nx_ / 2) * dlon + rng.normal(0, dlon * .1)))
        return nid[(i, j)]
    wid = 1000
    def way(ns, **tags):
        nonlocal wid; wid += 1; els.append(dict(type="way", id=wid, nodes=ns, tags=tags))
    for j in range(ny_):                                   # long E-W footways: junction nodes only where a N-S way crosses
        way([node(i, j) for i in range(nx_)], highway="footway")
    for i in range(0, nx_, 2):                             # N-S paths on every 2nd column
        way([node(i, j) for j in range(ny_)], highway="service", name=f"Road {i}")
    way([node(1, 1), node(1, 2)], highway="steps")         # steps (inside the grid, shares nodes)
    way([node(3, 3), node(3, 4)], highway="footway", foot="no")      # must be ignored
    way([node(5, 5), node(5, 6)], highway="service", access="private")   # must be ignored
    a, b = 9001, 9002                                      # disconnected island: must be dropped
    els += [dict(type="node", id=a, lat=LAT0 + .02, lon=LON0), dict(type="node", id=b, lat=LAT0 + .0201, lon=LON0)]
    way([a, b], highway="footway")
    return dict(elements=els)
