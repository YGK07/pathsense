"""Synthetic campus + inertial trace generator.

IMPORTANT: everything here is SIMULATED. It exists so the whole pipeline can be built, tested and
demoed before real data exists. Numbers produced from it must NEVER be reported as research results.
Replace with real traces (docs/DATA_COLLECTION.md) as soon as the pilot is done.
"""
from dataclasses import dataclass
import numpy as np, networkx as nx
from .config import FS, G
from .labels import label_from_attrs


def _smooth_field(rng, xs, ys, n=5, scale=120.0):
    cx, cy = rng.uniform(xs.min(), xs.max(), n), rng.uniform(ys.min(), ys.max(), n)
    a = rng.uniform(0.3, 1.0, n)
    f = np.zeros_like(xs, dtype=float)
    for i in range(n):
        f += a[i] * np.exp(-((xs - cx[i]) ** 2 + (ys - cy[i]) ** 2) / (2 * scale ** 2))
    return f / f.max()


def add_surface_attributes(G, seed=0, scale_frac=0.25):
    """Give any road graph an edge `idx` plus SIMULATED ground-truth surface attributes (rough, grad, kerbs, cls).
    Nodes need x,y in metres. Existing edge 'length' is kept; edges flagged steps=True get >=1 kerb.
    Replace with real ground truth (task P4) when it exists."""
    rng = np.random.default_rng(seed)
    E = list(G.edges)
    xs = np.array([G.nodes[n]["x"] for n in G.nodes]); ys = np.array([G.nodes[n]["y"] for n in G.nodes])
    ext = max(np.ptp(xs), np.ptp(ys), 1.0)
    mx = np.array([(G.nodes[u]["x"] + G.nodes[v]["x"]) / 2 for u, v in E])
    my = np.array([(G.nodes[u]["y"] + G.nodes[v]["y"]) / 2 for u, v in E])
    rough_f = _smooth_field(rng, mx, my, n=5, scale=scale_frac * ext)
    grad_f = _smooth_field(rng, mx, my, n=3, scale=scale_frac * 1.25 * ext)
    kerb_f = _smooth_field(rng, mx, my, n=4, scale=scale_frac * 0.75 * ext)
    for k, (u, v) in enumerate(E):
        d = G[u][v].get("length") or float(np.hypot(G.nodes[u]["x"] - G.nodes[v]["x"], G.nodes[u]["y"] - G.nodes[v]["y"]))
        rough = float(np.clip(0.9 * rough_f[k] ** 1.5 + rng.normal(0, 0.04), 0, 1))
        grad = float(np.clip(9.5 * grad_f[k] ** 2 + abs(rng.normal(0, 0.4)), 0, 12))
        kerbs = int(rng.poisson(0.5 * kerb_f[k] ** 2.5))
        if G[u][v].get("steps"):
            kerbs = max(kerbs, 1)
        G[u][v].update(idx=k, length=float(d), rough=rough, grad=grad, kerbs=kerbs,
                       cls=int(label_from_attrs(grad, rough, kerbs)))
    return G


def make_campus_graph(nx_=9, ny_=7, spacing=60.0, seed=0, drop=0.12):
    """Jittered grid road network (FAKE geometry) -- only for tests/offline demo. Use pathsense.osm for real paths."""
    rng = np.random.default_rng(seed)
    G0 = nx.grid_2d_graph(nx_, ny_)
    pos = {n: (n[0] * spacing + rng.normal(0, spacing * .12), n[1] * spacing + rng.normal(0, spacing * .12)) for n in G0}
    ids = {n: i for i, n in enumerate(G0.nodes)}
    G = nx.Graph()
    for n, i in ids.items():
        G.add_node(i, x=pos[n][0], y=pos[n][1])
    for u, v in G0.edges:
        G.add_edge(ids[u], ids[v])
    for u, v in list(G.edges):                         # randomly drop edges, keep connectivity
        if rng.random() < drop:
            G.remove_edge(u, v)
            if not nx.is_connected(G):
                G.add_edge(u, v)
    return add_surface_attributes(G, seed=seed)


@dataclass
class Trace:
    acc: np.ndarray      # (N,3) m/s^2, device frame: x forward, y left, z up
    gyro: np.ndarray     # (N,3) rad/s
    seg: np.ndarray      # (N,) edge idx per sample (stand-in for GPS map-matching)
    device: int
    contributor: int
    gps_acc: float       # metres; larger = worse
    reliability: float   # simulated truth, for evaluating the fusion model only


def make_devices(n=5, seed=1):
    rng = np.random.default_rng(seed)
    return [dict(gain=float(rng.normal(1, .15)), noise=float(rng.uniform(.04, .2)),
                 tilt=float(rng.normal(0, .6))) for _ in range(n)]


def make_contributors(n=12, n_devices=5, seed=2):
    rng = np.random.default_rng(seed)
    return [dict(device=int(rng.integers(n_devices)),
                 reliability=float(rng.choice([.95, .85, .6, .35], p=[.35, .3, .2, .15]))) for _ in range(n)]


def _segment_signal(rng, e, dev, rel, fs, speed, sign):
    n = max(int(e["length"] / speed * fs), 8)
    mult = 1 + (1 - rel) * 2.0                       # unreliable contributors are noisier
    if rng.random() < 0.4 * (1 - rel):               # phone loose in bag/pocket on this segment: burst of vibration
        mult *= rng.uniform(2.0, 3.5)
    th = np.deg2rad(e["grad"] * sign + dev["tilt"])
    sig = (0.15 + 2.5 * e["rough"]) * dev["gain"] * mult
    az = G * np.cos(th) + rng.normal(0, sig, n) + rng.normal(0, dev["noise"], n)
    ax = -G * np.sin(th) + rng.normal(0, sig * .3 + dev["noise"], n)
    ay = rng.normal(0, sig * .3 + dev["noise"], n)
    gy = rng.normal(0, .02 + .25 * e["rough"] * mult, n)
    t = np.arange(n) / fs

    def spike(at):
        amp = rng.uniform(5, 9) * dev["gain"]
        m = t >= at
        z = np.zeros(n)
        z[m] = amp * np.exp(-(t[m] - at) / .08) * np.cos(2 * np.pi * 9 * (t[m] - at))
        return z
    for _ in range(e["kerbs"]):
        if rng.random() < (0.05 + 0.1 * (1 - rel)):      # kerb missed (user bypassed / glitch)
            continue
        az += spike(rng.uniform(.2, max(t[-1] - .2, .3)))
    if rng.random() < 0.3 * (1 - rel):                    # spurious bump from sloppy carrying
        az += spike(rng.uniform(.2, max(t[-1] - .2, .3)))
    return (np.stack([ax, ay, az], 1),
            np.stack([rng.normal(0, .02, n), gy, rng.normal(0, .02, n)], 1))


def make_traces(G, n_traces=70, n_contrib=12, n_dev=5, seed=3, fs=FS, hub_bias=0.7):
    """Random shortest-path walks. hub_bias>0 concentrates traffic so some segments stay unobserved."""
    rng = np.random.default_rng(seed)
    devs, cons = make_devices(n_dev), make_contributors(n_contrib, n_dev)
    nodes = list(G.nodes)
    xs = np.array([G.nodes[n]["x"] for n in nodes]); ys = np.array([G.nodes[n]["y"] for n in nodes])
    cx, cy = xs.mean(), ys.mean()
    w = np.exp(-hub_bias * ((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * (xs.std() * 0.8) ** 2))
    w /= w.sum()
    out = []
    while len(out) < n_traces:
        a, b = rng.choice(nodes, 2, replace=False, p=w)
        path = nx.shortest_path(G, int(a), int(b), weight="length")
        if len(path) < 3:
            continue
        ci = int(rng.integers(len(cons))); c = cons[ci]; dev = devs[c["device"]]
        speed = float(rng.uniform(.8, 1.4)); acc, gyro, seg = [], [], []
        for u, v in zip(path[:-1], path[1:]):
            e = G[u][v]
            A, Gy = _segment_signal(rng, e, dev, c["reliability"], fs, speed, rng.choice([-1, 1]))
            acc.append(A); gyro.append(Gy); seg.append(np.full(len(A), e["idx"]))
        out.append(Trace(np.vstack(acc), np.vstack(gyro), np.concatenate(seg), c["device"], ci,
                         float(rng.uniform(3, 8) * (1 + (1 - c["reliability"]))), c["reliability"]))
    return out
