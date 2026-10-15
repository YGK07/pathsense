"""Real data (roadmap tasks P2 + P3): Sensor Logger exports -> aligned arrays, kerb events, map-matched Trace.

Works on an unzipped export folder or the .zip itself (read in memory, nothing is extracted), iOS or Android.

Two quirks of the real app that this module handles:
- Gravity.csv / Gyroscope.csv rows don't line up with Accelerometer.csv, so they are interpolated onto its clock.
- Annotation.csv is stamped with a different clock (time since phone boot), so presses can't be placed directly.
  Protocol fix: the 3 start taps are presses of the Annotation button. align_annotations finds the clock offset by
  matching those presses to the jolts they make; without a confident match, annotations are left unplaced.
"""
import csv, io, math, os, zipfile
from dataclasses import dataclass, field
import numpy as np
from scipy.signal import find_peaks
from .config import FS
from .features import highpass
from .simulate import Trace


@dataclass
class Recording:
    name: str
    platform: str
    device: str
    epoch: float              # recording start, seconds since 1970
    t: np.ndarray             # (N,) seconds since Start
    acc: np.ndarray           # (N,3) user acceleration, gravity removed, phone frame (x right, y top edge, z out of screen)
    grav: np.ndarray          # (N,3) gravity, same frame
    gyro: np.ndarray          # (N,3) rad/s
    gps: np.ndarray           # (M,4) seconds, lat, lon, horizontal accuracy (m)
    ann_raw: np.ndarray       # (K,) annotation times on the annotation clock (s)
    ann_labels: list = field(default_factory=list)   # 'up' (My Annotation 1) / 'down' (My Annotation 2) / other text

    @property
    def fs(self):
        return float(1 / np.median(np.diff(self.t)))


def _reader(path):
    """name -> csv text, for a folder or a zip (files may sit in a subfolder of the zip)."""
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        names = {os.path.basename(n): n for n in z.namelist() if n.lower().endswith(".csv")}
        return lambda n: z.read(names[n]).decode("utf-8-sig") if n in names else None
    return lambda n: open(os.path.join(path, n), encoding="utf-8-sig").read() if os.path.exists(os.path.join(path, n)) else None


def _xyz(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    t = np.array([float(r["seconds_elapsed"]) for r in rows])
    return t, np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])


def _onto(t_src, x_src, t):
    return np.stack([np.interp(t, t_src, x_src[:, i]) for i in range(x_src.shape[1])], 1)


def load_recording(path):
    read = _reader(path)
    meta = next(csv.DictReader(io.StringIO(read("Metadata.csv") or "a\n")), {})
    t, acc = _xyz(read("Accelerometer.csv"))
    if read("Gravity.csv"):
        grav = _onto(*_xyz(read("Gravity.csv")), t)
    elif read("TotalAcceleration.csv"):              # some Android phones: gravity = total - user acceleration
        grav = _onto(*_xyz(read("TotalAcceleration.csv")), t) - acc
    else:
        raise ValueError(f"{path}: needs Gravity.csv (switch Gravity on in Sensor Logger)")
    gyro = _onto(*_xyz(read("Gyroscope.csv")), t) if read("Gyroscope.csv") else np.zeros_like(acc)
    gps = np.zeros((0, 4))
    if read("Location.csv"):
        rows = list(csv.DictReader(io.StringIO(read("Location.csv"))))
        gps = np.array([[float(r["seconds_elapsed"]), float(r["latitude"]), float(r["longitude"]),
                         float(r.get("horizontalAccuracy") or 99)] for r in rows])
    ann, labels = np.zeros(0), []
    if read("Annotation.csv"):
        rows = list(csv.DictReader(io.StringIO(read("Annotation.csv"))))
        ann = np.array([int(r["time"]) / 1e9 for r in rows])
        labels = ["up" if r["text"].strip().endswith("1") else "down" if r["text"].strip().endswith("2") else r["text"]
                  for r in rows]
    epoch = float(meta.get("recording epoch time") or 0) / 1000
    return Recording(os.path.basename(str(path).rstrip("/\\")).removesuffix(".zip"), meta.get("platform", "?"),
                     meta.get("device name", "?"), epoch, t, acc, grav, gyro, gps, ann, labels)


def vertical(r):
    """User acceleration along gravity (true vertical, m/s2), whatever way the phone is held."""
    return (r.acc * (r.grav / np.linalg.norm(r.grav, axis=1, keepdims=True))).sum(1)


def walking_window(r, win=2.0, thr=0.45, sustain=6):
    """First and last moment of sustained walking: `sustain` consecutive 2 s windows (1 s apart) whose robust spread
    (1.4826 x median absolute deviation) of |user acc| is above thr. Taps are brief impulses that barely move the
    median, so a burst of sync taps right before walking is not counted as walking."""
    a = np.linalg.norm(r.acc, axis=1)
    edges = np.arange(r.t[0], r.t[-1] - win, 1.0)
    def spread(x):
        return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else 0.0
    act = np.array([spread(a[(r.t >= s) & (r.t < s + win)]) > thr for s in edges])
    run = np.convolve(act.astype(int), np.ones(sustain, int), "valid") == sustain
    if not run.any():
        return float(r.t[0]), float(r.t[-1])
    i = np.flatnonzero(run)
    return float(edges[i[0]]), float(edges[i[-1] + sustain - 1] + win)


def _groups(x, gap=4.0):
    g = [[0]] if len(x) else []
    for i in range(1, len(x)):
        (g[-1].append(i) if x[i] - x[g[-1][-1]] < gap else g.append([i]))
    return g


def align_annotations(r, min_conf=3.0):
    """Offset that maps annotation clock -> seconds since Start, from the leading group of >=3 sync presses.
    Returns (offset_for_first_press, confidence) or (None, confidence) when the presses can't be found in the signal."""
    if len(r.ann_raw) < 3:
        return None, 0.0
    rel = r.ann_raw - r.ann_raw[0]
    sync = _groups(rel)[0]
    if len(sync) < 3:
        return None, 0.0
    fs = r.fs
    sa = np.abs(highpass(np.linalg.norm(r.acc, axis=1), 3.0, fs))
    sg = np.abs(highpass(np.linalg.norm(r.gyro, axis=1), 3.0, fs))
    sa, sg = sa / (np.median(sa) + 1e-9), sg / (np.median(sg) + 1e-9)
    w = int(0.15 * fs)
    # max over +-0.15 s, computed once, then sampled at candidate press times
    from scipy.ndimage import maximum_filter1d
    burst = maximum_filter1d(sa, 2 * w + 1) + maximum_filter1d(sg, 2 * w + 1)
    a, b = walking_window(r)                        # start sync presses come before (or as) the walk begins
    offs = np.arange(r.t[0], max(r.t[0] + 0.01, min(a + 3.0, r.t[-1]) - rel[sync[-1]]), 0.01)
    sc = sum(np.interp(offs + rel[i], r.t, burst) for i in sync)
    last = _groups(rel)[-1] if len(_groups(rel)) > 1 else []
    for i in last:                                  # end presses (after the walk) are taps too: they must also hit jolts
        on = (offs + rel[i] > b - 1) & (offs + rel[i] <= r.t[-1])
        sc = sc + np.where(on, np.interp(offs + rel[i], r.t, burst), 0)
    conf = float(sc.max() / (np.median(sc) + 1e-9))
    return (float(offs[np.argmax(sc)]), conf) if conf >= min_conf else (None, conf)


def kerb_presses(r, offset):
    """[(seconds since Start, 'up'/'down')] for annotation presses that fall inside the walk (sync presses dropped)."""
    if offset is None:
        return []
    a, b = walking_window(r)
    t = r.ann_raw - r.ann_raw[0] + offset
    sync = {i for g in (_groups(t)[0], _groups(t)[-1]) if len(g) >= 3 for i in g}   # start and end sync taps
    return [(float(t[i]), r.ann_labels[i]) for i in range(len(t)) if i not in sync and a <= t[i] <= b]


def detect_events(r, mult=2.0):
    """Kerb-like jolts: peaks of high-passed vertical acceleration above `mult` x the median footstep, inside the walk."""
    a, b = walking_window(r)
    inside = (r.t > a + 1) & (r.t < b - 1)
    hp = np.abs(highpass(vertical(r), 3.0, r.fs)) * inside
    steps, _ = find_peaks(hp, distance=int(0.4 * r.fs))
    step = float(np.median(hp[steps])) if len(steps) else 1.0
    pk, _ = find_peaks(hp, height=mult * step, distance=int(0.5 * r.fs))
    return [dict(t=float(r.t[p]), strength=float(hp[p] / step)) for p in pk]


def position(r, t, max_acc=30.0):
    """lat, lon at time(s) t from GPS fixes with accuracy <= max_acc m (linear interpolation)."""
    g = r.gps[(r.gps[:, 3] <= max_acc) & (r.gps[:, 0] >= -5)]
    return np.interp(t, g[:, 0], g[:, 1]), np.interp(t, g[:, 0], g[:, 2])


# ---------------------------------------------------------------- map matching (P3: nearest edge + mode smoothing)
def _xy(lat, lon, lat0, lon0):
    return (np.asarray(lon) - lon0) * 111320 * math.cos(math.radians(lat0)), (np.asarray(lat) - lat0) * 111320


def edge_index(G):
    lat0, lon0 = G.graph.get("lat0"), G.graph.get("lon0")
    if lat0 is None:
        lat0 = float(np.mean([d["lat"] for _, d in G.nodes(data=True)])); lon0 = float(np.mean([d["lon"] for _, d in G.nodes(data=True)]))
    A, B, E = [], [], []
    for k, (u, v, d) in enumerate(G.edges(data=True)):   # idx = edge order, same as simulate.add_surface_attributes
        g = d.get("geom") or [[G.nodes[u]["lat"], G.nodes[u]["lon"]], [G.nodes[v]["lat"], G.nodes[v]["lon"]]]
        x, y = _xy([p[0] for p in g], [p[1] for p in g], lat0, lon0)
        for i in range(len(g) - 1):
            A.append((x[i], y[i])); B.append((x[i + 1], y[i + 1])); E.append(d.get("idx", k))
    return dict(A=np.array(A), B=np.array(B), E=np.array(E), lat0=lat0, lon0=lon0)


def match_points(ix, lat, lon, max_dist=25.0, smooth=5):
    """Edge idx (or -1 when farther than max_dist from any mapped path) and distance (m) per point."""
    px, py = _xy(lat, lon, ix["lat0"], ix["lon0"])
    P = np.stack([np.atleast_1d(px), np.atleast_1d(py)], 1)
    AB = ix["B"] - ix["A"]; L2 = (AB ** 2).sum(1) + 1e-9
    s = np.clip(((P[:, None, :] - ix["A"][None]) * AB[None]).sum(2) / L2[None], 0, 1)
    d = np.linalg.norm(P[:, None, :] - (ix["A"][None] + s[..., None] * AB[None]), axis=2)
    j = d.argmin(1); dist = d[np.arange(len(P)), j]; e = ix["E"][j].copy()
    if smooth > 1:                                     # mode filter: a lone fix on a side street doesn't switch edges
        h = smooth // 2; e0 = e.copy()
        for i in range(len(e)):
            vals, cnt = np.unique(e0[max(0, i - h):i + h + 1], return_counts=True); e[i] = vals[cnt.argmax()]
    e[dist > max_dist] = -1
    return e, dist


def to_trace(r, G, device=0, contributor=0, fs=FS, ix=None):
    """Recording -> pipeline Trace: resampled to fs, phone frame -> pipeline frame (x forward, y left, z up; phone held
    flat with its top edge forward), and `seg` = map-matched edge idx per sample (-1 off the map or outside the walk)."""
    a, b = walking_window(r)
    t = np.arange(a, b, 1 / fs)
    tot = _onto(r.t, r.acc + r.grav, t); gy = _onto(r.t, r.gyro, t)
    acc = np.stack([tot[:, 1], -tot[:, 0], tot[:, 2]], 1); gyro = np.stack([gy[:, 1], -gy[:, 0], gy[:, 2]], 1)
    g = r.gps[(r.gps[:, 3] <= 30) & (r.gps[:, 0] >= a - 5) & (r.gps[:, 0] <= b + 5)]
    e, _ = match_points(ix or edge_index(G), g[:, 1], g[:, 2])
    seg = e[np.clip(np.searchsorted(g[:, 0], t), 0, len(g) - 1)]
    return Trace(acc, gyro, seg.astype(int), device, contributor, float(np.median(g[:, 3])), float("nan"))
