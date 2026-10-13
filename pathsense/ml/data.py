"""Windowing for the deep models (numpy only; tested)."""
import numpy as np
from ..config import FS


def make_windows(traces, G, win_s=2.0, step_s=1.0, fs=FS):
    """Returns X (n,6,win) [acc xyz, gyro xyz], y (n,) segment class, dev (n,), seg (n,). Windows never span 2 segments."""
    cls = {d["idx"]: d["cls"] for _, _, d in G.edges(data=True)}
    w, s = int(win_s * fs), int(step_s * fs)
    X, y, dev, seg = [], [], [], []
    for t in traces:
        sig = np.hstack([t.acc, t.gyro])
        for a in range(0, len(sig) - w + 1, s):
            if t.seg[a] != t.seg[a + w - 1]:
                continue
            X.append(sig[a:a + w].T); y.append(cls[int(t.seg[a])]); dev.append(t.device); seg.append(int(t.seg[a]))
    X = np.asarray(X, np.float32)
    mu, sd = X.mean((0, 2), keepdims=True), X.std((0, 2), keepdims=True) + 1e-6
    return (X - mu) / sd, np.array(y), np.array(dev), np.array(seg)
