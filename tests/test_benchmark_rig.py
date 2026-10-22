"""Benchmarking-rig analysis on a synthetic pair: a phone recording and a reference IMU log that is 2.5 s out of sync."""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmark_rig import analyse


def _walk(seed=0, fs=100.0, dur=120.0):
    rng = np.random.default_rng(seed); t = np.arange(0, dur, 1 / fs)
    vert = 0.8 * np.sin(2 * np.pi * 1.8 * t) + 0.2 * rng.normal(size=len(t))          # rolling / walking vibration
    kerbs = [(20.0, 3.0), (45.0, 9.0), (70.0, 6.0), (95.0, 12.0)]                     # (time, height cm)
    for tk, h in kerbs:
        m = (t >= tk) & (t < tk + 0.4); vert[m] += h * np.exp(-(t[m] - tk) / 0.08) * np.cos(2 * np.pi * 9 * (t[m] - tk))
    return t, vert, kerbs


def test_rig_finds_offset_and_matches_kerbs(tmp_path):
    t, vert, kerbs = _walk()
    d = tmp_path / "phone"; d.mkdir()
    lat0, lon0 = 12.97, 79.16
    def w(name, header, rows):
        with open(d / name, "w", newline="") as f:
            c = csv.writer(f); c.writerow(header); c.writerows(rows)
    w("Accelerometer.csv", ["time", "seconds_elapsed", "z", "y", "x"], [[0, f"{x:.3f}", f"{v:.4f}", 0, 0] for x, v in zip(t, vert)])
    w("Gravity.csv", ["time", "seconds_elapsed", "z", "y", "x"], [[0, f"{x:.3f}", 9.81, 0, 0] for x in t])
    w("Metadata.csv", ["device name", "platform"], [["test", "ios"]])
    gps_t = np.arange(0, t[-1], 1.0)
    w("Location.csv", ["seconds_elapsed", "latitude", "longitude", "horizontalAccuracy"],
      [[f"{x:.1f}", lat0 + x * 1e-5, lon0, 5] for x in gps_t])
    rng = np.random.default_rng(1); shift = 2.5                                         # reference starts 2.5 s later
    tr = t[t >= shift] - shift
    with open(tmp_path / "imu.csv", "w", newline="") as f:
        c = csv.writer(f); c.writerow(["t", "ax", "ay", "az"])
        for x, v in zip(tr, vert[t >= shift]):
            c.writerow([f"{x:.3f}", 0, 0, f"{9.81 + 1.1 * v + 0.1 * rng.normal():.4f}"])
    with open(tmp_path / "meas.csv", "w", newline="") as f:
        c = csv.writer(f); c.writerow(["id", "type", "lat", "lon", "value"])
        for i, (tk, h) in enumerate(kerbs):
            c.writerow([f"k{i}", "kerb", lat0 + (tk + 0.2) * 1e-5, lon0, h])
    r = analyse(str(d), str(tmp_path / "imu.csv"), str(tmp_path / "meas.csv"))
    assert abs(r["lag_s"] - shift) < 0.05
    assert r["vertical_corr"] > 0.9
    assert len(r["kerbs"]["items"]) == 4 and r["kerbs"]["phone_vs_height_spearman"] == 1.0
    assert r["kerbs"]["reference_vs_height_spearman"] == 1.0
