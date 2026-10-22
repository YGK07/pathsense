"""Benchmarking rig analysis (proposal: "validate inferred surface profiles against a high-precision ground truth").

A run = the pushed trolley with a REFERENCE sensor rigidly fixed to its frame (a calibrated IMU board such as a
BNO055 / MPU-6050 logging to SD, or a second phone in a rigid mount) plus the phone under test, recording at the same
time, along a route whose kerbs and slopes were measured by hand (tape and level app, docs/BENCHMARK_RIG.md).

    python scripts/benchmark_rig.py --phone data/raw/rig/run1_phone.zip --reference data/raw/rig/run1_imu.csv \
        --measurements data/ground_truth/rig_measurements.csv --out outputs/rig_run1.json

--reference is either a Sensor Logger export (folder / zip) or a CSV with columns t (s), ax, ay, az (m/s^2, gravity
INCLUDED, as IMU boards report it) and optionally gx, gy, gz (rad/s).

What it reports:
  lag_s                 clock offset between the two recordings, found by cross-correlating |acceleration|
  vertical_corr         correlation of the vertical (gravity-direction) acceleration after alignment, 10 Hz
  roughness_corr        Spearman of 5 Hz-band roughness in 5 s windows, phone vs reference
  kerbs                 per measured kerb: jolt peak on phone and on reference, and Spearman of each vs kerb height
  slopes                per measured slope: phone tilt offset and reference tilt offset vs the level-app slope
"""
import argparse, csv, json, math, os, sys
import numpy as np
from scipy.signal import correlate
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathsense.features import highpass
from pathsense.io import load_recording, position

FS = 50.0


def from_csv(path):
    rows = list(csv.DictReader(open(path)))
    t = np.array([float(r["t"]) for r in rows])
    a = np.array([[float(r["ax"]), float(r["ay"]), float(r["az"])] for r in rows])
    g = np.array([[float(r.get("gx", 0) or 0), float(r.get("gy", 0) or 0), float(r.get("gz", 0) or 0)] for r in rows])
    return t, a, g


def series(src):
    """-> t (50 Hz), vertical user acceleration, |total acc|, tilt (deg) about the sensor's x axis, recording or None."""
    if os.path.isdir(src) or src.lower().endswith(".zip"):
        r = load_recording(src); t0 = r.t
        total = r.acc + r.grav; grav = r.grav
    else:
        r = None; t0, total, _ = from_csv(src)
        k = max(1, int(0.5 * (1 / np.median(np.diff(t0)))))      # gravity = 1 s moving average of total acceleration
        grav = np.stack([np.convolve(total[:, i], np.ones(2 * k + 1) / (2 * k + 1), "same") for i in range(3)], 1)
    t = np.arange(t0[0], t0[-1], 1 / FS)
    tot = np.stack([np.interp(t, t0, total[:, i]) for i in range(3)], 1)
    gr = np.stack([np.interp(t, t0, grav[:, i]) for i in range(3)], 1)
    gu = gr / np.linalg.norm(gr, axis=1, keepdims=True)
    vert = ((tot - gr) * gu).sum(1)
    tilt = np.degrees(np.arctan2(gu[:, 1], np.sqrt(gu[:, 0] ** 2 + gu[:, 2] ** 2)))
    return t, vert, np.linalg.norm(tot, axis=1), tilt, r


def find_lag(t1, x1, t2, x2, max_lag=60.0):
    """Seconds to ADD to the reference clock so it lines up with the phone (cross-correlation of |acc| envelopes)."""
    lo, hi = min(t1[0], t2[0]), max(t1[-1], t2[-1]); grid = np.arange(lo, hi, 1 / FS)
    e1 = np.interp(grid, t1, np.abs(highpass(x1, 0.5, FS)), left=0, right=0)
    e2 = np.interp(grid, t2, np.abs(highpass(x2, 0.5, FS)), left=0, right=0)
    c = correlate(e1 - e1.mean(), e2 - e2.mean(), mode="full")
    lags = (np.arange(len(c)) - (len(e2) - 1)) / FS
    ok = np.abs(lags) <= max_lag
    return float(lags[ok][np.argmax(c[ok])])


def analyse(phone, reference, measurements=None):
    tp, vp, ap, tiltp, rp = series(phone)
    tr, vr, ar, tiltr, _ = series(reference)
    lag = find_lag(tp, ap, tr, ar)
    tr = tr + lag
    m = (tp >= tr[0]) & (tp <= tr[-1]); t = tp[m]
    v1 = vp[m]; v2 = np.interp(t, tr, vr)
    dec = slice(None, None, 5)                                    # 10 Hz for the waveform correlation
    out = dict(lag_s=round(lag, 3), overlap_s=round(float(t[-1] - t[0]), 1),
               vertical_corr=round(float(np.corrcoef(v1[dec], v2[dec])[0, 1]), 3))
    h1, h2 = highpass(v1, 5.0, FS), highpass(v2, 5.0, FS)
    w = int(5 * FS); r1 = [np.std(h1[i:i + w]) for i in range(0, len(t) - w, w)]; r2 = [np.std(h2[i:i + w]) for i in range(0, len(t) - w, w)]
    out["roughness_corr"] = round(float(spearmanr(r1, r2).correlation), 3) if len(r1) > 3 else None
    if measurements and rp is not None:
        rows = list(csv.DictReader(open(measurements)))
        kerbs, slopes = [], []
        for r in rows:
            lat, lon = float(r["lat"]), float(r["lon"])
            j = np.argmin((rp.gps[:, 1] - lat) ** 2 + (rp.gps[:, 2] - lon) ** 2)       # when the phone passed it
            tc = rp.gps[j, 0]; win = (t >= tc - 3) & (t <= tc + 3)
            if not win.any():
                continue
            if r["type"] == "kerb":
                kerbs.append(dict(id=r["id"], height_cm=float(r["value"]), phone_peak=round(float(np.abs(v1[win]).max()), 2),
                                  reference_peak=round(float(np.abs(v2[win]).max()), 2)))
            elif r["type"] == "slope":
                base1, base2 = np.median(tiltp[m]), np.median(np.interp(t, tr, tiltr))
                slopes.append(dict(id=r["id"], measured_deg=float(r["value"]),
                                   phone_tilt_offset=round(float(np.median(tiltp[m][win]) - base1), 2),
                                   reference_tilt_offset=round(float(np.median(np.interp(t[win], tr, tiltr)) - base2), 2)))
        def sp(xs, a, b):
            return round(float(spearmanr([x[a] for x in xs], [x[b] for x in xs]).correlation), 3) if len(xs) > 2 else None
        out["kerbs"] = dict(items=kerbs, phone_vs_height_spearman=sp(kerbs, "phone_peak", "height_cm"),
                            reference_vs_height_spearman=sp(kerbs, "reference_peak", "height_cm"))
        out["slopes"] = dict(items=slopes, phone_vs_measured_spearman=sp(slopes, "phone_tilt_offset", "measured_deg"),
                             reference_vs_measured_spearman=sp(slopes, "reference_tilt_offset", "measured_deg"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phone", required=True); ap.add_argument("--reference", required=True)
    ap.add_argument("--measurements"); ap.add_argument("--out")
    a = ap.parse_args()
    res = analyse(a.phone, a.reference, a.measurements)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
