"""Tiny fake Sensor Logger export (same files and columns the real app writes) so pathsense.io is testable offline.

The phone is held flat (gravity on +z), walks along a straight east-west line, gets three sync taps at 5.0/5.7/6.4 s
(pressed on the Annotation button) and steps down a kerb at 20.0 s, pressing "My Annotation 2" 0.6 s later.
Annotation times use a different clock from the sensors, exactly like the real app (seconds since phone boot).
"""
import csv, io, os, zipfile
import numpy as np

EPOCH = 1792068843.836          # recording start, seconds since 1970
BOOT_CLOCK = 2289196.349        # what the annotation clock reads at the first sync press
TAPS = (5.0, 5.7, 6.4)
KERB_DOWN = 20.0
PRESS_LAG = 0.6


def _rows(lat0, lon0, dur=30.0, fs=100.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 1 / fs)
    walking = (t > 8) & (t < 28)
    step = np.where(walking, 1.5 * np.sin(2 * np.pi * 1.8 * t), 0.0)          # ~108 steps/min bounce
    z = step + rng.normal(0, 0.05, len(t))
    for c in TAPS + (KERB_DOWN,):
        m = (t >= c) & (t < c + 0.25)
        amp = 9.0 if c == KERB_DOWN else 14.0
        z[m] += amp * np.exp(-(t[m] - c) / 0.04) * np.cos(2 * np.pi * 12 * (t[m] - c))
    acc = np.stack([rng.normal(0, .05, len(t)), rng.normal(0, .05, len(t)), z], 1)
    grav = np.tile([0.0, 0.3, 9.8], (len(t), 1))
    gyro = rng.normal(0, .02, (len(t), 3))
    gt = np.arange(-3, dur, 1.0)                                                # 1 Hz GPS, fix starts before recording
    speed = np.clip((gt - 8) / 20, 0, 1) * 25                                   # 25 m east over the walk
    glat = np.full(len(gt), lat0)
    glon = lon0 + speed / (111320 * np.cos(np.radians(lat0)))
    return t, acc, grav, gyro, gt, glat, glon


def write(path_or_zip, lat0, lon0, platform="ios", as_zip=False):
    t, acc, grav, gyro, gt, glat, glon = _rows(lat0, lon0)
    files = {}

    def sensor(name, arr, tt):
        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow(["time", "seconds_elapsed", "z", "y", "x"])
        for s, (x, y, z) in zip(tt, arr):
            w.writerow([int((EPOCH + s) * 1e9), s, z, y, x])
        files[name] = buf.getvalue()
    sensor("Accelerometer.csv", acc, t)
    sensor("Gravity.csv", grav[:-3], t[:-3])                                   # real files differ in length by a few rows
    sensor("Gyroscope.csv", gyro, t)
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["time", "seconds_elapsed", "altitude", "speedAccuracy", "bearingAccuracy", "latitude",
                "altitudeAboveMeanSeaLevel", "bearing", "horizontalAccuracy", "verticalAccuracy", "longitude", "speed"])
    for s, la, lo in zip(gt, glat, glon):
        w.writerow([int((EPOCH + s) * 1e9), s, 200, -1, -1, la, 200, -1, 6.0, 10, lo, -1])
    files["Location.csv"] = buf.getvalue()
    presses = [(c - TAPS[0], "My Annotation 1") for c in TAPS] + [(KERB_DOWN + PRESS_LAG - TAPS[0], "My Annotation 2")]
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["time", "seconds_elapsed", "text", "millisecond_press_duration"])
    for r, txt in presses:
        w.writerow([int((BOOT_CLOCK + r) * 1e9), -1789779647.487 + r, txt, 70])
    files["Annotation.csv"] = buf.getvalue()
    files["Metadata.csv"] = ("version,device name,recording epoch time,recording time,recording timezone,platform,appVersion,"
                             "device id,sensors,sampleRateMs,standardisation,platform version\n"
                             f"3,TestPhone,{int(EPOCH * 1000)},2026-10-15_12-54-03,Asia/Kolkata,{platform},1.68,x,"
                             "Accelerometer|Gravity|Gyroscope|Location|Annotation,10|10|10|0|,true,1\n")
    if as_zip:
        with zipfile.ZipFile(path_or_zip, "w") as z:
            for n, s in files.items():
                z.writestr(n, s)
    else:
        os.makedirs(path_or_zip, exist_ok=True)
        for n, s in files.items():
            open(os.path.join(path_or_zip, n), "w", newline="").write(s)
    return path_or_zip
