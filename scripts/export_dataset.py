"""Package the recorded walks as a shareable, anonymised dataset (proposal: "publish the annotated Indian pedestrian
infrastructure trace dataset").

    python scripts/export_dataset.py                       # -> dataset/pathsense-vit-v1/  (git-ignored; nothing is uploaded)
    python scripts/export_dataset.py --trim 30             # drop the first / last 30 s of every walk's GPS (default)

What goes in, per walk: 50 Hz inertial signals (user acceleration, gravity, gyroscope, phone frame), GPS trimmed at
both ends and rounded to ~1 m, and the walker's button-press labels (kerb up / down, stairs up / down) on the sensor
clock. Plus the path segments with their real conditions. What stays out: phone names and exact recording times
(only the date is kept), raw Annotation clocks, and the start / end of each GPS track near where people live.
Publish only after every contributor has agreed (see CONSENT in the README this writes).
"""
import argparse, csv, json, os, shutil, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from pathsense.io import load_recording, walk_bounds, align_annotations, labelled_presses
from pathsense.osm import load_graph
from pathsense import conditions as CND
from process_walks import find_walks, load_options

FS = 50.0
README = """# PathSense VIT pedestrian accessibility traces, v1

Smartphone inertial + GPS traces of walks on the VIT Vellore campus (Tamil Nadu, India) and the adjoining
Katpadi-Thiruvalam Road, with walker-marked kerbs and stairs, and the campus path network with surface conditions.
Collected {date_range} by {n_contributors} contributors on {n_phones} phone models ({platforms}).

## Contents
- `walks.csv`: one row per walk: walk id, contributor id, phone model, platform, carry (hand / pocket / trolley),
  date, duration, GPS length, whether the button presses could be synchronised to the sensor clock.
- `walks/<walk>/imu.csv`: 50 Hz. `t` (s since the walk's start), `acc_x..z` user acceleration (m/s^2, gravity
  removed), `grav_x..z` gravity (m/s^2), `gyro_x..z` (rad/s), all in the phone's frame (x right, y top edge, z out of
  the screen). Recorded with Sensor Logger, "standardise units & frames" on.
- `walks/<walk>/gps.csv`: `t`, `lat`, `lon` (rounded to 5 decimals, about 1 m), `h_acc_m`. The first and last
  {trim} s of each walk are removed.
- `walks/<walk>/labels.csv`: `t`, `label` (kerb_up, kerb_down, stairs_up, stairs_down), `lat`, `lon`. Only for walks
  whose presses could be synchronised. Protocol: press 1 just after stepping up a kerb, 2 after stepping down;
  double-press 1 / 2 just before stairs up / down. Known exception: contributor c03 pressed before the kerb (the jolt
  follows the press by about 1 s); see `press_lag_s` in walks.csv.
- `segments.geojson` / `segments.csv`: the {n_segments} campus path segments (from OpenStreetMap) with surface,
  where that surface comes from (team survey, satellite imagery check, OSM tag, road type), kerbs marked on walks,
  possible kerbs (jolts with no press), steps.

## Licence
Traces and labels: CC BY 4.0. Segment geometry and OSM-derived attributes: (c) OpenStreetMap contributors, ODbL 1.0
(share-alike); keep this notice with the segments.

## CONSENT (fill in before publishing)
Every contributor below must agree to their traces being published in this anonymised form.
{consent}

## Known limitations
- {n_walks} walks, one campus; walking with a hand-held phone (no wheelchair or trolley traces yet).
- GPS horizontal accuracy is typically 5-15 m; segment assignment can be off near junctions.
- Walks marked synced = no have no usable labels (their start/end presses were missing).
"""


def main(a):
    out = os.path.join(ROOT, "dataset", a.name)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "walks"))
    real = json.load(open(os.path.join(ROOT, "outputs", "real_walks.json")))
    by_id = {w["id"]: w for w in real["walks"]}
    options = load_options(os.path.join(ROOT, "data", "raw", "walk_options.json"))
    rows = []
    walks = sorted((w for w in real["walks"]), key=lambda w: w["date"])
    for k, w in enumerate(walks, 1):
        path = next(p for p in find_walks(os.path.join(ROOT, "data", "raw")) if os.path.basename(p).removesuffix(".zip") == w["id"])
        r = load_recording(path)
        off = align_annotations(r)[0]
        a0, b0 = walk_bounds(r, off)
        wid = f"w{k:02d}"; d = os.path.join(out, "walks", wid); os.makedirs(d)
        t = np.arange(a0, b0, 1 / FS)
        cols = [np.interp(t, r.t, x[:, i]) for x in (r.acc, r.grav, r.gyro) for i in range(3)]
        with open(os.path.join(d, "imu.csv"), "w", newline="") as f:
            cw = csv.writer(f); cw.writerow(["t", "acc_x", "acc_y", "acc_z", "grav_x", "grav_y", "grav_z", "gyro_x", "gyro_y", "gyro_z"])
            for i in range(len(t)):
                cw.writerow([f"{t[i] - a0:.3f}"] + [f"{c[i]:.5f}" for c in cols])
        g = r.gps[(r.gps[:, 0] >= a0 + a.trim) & (r.gps[:, 0] <= b0 - a.trim)]
        with open(os.path.join(d, "gps.csv"), "w", newline="") as f:
            cw = csv.writer(f); cw.writerow(["t", "lat", "lon", "h_acc_m"])
            for row in g:
                cw.writerow([f"{row[0] - a0:.2f}", f"{row[1]:.5f}", f"{row[2]:.5f}", f"{row[3]:.1f}"])
        n_lab = 0
        if off is not None:
            opt = options.get(r.name, {})
            labs = labelled_presses(r, off, pairs=opt.get("pairs", "stairs"), stairs_at=tuple(opt.get("stairs_at", ())))
            keep = [(tt, f"{kind}_{lab}") for tt, lab, kind in labs if a0 + a.trim <= tt <= b0 - a.trim]
            with open(os.path.join(d, "labels.csv"), "w", newline="") as f:
                cw = csv.writer(f); cw.writerow(["t", "label", "lat", "lon"])
                for tt, lab in keep:
                    j = int(np.argmin(np.abs(r.gps[:, 0] - tt)))
                    cw.writerow([f"{tt - a0:.2f}", lab, f"{r.gps[j, 1]:.5f}", f"{r.gps[j, 2]:.5f}"])
            n_lab = len(keep)
        rows.append(dict(walk_id=wid, title=w["title"], contributor=w["contributor"], phone_model=w["device"],
                         platform=w["platform"], carry=w["carry"], date=w["date"][:10], duration_s=round(b0 - a0),
                         length_m=w["length_m"], synced=w["sync_ok"], labels=n_lab, gps_median_acc_m=w["gps_median_acc_m"]))
        print(wid, w["title"], n_lab, "labels", flush=True)
    lag = {}
    rr = os.path.join(ROOT, "outputs", "real_results.json")
    if os.path.exists(rr):
        lag = json.load(open(rr)).get("F3_F4", {}).get("press_lags", {})
    for row in rows:
        row["press_lag_s"] = lag.get(row["title"], {}).get("lag_s", "")
    with open(os.path.join(out, "walks.csv"), "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=list(rows[0])); cw.writeheader(); cw.writerows(rows)
    # segments with conditions
    G = load_graph(os.path.join(ROOT, "data", "osm_vit.json"))
    for k, (*_, d) in enumerate(G.edges(data=True)):
        d["idx"] = k
    cond, _, _ = CND.build(G, real, CND.load(os.path.join(ROOT, "data", "path_conditions.json"), {}),
                           CND.load(os.path.join(ROOT, "data", "satellite_review.json"), {}))
    feats, seg_rows = [], []
    for *_, d in G.edges(data=True):
        c = cond[d["idx"]]
        props = dict(segment_id=d["idx"], name=d.get("name") or "", highway=d.get("osm_highway"), length_m=round(d["length"], 1),
                     surface=c["surface"], surface_source=c["surface_src"] or "none", kerbs=c["kerbs"],
                     kerb_walks=len(c["kerb_walks"]), possible_kerbs=c["maybe"], steps=c["steps"],
                     slope_deg=c["grad"] if c["grad"] is not None else "", team_note=c["note"])
        seg_rows.append(props)
        feats.append(dict(type="Feature", properties=props,
                          geometry=dict(type="LineString", coordinates=[[round(lo, 6), round(la, 6)] for la, lo in d["geom"]])))
    json.dump(dict(type="FeatureCollection", features=feats), open(os.path.join(out, "segments.geojson"), "w"))
    with open(os.path.join(out, "segments.csv"), "w", newline="", encoding="utf-8") as f:
        cw = csv.DictWriter(f, fieldnames=list(seg_rows[0])); cw.writeheader(); cw.writerows(seg_rows)
    contributors = sorted({r["contributor"] for r in rows})
    open(os.path.join(out, "README.md"), "w", encoding="utf-8").write(README.format(
        date_range=f"{rows[0]['date']} to {rows[-1]['date']}", n_contributors=len(contributors),
        n_phones=len({r['phone_model'] for r in rows}), platforms=", ".join(sorted({r['platform'] for r in rows})),
        trim=a.trim, n_segments=len(seg_rows), n_walks=len(rows),
        consent="\n".join(f"- [ ] {c} agrees (name/date kept by the team, not in this file)" for c in contributors)))
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(out) for f in fs)
    print(f"wrote {out}: {len(rows)} walks, {len(seg_rows)} segments, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="pathsense-vit-v1"); ap.add_argument("--trim", type=float, default=30.0)
    main(ap.parse_args())
