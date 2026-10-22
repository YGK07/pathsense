# PathSense VIT pedestrian accessibility traces, v1

Smartphone inertial + GPS traces of walks on the VIT Vellore campus (Tamil Nadu, India) and the adjoining
Katpadi-Thiruvalam Road, with walker-marked kerbs and stairs, and the campus path network with surface conditions.
Collected 2026-10-15 to 2026-10-21 by 3 contributors on 3 phone models (android, ios).

## Contents
- `walks.csv`: one row per walk: walk id, contributor id, phone model, platform, carry (hand / pocket / trolley),
  date, duration, GPS length, whether the button presses could be synchronised to the sensor clock.
- `walks/<walk>/imu.csv`: 50 Hz. `t` (s since the walk's start), `acc_x..z` user acceleration (m/s^2, gravity
  removed), `grav_x..z` gravity (m/s^2), `gyro_x..z` (rad/s), all in the phone's frame (x right, y top edge, z out of
  the screen). Recorded with Sensor Logger, "standardise units & frames" on.
- `walks/<walk>/gps.csv`: `t`, `lat`, `lon` (rounded to 5 decimals, about 1 m), `h_acc_m`. The first and last
  30.0 s of each walk are removed.
- `walks/<walk>/labels.csv`: `t`, `label` (kerb_up, kerb_down, stairs_up, stairs_down), `lat`, `lon`. Only for walks
  whose presses could be synchronised. Protocol: press 1 just after stepping up a kerb, 2 after stepping down;
  double-press 1 / 2 just before stairs up / down. Known exception: contributor c03 pressed before the kerb (the jolt
  follows the press by about 1 s); see `press_lag_s` in walks.csv.
- `segments.geojson` / `segments.csv`: the 525 campus path segments (from OpenStreetMap) with surface,
  where that surface comes from (team survey, satellite imagery check, OSM tag, road type), kerbs marked on walks,
  possible kerbs (jolts with no press), steps.

## Licence
Traces and labels: CC BY 4.0. Segment geometry and OSM-derived attributes: (c) OpenStreetMap contributors, ODbL 1.0
(share-alike); keep this notice with the segments.

## Consent
Every contributor must agree to their traces being published in this anonymised form.
- [x] c01 agreed (2026-10-22)
- [x] c02 agreed (2026-10-22)
- [x] c03 agreed (2026-10-22)

## Known limitations
- 12 walks, one campus; walking with a hand-held phone (no wheelchair or trolley traces yet).
- GPS horizontal accuracy is typically 5-15 m; segment assignment can be off near junctions.
- Walks marked synced = no have no usable labels (their start/end presses were missing).
