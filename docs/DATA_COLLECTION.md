# Data collection protocol (pilot, task P1)

Status: **no real data has been collected yet.** Every trace used so far comes from `pathsense/simulate.py`.
This document is the step-by-step procedure for collecting the first real dataset and adding it to the project.

---

## 0. Ethics and privacy (do this first)
- Recording yourselves (the three team members) with your own phones is fine.
- **Anyone else**, and especially wheelchair users, needs written informed consent and approval from the institution or
  campus accessibility office **before** any recording. No wheelchair/trolley data is planned; all pilot data is walking data by team members.
- GPS traces are personal data. Use contributor IDs (`c01`, `c02`, `c03`), never names. Do not record routes that start or
  end at someone's room. Raw recordings go in `data/raw/` (git-ignored) and are shared through the team drive, not GitHub.

## 1. Equipment
| Item | Why |
|---|---|
| ≥3 **different** Android phone models (one per member is fine) | DANN (functionality 4) needs device diversity; leave-one-device-out evaluation needs ≥3 |
| **Sensor Logger** app (free, by Tsz Hei Choi), same version on every phone | Exports timestamped Accelerometer / Gravity / Gyroscope / Location CSVs |
| (optional) a second phone per walker | Record hand-held and pocket at the same time |
| Measuring tape or ruler | Kerb/step heights |
| A 1 m straight board/ruler + a second phone with an inclinometer app (or a spirit level) | Ground-truth gradient |
| Phone camera | Timestamped photos of each feature for the ground-truth sheet |

## 2. App settings (check on every phone before the first run)
1. In Sensor Logger enable: **Accelerometer, Gravity, Gyroscope, Location, Annotation** (and **Total Acceleration** if listed).
   *Why:* on Android, `Accelerometer.csv` has gravity **removed**. PathSense estimates slope from the gravity direction, so we need
   `Gravity.csv` (or `TotalAcceleration.csv`) as well. Without it the gradient feature cannot be computed.
2. Settings → Sensor Configuration → turn **on "Standardise Units & Frames"** so iOS/Android and units are consistent.
3. Set the sampling rate to the **highest available (≥100 Hz)**. We will downsample in code; we cannot upsample lost detail.
4. Export format: **Zipped CSV**.
5. After a 10-second test recording, open `Accelerometer.csv` and check the real rate:
   `rows / (last seconds_elapsed − first seconds_elapsed)`. Write it down per phone. If a phone gives < 50 Hz, note it; it can
   still be used but flag it in the run log.

Units (from the app's documentation): acceleration m/s², gyroscope rad/s, `time` = UNIX epoch in **nanoseconds** (UTC),
`seconds_elapsed` = seconds since Start. Location: lat/lon degrees, `horizontalAccuracy` metres.

## 3. Carrying the phone (most important factor for clean data)
No trolley is available, so all data is **walking** data. The paper must say so (walking as a proxy for wheeled motion).
- **Hand-held (primary):** hold the phone flat in front of you like reading a map, **screen up, top edge pointing forward**,
  elbow bent and tucked against your side, other hand free. Same grip for the whole run, do not switch hands or check messages.
  In the Android phone frame this means **y = forward, x = right, z = up**; the loader (task P2) remaps to the pipeline's
  frame (x forward, y left, z up). This keeps the gravity direction stable, which the slope estimate needs.
- **Pocket (secondary, recorded at the same time on a second phone):** front trouser pocket, screen facing the leg, top edge up.
  The thigh swings, so slope cannot be read from it directly, but the step pattern (stumbles, step-ups at kerbs) is very clear.
- Best practice: **one person carries two phones per run, one in hand and one in pocket.** Same walk, two devices, two mountings
  (free data for DANN and the mounting comparison). Never change mounting during a run.
- Walking changes what the signals mean: a kerb shows up as a **step-up/step-down irregularity in the gait**, not a wheel jolt,
  and roughness shows up as **less regular steps**. The detectors will be re-tuned for this (tasks F1, F7).

## 4. Choosing routes
Pick **6–10 routes on campus**, each 150–500 m, together covering:
- at least 2 **flat smooth** stretches (negative control: should be "accessible"),
- at least 3 **kerbs or steps** (with and without a ramp/cut alternative nearby),
- at least 1 **ramp or slope** you can measure (ideally one 4–7° and one > 7°),
- at least 1 **rough/unpaved/broken** stretch,
- some routes that **overlap**, so the same segment gets traces from different routes (fusion needs this).

Prefer routes on paths that appear in OpenStreetMap (`python scripts/fetch_osm.py`, then check the map), because traces are
later snapped to those segments (task P3).

**Scope: one compact study zone, not the whole campus.** Pick a zone of roughly 500 m × 500 m (e.g. men's hostel blocks →
food court / academic blocks). Inside it, 6–8 overlapping routes of 200–400 m, each with a fixed start and end landmark.
The rest of the campus is covered by the OSM graph and the GCN (F5) predicts it; it just isn't evaluated.
Also survey (ground truth only, **no runs**) 8–10 extra segments inside the zone: these are the "never walked" segments the
GCN is tested on.

**Marking routes:** before collection day, draw each route on Google My Maps (or a screenshot of the PathSense demo map),
number it `r01…r08`, and write its start/end landmark. On the day, every run starts and ends at exactly those landmarks.
The GPS in the recording records the actual path; the route ID goes in the filename and `runs_log.csv`. Nothing needs to be
marked during the walk itself.

## 5. Ground truth (separate survey walk, once per route, NOT during the recorded runs)
Do the survey on a different walk from the recordings so the recorded walks stay natural. Phone camera location tagging on:
every photo then carries its own GPS coordinates and time, which is enough to place the feature on the map.
Walk the route slowly and fill **`data/ground_truth/features.csv`** (one row per point obstacle) and
**`data/ground_truth/stretches.csv`** (one row per stretch of uniform surface). Templates are in `data/templates/`.
- **Location:** stand on the feature, drop a pin / read lat,lon from Google Maps or the Sensor Logger Location screen. Also write a landmark ("kerb at SJT gate 2").
- **Kerb/step height:** tape from lower to upper surface, in cm. A ramped kerb cut with ≤ 1 cm lip counts as *no kerb*.
- **Gradient:** lay the 1 m board along the direction of travel at 3 spots on the slope, read the inclinometer, record the
  **mean in degrees** and the **uphill direction** (compass bearing or "towards landmark X"). Rise/run also works:
  `deg = atan(rise/run)`.
- **Roughness (0–1)**, use this scale consistently, and take a photo for every stretch:
  | score | surface |
  |---|---|
  | 0.0 | polished tile / smooth new tarmac |
  | 0.2 | normal tarmac or concrete, small joints |
  | 0.4 | paver blocks, worn concrete, frequent joints |
  | 0.6 | broken patches, gravel over tarmac, tree roots |
  | 0.8 | unpaved / loose gravel / mud |
  | 1.0 | not passable with a wheelchair |
  Two members score each stretch independently; record both. Disagreement > 0.2 → re-check together.
- Class is **not** written by hand; it is computed in code from the thresholds in `pathsense/config.py` so the rule is the same everywhere.

## 6. Running a recording (each run)
1. Phone mounted, GPS on, wait outdoors until Location shows `horizontalAccuracy` **≤ 10 m** (usually 30–60 s).
2. **Calibration:** stand still on flat ground holding the phone in the walking grip, press Start, stay perfectly still for **10 s**
   (gives mount tilt + sensor bias).
3. **Sync mark:** three hard taps on the back of the phone (both phones if carrying two), a second apart. Add an
   Annotation "start".
4. Walk the route at a **steady normal pace (~1 m/s)**. Do not stop to look at the phone. If you must stop, stand still and
   add an Annotation ("stop – traffic").
5. At the end: three sync taps again, Annotation "end", 10 s still, Stop.
6. Immediately fill one row in **`data/runs_log.csv`** (template in `data/templates/`), including anything odd.

Repeat each route **≥3 times**, ideally by **different people with different phones**, and in **both directions** for slopes.

Target for the pilot (definition of done for P1): ≥3 phone models × ≥6 routes × ≥3 runs (≥54 runs), with ground truth for all routes.

## 7. File naming and where data goes
```
data/
  raw/                         # git-ignored. Unzipped Sensor Logger exports, one folder per run:
    c01_phoneA_hand_r03_run2_fwd/
        Accelerometer.csv  Gravity.csv  Gyroscope.csv  Location.csv  Annotation.csv  Metadata.csv ...
  runs_log.csv                 # git-ignored by default (data/*.csv); share via drive, or decide to commit (see below)
  ground_truth/
    features.csv               # committed: kerbs/steps (no personal data)
    stretches.csv              # committed: gradient + roughness per stretch
    photos/                    # optional, share via drive if large
  templates/                   # committed: blank templates
```
Name pattern: `<contributor>_<phone>_<hand|pocket>_r<route>_run<n>_<fwd|rev>`. Keep a `phones` list (phoneA = "Redmi Note 12", ...) in `runs_log.csv`.

Raw traces stay **out of Git** (privacy + size). Upload `data/raw/` as a zip to the team drive after each session.
Ground-truth CSVs contain no personal data and should be committed so everyone uses the same labels.

## 8. Quality checklist after each session
- [ ] Each run folder has Accelerometer, Gravity (or TotalAcceleration), Gyroscope and Location CSVs.
- [ ] Effective accelerometer rate ≥ 50 Hz; GPS has no gap > 10 s in the middle of the route.
- [ ] The three sync spikes are visible at start and end of `Accelerometer.csv` z axis.
- [ ] Median `horizontalAccuracy` ≤ 10 m.
- [ ] `runs_log.csv` row filled; ground truth exists for the route.

## 9. What happens next in code
- **P2** `pathsense/io.py` reads a run folder → resamples accel+gravity+gyro to `config.FS` → remaps axes → returns a `Trace`.
- **P3** map matching snaps GPS to OSM segments → fills `Trace.seg`.
- **P4** `pathsense/groundtruth.py` converts `features.csv`/`stretches.csv` into per-segment `rough/grad/kerbs/cls`.
After that the existing pipeline runs unchanged on real data.
