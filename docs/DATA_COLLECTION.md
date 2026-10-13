# Data collection protocol (pilot)

**Ethics first.** Collecting on yourselves with your own phones is fine. Anyone else (especially wheelchair users) needs informed
consent and the relevant institutional approval *before* collection. Do not record or store personal identifiers; use contributor
IDs (c01, c02...). Raw files go in `data/raw/` (git-ignored); share them via a team drive, not Git.

## What to record
- Android phones (≥3 different models, needed for DANN) running a sensor-logging app that exports CSV with timestamps for
  accelerometer, gyroscope and GPS. A free option is the "Sensor Logger" app; verify it exports raw accelerometer/gyroscope/location at ≥50 Hz on your phones.
- Mounting: (a) pocket, (b) mounted on a manually pushed trolley/wheelchair frame (phone fixed, screen up, top edge = forward).
  Record the mounting in the filename: `c01_phoneA_trolley_route3_run2.csv`.
- Routes: pick 6–10 campus/Katpadi routes with known kerbs, a ramp, rough/unpaved stretch and flat smooth path. Walk/push each route ≥3 times, at normal speed, ideally by different people/devices.
- **Ground truth:** for each route, note (on paper or a sheet) every kerb/step (with a landmark), slope (measure with a phone inclinometer or a spirit level + ruler: rise/run), and a 0–1 roughness rating (0 smooth tile … 1 broken/unpaved). Take a timestamped photo or voice note at each feature. Later: calibrated IMU/LiDAR rig if available.
- Sync: start each run with 3 hard taps/lifts of the trolley to create an alignment spike.

## What to deliver to the code
A folder `data/raw/` with CSVs and `data/segments.csv` (`segment_id, node_u, node_v, kerbs, gradient_deg, roughness, class`). Then see tasks P2/P3.
