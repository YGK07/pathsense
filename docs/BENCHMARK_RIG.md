# Benchmarking rig (validation of the phone's surface profiling)

Required by the accepted proposal: validate what the phone infers against a high-precision reference on the pushed
trolley. Analysis: `scripts/benchmark_rig.py` (tested on synthetic data in `tests/test_benchmark_rig.py`).

## Equipment (cheapest first)
| Reference | Cost | What it gives |
|---|---|---|
| Tape measure + a level app (iPhone Measure > Level, or any Android level app) | free | kerb heights (cm) and slopes (deg) at fixed points: the ground truth for every run |
| Second phone in a **rigid** mount on the trolley frame (bolted / cable-tied, not in a hand) | free | a reference IMU without hand motion |
| IMU board (BNO055, or MPU-6050) + Arduino / ESP32 logging to SD at 100 Hz | about Rs 600-1500 | calibrated accelerometer + gyroscope (columns `t,ax,ay,az,gx,gy,gz`) |
| Phone with LiDAR (iPhone Pro, with a 3D-scan app) | borrow | scans of kerbs and ramps; read heights off the scan |

## Trolley runs
1. Mount the reference rigidly near the trolley's axle. The phone under test goes where a user would carry it
   (hand, pocket, or a phone holder on the handle). Start both, then tap the trolley frame firmly three times:
   the script lines the two recordings up on those taps (cross-correlation), so clocks don't need to match.
2. Push along the measured route at walking pace. Press the Annotation buttons as on walks (1 after a kerb up,
   2 after a kerb down). Upload the phone's zip on the site with **Phone carried = On a trolley**.
3. Repeat each route 3 times in each direction, and with 2-3 different phones under test if possible.

## Measurement sheet
Fill `data/ground_truth/rig_measurements.csv` (template provided) while walking the route once before the runs:
`id,type,lat,lon,value,notes` with `type` = `kerb` (value = height in cm) or `slope` (value = degrees, positive
uphill in the direction of the run). Take the position from the site's map (right-click / long-press a spot,
or "Report a problem" shows coordinates) and add a photo in Edit paths.

## Run the analysis
    python scripts/benchmark_rig.py --phone <phone zip> --reference <reference zip or imu.csv> \
        --measurements data/ground_truth/rig_measurements.csv --out outputs/rig_run1.json
It reports the clock offset, the waveform correlation of vertical acceleration, roughness agreement in 5 s windows,
and, for every measured kerb and slope, the phone's and the reference's response with Spearman correlations against
the tape / level values. Put those numbers in the paper's validation table.
