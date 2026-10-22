# PathSense

**Inferring pedestrian accessibility from crowdsourced smartphone inertial data** — VIT Vellore (SCOPE) group project.
Domain: Route Finding / Person with Disability.

Phone accelerometer + gyroscope traces → kerbs, gradient and roughness per road segment → probabilistic fusion of many
unreliable traces → graph propagation to segments nobody walked → accessibility-aware routing per mobility profile.

> **What is real:** the website uses only real path data (see "Path data" below): OpenStreetMap paths and tags, the
> recorded walks in `data/raw/` (12 walks, 3 phones, 8.1 km) with the kerbs walkers marked, a satellite imagery check,
> and what the team records on the site. Every metric in `outputs/metrics.json` still comes from the **simulator**
> (`pathsense/simulate.py`) and is for developing and testing the pipeline only. See `docs/HANDOFF.md`.

## Run the website
```bash
start_pathsense.bat                   # Windows: double-click it; opens http://127.0.0.1:5000
python app/server.py --lan            # same, and prints an address for phones on the same Wi-Fi
```
The first start builds the map and processes the walks (about a minute). The site has four pages:
**Plan a route** (search a place or click the map, pick who is travelling, compare with the shortest route, report a
problem on the map), **Walks & data** (every walk and every reported problem), **Contribute** (how to record a walk with
Sensor Logger, and upload its .zip) and **About**. Reports are saved in `data/reports.json`, uploads in `data/raw/uploads/`.
**Edit paths** (on the map) lets the team record what a path is really like: surface, kerbs or ramps, steps, slope,
narrow, a note and a photo. Edits go to `data/path_conditions.json`, photos to `data/photos/` (re-saved without the
location and phone details inside the file).

### Path data
Each segment's surface comes from the strongest source available (`pathsense/conditions.py`): team edit >
satellite imagery check (`data/satellite_review.json`) > OpenStreetMap surface tag > OSM road type (campus roads are
taken as paved, tracks as unpaved) > "not checked yet". Kerbs are the ones walkers marked with the Annotation button
(press positions grouped within 8 m, put on the nearest footpath); jolts the phone felt on 2+ walks with no press
nearby are only "possible kerbs". Unpaved paths cost wheelchairs 7x their length, so long dirt shortcuts are avoided.
Phones on the Wi-Fi can use everything except "Use my location", which browsers only allow over https: start with
`python app/server.py --lan --https` (self-signed certificate; each phone asks once to trust it), then the phone can
also install PathSense as an app from the browser menu.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/fetch_osm.py           # download the REAL campus footpaths from OpenStreetMap -> data/osm_vit.json (needs internet)
python scripts/run_pipeline.py --osm data/osm_vit.json   # real geometry + simulated surface data -> outputs/
# (without --osm the pipeline uses a fake grid: fine for tests, but paths cut through buildings/lake)
python scripts/process_walks.py      # REAL Sensor Logger walks in data/raw/ -> "Real walks" layer (optional)
python app/server.py                  # the website at http://127.0.0.1:5000 (see "Run the website")
python scripts/real_experiments.py    # F1-F5, F7 and the Western comparison on the real walks (~15 min)
python -m pytest -q tests             # 30 tests
```
Deep-learning part (functionalities 3 & 4): `pip install -r requirements-ml.txt && python scripts/train_ssl_dann.py`
(results in `outputs/ssl_dann.json`).

## Pipeline
![architecture](docs/architecture.png)

| # | Functionality | Code | Status on REAL data (`scripts/real_experiments.py`, `docs/RESULTS.md`) |
|---|---|---|---|
| 1 | Inertial surface & obstacle inference | `pathsense/io.py`, `pathsense/features.py` | kerbs: jolt detector + a learned detector (better); roughness and slope from a hand-held phone did not separate surfaces / slopes; steps not yet testable |
| 2 | EM / Dawid–Skene trace fusion | `pathsense/fusion.py` | runs on real walks, optional trace-quality weights; ties majority vote with 1-3 walks per segment |
| 3 | Contrastive (SimCLR) pretraining | `pathsense/ml/models.py`, `scripts/real_experiments.py` | pretrained on all 12 walks; small gain with few labels |
| 4 | DANN device invariance | same | iPhone 15 <-> iPhone 16 cross-phone test; no gain from DANN yet |
| 5 | GCN propagation to unobserved segments | `scripts/real_experiments.py` (torch GCN), `pathsense/graph_prop.py` | surface type of hidden segments: GCN best; kerbs are local, neighbours don't help |
| 6 | Profile-conditioned A* routing | `pathsense/routing.py`, `pathsense/conditions.py` | works on the real map (website); beats a Western-centric profile |
| 7 | Pre-processing re-derived from pilot data | `scripts/real_experiments.py` (F7), `pathsense/io.py` | detector settings and window length chosen from the walks; **walking only, trolley runs still to record** |

**Dataset:** the anonymised traces are published in [`dataset/pathsense-vit-v1/`](dataset/pathsense-vit-v1/) (12 walks,
120 marked kerb crossings, 525 segments with conditions; CC BY 4.0, segments ODbL) and as a zip under Releases.
Rebuild it with `python scripts/export_dataset.py --consented <date>`.
Also: `scripts/benchmark_rig.py`
+ `docs/BENCHMARK_RIG.md` (validation against a reference IMU / tape and level), the Western-centric comparison
(in `real_experiments.py`). The simulator results (`scripts/run_pipeline.py`, `scripts/train_ssl_dann.py`) remain for
development only.

Website: `app/` (Flask + Leaflet, no build step). Handoff/plan: `docs/HANDOFF.md`, `docs/ROADMAP.md`. Paper draft: `docs/paper/`.

## Layout
```
pathsense/        library (osm, simulate, features, fusion, graph_prop, routing, ml/)
scripts/          process_walks.py, real_experiments.py, export_dataset.py, benchmark_rig.py, fetch_osm.py,
                  run_pipeline.py (simulator), train_ssl_dann.py (simulator), create_issues.sh
app/              Flask server (server.py) + front end (static/index.html, app.css, app.js)
tests/            pytest
docs/             RESULTS.md, BENCHMARK_RIG.md, HANDOFF.md, ROADMAP.md, DATA_COLLECTION.md, paper/
```
