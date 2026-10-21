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
Phones on the Wi-Fi can use everything except "Use my location", which browsers only allow over https.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/fetch_osm.py           # download the REAL campus footpaths from OpenStreetMap -> data/osm_vit.json (needs internet)
python scripts/run_pipeline.py --osm data/osm_vit.json   # real geometry + simulated surface data -> outputs/
# (without --osm the pipeline uses a fake grid: fine for tests, but paths cut through buildings/lake)
python scripts/process_walks.py      # REAL Sensor Logger walks in data/raw/ -> "Real walks" layer (optional)
python app/server.py                  # the website at http://127.0.0.1:5000 (see "Run the website")
python -m pytest -q tests             # 28 tests
```
Deep-learning part (functionalities 3 & 4): `pip install -r requirements-ml.txt && python scripts/train_ssl_dann.py`
(results in `outputs/ssl_dann.json`).

## Pipeline
![architecture](docs/architecture.png)

| # | Functionality | Code | Status |
|---|---|---|---|
| 1 | Inertial surface & obstacle inference | `pathsense/features.py` | works on simulated data, tested |
| 2 | EM / Dawid–Skene trace fusion | `pathsense/fusion.py` | works, tested; **not yet better than majority vote on sim** (open task F2) |
| 3 | Contrastive (SimCLR) pretraining | `pathsense/ml/models.py`, `scripts/train_ssl_dann.py` | trained on the recorded walks |
| 4 | DANN device invariance | same | trained, leave-one-phone-out comparison |
| 5 | GCN propagation to unobserved segments | `pathsense/graph_prop.py` | numpy GCN works, tested; torch/GAT version TODO |
| 6 | Profile-conditioned A* routing | `pathsense/routing.py` | works, tested |
| 7 | Wheeled-motion preprocessing | `pathsense/features.py` (+ `config.py`) | provisional params; needs pilot data |

Website: `app/` (Flask + Leaflet, no build step). Handoff/plan: `docs/HANDOFF.md`, `docs/ROADMAP.md`. Paper draft: `docs/paper/`.

## Layout
```
pathsense/        library (osm, simulate, features, fusion, graph_prop, routing, ml/)
scripts/          fetch_osm.py, run_pipeline.py, train_ssl_dann.py, create_issues.sh
app/              Flask server (server.py) + front end (static/index.html, app.css, app.js)
tests/            pytest
docs/             HANDOFF.md, ROADMAP.md, DATA_COLLECTION.md, paper/
```
