# HANDOFF — read this if you are picking the project up (new teammate, or a fresh Claude/AI session)

## 1. What this project is (one paragraph)
PathSense turns smartphone accelerometer/gyroscope traces into an accessibility map of walking surfaces (kerbs, gradient,
roughness) and routes wheelchair users / people on crutches / low-vision users around bad segments. Core ideas:
(F1) infer surface from raw IMU, (F2) fuse many unreliable traces with Dawid–Skene EM, (F3) SimCLR pretraining on
unlabelled traces, (F4) DANN so any phone model works, (F5) GCN to predict never-walked segments, (F6) profile-conditioned
A* routing, (F7) preprocessing retuned for wheeled motion. Target: working demo + conference paper + released dataset.
Team: George C Mathew (coordinator), Yohan George, Siddhant Gaddi. School: SCOPE, VIT Vellore. Study area: VIT campus + Katpadi roads.

## 2. Ground rules that matter
- **All numbers right now are from a simulator.** Nothing in `outputs/metrics.json` is a research result. The paper's
  Results table (Table 2 in `docs/paper/PathSense_Springer_Format.docx`) is deliberately TBD. Fill it only with real,
  measured numbers. Fabricated results = academic misconduct.
- Wheelchair data needs consent + approval from the campus accessibility office / institution. The fallback is a
  manually pushed trolley (already written into the proposal's feasibility note).
- Functionality 5 needs a segment graph of Katpadi; OSM footpaths there are patchy. If the graph can't be built
  (task D3), swap F5 for the multi-task model (mobility state + surface attributes from one window; shares the encoder)
  and tell the teacher, because it changes the agreed list of 7.

## 3. Current state (verified)
| Item | State |
|---|---|
| Simulator, features, fusion, GCN, routing, pipeline, Flask+Leaflet demo | **built, 9 tests pass, pipeline runs end to end** |
| `scripts/train_ssl_dann.py`, `pathsense/ml/models.py` (PyTorch) | **written but never executed** (author's sandbox had no torch). Expect small bugs. Task M1 |
| Real data, map matching, Android app, real evaluation | **not started** |
| Paper | draft in Springer-style docx; Results TBD; author block emails + Authors' contributions TBD |

Simulated-data findings worth knowing (seed 0, `python scripts/run_pipeline.py`):
- Single-observation classifier ~0.88 accuracy; **Dawid–Skene fusion is only roughly equal to majority vote** (0.84 vs 0.83–0.89
  depending on seed) and contributor-reliability estimates correlate weakly with truth. So F2 is *not yet showing its value* — task F2
  below is the most important research task.
- GCN beats the "predict the mean" baseline on never-observed segments for roughness and especially gradient (MAE ~0.6° vs ~2.6°);
  class accuracy on unobserved segments is variable across seeds.
- Routing: for the wheelchair profile, PathSense routes cut blocked segments (~2.0 vs ~2.9 for shortest path on average) at modest length cost.

## 4. Suggested order of work (critical path first)
1. **Run tests + pipeline + demo locally** (5 min). Push repo to GitHub, run `bash scripts/create_issues.sh` to create the issue list.
2. **Pilot data collection (P1)** — everything real depends on it. Needs only phones + a trolley/walking. See `docs/DATA_COLLECTION.md`.
3. **Real data loader + map matching (P2, P3)** so traces become the same `Obs` objects the pipeline already uses.
4. **Run the deep models (M1) and the ablation table (M2).** Cheap, gives paper tables.
5. **Fix/justify fusion (F2).** 6. **Real graph (D3) → F5.** 7. **Demo polish + deploy (A-tasks).** 8. **Paper (W-tasks).**

## 5. How the code fits together
`simulate.py` → traces (acc, gyro, seg id, device, contributor) → `features.segment_observations` → per-obs class
(RandomForest on [kerbs, gradient, roughness-σ]; later: encoder embeddings) → `fusion.dawid_skene` (annotator = contributor)
→ fused per-segment class + continuous estimates → `graph_prop.propagate` (GCN on the line graph; predicts unobserved
segments) → edge attribute `est` on the graph → `routing.route` (A*, cost = d + w_g·grad + w_r·rough + w_k·kerbs; weights in
`config.PROFILES`) → `outputs/map.json` → `app/server.py`. To use real data, only the first two arrows change: produce `Trace`-like
objects (or `Obs` directly) from real CSVs with a real map-matched segment id; the rest is unchanged.

## 6. To continue with an AI assistant (paste this)
> I'm working on the PathSense repo (see README.md and docs/HANDOFF.md). Read docs/HANDOFF.md and docs/ROADMAP.md first. Today I want to
> do task <ID> from ROADMAP.md. Run `python -m pytest -q tests` before and after your changes, keep changes small, add tests for new
> pathsense/ code, do not invent results or citations, and label anything simulated as simulated.

## 7. Open decisions for the team
- Android app vs web app for the demo? (Recommended: keep the web demo as the evaluation demo; add an Android *collector* only if time allows — task A1.)
- Final sampling rate / window size (decide after P1, then update `config.py` and the paper).
- Whether F5 stays or is swapped (see §2).
