# ROADMAP — every task with ID, owner slot, definition of done

Legend: ☐ todo · ◐ partly done · ✅ done. Put a name next to a task when you take it.
Priority: **P0** critical path, P1 important, P2 nice to have.

## Phase 0 — Setup
- ☐ S1 (P0) Create GitHub repo, push, add teammates as collaborators, enable branch protection on `main`, run `bash scripts/create_issues.sh`. DoD: CI green.
- ☐ S2 (P0) Each member runs README quick start and the demo locally.

## Phase 1 — Data (nothing real exists until this is done)
- ☐ P1 (P0) Pilot collection per `docs/DATA_COLLECTION.md`. DoD: ≥3 phone models, ≥6 routes, ≥3 runs each, ground-truth sheet.
- ☐ P2 (P0) `pathsense/io.py`: load Sensor-Logger-style CSVs → resample to fixed FS, align accel/gyro/GPS, return `Trace`-compatible objects. DoD: test with a small fixture CSV.
- ☐ P3 (P0) Map matching: snap GPS to road-graph edges, fill `Trace.seg`. Start with nearest-edge + hysteresis; try Valhalla/OSRM/`leuvenmapmatching` if noisy. DoD: ≥90% of samples on correct edge on a hand-checked route.
- ☐ P4 (P1) Build `data/segments.csv` ground truth from the field notes; loader `pathsense/groundtruth.py`.
- ☐ P5 (P2) Dataset release: datasheet, anonymisation check, licence, Zenodo/Kaggle upload. Required by the accepted methodology.

## Phase 2 — Core methods on real data
- ◐ F1 (P0) Calibrate on real data: kerb detector thresholds, gradient bias correction (mount tilt), roughness→0–1 mapping. DoD: precision/recall/F1 vs ground truth reported per route.
- ☐ F7 (P0) Preprocessing retuned for wheeled motion: choose FS, window length, filter cut-offs from pilot spectra. Ablation: gait-style windows (e.g. 2 s, 50 % overlap, 0.3–3 Hz gait band) vs retuned. DoD: table + spectrum figure.
- ◐ F2 (P0) Fusion. Currently ≈ majority vote on simulated data. Try: (a) weight by trace quality (GPS accuracy, speed consistency) as priors in the E-step; (b) run DS on discretised *features* (kerb yes/no, grade bins, rough bins) rather than a class predicted by one shared classifier; (c) contributor×device confusion matrices; (d) compare with majority/weighted voting at different numbers of traces per segment. DoD: honest comparison on real data with CIs; if it doesn't help, report that.
- ◐ F5 (P1) Replace numpy GCN with PyTorch GCN/GAT (`pathsense/ml/gnn_torch.py`), keep the numpy one as a baseline. DoD: held-out-segment MAE vs mean/neighbour-average baselines.
- ◐ D3 (P0) [loader + fetch script + demo done; run `scripts/fetch_osm.py` and check campus coverage, hand-fix in JOSM] Real road graph for VIT + Katpadi: OSM via `osmnx`; hand-correct footpaths in JOSM if needed. Decide F5 vs multi-task swap by this task's outcome. DoD: graph with ≥100 segments and `length`, node coordinates.
- ☐ F6 (P1) Tune `config.PROFILES` weights with real users or accessibility guidelines; optional hard constraints (e.g. never cross a kerb for manual wheelchair unless no alternative). DoD: documented rationale per weight.

## Phase 3 — Deep learning (F3, F4)
- ☐ M1 (P0) Install torch, run `python scripts/train_ssl_dann.py`, fix bugs (it has never been executed). DoD: runs end to end on simulated data.
- ☐ M2 (P0) Ablation on real windows: no-SSL vs SSL at 10/25/50/100 % labels; no-DANN vs DANN leave-one-device-out. DoD: Table 1 rows filled with real numbers + plots.
- ☐ M3 (P1) Plug encoder embeddings into the observation classifier (replace RandomForest on 3 handcrafted features); compare.
- ☐ M4 (P2, fallback for F5) Multi-task head: mobility state (walk/wheelchair/trolley) + surface attributes; feeds automatic profile detection.

## Phase 4 — App & demo
- ◐ A0 (P0) Web demo exists (`app/`). Make it load the real map (`outputs/map.json` from the real pipeline); remove the "synthetic" banner only when true.
- ☐ A1 (P1) Android collector app (Kotlin or Flutter): record IMU+GPS, upload CSV. Alternative: continue using a ready-made logger app and say so in the paper.
- ☐ A2 (P1) Demo polish: profile comparison view (same A→B, 4 profiles, 4 routes), "report a problem" button, mobile layout, screenshots for the report.
- ☐ A3 (P2) Deploy (Render/Fly.io/Railway) so the teacher can open a link.
- ☐ A4 (P2) Live "upload a trace → map updates" flow in the demo.

## Phase 5 — Paper & report (docs/paper/)
- ☐ W1 (P0) Fill Table 2 and Section 6 with real numbers only. Replace the author-block email and Authors' contributions placeholders.
- ☐ W2 (P0) Figures: spectra, kerb detection example, fusion vs voting curve, GCN held-out error, route comparison screenshot.
- ☐ W3 (P1) Comparison of Western-centric vs Indian mobility profiles on unpaved/irregular surfaces (accepted methodology item).
- ☐ W4 (P1) Re-verify every reference (authors, venue, year) before submission.
- ☐ W5 (P1) Convert to the official Springer Nature LaTeX template (sn-jnl) on Overleaf if the teacher requires the .tex source.
