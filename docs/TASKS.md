# PathSense — ordered priority task list (as of 2026-10-13)

Ordered by what blocks what. IDs match `docs/ROADMAP.md`; new items are marked **NEW**.
Done today: repo cloned, 11/11 tests pass, simulated pipeline + demo run locally; data-collection protocol + templates written.

## Tier 1 — critical path (nothing real exists without these)
1. **D3 — Fetch the real campus graph.** Run `python scripts/fetch_osm.py`, check coverage on the demo map, note paths missing in OSM.
   Pick pilot routes on paths that exist in the graph. *(1 hour; needed before choosing routes.)*
2. **P1 — Pilot data collection** per `docs/DATA_COLLECTION.md`: 3 phones × ≥6 routes × ≥3 runs + ground truth sheets.
   Start with **one route, one phone, 2 runs** as a dry run and send it through steps 3–4 before collecting the rest.
3. **P2 — `pathsense/io.py` loader** for Sensor Logger folders: read Accelerometer+Gravity(or TotalAcceleration)+Gyroscope+Location,
   resample to `config.FS`, remap axes (phone y-forward → pipeline x-forward), trim calibration/sync sections, return `Trace`. Test with a tiny fixture.
4. **P3 — Map matching** GPS → OSM segment ids (`Trace.seg`). Nearest-edge + hysteresis first; ≥90 % correct on a hand-checked route.
5. **P4 — `pathsense/groundtruth.py`**: turn `data/ground_truth/features.csv` + `stretches.csv` into per-segment rough/grad/kerbs/cls.
6. **NEW — `run_pipeline.py --real`** mode that uses steps 3–5 instead of the simulator, and writes a `map.json` with `synthetic=False`.

6b. **NEW — Tell the teacher about the wheelchair/trolley change.** All data will be walking data (hand + pocket). F7
   "wheeled-motion preprocessing" cannot be done on real wheeled data; proposed replacement: "preprocessing retuned per carrying
   mode (hand vs pocket) instead of gait-style defaults", with wheeled motion kept as simulated only / future work. Get approval.

## Tier 2 — core methods on real data (gives the paper its results)
7. **F7 — Wheeled-motion preprocessing**: choose FS / window / filter cut-offs from pilot spectra; ablation vs gait-style settings.
8. **F1 — Calibrate inference**: kerb thresholds, **gradient sign + mount-tilt correction (NEW: `pitch_deg` currently takes `abs()`, losing uphill/downhill, and ignores the 10 s calibration)**, roughness → 0–1 mapping. Report P/R/F1 per route.
9. **M1 — Install torch, run `scripts/train_ssl_dann.py`, fix bugs** (never executed). Add `torch` handling so CI still passes without it.
   **NEW:** normalisation in `ml/data.py` uses statistics from the held-out device too (minor leakage) — fix while debugging.
10. **M2 — SSL / DANN ablation** on real windows (label fractions 10/25/50/100 %, leave-one-device-out). → paper Table 1.
11. **F2 — Make fusion useful or report honestly.** Currently DS ≈ majority vote and reliability correlation ≈ 0. Most promising:
    run DS on discretised *features* instead of one shared classifier's output; add trace-quality priors (GPS accuracy). Compare at 1/3/5/10 traces per segment with CIs.
12. **F5 — GCN on the real graph** vs mean / neighbour-average baselines (torch GCN/GAT optional). If the real graph is too patchy, decide the M4 swap **and tell the teacher**.

## Tier 3 — routing, demo, paper
13. **F6 — Justify profile weights** (cite accessibility guidelines, e.g. 1:12 ramp ≈ 4.8°); optional hard "no kerb" rule for manual wheelchair.
14. **NEW — Doc/consistency fixes**: paper §4.2 says gradient uses the gyroscope (code doesn't); HANDOFF says 9 tests (11); `config.py` says "task R2" (F6);
    demo crashes with no `outputs/map.json` (add a clear message); CI should also run the pipeline on the OSM fixture.
15. **A0 — Demo on the real map**; remove the "simulated" banner only when the surface data is real.
16. **W1/W2 — Paper results + figures with real numbers only** (Table 2, Section 6, spectra, kerb example, fusion curve, GCN error, route screenshots).
17. **W4 — Verify every reference.** **W1 — fill author e-mails + Authors' contributions.**
18. **A2 — Demo polish**: 4-profile comparison view, mobile layout, screenshots.

## Tier 4 — nice to have
19. **P5 — Dataset release** (datasheet, anonymisation check, licence, Zenodo).
20. **W3 — Western vs Indian profile comparison.**
21. **A1 — Android collector app** (or just state "Sensor Logger" in the paper).
22. **A3 — Deploy demo** (Render/Fly.io). **A4 — upload-a-trace flow.** **M3 — encoder embeddings in classifier.** **M4 — multi-task head.**
23. **S1 housekeeping**: branch protection on `main`, run `scripts/create_issues.sh` (needs GitHub CLI `gh`, not installed on this PC).

### Parallelisable right now (while waiting on data)
- Steps 1, 3 (with a hand-made fixture), 9 and 14 do not need real data.
- One person: D3 + route planning. One person: io.py + map matching. One person: torch/M1.
