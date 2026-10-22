# PathSense results on real data (pilot, October 2026)

Data: 12 walks on the VIT Vellore campus and Katpadi-Thiruvalam Road, 8.1 km, 3 contributors, 3 phone models
(iPhone 15, iPhone 16, Oppo CPH2569), phone held in the hand. 6 walks have synchronised button presses: 120 marked
kerb crossings (89 on the iPhone 15, 31 on the iPhone 16). Map: 525 OpenStreetMap path segments (26.6 km).
Reproduce: `python scripts/real_experiments.py` -> `outputs/real_results.json` (5 seeds; about 12 minutes on a CPU).
Ground truth = walker button presses (kerbs) and real surfaces (OSM tags, satellite check, team survey).

## F1 Inertial surface and obstacle inference
| Kerb detection (balanced accuracy, iPhone 15, leave one walk out) | |
|---|---|
| Rule-based jolt detector (vertical-acceleration transient) | 0.74 |
| Learned 1D-CNN detector (F3 / F7 windows) | **0.88** |

Roughness (5 Hz band energy) did not separate smooth from paved segments (54 segments, Mann-Whitney p = 0.85; no
walk crossed an unpaved segment). Slope from the hand-held phone's tilt was not consistent between walks on the same
segment (correlation 0.0 for 106 opposite-direction pairs): hand motion swamps a campus slope of 1-3 degrees. Steps:
no stairs were marked on a synchronised walk yet, so not evaluated.

## F2 Probabilistic multi-trace fusion (Dawid-Skene EM)
Each walk = an annotator voting "kerb / no kerb" per segment from the jolt detector. 102 segments with a true answer
(32 with kerbs), 248 observations.

| Balanced accuracy | all 102 segments | 24 segments walked 3+ times |
|---|---|---|
| One walk | 0.75 | 0.74 |
| Majority vote | **0.77** | 0.73 |
| Dawid-Skene | 0.76 | **0.77** |
| Dawid-Skene, GPS-quality weighted | 0.76 | 0.77 |

Fusion helps over a single walk; Dawid-Skene beats voting only where segments have 3+ walks. Its per-walk reliability
estimates correlate with real per-walk accuracy at Spearman 0.35 (12 walks).

## F3 Self-supervised contrastive pretraining (SimCLR, 5,330 unlabelled windows from all 12 walks)
| Labels used (iPhone 15, leave one walk out) | from scratch | SimCLR-pretrained |
|---|---|---|
| 25% | 0.86 | 0.85 |
| 50% | 0.86 | 0.85 |
| 100% | 0.88 | 0.89 |

No benefit at this scale (with 2.56 s windows pretraining gave +2 points at 25% labels; with the better 1.28 s windows
the from-scratch model is already strong).

## F4 Device-invariant learning (DANN)
| Balanced accuracy on the unseen phone | train iPhone 15 -> test iPhone 16 | train iPhone 16 -> test iPhone 15 |
|---|---|---|
| Rule-based detector | 0.54 | 0.74 |
| From scratch | 0.53 | 0.62 |
| + DANN | 0.59 | 0.60 |
| SimCLR | 0.54 | 0.61 |
| SimCLR + DANN | **0.67** (AUC 0.83) | 0.59 |

Cross-phone accuracy drops a lot (0.88 same-phone -> 0.53-0.67). SimCLR + DANN gives +14 points when training on the
larger iPhone 15 set and no gain from the 31-example iPhone 16 set. Label timing differs per contributor (c03 presses
~1 s before the kerb, c01 ~0.2-0.9 s after) and is corrected per walk from the data.

## F5 Graph propagation to unobserved segments (GCN, 5-fold, labels hidden)
| | surface type, 264 segments (accuracy / macro-F1) | kerb on segment, 102 segments |
|---|---|---|
| Majority class | 0.50 / 0.22 | 0.69 / 0.41 |
| Segment features only | 0.50 / 0.44 | 0.72 / 0.70 |
| Neighbours vote | 0.80 / 0.77 | 0.62 / 0.55 |
| GCN (features + neighbours) | **0.82** / 0.75 | **0.72** / 0.70 |

## F6 Profile-conditioned routing vs a Western-centric profile (300 random campus trips, manual wheelchair)
| Scored on the real map | PathSense | Western-centric (OSM + assumed kerb ramps) | Shortest |
|---|---|---|---|
| Trips usable by a wheelchair (no stairs, closed or vehicle-only path, unpaved <= 20 m) | **91%** | 43% | 38% |
| Trips with stairs | 1% | 33% | 57% |
| Mean kerbs per trip | 1.45 | 2.65 | 5.13 |
| Mean length | 1,096 m | 1,009 m | 883 m |

## F7 Pre-processing re-derived from the pilot data
| Jolt detector (F1 score, kerb events) | |
|---|---|
| Previous setting (3 Hz high-pass, 2x footstep, 0.5 s) | 0.34 |
| Generic HAR setting (0.3 Hz gravity separation) | 0.47 |
| Chosen by leave-one-walk-out | 0.42 |
Adopted: 0.3 Hz / 2x / 0.3 s (`pathsense/io.py`).

| Learned detector window length | balanced accuracy |
|---|---|
| 1.28 s | **0.89** |
| 2.56 s (HAR default) | 0.82 |
| 5.12 s | 0.82 |
Derived from walking; the wheeled-motion (trolley) runs are still to be recorded and the same script re-derives them.

## Not yet done
Benchmarking rig runs (protocol and analysis ready: `docs/BENCHMARK_RIG.md`, `scripts/benchmark_rig.py`); dataset
publication (anonymised export ready: `scripts/export_dataset.py`, waiting for contributor consent); trolley traces.
