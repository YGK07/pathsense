#!/usr/bin/env bash
# Creates GitHub issues from the ROADMAP task list. Requires the GitHub CLI: https://cli.github.com  (gh auth login)
# Run once from the repo root after pushing: bash scripts/create_issues.sh
set -e
mk() { gh issue create --title "$1" --body "$2 See docs/ROADMAP.md (task ID in the title)." --label "$3" 2>/dev/null || gh issue create --title "$1" --body "$2 See docs/ROADMAP.md (task ID in the title)."; }
for l in P0 P1 P2; do gh label create "$l" --force >/dev/null 2>&1 || true; done
mk "P1: Pilot data collection" "Collect the pilot dataset per docs/DATA_COLLECTION.md." P0
mk "P2: Real CSV loader (pathsense/io.py)" "Load logger CSVs, resample, align, return Trace objects." P0
mk "P3: GPS map matching to road graph" "Fill Trace.seg for real traces." P0
mk "D3: Real VIT/Katpadi road graph (OSM)" "Decide F5 vs multi-task swap based on graph quality." P0
mk "F1: Calibrate inference on real data" "Kerb thresholds, gradient bias, roughness mapping." P0
mk "F7: Wheeled-motion preprocessing + ablation" "Choose FS/window/filters from pilot spectra." P0
mk "F2: Make fusion demonstrably useful (or report honestly)" "DS currently ~ majority vote on simulated data." P0
mk "M1: Run and debug scripts/train_ssl_dann.py" "Never executed; install torch and fix." P0
mk "M2: SSL and DANN ablation tables" "Real numbers for paper Table 1 rows." P0
mk "W1: Fill paper results with real numbers only" "No simulated numbers in the paper." P0
mk "A0: Demo uses the real map" "Swap simulated map for real pipeline output." P0
mk "F5: Torch GCN/GAT propagation" "Compare with numpy GCN and baselines." P1
mk "F6: Tune mobility profile weights" "Document rationale." P1
mk "M3: Encoder embeddings in observation classifier" "Replace RandomForest on 3 features." P1
mk "A1: Android collector or documented logger app" "Decide and implement." P1
mk "A2: Demo polish and comparison view" "Profile comparison, mobile layout." P1
mk "W3: Western vs Indian profile comparison" "Accepted methodology item." P1
mk "W4: Verify all references" "Check authors/venue/year." P1
mk "P4: Ground-truth segments file" "Build data/segments.csv from field notes." P1
mk "P5: Dataset release" "Datasheet, anonymisation, upload." P2
mk "A3: Deploy the demo" "Public link for evaluation." P2
mk "M4: Multi-task fallback head" "Mobility state + surface attributes." P2
