# data/

**No real data has been collected yet.** Every result so far comes from the simulator (`pathsense/simulate.py`).

- `templates/` - blank CSV templates with one EXAMPLE row each (runs log, ground-truth features, ground-truth stretches).
- `ground_truth/features.csv`, `ground_truth/stretches.csv` - header-only files; fill them during the pilot (committed, no personal data).
- `raw/` - unzipped Sensor Logger exports, one folder per run (git-ignored; share via the team drive).
- `runs_log.csv` - copy of `templates/runs_log.csv`, one row per run.
- `osm_vit.json` - created by `python scripts/fetch_osm.py`.

Full procedure: `docs/DATA_COLLECTION.md`.
