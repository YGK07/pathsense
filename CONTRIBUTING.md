# Contributing

1. Pick an issue (or a task ID from `docs/ROADMAP.md`) and comment that you are taking it.
2. Branch from `main`: `git checkout -b feat/<task-id>-short-name`.
3. Keep changes small; add or update a test in `tests/` when you touch `pathsense/`.
4. `python -m pytest -q tests` must pass before you open a PR (CI runs it too).
5. Open a PR to `main`, one reviewer from the team approves, then squash-merge.
6. Never commit raw participant data or personal info (`data/raw/` is git-ignored). Anything involving wheelchair
   users needs consent first (docs/DATA_COLLECTION.md).
7. Never present simulated output as a real result. Label it "simulated" in code, plots and the paper.
