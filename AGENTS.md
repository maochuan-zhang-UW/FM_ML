# Repository Guidelines

## Project Structure & Module Organization
- `01-scripts/`: main analysis and ML code.
- MATLAB pipeline scripts live at `01-scripts/*.m` and commonly use stage prefixes like `A_`, `G_`, `X_`, `Z_`.
- Python model subprojects are grouped under `01-scripts/APP`, `CFM`, `RPNet`, `Stone_polarPicker`, `eqpolarity`, and `DiTing-FOCALFLOW`.
- `02-data/`: datasets and model artifacts used by scripts.
- `03-output-graphics/`: generated figures (see `savemyfigureFM5_ML.m`).
- `04-manuscripts/`: figure-building scripts for paper assets.
- `05-mise/`: design/source assets.

## Build, Test, and Development Commands
- Initialize MATLAB paths:
  - `matlab -batch "run('FM_buildpath5.m')"` (update absolute paths for your machine).
- Run a representative MATLAB workflow:
  - `matlab -batch "run('01-scripts/Z_AGU2025_workflow.m')"`
- APP module:
  - `cd 01-scripts/APP && pip install -r requirements.txt`
  - `python APP_Run.py --mode='train'` (or `test`, `predict`)
- CFM module:
  - `conda env create -f 01-scripts/CFM/CFM_env.yml`
  - `python 01-scripts/CFM/predict.py --help`

## Coding Style & Naming Conventions
- MATLAB: keep existing script-first style and filename prefixes by workflow stage (`A_...`, `G_...`, etc.).
- Python: use 4-space indentation, `snake_case` for functions/files, and keep CLI scripts runnable from terminal.
- No repo-wide formatter/linter is enforced; follow the style already present in the directory you edit.

## Testing Guidelines
- There is no single top-level test runner; validate in the module you changed.
- Python checks: run available test/demo scripts such as `01-scripts/eqpolarity/demos/test_*.py` or `01-scripts/Stone_polarPicker/predict_test*.py`.
- MATLAB checks: rerun affected `.m` scripts and verify expected outputs in `03-output-graphics/` and related data outputs.

## Commit & Pull Request Guidelines
- Commit messages in this repo are brief and imperative (example from history: `updateAPP`).
- Keep subjects short, scoped, and action-oriented; add details in the body when data/model files change.
- PRs should include: objective, changed paths, run/repro commands, and before/after figures or metrics for behavior changes.

## Data & Configuration Tips
- Avoid committing generated artifacts unless required; `.gitignore` already excludes `*.mat`, `*.mp4`, `*.png`, `*.pdf`.
- Prefer configurable/local paths over hard-coded absolute paths before sharing cross-machine changes.
