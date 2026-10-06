# Progress

Working brief: audit, rebuild and ship this project in phases (0 audit → 1 plan → 2 pipeline →
3 insights → 4 teaching → 5 README → 6 web app), stopping at each checkpoint for approval.

## Status

| Phase | State |
|---|---|
| 0. Audit | Done, waiting for checkpoint approval. Result: [docs/AUDIT.md](docs/AUDIT.md) |
| 1. Plan | Not started |
| 2–6 | Not started |

## How to resume

1. `git lfs pull` (the CSVs in `data/` are LFS pointers until then).
2. Clone the reference next to this repo: `git clone https://github.com/adzict/yelp_sentiment_analysis ../yelp_sentiment_analysis`.
3. Audit environment: see "How to reproduce this audit" at the end of `docs/AUDIT.md`
   (Python 3.11, `docs/audit/requirements-audit.txt`, `pip install --no-deps xgboost==2.1.0`).
4. All audit numbers are in `docs/audit/results/*.json`; `python docs/audit/make_report.py`
   regenerates the tables and figure in `docs/AUDIT.md`.

## Done

- Tagged the original `main` (commit `849616c`) as `v0-original`. The tag exists locally, but this
  session's git proxy refused to push tags; push it with `git push origin v0-original` from a normal
  clone (or create it on GitHub at commit `849616c`). `main` itself is unchanged.
- Phase 0 audit: inventory, cell-by-cell provenance against the reference, re-run of all four
  notebooks (as written and minimally patched), independent checks of data, modelling, VADER and
  business claims, verdict table. Scripts in `docs/audit/`, results in `docs/audit/results/`.

## Decisions and why

- **Audit code lives in `docs/audit/` and its JSON output is committed.** The brief requires every
  number to be traceable to code in this repo; the tables in AUDIT.md are generated from the JSON.
- **The original environment was reconstructed from the notebooks' `!pip install` output**
  (no requirements file existed) so the original numbers could be reproduced exactly.
  xgboost is installed with `--no-deps` to skip ~200 MB of GPU libraries.
- **Original files were not modified.** Notebook re-runs used copies of the notebooks and of `data/`.
- **The reference's `Classification.py` is loaded from a local clone, not copied in.** The
  reference repo has no licence file.
- **The original test split was evaluated once, for the audit only** (logistic regression).
  Phase 2 will create a new split, so this number cannot influence modelling choices.
- **Executed notebook copies are not committed** (large; their numbers are in the JSON).

## Open questions for the user

1. **Branch.** The brief says to work on `revamp`; this cloud session is set up to push to
   `claude/amazing-edison-j0t3kb`. Phase 0 is pushed there. Should it also go to `revamp`?
2. **Raw data.** The raw Yelp files are in neither repo; the committed CSVs are already "prepared"
   (the reference's cleaning + `review_length`). Treat the committed CSVs as the raw input for
   `make data`, or obtain the original Yelp release?
3. **Data licence.** Needs to be settled before Phase 6 shows any review text (see AUDIT.md §6).
4. **Git LFS bandwidth.** Every clone that pulls the data downloads ~230 MB from GitHub's LFS quota.
   CI should run on a small committed sample instead of the full data.
5. **The original notebooks.** Keep them (e.g. under `archive/`) for transparency, or remove them
   from the main tree (they stay in history and under the `v0-original` tag)?

## Still running at the time of writing

- `docs/audit/rerun_reference_models.py` for AdaBoost, random forest, XGBoost and KNN (the decision
  tree is done and reproduces the pasted numbers exactly). Results are appended to
  `docs/audit/results/reference_models_rerun.json`; re-run `make_report.py` afterwards.
