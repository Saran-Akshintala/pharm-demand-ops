# pharm-demand-ops — Project Conventions

A pharmacy demand-forecasting system: EDA → trained model → Streamlit UI + FastAPI service.
`README.md` covers features and structure. This file is about the parts that bite.

**Status: dormant.** Last commit 02-Dec-2025. It is a portfolio project, not something running in
production — so treat a change here as a change to a demo someone may show, and do not assume any
scheduled job or deployment depends on it.

## ⚠️ `data/` holds real pharmacy sales data and is gitignored

`.gitignore` excludes `data/`, `*.xlsx` and `*.csv`. That is a **privacy boundary, not tidiness** —
the inputs are a real pharmacy's order history. Never commit a sample "just for testing", never
paste rows into a commit message or a README, and never move a spreadsheet out of `data/` to make a
path simpler.

If an example is genuinely needed, generate synthetic rows.

## `models/*.pkl` IS committed, and that is a deliberate trade

`models/order_predictor.pkl`, `model_info.json` and `registry.json` are tracked, so the app runs on
a fresh clone without a training pass. Two things follow:

- **A pickle is executable.** Only load a model this repo produced; never load one from anywhere
  else, and never add a code path that unpickles a user-supplied file.
- **Retraining changes a binary in git.** `app/retrain.py` rewrites the `.pkl`. Commit it with the
  metrics from that run in the message, or the registry and the file drift apart with no way to
  tell which model is in there.

## The `test_*.py` files in `app/` are not a test suite

There are ~30 of them beside the application code — `test_green_highlighting_fix.py`,
`test_single_button_download.py`, `test_edit_persistence_and_coloring_fixes.py`. They are one-off
validation scripts written while chasing specific UI bugs, not a suite anything runs, and many
assume a particular Excel file or a live Streamlit session.

⚠️ So `pytest app/` is not a meaningful health check here, and a green run does not mean the app
works. If real coverage is wanted, that is a piece of work to do deliberately — creating `tests/`
and moving the genuine cases into it — not something to assume already exists.

`app/debug_*.py`, `app/final_validation.py` and `app/validate_edge_case.py` are the same kind of
artefact.

## Scheme parsing is the domain logic worth being careful with

Promotional schemes (`"9+1"`, `"12+2"`) drive the predicted quantities, and the grid applies
colour-coded business rules on top. Changing a multiplier or a rounding rule changes what a
pharmacist would order. Anything touching scheme parsing, box adjustment or the priority rules
needs a worked example in the commit message showing the before/after quantity.
