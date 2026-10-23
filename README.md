# DeadRoute

DeadRoute is an AI Build Arena demonstration of road accident-risk prediction. It combines a trained regression model with a small Flask API and a browser-based route visualization.

> **Demo limitation:** The dataset and route network are synthetic. Map roads, locations, and simulated road conditions are illustrative; the dataset contains no geographic coordinates. The displayed score is a model prediction on a 0–1 scale, not a calibrated real-world accident probability.

## Project layout

- `frontend/` — standalone browser UI (`index.html`)
- `backend/` — Flask prediction API, API checks, and Python requirements
- `ml/` — training and optimization scripts, saved model artifacts, experiment workbooks, and run summaries
- `dataset/` — training and unlabeled test CSVs plus the dataset audit

The frontend expects the backend at `http://127.0.0.1:5000`.

## Run locally (Windows)

Use Python 3.12 or a compatible Python version with wheels for the pinned ML packages.

From the project root, create an environment and install dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Start the backend in one terminal:

```powershell
.\.venv\Scripts\python.exe backend\app.py
```

Start the static frontend in a second terminal:

```powershell
Set-Location frontend
py -m http.server 8000
```

Open <http://127.0.0.1:8000/>. The page uses Google Fonts when available; the prediction API itself runs locally.

## Verify the API

With dependencies installed, run from the project root:

```powershell
.\.venv\Scripts\python.exe backend\test_api.py
```

The API exposes `/health`, `/metadata`, and `/predict` on port 5000.

## Model and evaluation

The API loads `ml/best_model.pkl`, the baseline XGBoost pipeline selected on a fixed 80/20 validation split (seed 42). Its recorded validation metrics are MAE **0.04365**, RMSE **0.05625**, and R² **0.88542**. The separate optimized artifact and experiment are retained in `ml/`; the measured RMSE improvement over baseline was negligible, so the baseline remains the API model.

`accident_risk` is the regression target. The `id` column is excluded. The experiments compare models with and without `num_reported_accidents`; check that field’s availability at prediction time before using the model prospectively, since its timing could make it a leakage risk. `test.csv` has no target labels, so no independent test score is reported.

Re-run the baseline comparison or controlled XGBoost optimization with:

```powershell
.\.venv\Scripts\python.exe ml\model_experiments.py
.\.venv\Scripts\python.exe ml\model_optimization.py
```

The scripts write their result sheets, summaries, and model artifacts under `ml/`.
