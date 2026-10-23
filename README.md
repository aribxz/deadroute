# DeadRoute

DeadRoute is an AI Build Arena demonstration of road accident-risk prediction. It combines a trained regression model with a Flask API and a browser-based route visualization.

> **Demo limitation:** The dataset and route network are synthetic. Map roads, locations, and simulated road conditions are illustrative; the dataset contains no geographic coordinates. The displayed score is a model prediction on a 0–1 scale, not a calibrated real-world accident probability.

## Project layout

- `frontend/` — standalone browser UI (`index.html`) and API URL configuration (`config.js`)
- `backend/` — Flask prediction API, API checks, and Python requirements
- `ml/` — training and optimization scripts, saved model artifacts, experiment workbooks, and run summaries
- `dataset/` — training and unlabeled test CSVs plus the dataset audit

## Run locally (Windows)

Use Python 3.12 or a compatible Python version with wheels for the pinned ML packages.

From the project root, create an environment and install dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
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

## Deploy on Render

Create two services connected to this repository and the `main` branch.

### Flask backend — Web Service

- **Root Directory:** leave blank (repository root)
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn --chdir backend --bind 0.0.0.0:$PORT app:app`
- **Health Check Path:** `/health`
- **Environment Variable:** `FRONTEND_ORIGIN` = the exact public URL of the Static Site, such as `https://deadroute-ui.onrender.com`

The backend accepts local development origins and the single exact origin configured in `FRONTEND_ORIGIN`.

### Browser frontend — Static Site

- **Root Directory:** leave blank (repository root)
- **Build Command:** leave blank
- **Publish Directory:** `frontend`

After creating the backend, copy its public `https://...onrender.com` URL into `frontend/config.js` as `window.DEADROUTE_API_BASE`. Then create or redeploy the Static Site. Set `FRONTEND_ORIGIN` on the backend to the Static Site’s exact public URL and redeploy the backend if you add or change that URL.

Render web services must bind to `0.0.0.0` and Render’s assigned `$PORT`; the start command above does that. The backend must be a Web Service, while the static HTML frontend is a Static Site.

## Verify the API

With dependencies installed, run from the project root:

```powershell
.\.venv\Scripts\python.exe backend\test_api.py
```

The API exposes `/health`, `/metadata`, and `/predict`.

## Model and evaluation

The API loads `ml/best_model.pkl`, the baseline XGBoost pipeline selected on a fixed 80/20 validation split (seed 42). Its recorded validation metrics are MAE **0.04365**, RMSE **0.05625**, and R² **0.88542**. The separate optimized artifact and experiment are retained in `ml/`; the measured RMSE improvement over baseline was negligible, so the baseline remains the API model.

`accident_risk` is the regression target. The `id` column is excluded. The experiments compare models with and without `num_reported_accidents`; check that field’s availability at prediction time before using the model prospectively, since its timing could make it a leakage risk. `test.csv` has no target labels, so no independent test score is reported.

Re-run the baseline comparison or controlled XGBoost optimization with:

```powershell
.\.venv\Scripts\python.exe ml\model_experiments.py
.\.venv\Scripts\python.exe ml\model_optimization.py
```

The scripts write their result sheets, summaries, and model artifacts under `ml/`.
