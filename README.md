# DeadRoute

DeadRoute is an AI Build Arena demonstration of road accident-risk prediction. It combines a trained regression model with a Flask API and a browser-based route visualization.

> **Demo limitation:** The dataset and route network are synthetic. Map roads, locations, and simulated road conditions are illustrative; the dataset contains no geographic coordinates. The displayed score is a model prediction on a 0–1 scale, not a calibrated real-world accident probability.

## Project layout

- `frontend/` — browser UI assets (`index.html`, `config.js`), served by Flask at `/`
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

Start the full app from the project root:

```powershell
.\.venv\Scripts\python.exe app.py
```

Optional: start a separate static frontend for UI-only work in a second terminal:

```powershell
Set-Location frontend
py -m http.server 8000
```

With the app running, open <http://127.0.0.1:5000/> to use the combined frontend and API. If you use the optional static server, open <http://127.0.0.1:8000/>; its config points API requests to the local backend on port 5000. Google Fonts load when internet access is available.

## Deploy on Render

Create one **Web Service** connected to this repository and the `main` branch. Flask serves both the browser UI at `/` and the prediction API from the same service.

- **Root Directory:** leave blank (repository root)
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `python app.py`
- **Health Check Path:** `/health`

The root `app.py` starts the app on `127.0.0.1:5000` locally. On Render, it uses the provided `PORT` and binds to `0.0.0.0`; Waitress is used as the WSGI server after dependencies install. Open the Web Service URL to see the frontend. Its API calls use the same origin, so no separate Static Site or frontend CORS setting is needed.

## Verify the API

With dependencies installed, run from the project root:

```powershell
.\.venv\Scripts\python.exe backend\test_api.py
```

The app is served at `/`; API endpoints are `/health`, `/metadata`, and `/predict`.

## Model and evaluation

The API loads `ml/best_model.pkl`, the baseline XGBoost pipeline selected on a fixed 80/20 validation split (seed 42). Its recorded validation metrics are MAE **0.04365**, RMSE **0.05625**, and R² **0.88542**. The separate optimized artifact and experiment are retained in `ml/`; the measured RMSE improvement over baseline was negligible, so the baseline remains the API model.

`accident_risk` is the regression target. The `id` column is excluded. The experiments compare models with and without `num_reported_accidents`; check that field’s availability at prediction time before using the model prospectively, since its timing could make it a leakage risk. `test.csv` has no target labels, so no independent test score is reported.

Re-run the baseline comparison or controlled XGBoost optimization with:

```powershell
.\.venv\Scripts\python.exe ml\model_experiments.py
.\.venv\Scripts\python.exe ml\model_optimization.py
```

The scripts write their result sheets, summaries, and model artifacts under `ml/`.

## Deploy the frontend on Netlify

Netlify hosts the browser UI as a static site; it does not run the Flask/XGBoost API. Keep a working Python API deployed on a Python-capable host, then connect it to this site:

1. Import this GitHub repository into Netlify. Keep the **Base directory** at the repository root.
2. The included `netlify.toml` sets the build command to `node scripts/netlify_build.js` and publish directory to `netlify-build`.
3. In Netlify site environment variables, set `DEADROUTE_API_BASE` to the API's public HTTPS origin, for example `https://your-api-host.example` (no `/predict` suffix or trailing path).
4. Deploy. The build generates the static frontend and a same-origin `/api/*` proxy to the Python API, so browser CORS configuration is not needed.
5. Verify the Netlify URL loads the page and `https://<api-origin>/health` reports `"status":"ok"`; submit a prediction to confirm the proxy works.

The build intentionally stops with a clear error if `DEADROUTE_API_BASE` is missing or invalid. The current Render deployment must be healthy before Netlify predictions can work; deploying only the static site does not include the saved model or Flask server. Netlify proxy requests have a 26-second timeout, which is sufficient for this prediction endpoint.
