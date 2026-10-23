from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
# Supports the project-local dependency directory used for the hackathon setup.
LOCAL_PACKAGES = ROOT / '.packages'
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = ROOT.parent / 'ml' / 'best_model.pkl'
DATA_PATH = ROOT.parent / 'dataset' / 'train.csv'
FRONTEND_ORIGIN = os.environ.get('FRONTEND_ORIGIN', '').strip().rstrip('/')
DATA_NOTICE = 'Model trained on a synthetic road accident risk dataset.'
SCORE_INTERPRETATION = 'Model-predicted risk score, not a calibrated real-world accident probability.'

# Values/ranges are verified against dataset/train.csv.
CATEGORY_VALUES = {
    'road_type': {'urban', 'rural', 'highway'},
    'lighting': {'daylight', 'dim', 'night'},
    'weather': {'clear', 'rainy', 'foggy'},
    'time_of_day': {'morning', 'afternoon', 'evening'},
}
INTEGER_RANGES = {'num_lanes': (1, 4), 'speed_limit': (25, 70), 'num_reported_accidents': (0, 7)}
FLOAT_RANGES = {'curvature': (0.0, 1.0)}
BOOLEAN_FEATURES = {'road_signs_present', 'public_road', 'holiday', 'school_season'}

model_artifact: dict[str, Any] | None = None
model_load_error: Exception | None = None
try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError('Expected baseline model artifact best_model.pkl was not found.')
    loaded = joblib.load(MODEL_PATH)
    required = {'model', 'algorithm', 'feature_columns', 'validation_metrics'}
    if not isinstance(loaded, dict) or not required.issubset(loaded):
        raise ValueError('The model artifact does not have the expected saved-artifact format.')
    if not hasattr(loaded['model'], 'predict'):
        raise ValueError('The artifact does not contain a prediction pipeline.')
    model_artifact = loaded
except Exception as exc:  # Keep API available to report unhealthy state, without a mock fallback.
    model_load_error = exc
    app_logger = None
    print(f'Model loading failed: {type(exc).__name__}: {exc}', file=sys.stderr)

app = Flask(__name__)


def model_is_loaded() -> bool:
    return model_artifact is not None


def allowed_origin(origin: str) -> bool:
    # Keep local development origins and allow the exact deployed frontend origin.
    from urllib.parse import urlparse
    parsed = urlparse(origin)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
        return False
    if parsed.hostname in {'localhost', '127.0.0.1'}:
        return True
    return bool(FRONTEND_ORIGIN) and origin.rstrip('/') == FRONTEND_ORIGIN


@app.after_request
def add_local_cors(response):
    origin = request.headers.get('Origin')
    if origin and allowed_origin(origin):
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Vary'] = 'Origin'
    return response


def unavailable_response():
    return jsonify({'error': 'The trained model is not available.'}), 503


@app.get('/')
def index():
    healthy = model_is_loaded()
    return jsonify({
        'service': 'DeadRoute prediction API',
        'status': 'ok' if healthy else 'error',
        'message': 'The browser frontend is hosted separately from this API.',
        'endpoints': {
            'health': '/health',
            'metadata': '/metadata',
            'predict': '/predict',
        },
    }), (200 if healthy else 503)


@app.get('/health')
def health():
    healthy = model_is_loaded()
    body = {
        'status': 'ok' if healthy else 'error',
        'model_loaded': healthy,
        'model_type': 'XGBoost' if healthy else None,
        'mode': 'trained_model' if healthy else 'unavailable',
    }
    return jsonify(body), (200 if healthy else 503)


@app.get('/metadata')
def metadata():
    if not model_is_loaded():
        return unavailable_response()
    artifact = model_artifact
    metrics = artifact['validation_metrics']
    return jsonify({
        'model': 'XGBoost',
        'model_name': artifact.get('algorithm', 'XGBoost Regressor'),
        'required_features': artifact['feature_columns'],
        'uses_num_reported_accidents': 'num_reported_accidents' in artifact['feature_columns'],
        'metrics': {
            'mae': float(metrics['MAE']),
            'rmse': float(metrics['RMSE']),
            'r2': float(metrics.get('R²', metrics.get('R2'))),
        },
        'data_notice': DATA_NOTICE,
    })


def validate_payload(payload: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(payload, dict):
        return None, 'JSON body must be an object.'
    features = model_artifact['feature_columns']
    missing = [feature for feature in features if feature not in payload]
    if missing:
        return None, f'Missing required feature(s): {", ".join(missing)}.'
    extra = sorted(set(payload) - set(features))
    if extra:
        return None, f'Unsupported feature(s): {", ".join(extra)}.'

    clean: dict[str, Any] = {}
    for feature in features:
        value = payload[feature]
        if feature in CATEGORY_VALUES:
            if not isinstance(value, str) or value not in CATEGORY_VALUES[feature]:
                allowed = ', '.join(sorted(CATEGORY_VALUES[feature]))
                return None, f'{feature} must be one of: {allowed}.'
        elif feature in BOOLEAN_FEATURES:
            if type(value) is not bool:
                return None, f'{feature} must be a JSON boolean.'
        elif feature in INTEGER_RANGES:
            low, high = INTEGER_RANGES[feature]
            # Browser/demo payloads can serialize whole-number controls as 40.0.
            # Accept integer-valued finite numbers, but reject booleans/fractions.
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not float(value).is_integer()
                or not low <= value <= high
            ):
                return None, f'{feature} must be an integer from {low} to {high}.'
            value = int(value)
        elif feature in FLOAT_RANGES:
            low, high = FLOAT_RANGES[feature]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
                return None, f'{feature} must be a finite number from {low} to {high}.'
            value = float(value)
        else:
            return None, f'No validation rule is configured for required feature {feature}.'
        clean[feature] = value
    return clean, None


@app.post('/predict')
def predict():
    if not model_is_loaded():
        return unavailable_response()
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({'error': 'Request body must be valid JSON with application/json content type.'}), 400
    values, error = validate_payload(payload)
    if error:
        return jsonify({'error': error}), 400
    try:
        row = pd.DataFrame([values], columns=model_artifact['feature_columns'])
        prediction = float(model_artifact['model'].predict(row)[0])
        if not np.isfinite(prediction):
            raise ValueError('Prediction was not finite.')
        return jsonify({
            'accident_risk': prediction,
            'model': 'XGBoost',
            'mode': 'trained_model',
            'score_interpretation': SCORE_INTERPRETATION,
        })
    except Exception:
        app.logger.exception('Prediction failed')
        return jsonify({'error': 'Prediction could not be generated.'}), 500


if __name__ == '__main__':
    # Local hackathon development only; use a production WSGI server when deploying.
    app.run(host='127.0.0.1', port=5000, debug=False)
