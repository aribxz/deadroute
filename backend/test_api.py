import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import app as backend

VALID = {
    'road_type': 'urban', 'num_lanes': 2, 'curvature': 0.35, 'speed_limit': 40,
    'lighting': 'daylight', 'weather': 'clear', 'road_signs_present': True,
    'public_road': True, 'time_of_day': 'afternoon', 'holiday': False,
    'school_season': True, 'num_reported_accidents': 2,
}

class PredictionApiTests(unittest.TestCase):
    def setUp(self):
        self.client = backend.app.test_client()

    def test_root_describes_api_and_frontend(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['service'], 'DeadRoute prediction API')
        self.assertIn('separately', body['message'])
        self.assertEqual(body['endpoints']['health'], '/health')
        self.assertEqual(body['endpoints']['metadata'], '/metadata')
        self.assertEqual(body['endpoints']['predict'], '/predict')

    def test_health_reports_loaded_model(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['status'], 'ok')
        self.assertTrue(body['model_loaded'])
        self.assertEqual(body['model_type'], 'XGBoost')

    def test_metadata_matches_artifact(self):
        response = self.client.get('/metadata')
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        artifact = backend.model_artifact
        self.assertEqual(body['required_features'], artifact['feature_columns'])
        self.assertEqual(body['uses_num_reported_accidents'], artifact['accident_count_included'])
        self.assertAlmostEqual(body['metrics']['mae'], artifact['validation_metrics']['MAE'])
        self.assertAlmostEqual(body['metrics']['rmse'], artifact['validation_metrics']['RMSE'])
        self.assertAlmostEqual(body['metrics']['r2'], artifact['validation_metrics']['R²'])

    def test_prediction_returns_real_finite_model_output(self):
        response = self.client.post('/predict', json=VALID)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIsInstance(body['accident_risk'], (float, int))
        self.assertTrue(math.isfinite(body['accident_risk']))
        self.assertEqual(body['model'], 'XGBoost')
        self.assertIn('not a calibrated real-world accident probability', body['score_interpretation'])
        self.assertNotIn('prediction', body)
        self.actual_prediction = body

    def test_missing_feature_is_rejected(self):
        payload = dict(VALID); payload.pop('num_reported_accidents')
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn('Missing required feature', response.get_json()['error'])

    def test_invalid_category_and_range_are_rejected(self):
        payload = dict(VALID); payload['road_type'] = 'suburban'
        self.assertEqual(self.client.post('/predict', json=payload).status_code, 400)
        payload = dict(VALID); payload['speed_limit'] = 500
        self.assertEqual(self.client.post('/predict', json=payload).status_code, 400)

    def test_whole_number_float_is_accepted_but_fraction_is_rejected(self):
        payload = dict(VALID); payload['speed_limit'] = 40.0
        self.assertEqual(self.client.post('/predict', json=payload).status_code, 200)
        payload['speed_limit'] = 40.5
        self.assertEqual(self.client.post('/predict', json=payload).status_code, 400)

    def test_invalid_boolean_and_malformed_json_are_rejected(self):
        payload = dict(VALID); payload['holiday'] = 'false'
        self.assertEqual(self.client.post('/predict', json=payload).status_code, 400)
        response = self.client.post('/predict', data='{bad json', content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_local_and_configured_frontend_cors(self):
        local = self.client.get('/health', headers={'Origin':'http://localhost:5500'})
        self.assertEqual(local.headers.get('Access-Control-Allow-Origin'), 'http://localhost:5500')

        previous = backend.FRONTEND_ORIGIN
        try:
            backend.FRONTEND_ORIGIN = 'https://deadroute-ui.onrender.com'
            deployed = self.client.get('/health', headers={'Origin':'https://deadroute-ui.onrender.com'})
            blocked = self.client.get('/health', headers={'Origin':'https://example.com'})
            self.assertEqual(deployed.headers.get('Access-Control-Allow-Origin'), 'https://deadroute-ui.onrender.com')
            self.assertIsNone(blocked.headers.get('Access-Control-Allow-Origin'))
        finally:
            backend.FRONTEND_ORIGIN = previous

if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PredictionApiTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        with backend.app.test_client() as client:
            print('SAMPLE_RESPONSE=' + json.dumps(client.post('/predict', json=VALID).get_json(), sort_keys=True))
    raise SystemExit(not result.wasSuccessful())
