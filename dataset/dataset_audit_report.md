# DeadRoute Dataset Audit

**Scope:** Read-only audit of `dataset/train.csv` and `dataset/test.csv` before model training. Dtypes below were inferred from CSV values; CSV itself does not store a schema.

## Files and schema

| File | Rows | Columns | Notes |
|---|---:|---:|---|
| `train.csv` | 517,754 | 14 | Includes `accident_risk` target |
| `test.csv` | 172,585 | 13 | Same 13 input columns; target omitted |

| Column | Inferred type | Column | Inferred type |
|---|---|---|---|
| `id` | integer | `road_type` | categorical/string |
| `num_lanes` | integer | `curvature` | float |
| `speed_limit` | integer | `lighting` | categorical/string |
| `weather` | categorical/string | `road_signs_present` | boolean |
| `public_road` | boolean | `time_of_day` | categorical/string |
| `holiday` | boolean | `school_season` | boolean |
| `num_reported_accidents` | integer | `accident_risk` (train only) | float |

## Data quality and consistency

- **Missing values:** 0 in every column in both files.
- **Exact duplicate rows:** 0 in either file. `id` is unique in each file, and train/test IDs do not overlap.
- **Observed value checks:** categories are consistent (`road_type`: urban/rural/highway; `lighting`: daylight/dim/night; `weather`: clear/rainy/foggy; `time_of_day`: morning/afternoon/evening). Boolean columns contain only `True`/`False`.
- Numeric ranges are plausible for the apparent fields: `num_lanes` 1–4, `curvature` 0–1, `speed_limit` 25–70, `num_reported_accidents` 0–7. No obvious malformed values were found.
- IDs run 0–517,753 in train and 517,754–690,338 in test. They look like row identifiers; exclude `id` from model features.

## Target: `accident_risk`

- Training range: **0.00–1.00**; mean **0.352**, median **0.34**; 98 distinct values, mostly in 0.01 increments.
- Approximate distribution (risk bins):

| Risk interval | Rows | Risk interval | Rows |
|---|---:|---|---:|
| [0.0, 0.1) | 28,040 | [0.5, 0.6) | 52,343 |
| [0.1, 0.2) | 65,995 | [0.6, 0.7) | 29,074 |
| [0.2, 0.3) | 105,635 | [0.7, 0.8) | 11,670 |
| [0.3, 0.4) | 125,020 | [0.8, 0.9) | 3,466 |
| [0.4, 0.5) | 96,021 | [0.9, 1.0] | 490 |

**Regression suitability:** Yes. The target is numeric, bounded, and has many distinct levels; regression is appropriate. Use metrics such as MAE/RMSE and keep predictions/interpretation within the 0–1 risk scale. Its 2-decimal granularity means it is rounded/discrete rather than truly continuous.

## Features and leakage

- Candidate predictors: road/geometry (`road_type`, `num_lanes`, `curvature`, `speed_limit`), conditions (`lighting`, `weather`, `time_of_day`), and context (`road_signs_present`, `public_road`, `holiday`, `school_season`).
- **Leakage risk: `num_reported_accidents`.** It may be useful, but if it records accidents from the same period/event used to define `accident_risk`, it is post-outcome information and would inflate offline performance. Its correlation with the target is about **0.214**, which alone does not prove leakage. Confirm its time window and availability at prediction time; exclude it if the goal is prospective risk prediction and the count would not yet be known.
- No other obvious target-like column is present. `id` should be excluded as an identifier, not treated as a meaningful predictor. Feature importance cannot be established reliably without a model/validation analysis.

## Geographic mapping

Neither dataset has latitude/longitude, coordinates, address, road-segment key, or other explicit geographic field. The current data cannot place risk on a map; location identifiers or coordinates would need to be joined from another source.

## Train/test comparison

- Schemas align as expected: test has the same 13 inputs and omits only `accident_risk`.
- Input ranges and category levels match. Numeric means are nearly identical (for example, `num_lanes` 2.492 train vs 2.494 test; `curvature` 0.489 vs 0.487; reported accidents 1.188 vs 1.186). Categorical proportions differ by at most 0.38 percentage points among checked fields.
- The ID ranges are contiguous and non-overlapping, consistent with a row split. No major train/test distribution shift is apparent from these basic checks.

## Recommendation

**Proceed with model training after confirming the prediction-time meaning of `num_reported_accidents`.** Keep `id` out of the feature matrix, use a validation split representative of the intended deployment setting, and compare results with/without the accident-count feature if its availability is uncertain. Dataset audit only; no model was trained and original CSV files were not modified.
