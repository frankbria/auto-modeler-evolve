# Learnings

Cached knowledge from research. Check here before searching the web.

## XGBoost and LightGBM Integration (Day 3, 04:31)

- Both xgboost and lightgbm are sklearn-compatible — they implement `fit()`, `predict()`, `predict_proba()`, and `feature_importances_`
- XGBoost's `feature_importances_` sums to ~1.0; LightGBM's are raw gain values (unnormalized) — but `explainer.py` normalizes via `sum(abs(importances))` so both work fine
- Use `verbosity=0` for XGBClassifier/XGBRegressor to suppress training logs
- Use `verbose=-1` for LGBMClassifier/LGBMRegressor to suppress training logs
- LightGBM emits a sklearn warning "X does not have valid feature names but LGBMRegressor was fitted with feature names" when you call `predict()` with a plain numpy array on a model trained on a DataFrame — harmless in tests, safe in production (our predict pipeline uses numpy arrays consistently)
- XGBClassifier needs `eval_metric="logloss"` to avoid a warning about eval metric defaulting
- Optional import pattern: `try: from xgboost import X; _XGBOOST_AVAILABLE = True \n except ImportError: _XGBOOST_AVAILABLE = False` — allows code to work without the dependency

## Performance Baseline — Initial Measurements (Day 3, 04:31)

On a GitHub Actions Linux runner (2-core, 7GB RAM):
- Upload 200-row CSV + full profile: ~28ms
- Upload 1000-row CSV + full profile: ~27ms
- Cached profile endpoint (DB hit only): ~2ms
- Correlations heatmap endpoint: ~2ms
- Feature suggestions: ~6ms
- Linear regression train + poll (200 rows): ~218ms total
- Model recommendations endpoint: ~3ms
- Single prediction (post-deploy): ~4ms

These are fast because: SQLite is in-process, pandas operations are vectorized, models are tiny (200 rows). Expect 5-10x slower on real hardware with larger datasets (10k+ rows).
