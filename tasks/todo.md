# Issue #5 — Fix train/test data leakage inflating all model metrics

**Severity:** HIGH (blocker) · Labels: blocker, data-integrity, ml-correctness
**Branch:** `fix/5-train-test-leakage`

## The problem (verified in code)
- `core/trainer.py:573-600` `prepare_features()` fits numeric-median imputation and
  per-feature `LabelEncoder` on the **full dataset**, then `train_single_model` splits
  that already-leaked matrix (`trainer.py:965-979`) and also hands the whole matrix to
  `cross_val_score` (`trainer.py:1112-1119`).
- `api/validation.py` `_build_Xy` re-loads the **entire CSV** and every diagnostic
  (confusion matrix, calibration, threshold, segment, errors — 15 endpoints) predicts on
  **all rows** → in-sample, not held-out.
- `trainer.py:976-979` n<10 fallback sets `X_train = X_test = X`, yet
  `_classification_summary` (`trainer.py:1237`) still prints "% accuracy on the held-out
  test set".

## Key architectural constraints (from exploration — do NOT break these)
1. The saved `.joblib` model is a **bare estimator over a numeric feature space**
   (one column per feature). `core/deployer.py` has ~30 functions (predict, explain,
   sensitivity, sweep, goal-seek) that build/perturb **numeric** vectors and call
   `model.predict`/`predict_proba`. We must NOT change the model's numeric input contract.
2. Serving preprocessing lives in `deployer.PredictionPipeline` (medians + per-feature
   `LabelEncoder`), built by `build_prediction_pipeline(full_df)`. The model's ordinal
   codes MUST match serving's codes or predictions silently corrupt.
3. `prepare_features` is called by ~12 test files + analysis fns expecting a numeric
   ndarray — changing its return type is high blast radius.
4. DB has an inline-migration mechanism (`db._apply_migrations`) — adding a nullable
   `ModelRun` column is cheap and safe for existing DBs.

## Design (contained, correct): split-first + train-fold-fit preprocessor + persisted test indices

### Step 1 — `core/preprocessing.py` (new): single source of truth for the transform
- `build_preprocessor(df, feature_cols) -> ColumnTransformer` (UNFITTED):
  `SimpleImputer(strategy="median")` for numeric cols + `OrdinalEncoder(handle_unknown=
  "use_encoded_value", unknown_value=-1)` for categorical. Output = one column per
  feature, order = `feature_cols` (preserves `feature_importances_` alignment).
- `extract_clean_xy(df, feature_cols, target_col, problem_type) -> (X_raw_df, y, le_target)`:
  drop missing-target rows (reset_index), encode **target** globally (standard practice,
  not leakage), return the **raw** (untransformed) feature frame.
- Tests: `tests/test_preprocessing.py`.

### Step 2 — `core/trainer.py`: split first, fit preprocessing on train fold only
- Flow: `X_raw, y, le = extract_clean_xy(...)` → split the **raw** frame (random or
  chronological) → `prep = build_preprocessor(...)` → `prep.fit(X_train_raw)` →
  `X_train = prep.transform(...)`, `X_test = prep.transform(X_test_raw)` → fit bare
  estimator on numeric `X_train` → metrics on `prep.transform(X_test_raw)` (LEAK-FREE).
- CV: `cross_val_score(Pipeline([("prep", build_preprocessor(...)), ("model",
  fresh_estimator)]), X_raw, y, cv=...)` so prep refits each fold (LEAK-FREE).
- Persist the fitted `prep` as sidecar `{model_run_id}.prep.joblib` (next to the model).
- Return `test_indices` (positional indices into the clean frame) in the result dict.
- Apply the same split-first treatment to ensemble / goal-driven / tuning training paths.

### Step 3 — n<10 honesty (`trainer.py`)
- When `n < MIN_HELDOUT` (=10) → mark `metrics["evaluation"] = "in_sample"`,
  `metrics["n_too_small"] = True`; summaries emit "on the training data (dataset too
  small for a held-out test — treat as optimistic)" instead of "held-out test set".
  Held-out wording only when a real test split exists.

### Step 4 — persist held-out test indices on `ModelRun`
- `models/model_run.py`: add `test_indices: Optional[str]` (JSON list) + `evaluation`
  (str) nullable columns.
- `db._apply_migrations`: add `("modelrun", "test_indices", "TEXT")`,
  `("modelrun", "evaluation", "TEXT")`.
- `api/models.py`: store `result["test_indices"]` JSON on the run after training.

### Step 5 — `api/validation.py`: diagnostics on held-out rows only
- `_build_Xy` → add `held_out_only` path. When the run has `test_indices`, slice the
  clean frame to those positional rows and transform via the persisted `{run}.prep.joblib`
  (NOT a refit on full data). Confusion / calibration / threshold / segment / errors all
  use the held-out slice.
- Legacy models with no `test_indices`: fall back to full-data prediction but tag the
  response `evaluation: "in_sample"` + a plain-English note (no silent inflation).

### Step 6 — serving alignment (`core/deployer.py`)
- `build_prediction_pipeline`: when a persisted `{run}.prep.joblib` exists, source the
  `medians` + per-feature category orderings FROM it (so serving ordinal codes == training
  codes). Keep computing means/stds/ranges from the df for UX warnings. Falls back to
  current full-df fit when no preprocessor sidecar (legacy/retrain edge).

### Step 7 — leakage regression test (AC4)
- `tests/test_train_test_leakage.py`: data where a categorical feature has a test-only
  category and numeric NaNs; assert (a) the preprocessor passed to CV is unfitted/refit
  per fold, (b) held-out metrics from the new path are NOT inflated vs a manual leak-free
  baseline, (c) the old global-fit path would inflate (documents the bug), (d) n<10 result
  is tagged `in_sample` and summary omits "held-out".

## Acceptance criteria (from issue #5) — ALL MET
- [x] Imputation/encoding in an sklearn Pipeline (SimpleImputer + OrdinalEncoder w/
      handle_unknown) fit only on each training fold; unfitted Pipeline passed to the
      train/test fit (via `_split_and_preprocess`) and to `cross_val_score`
      (`trainer.train_single_model` CV block + `validation._leakfree_cv`).
- [x] Calibration/threshold/confusion/segment/error/fairness diagnostics computed on
      persisted held-out indices (`validation._build_eval_Xy`), never in-sample.
- [x] Never emit "held-out test set" when X_train is X_test; `_tag_evaluation` marks
      `evaluation=in_sample` + summaries say "on the training data"; held-out requires
      n >= MIN_HELDOUT_ROWS (10).
- [x] Leakage regression test (`tests/test_train_test_leakage.py`, 6 tests): prep refit
      per CV fold, held-out metric == manual leak-free baseline, leaky path diverges,
      n<10 tagged in_sample, serving codes/unknown-sentinel match the train-fold prep.

## Status: implementation complete; full backend suite green (6678 passed, 1 fixed).
The sole full-suite failure (`test_calibration_check_endpoint_200_classification`) was a
real behavior change, not a flake: held-out calibration on the 40-row fixture left only 8
rows (< calibration's 10-sample floor). Fixed correctly by raising `_MIN_HELDOUT_EVAL_ROWS`
to 10 so too-small held-out sets fall back to in-sample (honestly tagged) instead of
erroring — matching the trainer's MIN_HELDOUT_ROWS and the UX "fail gracefully" rule.

## Test strategy
- Pure-function tests for `preprocessing` + `trainer` leak-free path (no DB).
- REST tests via `client` for validation endpoints returning held-out diagnostics +
  `evaluation` tag; `anon_client` unaffected.
- Regression test (Step 7) is the AC4 gate.
- Full `uv run pytest` must stay green; ruff + black clean.

## Deviations / decisions (self-adapted; issue body had the plan)
- **OrdinalEncoder over OneHotEncoder**: preserves one-column-per-feature so
  `feature_importances_`/`coef_` alignment and deployer's numeric perturbation machinery
  keep working. OneHot would explode the feature space and break ~30 deployer functions.
- **Model stays a bare estimator + sidecar preprocessor** (NOT a wrapped sklearn Pipeline
  artifact): wrapping would change the model's input contract from numeric→raw and break
  the entire serving/explanation layer. Sidecar preprocessor gives leak-free training and
  train==serve code alignment with minimal blast radius.
- Learning-curve / overfitting / data-quality diagnostics (`compute_learning_curve` etc.)
  are NOT in the issue's AC list — left as a documented Known Limitation unless trivially
  covered, to contain risk.
