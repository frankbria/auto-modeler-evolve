"""Model validation & explainability API endpoints.

Routes:
  GET /api/validate/{model_run_id}/metrics      — cross-val, confusion matrix / residuals
  GET /api/validate/{model_run_id}/explain      — global feature importance
  GET /api/validate/{model_run_id}/explain/{row_index} — single-row explanation
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from core.explainer import (
    compute_class_conditional_importance,
    compute_feature_importance,
    compute_partial_dependence,
    explain_single_prediction,
)
from core.feature_engine import apply_transformations
from core.trainer import (
    CLASSIFICATION_ALGORITHMS,
    REGRESSION_ALGORITHMS,
    prepare_features,
)
from core.validator import (
    assess_confidence_limitations,
    compute_calibration_check,
    compute_confidence_distribution,
    compute_confusion_matrix,
    compute_error_distribution,
    compute_fairness_metrics,
    compute_min_viable_feature_set,
    compute_per_class_threshold_analysis,
    compute_prediction_error_correlation,
    compute_prediction_errors,
    compute_residuals,
    compute_segment_performance,
    compute_threshold_analysis,
    run_cross_validation,
)
from db import get_session
from models.dataset import Dataset
from models.feature_set import FeatureSet
from models.model_run import ModelRun

from auth.dependencies import require_owner

router = APIRouter(
    tags=["validation"],
    dependencies=[Depends(require_owner)],
)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_run_context(model_run_id: str, session: Session):
    """Load ModelRun + FeatureSet + Dataset from DB."""
    run = session.get(ModelRun, model_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found")

    if run.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Model run status is '{run.status}'. Validation requires a completed run.",
        )

    if not run.model_path or not Path(run.model_path).exists():
        raise HTTPException(
            status_code=404, detail="Serialized model file not found on disk"
        )

    feature_set = session.get(FeatureSet, run.feature_set_id)
    if not feature_set:
        raise HTTPException(status_code=404, detail="Feature set not found")

    dataset = session.exec(
        select(Dataset).where(Dataset.id == feature_set.dataset_id)
    ).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    return run, feature_set, dataset, file_path


def _build_Xy(
    file_path: Path,
    feature_set: FeatureSet,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load CSV, apply transforms, return (X, y, feature_names).

    Full-dataset numeric matrix — used for model-property views (feature
    importance, partial dependence, single-row explanations) that are not
    held-out evaluations.
    """
    df = pd.read_csv(file_path)
    transforms = json.loads(feature_set.transformations or "[]")
    if transforms:
        df, _ = apply_transformations(df, transforms)

    target_col = feature_set.target_column
    feature_cols = [c for c in df.columns if c != target_col]
    problem_type = feature_set.problem_type or "regression"

    X, y, _ = prepare_features(df, feature_cols, target_col, problem_type)
    return X, y, feature_cols


def _prep_sidecar_path(model_path: str | None) -> Path | None:
    """Path of the persisted train-fold preprocessor for a model, if any.

    Saved next to the model as ``{model_run_id}.prep.joblib`` (see trainer #5).
    """
    if not model_path:
        return None
    p = Path(model_path)
    return p.with_name(f"{p.stem}.prep.joblib")


# Below this many held-out rows the per-bin / per-segment / calibration
# diagnostics aren't meaningful (calibration itself requires >=10 samples), so we
# fall back to the full dataset tagged "in_sample" rather than erroring on a
# sliver. Matches the trainer's MIN_HELDOUT_ROWS so a dataset big enough to carve
# a held-out split is big enough to diagnose on it.
_MIN_HELDOUT_EVAL_ROWS = 10


def _build_eval_Xy(
    run: ModelRun,
    file_path: Path,
    feature_set: FeatureSet,
) -> tuple[np.ndarray, np.ndarray, list[str], str, pd.DataFrame]:
    """Return the matrix diagnostics should score on + how it was obtained.

    When the run persisted held-out ``test_indices`` AND its preprocessor
    sidecar is present, load the cleaned raw frame, slice to the held-out rows,
    and transform with the **train-fold** preprocessor — so calibration /
    threshold / confusion / segment / error diagnostics are computed on data the
    model never trained on (issue #5).

    Falls back to the full dataset (tagged ``"in_sample"``) for legacy runs that
    predate held-out persistence, so old models still render — but honestly
    labelled, never silently inflated.

    Returns ``(X_numeric, y, feature_cols, evaluation, eval_df)`` where
    ``eval_df`` is the transformed dataframe rows aligned 1:1 with ``X``/``y``
    (so callers can pull display values / segment columns for exactly the rows
    scored), and ``evaluation`` is ``"held_out"`` or ``"in_sample"``.
    """
    df = pd.read_csv(file_path)
    transforms = json.loads(feature_set.transformations or "[]")
    if transforms:
        df, _ = apply_transformations(df, transforms)

    # Reproduce the EXACT frame ordering/sampling the trainer used so positional
    # test_indices line up (#5). api/models.py samples large datasets first, then
    # sorts chronologically — mirror that order or held-out rows would be
    # mis-identified for chronological / >50k-row runs.
    run_metrics = json.loads(run.metrics) if run.metrics else {}
    from core.trainer import sample_large_dataset as _sample_large_dataset

    df, _sample_info = _sample_large_dataset(df)
    if run_metrics.get("split_strategy") == "chronological" and run_metrics.get(
        "date_col_used"
    ):
        _date_col = run_metrics["date_col_used"]
        if _date_col in df.columns:
            df = df.sort_values(_date_col).reset_index(drop=True)

    target_col = feature_set.target_column
    feature_cols = [c for c in df.columns if c != target_col]
    problem_type = feature_set.problem_type or "regression"

    # Positions in the transformed df that survive target-dropna — both
    # extract_clean_xy and prepare_features drop the same rows, so this maps
    # cleaned-frame position -> transformed-df position for row alignment.
    kept = np.flatnonzero(df[target_col].notna().to_numpy())

    test_idx = json.loads(run.test_indices) if run.test_indices else None
    prep_path = _prep_sidecar_path(run.model_path)
    if test_idx and prep_path is not None and prep_path.exists():
        from core.preprocessing import extract_clean_xy

        X_raw, y, _le = extract_clean_xy(df, feature_cols, target_col, problem_type)
        # Guard against frame drift (dataset edited after training).
        valid = [i for i in test_idx if 0 <= i < len(X_raw) and i < len(kept)]
        if len(valid) >= _MIN_HELDOUT_EVAL_ROWS:
            prep = joblib.load(str(prep_path))
            X = prep.transform(X_raw.iloc[valid])
            eval_df = df.iloc[kept[valid]].reset_index(drop=True)
            return X, np.asarray(y)[valid], feature_cols, "held_out", eval_df

    # Legacy / too-few-held-out: full-data prediction (optimistic, but tagged).
    X, y, _ = prepare_features(df, feature_cols, target_col, problem_type)
    eval_df = df.iloc[kept].reset_index(drop=True)
    return X, y, feature_cols, "in_sample", eval_df


_IN_SAMPLE_NOTE = (
    "These diagnostics were computed on the training data (this model predates "
    "held-out tracking or the dataset is too small), so they are optimistic. "
    "Retrain to get held-out diagnostics."
)


def _eval_note(evaluation: str) -> str | None:
    """Plain-English caveat to attach when diagnostics are in-sample."""
    return None if evaluation == "held_out" else _IN_SAMPLE_NOTE


def _build_raw_xy(
    file_path: Path,
    feature_set: FeatureSet,
) -> tuple[pd.DataFrame, np.ndarray, list[str], str]:
    """Load CSV + transforms and return the **raw** (untransformed) feature
    frame, target, feature names, and problem type — for wrapping an unfitted
    preprocessor in a CV Pipeline so cross-validation stays leak-free (#5).
    """
    from core.preprocessing import extract_clean_xy

    df = pd.read_csv(file_path)
    transforms = json.loads(feature_set.transformations or "[]")
    if transforms:
        df, _ = apply_transformations(df, transforms)
    target_col = feature_set.target_column
    feature_cols = [c for c in df.columns if c != target_col]
    problem_type = feature_set.problem_type or "regression"
    X_raw, y, _le = extract_clean_xy(df, feature_cols, target_col, problem_type)
    return X_raw, y, feature_cols, problem_type


def _leakfree_cv(run: ModelRun, file_path: Path, feature_set: FeatureSet) -> dict:
    """Run cross-validation with preprocessing refit per fold (leak-free, #5)."""
    from sklearn.pipeline import Pipeline

    from core.preprocessing import build_preprocessor

    X_raw, y, feature_cols, problem_type = _build_raw_xy(file_path, feature_set)
    unfitted = _get_unfitted_model(run.algorithm, problem_type)
    cv_estimator = Pipeline(
        [("prep", build_preprocessor(X_raw, feature_cols)), ("model", unfitted)]
    )
    return run_cross_validation(cv_estimator, X_raw, y, problem_type)


def _get_unfitted_model(algorithm: str, problem_type: str):
    """Return a fresh unfitted estimator for cross-validation."""
    registry = (
        REGRESSION_ALGORITHMS
        if problem_type == "regression"
        else CLASSIFICATION_ALGORITHMS
    )
    if algorithm not in registry:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithm '{algorithm}'",
        )
    info = registry[algorithm]
    return info["class"](**info["params"])


# ---------------------------------------------------------------------------
# 1. Validation metrics
# ---------------------------------------------------------------------------


@router.get("/api/validate/{model_run_id}/metrics")
def get_validation_metrics(
    model_run_id: str,
    session: Session = Depends(get_session),
):
    """Cross-validation + confusion matrix (classification) or residuals (regression)."""
    run, feature_set, _dataset, file_path = _load_run_context(model_run_id, session)

    problem_type = feature_set.problem_type or "regression"
    metrics = json.loads(run.metrics) if run.metrics else {}

    # Cross-validation with preprocessing refit per fold (leak-free, #5).
    cv_result = _leakfree_cv(run, file_path, feature_set)

    # Confusion matrix / residuals on the held-out slice (never in-sample, #5).
    X, y, feature_cols, evaluation, _eval_df = _build_eval_Xy(
        run, file_path, feature_set
    )
    fitted_model = joblib.load(run.model_path)
    y_pred = fitted_model.predict(X)

    if problem_type == "classification":
        cm_result = compute_confusion_matrix(y, y_pred)
        error_analysis = {"type": "confusion_matrix", **cm_result}
    else:
        residual_result = compute_residuals(y, y_pred)
        error_analysis = {"type": "residuals", **residual_result}

    # Confidence assessment
    cv_std = cv_result.get("std")
    confidence = assess_confidence_limitations(
        metrics=metrics,
        problem_type=problem_type,
        n_rows=len(X),
        n_features=len(feature_cols),
        cv_std=cv_std,
    )

    return {
        "model_run_id": model_run_id,
        "algorithm": run.algorithm,
        "problem_type": problem_type,
        "held_out_metrics": metrics,
        "cross_validation": cv_result,
        "error_analysis": error_analysis,
        "evaluation": evaluation,
        "evaluation_note": _eval_note(evaluation),
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# 2. Global feature importance
# ---------------------------------------------------------------------------


@router.get("/api/validate/{model_run_id}/explain")
def get_global_explanation(
    model_run_id: str,
    session: Session = Depends(get_session),
):
    """Return global feature importance for the trained model."""
    run, feature_set, _dataset, file_path = _load_run_context(model_run_id, session)

    problem_type = feature_set.problem_type or "regression"
    X, y, feature_cols = _build_Xy(file_path, feature_set)

    fitted_model = joblib.load(run.model_path)
    importance = compute_feature_importance(fitted_model, feature_cols)

    top_3 = [item["feature"] for item in importance[:3]]
    summary = (
        f"The top factors driving predictions are: {', '.join(top_3)}. "
        "Higher bars mean a feature has more influence on the model's output."
    )

    return {
        "model_run_id": model_run_id,
        "algorithm": run.algorithm,
        "problem_type": problem_type,
        "feature_importance": importance,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 3. Individual prediction explanation
# ---------------------------------------------------------------------------


@router.get("/api/validate/{model_run_id}/explain/{row_index}")
def get_row_explanation(
    model_run_id: str,
    row_index: int,
    session: Session = Depends(get_session),
):
    """Explain a single prediction using feature contributions."""
    run, feature_set, _dataset, file_path = _load_run_context(model_run_id, session)

    problem_type = feature_set.problem_type or "regression"
    target_col = feature_set.target_column or "target"
    X, y, feature_cols = _build_Xy(file_path, feature_set)

    if row_index < 0 or row_index >= len(X):
        raise HTTPException(
            status_code=400,
            detail=f"row_index {row_index} out of range (dataset has {len(X)} rows).",
        )

    fitted_model = joblib.load(run.model_path)
    x_row = X[row_index]

    explanation = explain_single_prediction(
        model=fitted_model,
        x_row=x_row,
        X_train=X,
        feature_names=feature_cols,
        problem_type=problem_type,
        target_name=target_col,
    )

    return {
        "model_run_id": model_run_id,
        "row_index": row_index,
        "actual_value": round(float(y[row_index]), 4) if y is not None else None,
        **explanation,
    }


# ---------------------------------------------------------------------------
# 4. Segment performance breakdown
# ---------------------------------------------------------------------------


@router.get("/api/models/{model_run_id}/segment-performance")
def get_segment_performance(
    model_run_id: str,
    col: str,
    session: Session = Depends(get_session),
):
    """Break down model performance (R² or accuracy) by a categorical column.

    The column must exist in the original dataset (before feature transforms).
    Groups with fewer than 2 rows are excluded from metric computation.
    """
    run, feature_set, dataset, file_path = _load_run_context(model_run_id, session)

    # Load raw CSV to get group column values (before transforms)
    df_raw = pd.read_csv(file_path)
    if col not in df_raw.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{col}' not found in dataset. "
            f"Available columns: {', '.join(df_raw.columns.tolist())}",
        )

    # Cap cardinality — prevent useless breakdown on high-cardinality or near-unique columns
    n_unique = df_raw[col].nunique()
    n_rows = len(df_raw)
    is_high_cardinality = n_unique > 50
    is_near_unique = n_rows > 5 and n_unique >= n_rows * 0.8
    if is_high_cardinality or is_near_unique:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column '{col}' has {n_unique} unique values — too many to break down. "
                "Choose a categorical column with a small number of distinct values "
                "(e.g., region, product type, segment)."
            ),
        )

    # Build held-out features + predictions (never in-sample, #5).
    problem_type = feature_set.problem_type or "regression"
    X, y, _feature_cols, evaluation, eval_df = _build_eval_Xy(
        run, file_path, feature_set
    )

    fitted_model = joblib.load(run.model_path)
    y_pred = fitted_model.predict(X)

    # Pull group values for exactly the scored rows. ``eval_df`` is aligned 1:1
    # with X/y; fall back to the raw column if a transform dropped it.
    if col in eval_df.columns:
        group_values = eval_df[col].tolist()
    else:
        group_values = df_raw[col].tolist()[: len(y)]

    result = compute_segment_performance(
        group_values=group_values,
        y_true=y,
        y_pred=y_pred,
        problem_type=problem_type,
    )

    return {
        "model_run_id": model_run_id,
        "group_col": col,
        "algorithm": run.algorithm,
        "problem_type": problem_type,
        "evaluation": evaluation,
        "evaluation_note": _eval_note(evaluation),
        **result,
    }


# ---------------------------------------------------------------------------
# 4. Prediction error analysis
# ---------------------------------------------------------------------------


@router.get("/api/models/{model_run_id}/prediction-errors")
def get_prediction_errors(
    model_run_id: str,
    n: int = 10,
    session: Session = Depends(get_session),
):
    """Return the top-N largest prediction errors on training data.

    For regression: rows with largest absolute residuals.
    For classification: rows where the prediction was wrong.

    Query params:
        n: Number of error rows to return (1-50, default 10).
    """
    n = max(1, min(50, n))
    run, feature_set, dataset, file_path = _load_run_context(model_run_id, session)

    problem_type = feature_set.problem_type or "regression"
    X, y, feature_cols, evaluation, eval_df = _build_eval_Xy(
        run, file_path, feature_set
    )

    fitted_model = joblib.load(run.model_path)
    y_pred = fitted_model.predict(X)

    # Build display rows from ``eval_df`` (aligned 1:1 with the scored rows) so
    # analysts see original values for exactly the held-out errors.
    target_col = feature_set.target_column
    display_cols = [c for c in feature_cols if c in eval_df.columns]
    feature_rows = [
        {col: row[col] for col in display_cols if col in row}
        for row in eval_df.to_dict(orient="records")
    ]

    # Resolve classification class labels from pipeline if available
    from core.deployer import load_pipeline as _lp

    target_classes = None
    if run.model_path:
        pipeline_path = run.model_path.replace("_model.joblib", "_pipeline.joblib")
        try:
            from pathlib import Path as _Path

            if _Path(pipeline_path).exists():
                _pipe = _lp(pipeline_path)
                target_classes = getattr(_pipe, "target_classes", None)
        except Exception:  # noqa: BLE001
            pass

    result = compute_prediction_errors(
        y_true=y,
        y_pred=y_pred,
        problem_type=problem_type,
        n=n,
        feature_rows=feature_rows,
        target_classes=target_classes,
    )

    return {
        "model_run_id": model_run_id,
        "algorithm": run.algorithm,
        "problem_type": problem_type,
        "target_col": target_col,
        "n_requested": n,
        "evaluation": evaluation,
        "evaluation_note": _eval_note(evaluation),
        **result,
    }


# ---------------------------------------------------------------------------
# 5. Prediction error distribution
# ---------------------------------------------------------------------------


@router.get("/api/models/{model_run_id}/error-distribution")
def get_error_distribution(
    model_run_id: str,
    n_bins: int = 10,
    session: Session = Depends(get_session),
):
    """Return the distribution of prediction errors across all training rows.

    For regression: histogram of residuals with bias and spread stats.
    For classification: per-class error rates showing which classes struggle.

    Query params:
        n_bins: Histogram bin count for regression (5-30, default 10).
    """
    n_bins = max(5, min(30, n_bins))
    run, feature_set, dataset, file_path = _load_run_context(model_run_id, session)

    problem_type = feature_set.problem_type or "regression"
    X, y, feature_cols, evaluation, _eval_df = _build_eval_Xy(
        run, file_path, feature_set
    )

    fitted_model = joblib.load(run.model_path)
    y_pred = fitted_model.predict(X)

    # Resolve class labels from pipeline if available
    target_classes = None
    if run.model_path:
        pipeline_path = run.model_path.replace("_model.joblib", "_pipeline.joblib")
        try:
            from core.deployer import load_pipeline as _lp

            if Path(pipeline_path).exists():
                _pipe = _lp(pipeline_path)
                target_classes = getattr(_pipe, "target_classes", None)
        except Exception:  # noqa: BLE001
            pass

    result = compute_error_distribution(
        y_true=y,
        y_pred=y_pred,
        problem_type=problem_type,
        n_bins=n_bins,
        target_classes=target_classes,
    )

    result["model_run_id"] = model_run_id
    result["algorithm"] = run.algorithm
    result["target_col"] = feature_set.target_column
    result["evaluation"] = evaluation
    result["evaluation_note"] = _eval_note(evaluation)

    return result


# ---------------------------------------------------------------------------
# 6. Partial dependence plot
# ---------------------------------------------------------------------------


@router.get("/api/models/{model_run_id}/partial-dependence")
def get_partial_dependence(
    model_run_id: str,
    feature: str,
    steps: int = 20,
    session: Session = Depends(get_session),
):
    """Compute partial dependence of model output on one feature.

    Unlike sensitivity analysis (which fixes all other features at training means),
    PDP averages over the actual training distribution — giving a more accurate
    marginal effect estimate.

    Args:
        feature: Column name to sweep.
        steps:   Number of evenly-spaced grid points between p5 and p95 of training values.
                 Clamped to [5, 50].

    Returns JSON with grid_values, mean_predictions, std_predictions, and a summary.
    """
    run, feature_set, _dataset, file_path = _load_run_context(model_run_id, session)

    X, y, feature_cols = _build_Xy(file_path, feature_set)

    if feature not in feature_cols:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Feature '{feature}' not found in the trained model's feature set. "
                f"Available features: {', '.join(feature_cols)}"
            ),
        )

    feature_idx = feature_cols.index(feature)
    steps = max(5, min(50, steps))

    # Build sweep grid from p5 to p95 of training values (avoids extreme extrapolation)
    col_vals = X[:, feature_idx]
    p5 = float(np.percentile(col_vals, 5))
    p95 = float(np.percentile(col_vals, 95))
    if p5 == p95:
        # Constant feature — return trivial result
        grid = np.array([p5])
    else:
        grid = np.linspace(p5, p95, steps)

    problem_type = feature_set.problem_type or "regression"
    fitted_model = joblib.load(run.model_path)

    # Resolve class names for multiclass classification
    class_names: list[str] | None = None
    pipeline_path = run.model_path.replace("_model.joblib", "_pipeline.joblib")
    if Path(pipeline_path).exists():
        try:
            _pipe = joblib.load(pipeline_path)
            class_names = getattr(_pipe, "target_classes", None)
        except Exception:  # noqa: BLE001
            pass

    result = compute_partial_dependence(
        model=fitted_model,
        X_train=X,
        feature_idx=feature_idx,
        grid_values=grid,
        problem_type=problem_type,
        class_names=class_names,
    )

    return {
        "model_run_id": model_run_id,
        "feature": feature,
        "feature_idx": feature_idx,
        "algorithm": run.algorithm,
        "target_col": feature_set.target_column,
        **result,
    }


# ---------------------------------------------------------------------------
# 6. Fairness / bias analysis
# ---------------------------------------------------------------------------


@router.get("/api/models/{model_run_id}/fairness")
def get_fairness_metrics(
    model_run_id: str,
    col: str,
    session: Session = Depends(get_session),
):
    """Return fairness metrics across groups defined by a sensitive column.

    For classification: SPD, DIR, per-group accuracy and positive-prediction rate.
    For regression: per-group MAE and MAE disparity ratio.

    Query params:
      col: Column name to use as the sensitive attribute (e.g. "region", "gender").
    """
    run, feature_set, _dataset, file_path = _load_run_context(model_run_id, session)

    problem_type = feature_set.problem_type or "regression"

    # Load the raw CSV to get sensitive column values before encoding
    df_raw = pd.read_csv(file_path)

    if col not in df_raw.columns:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column '{col}' not found in the dataset. "
                f"Available columns: {', '.join(df_raw.columns.tolist())}"
            ),
        )

    # Cardinality guard — same limits as segment performance
    n_unique = df_raw[col].nunique()
    if n_unique > 50:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column '{col}' has {n_unique} unique values (max 50 for fairness analysis). "
                "Choose a column with fewer distinct groups (e.g. region, department, category)."
            ),
        )
    if n_unique < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Column '{col}' has only {n_unique} unique value — need at least 2 groups."
            ),
        )

    # Build held-out X, y for predictions (never in-sample, #5).
    X, y, feature_cols, evaluation, eval_df = _build_eval_Xy(
        run, file_path, feature_set
    )

    fitted_model = joblib.load(run.model_path)
    y_pred = fitted_model.predict(X)

    # Sensitive column for exactly the scored rows (eval_df is aligned 1:1).
    target_col = feature_set.target_column
    if col in eval_df.columns:
        sensitive_vals = eval_df[col].values
    else:
        df_aligned = df_raw.dropna(subset=[target_col]).reset_index(drop=True)
        sensitive_vals = df_aligned[col].values[: len(y)]

    result = compute_fairness_metrics(
        y_true=y,
        y_pred=y_pred,
        sensitive_values=np.array(sensitive_vals),
        problem_type=problem_type,
    )

    result["sensitive_col"] = col
    result["model_run_id"] = model_run_id
    result["algorithm"] = run.algorithm
    result["target_col"] = target_col
    result["evaluation"] = evaluation
    result["evaluation_note"] = _eval_note(evaluation)

    return result


# ---------------------------------------------------------------------------
# 8. Classification threshold analysis
# ---------------------------------------------------------------------------


@router.get("/api/models/{model_run_id}/threshold-analysis")
def get_threshold_analysis(
    model_run_id: str,
    session: Session = Depends(get_session),
):
    """Return precision/recall/F1 sweep across probability thresholds (0.05–0.95).

    Only available for binary classification models with predict_proba support.
    Returns 400 for regression models or models without probability output.
    """
    run, feature_set, _dataset, file_path = _load_run_context(model_run_id, session)

    problem_type = feature_set.problem_type or "regression"
    if problem_type != "classification":
        raise HTTPException(
            status_code=400,
            detail="Threshold analysis is only available for classification models.",
        )

    X, y, _feature_cols, evaluation, _eval_df = _build_eval_Xy(
        run, file_path, feature_set
    )

    fitted_model = joblib.load(run.model_path)

    if not hasattr(fitted_model, "predict_proba"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Algorithm '{run.algorithm}' does not support probability output "
                "required for threshold analysis."
            ),
        )

    proba = fitted_model.predict_proba(X)

    # Binary: use probability of positive class (index 1)
    # Multiclass: use maximum probability across classes (confidence proxy)
    if proba.shape[1] == 2:
        y_proba = proba[:, 1]
        classes = fitted_model.classes_.tolist()
        class_names = [str(c) for c in classes]
    else:
        y_proba = proba.max(axis=1)
        classes = fitted_model.classes_.tolist()
        class_names = [str(c) for c in classes]

    result = compute_threshold_analysis(
        y_true=y,
        y_proba=y_proba,
        class_names=class_names,
    )

    result["model_run_id"] = model_run_id
    result["algorithm"] = run.algorithm
    result["target_col"] = feature_set.target_column
    result["evaluation"] = evaluation
    result["evaluation_note"] = _eval_note(evaluation)

    return result


# ---------------------------------------------------------------------------
# 9. Per-class threshold analysis (multiclass one-vs-rest)
# ---------------------------------------------------------------------------


@router.get("/api/models/{model_run_id}/per-class-threshold")
def get_per_class_threshold(
    model_run_id: str,
    session: Session = Depends(get_session),
):
    """Return per-class optimal thresholds via one-vs-rest sweep.

    For each class independently finds the confidence threshold (0.05–0.95)
    that maximises F1 score in a one-vs-rest binary framing.

    Only available for classification models with predict_proba support.
    Returns 400 for regression models, models without probability output,
    or binary models (use /threshold-analysis for binary).
    """
    run, feature_set, _dataset, file_path = _load_run_context(model_run_id, session)

    problem_type = feature_set.problem_type or "regression"
    if problem_type != "classification":
        raise HTTPException(
            status_code=400,
            detail="Per-class threshold analysis is only available for classification models.",
        )

    X, y, _feature_cols, evaluation, _eval_df = _build_eval_Xy(
        run, file_path, feature_set
    )

    fitted_model = joblib.load(run.model_path)

    if not hasattr(fitted_model, "predict_proba"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Algorithm '{run.algorithm}' does not support probability output "
                "required for per-class threshold analysis."
            ),
        )

    proba = fitted_model.predict_proba(X)
    classes = fitted_model.classes_.tolist()
    class_names = [str(c) for c in classes]

    result = compute_per_class_threshold_analysis(
        y_true=y,
        y_proba=proba,
        class_names=class_names,
    )

    result["model_run_id"] = model_run_id
    result["algorithm"] = run.algorithm
    result["target_col"] = feature_set.target_column
    result["evaluation"] = evaluation
    result["evaluation_note"] = _eval_note(evaluation)

    return result


# ---------------------------------------------------------------------------
# 10. Confidence distribution analysis
# ---------------------------------------------------------------------------


@router.get("/api/models/{model_run_id}/confidence-distribution")
def get_confidence_distribution(
    model_run_id: str,
    n_bins: int = 10,
    session: Session = Depends(get_session),
):
    """Return the distribution of prediction confidence (max-class probability) across training data.

    Only available for classification models with predict_proba support.
    Returns 400 for regression models or models without probability output.

    Query params:
        n_bins: Histogram bin count (5-20, default 10).
    """
    n_bins = max(5, min(20, n_bins))
    run, feature_set, _dataset, file_path = _load_run_context(model_run_id, session)

    problem_type = feature_set.problem_type or "regression"
    if problem_type != "classification":
        raise HTTPException(
            status_code=400,
            detail="Confidence distribution is only available for classification models.",
        )

    X, y, _feature_cols, evaluation, _eval_df = _build_eval_Xy(
        run, file_path, feature_set
    )

    fitted_model = joblib.load(run.model_path)

    if not hasattr(fitted_model, "predict_proba"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Algorithm '{run.algorithm}' does not support probability output "
                "required for confidence distribution."
            ),
        )

    proba = fitted_model.predict_proba(X)
    y_proba_cd = proba.max(axis=1)
    y_pred_cd = fitted_model.predict(X)

    class_names_cd = [str(c) for c in fitted_model.classes_.tolist()]

    result = compute_confidence_distribution(
        y_proba=y_proba_cd,
        y_pred=y_pred_cd,
        class_names=class_names_cd,
        n_bins=n_bins,
    )

    result["model_run_id"] = model_run_id
    result["algorithm"] = run.algorithm
    result["target_col"] = feature_set.target_column
    result["evaluation"] = evaluation
    result["evaluation_note"] = _eval_note(evaluation)

    return result


@router.get("/api/models/{model_run_id}/class-feature-importance")
async def get_class_feature_importance(
    model_run_id: str,
    session: Session = Depends(get_session),
):
    """Per-predicted-class feature importance breakdown.

    For each class the model predicts, shows which features deviate most from the
    global training mean (importance-weighted), answering "what makes the model
    predict X vs Y?"

    Classification only. Returns 400 for regression runs.
    """
    run, feature_set, _dataset, file_path = _load_run_context(model_run_id, session)

    problem_type = feature_set.problem_type or "regression"
    if problem_type != "classification":
        raise HTTPException(
            status_code=400,
            detail="Class-conditional feature importance is only available for classification models.",
        )

    X, y, feature_cols = _build_Xy(file_path, feature_set)

    fitted_model = joblib.load(run.model_path)
    y_pred = fitted_model.predict(X)

    class_names: list[str] | None = None
    if hasattr(fitted_model, "classes_"):
        class_names = [str(c) for c in fitted_model.classes_.tolist()]

    try:
        result = compute_class_conditional_importance(
            model=fitted_model,
            X=X,
            y_pred=y_pred,
            feature_names=feature_cols,
            class_names=class_names,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["model_run_id"] = model_run_id
    result["algorithm"] = run.algorithm
    result["target_col"] = feature_set.target_column

    return result


@router.get("/api/models/{model_run_id}/calibration-check")
async def get_calibration_check(
    model_run_id: str,
    n_bins: int = 10,
    session: Session = Depends(get_session),
):
    """Check how well the model's predicted probabilities match actual outcome rates.

    A well-calibrated model that predicts 70% probability is correct ~70% of the time.
    Returns a reliability diagram (calibration curve), Expected Calibration Error (ECE),
    and Brier score.

    Classification only. Returns 400 for regression or non-probabilistic models.

    Query params:
        n_bins: Calibration bin count (5-20, default 10).
    """
    n_bins = max(5, min(20, n_bins))
    run, feature_set, _dataset, file_path = _load_run_context(model_run_id, session)

    problem_type = feature_set.problem_type or "regression"
    if problem_type != "classification":
        raise HTTPException(
            status_code=400,
            detail="Calibration check is only available for classification models.",
        )

    X, y, _feature_cols, evaluation, _eval_df = _build_eval_Xy(
        run, file_path, feature_set
    )

    fitted_model = joblib.load(run.model_path)

    if not hasattr(fitted_model, "predict_proba"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Algorithm '{run.algorithm}' does not support probability output "
                "required for calibration check."
            ),
        )

    y_proba_matrix = fitted_model.predict_proba(X)
    class_names = [str(c) for c in fitted_model.classes_.tolist()]

    try:
        result = compute_calibration_check(
            y_true=y,
            y_proba_matrix=y_proba_matrix,
            class_names=class_names,
            n_bins=n_bins,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["model_run_id"] = model_run_id
    result["algorithm"] = run.algorithm
    result["target_col"] = feature_set.target_column
    result["evaluation"] = evaluation
    result["evaluation_note"] = _eval_note(evaluation)

    return result


# ---------------------------------------------------------------------------
# 12. Prediction error correlation
# ---------------------------------------------------------------------------


@router.get("/api/models/{model_run_id}/error-correlation")
def get_error_correlation(
    model_run_id: str,
    session: Session = Depends(get_session),
):
    """Identify which input features most correlate with prediction errors.

    For regression: correlates each feature with the absolute residual.
    For classification: correlates each feature with whether the prediction was wrong.

    Returns 404 for unknown model run.
    """
    run, feature_set, _dataset, file_path = _load_run_context(model_run_id, session)

    problem_type = feature_set.problem_type or "regression"
    X, y, feature_cols, evaluation, _eval_df = _build_eval_Xy(
        run, file_path, feature_set
    )

    fitted_model = joblib.load(run.model_path)
    y_pred = fitted_model.predict(X)

    try:
        result = compute_prediction_error_correlation(
            X=X,
            y_true=y,
            y_pred=y_pred,
            feature_names=feature_cols,
            problem_type=problem_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["model_run_id"] = model_run_id
    result["algorithm"] = run.algorithm
    result["target_col"] = feature_set.target_column
    result["evaluation"] = evaluation
    result["evaluation_note"] = _eval_note(evaluation)

    return result


# ---------------------------------------------------------------------------
# 13. Minimum viable feature set
# ---------------------------------------------------------------------------


@router.get("/api/models/{model_run_id}/min-feature-set")
def get_min_feature_set(
    model_run_id: str,
    tolerance: float = 0.02,
    session: Session = Depends(get_session),
):
    """Greedy backward elimination: find the smallest feature subset that
    preserves model accuracy within ``tolerance``.

    Returns 404 for unknown model run, 400 for unsupported conditions.
    """
    run, feature_set, _dataset, file_path = _load_run_context(model_run_id, session)

    if tolerance < 0 or tolerance > 1:
        raise HTTPException(
            status_code=400,
            detail="tolerance must be between 0 and 1 (e.g., 0.02 for 2%).",
        )

    problem_type = feature_set.problem_type or "regression"
    X, y, feature_cols = _build_Xy(file_path, feature_set)

    registry = (
        REGRESSION_ALGORITHMS
        if problem_type == "regression"
        else CLASSIFICATION_ALGORITHMS
    )
    if run.algorithm not in registry:
        raise HTTPException(
            status_code=400,
            detail=f"Algorithm '{run.algorithm}' is not in the registry.",
        )
    algo_info = registry[run.algorithm]
    model_class = algo_info["class"]
    model_params = algo_info["params"]

    try:
        result = compute_min_viable_feature_set(
            X=X,
            y=y,
            feature_names=feature_cols,
            model_class=model_class,
            model_params=model_params,
            problem_type=problem_type,
            tolerance=tolerance,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["model_run_id"] = model_run_id
    result["algorithm"] = run.algorithm
    result["target_col"] = feature_set.target_column

    return result
