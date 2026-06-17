"""Regression tests for train/test data leakage (issue #5).

These pin the core promise of the #5 fix: imputation/encoding is fit on the
training fold ONLY, so reported held-out metrics are not inflated by test-fold
statistics, and the unfitted preprocessor handed to cross-validation refits per
fold. Also covers the n<10 in-sample honesty labelling.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from core.preprocessing import build_preprocessor, extract_clean_xy
from core.trainer import _split_and_preprocess, train_single_model


def _leaky_signal_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """A learnable regression frame with NaNs + a categorical feature.

    Numeric features carry real signal; ``region`` is informative too. NaNs in a
    numeric column force median imputation — the classic leakage vector.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    region = rng.choice(["north", "south", "east", "west"], n)
    region_effect = pd.Series(region).map(
        {"north": 1.0, "south": -1.0, "east": 2.0, "west": 0.0}
    )
    y = 3 * x1 + 0.5 * x2 + region_effect.to_numpy() + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "region": region, "target": y})
    # Punch holes so imputation actually does something.
    nan_idx = rng.choice(n, size=n // 10, replace=False)
    df.loc[nan_idx, "x2"] = np.nan
    return df


# ---------------------------------------------------------------------------
# AC1: preprocessing is fit on the training fold only
# ---------------------------------------------------------------------------


def test_preprocessor_for_cv_is_unfitted_and_refit_per_fold():
    """The Pipeline handed to cross_val_score must be unfitted, so each fold
    refits its own preprocessing (no global statistics leak)."""
    df = _leaky_signal_df()
    feature_cols = ["x1", "x2", "region"]
    X_raw, y, _le = extract_clean_xy(df, feature_cols, "target", "regression")

    prep = build_preprocessor(X_raw, feature_cols)
    # Unfitted before CV.
    try:
        check_is_fitted(prep)
        fitted = True
    except Exception:
        fitted = False
    assert not fitted

    from sklearn.linear_model import LinearRegression

    pipe = Pipeline([("prep", prep), ("model", LinearRegression())])
    scores = cross_val_score(pipe, X_raw, y, cv=5, scoring="r2")
    assert len(scores) == 5
    # The transformer object we passed in stays unfitted (clones run per fold).
    try:
        check_is_fitted(prep)
        still_fitted = True
    except Exception:
        still_fitted = False
    assert not still_fitted


def test_split_and_preprocess_fits_only_on_training_rows():
    """The fitted preprocessor's median must equal the TRAIN-fold median, not
    the whole-column median — proving test rows didn't leak into imputation."""
    df = _leaky_signal_df(n=120, seed=3)
    feature_cols = ["x1", "x2", "region"]
    X_raw, y, _le = extract_clean_xy(df, feature_cols, "target", "regression")

    (
        _X_train,
        _X_test,
        _y_train,
        _y_test,
        prep,
        test_indices,
        has_heldout,
    ) = _split_and_preprocess(X_raw, np.asarray(y), "random", feature_cols)

    assert has_heldout
    assert test_indices is not None
    train_idx = [i for i in range(len(X_raw)) if i not in set(test_indices)]
    train_median = X_raw.iloc[train_idx]["x2"].median()
    full_median = X_raw["x2"].median()

    # SimpleImputer for x2 stores the train-fold median.
    assert isinstance(prep, ColumnTransformer)
    medians = {}
    for _name, trans, cols in prep.transformers_:
        if cols == ["x2"] and hasattr(trans, "statistics_"):
            medians["x2"] = float(trans.statistics_[0])
    assert "x2" in medians
    assert abs(medians["x2"] - float(train_median)) < 1e-9
    # And in this dataset the two medians differ, so this is a real check.
    assert abs(full_median - train_median) > 0 or len(test_indices) == 0


# ---------------------------------------------------------------------------
# AC1/AC4: leak-free held-out metric is not inflated vs a manual baseline
# ---------------------------------------------------------------------------


def test_heldout_metric_matches_manual_leakfree_baseline():
    """The trainer's reported held-out R² must equal a hand-rolled leak-free
    computation (split first → fit prep on train → score test), confirming no
    optimistic inflation from global preprocessing."""
    df = _leaky_signal_df(n=200, seed=7)
    feature_cols = ["x1", "x2", "region"]
    X_raw, y, _le = extract_clean_xy(df, feature_cols, "target", "regression")
    y = np.asarray(y)

    with tempfile.TemporaryDirectory() as tmp:
        result = train_single_model(
            X_raw,
            y,
            "linear_regression",
            "regression",
            Path(tmp),
            "leak_test",
            feature_cols=feature_cols,
        )

    reported_r2 = result["metrics"]["r2"]
    assert result["metrics"]["evaluation"] == "held_out"
    assert result["test_indices"] is not None

    # Manual leak-free baseline using the SAME split the trainer used.
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    test_idx = result["test_indices"]
    train_idx = [i for i in range(len(X_raw)) if i not in set(test_idx)]
    prep = build_preprocessor(X_raw, feature_cols)
    prep.fit(X_raw.iloc[train_idx])
    Xtr = prep.transform(X_raw.iloc[train_idx])
    Xte = prep.transform(X_raw.iloc[test_idx])
    m = LinearRegression().fit(Xtr, y[train_idx])
    manual_r2 = round(float(r2_score(y[test_idx], m.predict(Xte))), 4)

    assert abs(reported_r2 - manual_r2) < 1e-6


def test_global_fit_path_diverges_from_leakfree():
    """Documents the bug: fitting median imputation on the FULL data before the
    split (the old behaviour) feeds test-fold information into the test rows'
    imputed values, so the held-out score differs from the leak-free path. We
    construct the split first, then place NaNs and a distribution skew so the
    full-data median and the train-fold median genuinely differ — proving the
    leak was real and is now closed."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    from core.preprocessing import build_preprocessor as _bp

    n = 60
    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.2, random_state=42)
    rng = np.random.default_rng(11)
    # Underlying signal for every row.
    true_x = np.empty(n)
    true_x[train_idx] = rng.normal(0, 1, len(train_idx))
    true_x[test_idx] = rng.normal(50, 1, len(test_idx))
    y = 2 * true_x + rng.normal(0, 0.1, n)

    # Observed x: train fully observed (low). Of the test rows, half are
    # OBSERVED high (these pull the full-data median up — the leak source) and
    # half are NaN (imputed with whichever median is used).
    x = true_x.copy()
    test_half = test_idx[: len(test_idx) // 2]
    x[test_half] = np.nan
    df = pd.DataFrame({"x": x, "target": y})
    feature_cols = ["x"]
    X_raw, yv, _ = extract_clean_xy(df, feature_cols, "target", "regression")
    yv = np.asarray(yv)

    # Leak-free: fit prep on train only (test rows imputed with TRAIN median).
    p_free = _bp(X_raw, feature_cols)
    p_free.fit(X_raw.iloc[train_idx])
    free_pred = (
        LinearRegression()
        .fit(p_free.transform(X_raw.iloc[train_idx]), yv[train_idx])
        .predict(p_free.transform(X_raw.iloc[test_idx]))
    )
    r2_free = r2_score(yv[test_idx], free_pred)

    # Leaky: fit prep on the FULL data, then split the transformed matrix.
    p_leak = _bp(X_raw, feature_cols)
    Xall = p_leak.fit_transform(X_raw)
    leak_pred = (
        LinearRegression().fit(Xall[train_idx], yv[train_idx]).predict(Xall[test_idx])
    )
    r2_leak = r2_score(yv[test_idx], leak_pred)

    # The leak changes the imputed test values -> the two scores diverge.
    assert abs(r2_free - r2_leak) > 1e-6


# ---------------------------------------------------------------------------
# AC3: n<10 is in-sample and never labelled "held-out"
# ---------------------------------------------------------------------------


def test_tiny_dataset_is_tagged_in_sample_not_heldout():
    df = _leaky_signal_df(n=6, seed=1)
    feature_cols = ["x1", "x2", "region"]
    X_raw, y, _le = extract_clean_xy(df, feature_cols, "target", "regression")

    with tempfile.TemporaryDirectory() as tmp:
        result = train_single_model(
            X_raw,
            np.asarray(y),
            "linear_regression",
            "regression",
            Path(tmp),
            "tiny",
            feature_cols=feature_cols,
        )

    assert result["metrics"]["evaluation"] == "in_sample"
    assert result["metrics"].get("n_too_small") is True
    assert result["test_indices"] is None
    assert "held-out test set" not in result["summary"]
    assert "training data" in result["summary"]
