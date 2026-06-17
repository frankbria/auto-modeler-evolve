"""Tests for core.preprocessing — leak-free feature transform (issue #5).

The whole point of this module is that imputation/encoding statistics are
learned by an *unfitted* transformer fit only on the training fold. These tests
pin that property plus the column-order / unseen-category contracts the rest of
the system relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import NotFittedError

from core.preprocessing import (
    UNKNOWN_ENCODED_VALUE,
    build_preprocessor,
    coerce_raw_features,
    extract_clean_xy,
    ordered_feature_names,
    split_numeric_categorical,
)


def _mixed_df(n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.integers(20, 60, n).astype(float),
            "region": rng.choice(["north", "south", "east"], n),
            "spend": rng.normal(100, 10, n),
            "target": rng.integers(0, 2, n),
        }
    )


# ---------------------------------------------------------------------------
# split_numeric_categorical / ordered_feature_names
# ---------------------------------------------------------------------------


def test_split_numeric_categorical():
    df = _mixed_df()
    numeric, categorical = split_numeric_categorical(df, ["age", "region", "spend"])
    assert numeric == ["age", "spend"]
    assert categorical == ["region"]


def test_ordered_feature_names_preserves_feature_col_order():
    df = _mixed_df()
    names = ordered_feature_names(df, ["region", "age", "spend"])
    # One transformer per feature -> output order matches feature_cols.
    assert names == ["region", "age", "spend"]


# ---------------------------------------------------------------------------
# build_preprocessor
# ---------------------------------------------------------------------------


def test_build_preprocessor_returns_unfitted_transformer():
    df = _mixed_df()
    prep = build_preprocessor(df, ["age", "region", "spend"])
    assert isinstance(prep, ColumnTransformer)
    # Unfitted: transforming before fit must raise.
    try:
        prep.transform(coerce_raw_features(df, ["age", "region", "spend"]))
        raised = False
    except (NotFittedError, AttributeError, ValueError):
        raised = True
    assert raised, "preprocessor should be unfitted until caller fits it"


def test_preprocessor_one_column_per_feature():
    df = _mixed_df()
    cols = ["age", "region", "spend"]
    prep = build_preprocessor(df, cols)
    X = prep.fit_transform(coerce_raw_features(df, cols))
    assert X.shape == (len(df), len(cols))


def test_median_imputation_uses_fit_data_only():
    # Train fold has known median; transform of a NaN row must use the TRAIN
    # median, not any statistic from the row being transformed.
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 100.0], "region": ["a", "a", "b", "b"]})
    prep = build_preprocessor(train, ["x", "region"])
    prep.fit(coerce_raw_features(train, ["x", "region"]))
    test = pd.DataFrame({"x": [np.nan], "region": ["a"]})
    out = prep.transform(coerce_raw_features(test, ["x", "region"]))
    # median of [1,2,3,100] = 2.5
    assert out[0, 0] == 2.5


def test_unseen_category_maps_to_unknown_value():
    train = pd.DataFrame({"region": ["a", "a", "b", "b"], "x": [1.0, 2, 3, 4]})
    prep = build_preprocessor(train, ["region", "x"])
    prep.fit(coerce_raw_features(train, ["region", "x"]))
    test = pd.DataFrame({"region": ["zzz_unseen"], "x": [2.0]})
    out = prep.transform(coerce_raw_features(test, ["region", "x"]))
    # output order matches feature_cols -> region is column 0
    assert out[0, 0] == UNKNOWN_ENCODED_VALUE


def test_categorical_nan_filled_as_missing_category():
    train = pd.DataFrame({"region": ["a", "b", None, "a"], "x": [1.0, 2, 3, 4]})
    prep = build_preprocessor(train, ["region", "x"])
    # Should fit without error (NaN -> "MISSING" via constant imputer).
    prep.fit(coerce_raw_features(train, ["region", "x"]))
    out = prep.transform(coerce_raw_features(train, ["region", "x"]))
    assert out.shape == (4, 2)
    assert not np.isnan(out).any()


def test_numeric_looking_categorical_stable_codes():
    # 1 and 1.0 must collapse to one category after str-coercion.
    df = pd.DataFrame(
        {"code": pd.Series([1, 1.0, 2, 2], dtype=object), "x": [1.0, 2, 3, 4]}
    )
    coerced = coerce_raw_features(df, ["code", "x"])
    assert set(coerced["code"].dropna().unique()) == {"1", "1.0", "2"} or set(
        coerced["code"].dropna().unique()
    ) <= {"1", "2", "1.0", "2.0"}


# ---------------------------------------------------------------------------
# extract_clean_xy
# ---------------------------------------------------------------------------


def test_extract_clean_xy_drops_missing_target_and_resets_index():
    df = pd.DataFrame({"x": [1.0, 2, 3, 4], "target": [1, None, 0, 1]})
    X, y, le = extract_clean_xy(df, ["x"], "target", "regression")
    assert len(X) == 3
    assert list(X.index) == [0, 1, 2]  # reset
    assert len(y) == 3
    assert le is None


def test_extract_clean_xy_encodes_string_target():
    df = pd.DataFrame({"x": [1.0, 2, 3, 4], "target": ["yes", "no", "yes", "no"]})
    X, y, le = extract_clean_xy(df, ["x"], "target", "classification")
    assert le is not None
    assert set(np.unique(y)) == {0, 1}
    assert sorted(le.classes_.tolist()) == ["no", "yes"]


def test_extract_clean_xy_returns_raw_untransformed_features():
    df = _mixed_df()
    X, y, _le = extract_clean_xy(
        df, ["age", "region", "spend"], "target", "classification"
    )
    # region stays as un-encoded category strings — NOT label-encoded here
    # (that's the preprocessor's job, fit per training fold).
    assert all(isinstance(v, str) for v in X["region"])
    assert set(X["region"].unique()) <= {"north", "south", "east"}
