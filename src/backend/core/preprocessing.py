"""Leak-free feature preprocessing for model training.

The single source of truth for turning a raw feature DataFrame into the numeric
matrix the estimators consume. The key property — and the whole reason this
module exists (issue #5) — is that the imputation / encoding statistics are
learned by an *unfitted* sklearn transformer that callers fit **only on the
training fold**, never on the full dataset. The same unfitted transformer is
handed to ``cross_val_score`` so each CV fold refits it in isolation.

Design constraints (do not break):
- Output is **one column per feature**, in ``feature_cols`` order, so the saved
  estimator stays a bare numeric model and ``feature_importances_`` / ``coef_``
  stay aligned with the feature names (``core/deployer.py`` perturbs this numeric
  space directly). That rules out OneHotEncoder; we use OrdinalEncoder.
- Unseen categories at transform time map to ``-1`` (``handle_unknown=
  "use_encoded_value"``) instead of raising, mirroring serving's best-effort
  behaviour for novel inputs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Sentinel used for missing categorical values (matches the legacy
# ``fillna("MISSING")`` behaviour in trainer.prepare_features / deployer).
MISSING_CATEGORY = "MISSING"

# Encoded value handed to unseen categories at transform time. Must be an int
# for OrdinalEncoder(handle_unknown="use_encoded_value"); the encoder still emits
# a float matrix, so downstream comparisons see -1.0.
UNKNOWN_ENCODED_VALUE = -1


def split_numeric_categorical(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[str], list[str]]:
    """Partition ``feature_cols`` into (numeric, categorical) by dtype.

    Uses the same ``is_numeric_dtype`` test as the legacy trainer and the
    serving ``PredictionPipeline`` so the three preprocessing paths agree on
    which columns are categorical.
    """
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feature_cols if c not in numeric]
    return numeric, categorical


def build_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    """Return an **unfitted** ColumnTransformer for ``feature_cols``.

    Numeric columns -> median imputation. Categorical columns -> constant
    ``MISSING`` imputation then ordinal encoding (unseen -> ``-1``). The
    transformer must be fit on the training fold only; callers also pass a
    fresh one into ``cross_val_score`` so every fold refits independently.

    One transformer per feature, listed in ``feature_cols`` order, so the output
    column order **matches** ``feature_cols`` exactly — keeping the estimator's
    ``feature_importances_`` / ``coef_`` aligned with the feature names.
    """
    numeric, _categorical = split_numeric_categorical(df, feature_cols)
    numeric_set = set(numeric)

    transformers = []
    for i, col in enumerate(feature_cols):
        if col in numeric_set:
            transformers.append((f"f{i}", SimpleImputer(strategy="median"), [col]))
        else:
            cat_pipeline = Pipeline(
                steps=[
                    (
                        "impute",
                        SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY),
                    ),
                    (
                        "encode",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=UNKNOWN_ENCODED_VALUE,
                        ),
                    ),
                ]
            )
            transformers.append((f"f{i}", cat_pipeline, [col]))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def ordered_feature_names(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    """Feature names in the column order produced by :func:`build_preprocessor`.

    The preprocessor emits one column per feature in ``feature_cols`` order, so
    this is just ``feature_cols`` — provided as a named helper so callers don't
    re-encode that assumption.
    """
    return list(feature_cols)


def coerce_raw_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return a feature-only frame with categorical columns cast to ``str``.

    Casting matches the legacy ``astype(str)`` so ordinal codes are stable
    between training and serving (e.g. ``1`` and ``1.0`` collapse to one
    category). NaNs are preserved for the imputer to fill.
    """
    out = df[feature_cols].copy()
    _numeric, categorical = split_numeric_categorical(df, feature_cols)
    for col in categorical:
        # Preserve NaN; stringify everything else so codes are deterministic.
        out[col] = out[col].where(out[col].isna(), out[col].astype(str))
    return out


def extract_serving_stats(
    preprocessor: ColumnTransformer,
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Pull per-column medians and categorical category orderings out of a
    **fitted** preprocessor.

    Lets serving rebuild its transform from the *training-fold* statistics so
    the ordinal codes the deployed model receives match exactly what it was
    trained on — otherwise a category missing from the training fold would shift
    every code and silently corrupt predictions (issue #5).

    Returns ``(medians, categories)`` keyed by column name. ``categories`` lists
    are in the encoder's order, so ``categories[col].index(value)`` is the
    ordinal code the model expects.
    """
    medians: dict[str, float] = {}
    categories: dict[str, list[str]] = {}
    for _name, trans, cols in getattr(preprocessor, "transformers_", []):
        if not cols or trans in ("drop", "passthrough"):
            continue
        col = cols[0]
        if isinstance(trans, SimpleImputer):
            stats = getattr(trans, "statistics_", None)
            if stats is not None and len(stats) > 0:
                medians[col] = float(stats[0])
        elif isinstance(trans, Pipeline):
            encoder = trans.named_steps.get("encode")
            cats = getattr(encoder, "categories_", None)
            if cats is not None and len(cats) > 0:
                categories[col] = [str(c) for c in cats[0].tolist()]
    return medians, categories


def extract_clean_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    problem_type: str,
) -> tuple[pd.DataFrame, np.ndarray, Optional[LabelEncoder]]:
    """Build the **raw** (untransformed) feature frame + target vector.

    Unlike the legacy ``prepare_features``, this does NOT fit any feature
    imputation/encoding — that is deferred to :func:`build_preprocessor` so it
    can be fit on the training fold only (issue #5).

    - Drops rows where the target is missing (index reset so positional
      ``test_indices`` are stable).
    - Label-encodes a string classification target (this is standard practice
      and not train/test leakage — the label space must be globally consistent).

    Returns ``(X_raw_df, y, target_label_encoder_or_None)``.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    valid_features = [c for c in feature_cols if c in df.columns]
    if not valid_features:
        raise ValueError("No valid feature columns found in DataFrame.")

    df_clean = (
        df[valid_features + [target_col]]
        .dropna(subset=[target_col])
        .reset_index(drop=True)
    )
    if len(df_clean) < 2:
        raise ValueError("Not enough non-null rows to train a model.")

    X_raw = coerce_raw_features(df_clean, valid_features)

    y_series = df_clean[target_col]
    le_target: Optional[LabelEncoder] = None
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y_series):
        le_target = LabelEncoder()
        y = le_target.fit_transform(y_series.astype(str))
    else:
        y = (
            y_series.values.astype(float)
            if problem_type == "regression"
            else y_series.values
        )

    return X_raw, y, le_target
