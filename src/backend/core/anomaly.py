"""Multi-dimensional anomaly detection using IsolationForest.

Unlike per-column z-score outlier detection (which finds univariate outliers),
IsolationForest detects *multivariate* anomalies — rows that are unusual across
multiple features simultaneously.

For example:
  - Revenue of $500 is normal on its own
  - But $500 revenue for 10,000 units in the "Premium" category is anomalous

Design:
- Uses sklearn IsolationForest (O(n log n) random partitioning)
- Normalises score_samples() output to 0-100 (100 = most anomalous)
- Returns top-N anomalous rows with per-feature values for display
- Pure function — no DB, no file I/O, no LLM calls
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(
    df: pd.DataFrame,
    features: list[str],
    contamination: float = 0.05,
    n_top: int = 20,
) -> dict:
    """Run IsolationForest anomaly detection on selected numeric features.

    Args:
        df: Full dataset.
        features: Column names to use. Non-numeric columns are silently dropped.
        contamination: Expected fraction of anomalies (0.01–0.5).
        n_top: How many top anomalies to return in the result.

    Returns:
        {
            anomaly_count: int,
            total_rows: int,
            contamination_used: float,
            top_anomalies: list[AnomalyRecord],
            summary: str,
            features_used: list[str],
        }

    Raises:
        ValueError: If no numeric features remain after filtering.
    """
    contamination = float(max(0.01, min(0.5, contamination)))

    # Keep only numeric, non-all-NaN columns that exist in df
    numeric_features = [
        f
        for f in features
        if f in df.columns
        and pd.api.types.is_numeric_dtype(df[f])
        and not df[f].isna().all()
    ]
    if not numeric_features:
        raise ValueError(
            "No numeric features available for anomaly detection. "
            "Please select at least one numeric column."
        )

    X = df[numeric_features].copy()

    # Fill NaN with column median so IsolationForest can handle missing values
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    labels = model.fit_predict(X)  # -1 = anomaly, 1 = normal
    raw_scores = model.score_samples(X)  # More negative = more anomalous

    # Normalise to 0-100: 100 = most anomalous
    score_range = raw_scores.max() - raw_scores.min()
    if score_range < 1e-10:
        anomaly_scores = np.zeros(len(raw_scores))
    else:
        anomaly_scores = (raw_scores.max() - raw_scores) / score_range * 100

    anomaly_mask = labels == -1
    anomaly_count = int(anomaly_mask.sum())
    total = len(X)

    # Top-N most anomalous rows (sorted descending by score)
    top_indices = np.argsort(anomaly_scores)[::-1][:n_top]
    top_anomalies = []
    for idx in top_indices:
        row = X.iloc[idx]
        top_anomalies.append(
            {
                "row_index": int(
                    df.index[idx] if hasattr(df.index, "__getitem__") else idx
                ),
                "anomaly_score": round(float(anomaly_scores[idx]), 1),
                "is_anomaly": bool(anomaly_mask[idx]),
                "values": {
                    col: (None if np.isnan(row[col]) else round(float(row[col]), 4))
                    for col in numeric_features
                },
            }
        )

    # Plain-English summary
    pct = anomaly_count / total * 100 if total > 0 else 0
    if anomaly_count == 0:
        summary = f"No anomalous records found across {total} rows using {len(numeric_features)} features."
    else:
        top_score = round(float(anomaly_scores.max()), 1)
        summary = (
            f"Found {anomaly_count} unusual record(s) out of {total} "
            f"({pct:.1f}%). "
            f"The most anomalous record has a score of {top_score}/100 "
            f"(row index {top_anomalies[0]['row_index']})."
        )

    return {
        "anomaly_count": anomaly_count,
        "total_rows": total,
        "contamination_used": contamination,
        "top_anomalies": top_anomalies,
        "summary": summary,
        "features_used": numeric_features,
    }
