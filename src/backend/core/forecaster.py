"""forecaster.py

Time-series forecasting engine.  Given a DataFrame with a date column and a
numeric target, this module:

  1. Detects the data frequency (daily / weekly / monthly / quarterly).
  2. Engineers time-based features (trend index, cyclic sin/cos for month
     and day-of-week).
  3. Trains a LinearRegression model on the historical data.
  4. Extrapolates the next N periods and produces 95% prediction intervals
     (residual-std approach — same as deploy.py uses).
  5. Returns a self-contained ForecastResult dict ready for the frontend.

The chart_type is "forecast", distinguishing it from the existing "line" chart
so the frontend can render historical vs forecast with different styling.
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def detect_time_series(
    df: pd.DataFrame,
) -> dict[str, Any] | None:
    """Return {date_col, value_cols} if the DataFrame looks like time-series.

    A DataFrame qualifies when:
      • At least one column can be parsed as datetime.
      • At least one other column is numeric.
      • After sorting by the date column there are ≥ 4 distinct rows.

    Returns None when none of those conditions hold.
    """
    date_col = _pick_date_col(df)
    if date_col is None:
        return None

    value_cols = [
        c for c in df.select_dtypes(include="number").columns.tolist() if c != date_col
    ]
    if not value_cols:
        return None

    # Need at least 4 rows after dropping null dates
    try:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
        n_valid = parsed.notna().sum()
    except Exception:  # noqa: BLE001
        return None

    if n_valid < 4:
        return None

    return {"date_col": date_col, "value_cols": value_cols}


def forecast_next_periods(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    periods: int = 6,
) -> dict[str, Any]:
    """Forecast the next *periods* time steps beyond the historical data.

    Returns a ForecastResult dict with keys:
      date_col, value_col,
      historical: [{date, value}],       ← sorted historical points
      forecast:   [{date, value, lower, upper}],  ← future predictions + 95% CI
      period_label: "month" | "week" | "day" | "quarter",
      trend:    "up" | "down" | "stable",
      growth_pct: float,                 ← pct change from last historical to last forecast
      summary: str,
      chart_type: "forecast"

    Raises ValueError for bad inputs (caller should catch).
    """
    periods = max(1, min(periods, 24))  # clamp to [1, 24]

    # ---- 1. Parse and sort dates ------------------------------------------------
    series = df[[date_col, value_col]].copy()
    series[date_col] = pd.to_datetime(series[date_col], errors="coerce")
    series = series.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if len(series) < 4:
        raise ValueError("Need at least 4 non-null data points for forecasting.")

    dates: list[pd.Timestamp] = series[date_col].tolist()
    values: list[float] = series[value_col].astype(float).tolist()
    n = len(dates)

    # ---- 2. Detect frequency ---------------------------------------------------
    period_label, delta = _detect_frequency(dates)

    # ---- 3. Build feature matrix -----------------------------------------------
    X = _build_features(dates, start_index=0)
    y = np.array(values)

    # ---- 4. Train linear model --------------------------------------------------
    model = LinearRegression()
    model.fit(X, y)

    # Residual std for confidence interval
    y_pred_train = model.predict(X)
    residuals = y - y_pred_train
    residual_std = float(np.std(residuals, ddof=1)) if n > 1 else 0.0
    ci_half = 1.96 * residual_std  # 95% CI half-width

    # ---- 5. Generate future dates and predict -----------------------------------
    # Future trend indices MUST continue from n, not restart at 0.
    last_date: pd.Timestamp = dates[-1]
    future_dates = _next_dates(last_date, delta, periods)
    X_future = _build_features(future_dates, start_index=n)
    y_future = model.predict(X_future)

    # ---- 6. Assemble result -----------------------------------------------------
    historical = [
        {"date": _fmt_date(d, period_label), "value": round(float(v), 4)}
        for d, v in zip(dates, values)
    ]

    forecast = []
    for d, v in zip(future_dates, y_future):
        v_f = float(v)
        forecast.append(
            {
                "date": _fmt_date(d, period_label),
                "value": round(v_f, 4),
                "lower": round(v_f - ci_half, 4),
                "upper": round(v_f + ci_half, 4),
            }
        )

    # ---- 7. Trend + summary ----------------------------------------------------
    last_historical = float(values[-1])
    last_forecast = float(y_future[-1])

    if last_historical == 0:
        growth_pct = 0.0
    else:
        growth_pct = round(
            (last_forecast - last_historical) / abs(last_historical) * 100, 1
        )

    if growth_pct > 2:
        trend = "up"
        trend_word = "increase"
    elif growth_pct < -2:
        trend = "down"
        trend_word = "decrease"
    else:
        trend = "stable"
        trend_word = "remain stable"

    if trend == "stable":
        direction_phrase = "remain relatively stable"
    else:
        direction_phrase = f"{trend_word} by {abs(growth_pct)}%"

    summary = (
        f"{value_col} is expected to {direction_phrase} over the next "
        f"{periods} {period_label}{'s' if periods > 1 else ''}. "
        f"Last recorded value: {_fmt_value(last_historical)}. "
        f"Projected value: {_fmt_value(last_forecast)} "
        f"(95% range: {_fmt_value(last_forecast - ci_half)} – "
        f"{_fmt_value(last_forecast + ci_half)})."
    )

    return {
        "chart_type": "forecast",
        "date_col": date_col,
        "value_col": value_col,
        "historical": historical,
        "forecast": forecast,
        "period_label": period_label,
        "trend": trend,
        "growth_pct": growth_pct,
        "summary": summary,
        "ci_level": 0.95,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _pick_date_col(df: pd.DataFrame) -> str | None:
    """Return the first column that looks like dates/timestamps, or None.

    Strict rules to avoid false positives:
    1. Skip purely numeric columns (integers/floats — they parse as Unix ns timestamps).
    2. Only accept object/string/datetime columns where ≥80% of non-null values
       parse as dates AND the parsed range spans more than 1 day (rules out plain IDs).
    """
    date_hints = {"date", "time", "day", "week", "month", "year", "period", "dt", "ts"}

    # Put hint columns first, then the rest
    hint_cols = [c for c in df.columns if any(h in c.lower() for h in date_hints)]
    other_cols = [c for c in df.columns if c not in hint_cols]
    candidates = hint_cols + other_cols

    for col in candidates:
        # Skip numeric columns — integers and floats parse as Unix nanosecond timestamps,
        # causing false positives on datasets that have no dates at all.
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            valid_pct = parsed.notna().mean()
            if valid_pct < 0.8:
                continue
            # Range check: dates should span more than 1 day
            valid = parsed.dropna()
            if len(valid) < 2:
                continue
            span_days = (valid.max() - valid.min()).days
            if span_days <= 1:
                continue
            return col
        except Exception:  # noqa: BLE001
            continue
    return None


def _detect_frequency(dates: list[pd.Timestamp]) -> tuple[str, timedelta]:
    """Infer the dominant time frequency from a sorted list of timestamps.

    Returns (label, representative_delta) where label is one of:
      "day", "week", "month", "quarter"
    """
    if len(dates) < 2:
        return "month", timedelta(days=30)

    diffs = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    median_days = float(np.median(diffs))

    if median_days <= 1.5:
        return "day", timedelta(days=1)
    if median_days <= 8:
        return "week", timedelta(weeks=1)
    if median_days <= 45:
        return "month", timedelta(days=round(median_days))
    return "quarter", timedelta(days=round(median_days))


def _next_dates(
    last: pd.Timestamp,
    delta: timedelta,
    n: int,
) -> list[pd.Timestamp]:
    """Generate n future dates spaced by delta from last."""
    result = []
    current = last
    for _ in range(n):
        current = current + delta
        result.append(current)
    return result


def _build_features(dates: list[pd.Timestamp], start_index: int = 0) -> np.ndarray:
    """Create a feature matrix for the given dates.

    Features:
      - trend_index (start_index, start_index+1, …) — linear trend
      - sin_month, cos_month — annual seasonality
      - sin_dow, cos_dow — weekly seasonality

    Args:
        dates:       List of timestamps to build features for.
        start_index: The index value assigned to dates[0]. Use len(historical)
                     when calling for future dates so the trend is continuous.
    """
    rows = []
    for i, d in enumerate(dates):
        trend_idx = start_index + i
        sin_month = math.sin(2 * math.pi * d.month / 12)
        cos_month = math.cos(2 * math.pi * d.month / 12)
        sin_dow = math.sin(2 * math.pi * d.dayofweek / 7)
        cos_dow = math.cos(2 * math.pi * d.dayofweek / 7)
        rows.append([trend_idx, sin_month, cos_month, sin_dow, cos_dow])
    return np.array(rows, dtype=float)


def _fmt_date(d: pd.Timestamp, period_label: str) -> str:
    """Format a timestamp for the given frequency."""
    if period_label in ("day",):
        return d.strftime("%Y-%m-%d")
    if period_label == "week":
        return d.strftime("%Y-%m-%d")
    if period_label == "month":
        return d.strftime("%b %Y")
    return d.strftime("%b %Y")  # quarter → also month-year


def _fmt_value(v: float) -> str:
    """Format a numeric value compactly (K / M suffixes)."""
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{v:.2f}"
