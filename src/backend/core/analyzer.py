import numpy as np
import pandas as pd


def analyze_dataframe(df: pd.DataFrame) -> dict:
    """Return column statistics for a dataframe.

    For each column produces: dtype, non_null_count, null_count, null_pct,
    unique_count, and 5 sample_values. Numeric columns also get min, max,
    mean, and std.
    """
    columns = []

    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        null_count = int(series.isna().sum())
        total = len(series)
        null_pct = round(null_count / total * 100, 2) if total > 0 else 0.0

        sample_values = (
            series.dropna()
            .head(5)
            .apply(lambda v: v.item() if isinstance(v, np.generic) else v)
            .tolist()
        )

        stat: dict = {
            "name": col,
            "dtype": str(series.dtype),
            "non_null_count": non_null,
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": int(series.nunique()),
            "sample_values": sample_values,
        }

        if pd.api.types.is_numeric_dtype(series):
            stat["min"] = _safe_scalar(series.min())
            stat["max"] = _safe_scalar(series.max())
            stat["mean"] = _safe_scalar(series.mean())
            stat["std"] = _safe_scalar(series.std())

        columns.append(stat)

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
    }


def _safe_scalar(value):
    """Convert numpy scalars to native Python types for JSON serialization."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
