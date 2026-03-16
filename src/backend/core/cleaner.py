"""Conversational data cleaning operations.

Each function takes a DataFrame, applies one cleaning operation, and returns
a result dict describing what changed.  No DB or file I/O — callers handle
persistence.

Supported operations
--------------------
remove_duplicates     Remove exact duplicate rows.
fill_missing          Fill NaN values in one column with mean/median/mode/zero
                      or a literal value.
filter_rows           Remove rows where column op value is True
                      (e.g. quantity < 0 removes negative quantities).
cap_outliers          Clip a numeric column at the given percentile/100-pct.
drop_column           Remove a column entirely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summary_line(op: str, before: int, after: int, col: str | None = None) -> str:
    removed = before - after
    pct = removed / before * 100 if before > 0 else 0
    col_str = f" from column '{col}'" if col else ""
    if removed == 0:
        return f"No rows were changed by '{op}'{col_str} — dataset unchanged ({before} rows)."
    return (
        f"'{op}'{col_str}: removed {removed} row(s) ({pct:.1f}%), "
        f"{after} rows remain (was {before})."
    )


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicate rows (all columns must match).

    Returns:
        (cleaned_df, result_dict)
    """
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    after = len(cleaned)
    removed = before - after
    return cleaned, {
        "operation": "remove_duplicates",
        "before_rows": before,
        "after_rows": after,
        "modified_count": removed,
        "summary": (
            f"Removed {removed} duplicate row(s). {after} rows remain."
            if removed > 0
            else f"No duplicate rows found — dataset unchanged ({before} rows)."
        ),
    }


def fill_missing(
    df: pd.DataFrame,
    column: str,
    strategy: str,
    fill_value: float | str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fill NaN values in *column* using the given strategy.

    strategy: "mean" | "median" | "mode" | "zero" | "value"
    fill_value: used when strategy == "value"
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")

    before_nulls = int(df[column].isna().sum())
    if before_nulls == 0:
        return df.copy(), {
            "operation": "fill_missing",
            "column": column,
            "strategy": strategy,
            "before_rows": len(df),
            "after_rows": len(df),
            "modified_count": 0,
            "summary": f"Column '{column}' has no missing values — nothing to fill.",
        }

    cleaned = df.copy()

    if strategy == "mean":
        if not pd.api.types.is_numeric_dtype(cleaned[column]):
            raise ValueError(f"Strategy 'mean' requires a numeric column; '{column}' is not numeric.")
        val = cleaned[column].mean()
        cleaned[column] = cleaned[column].fillna(val)
        strategy_desc = f"mean ({val:.4g})"
    elif strategy == "median":
        if not pd.api.types.is_numeric_dtype(cleaned[column]):
            raise ValueError(f"Strategy 'median' requires a numeric column; '{column}' is not numeric.")
        val = cleaned[column].median()
        cleaned[column] = cleaned[column].fillna(val)
        strategy_desc = f"median ({val:.4g})"
    elif strategy == "mode":
        mode_vals = cleaned[column].mode()
        val = mode_vals.iloc[0] if len(mode_vals) > 0 else 0
        cleaned[column] = cleaned[column].fillna(val)
        strategy_desc = f"mode ({val!r})"
    elif strategy == "zero":
        cleaned[column] = cleaned[column].fillna(0)
        strategy_desc = "0"
    elif strategy == "value":
        if fill_value is None:
            raise ValueError("fill_value must be provided when strategy='value'.")
        cleaned[column] = cleaned[column].fillna(fill_value)
        strategy_desc = repr(fill_value)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Use: mean, median, mode, zero, value."
        )

    after_nulls = int(cleaned[column].isna().sum())
    filled = before_nulls - after_nulls
    return cleaned, {
        "operation": "fill_missing",
        "column": column,
        "strategy": strategy,
        "fill_value_used": strategy_desc,
        "before_rows": len(df),
        "after_rows": len(cleaned),
        "modified_count": filled,
        "summary": (
            f"Filled {filled} missing value(s) in '{column}' with {strategy_desc}. "
            f"Column is now complete."
        ),
    }


_VALID_OPERATORS = {"gt", "lt", "eq", "ne", "gte", "lte", "contains", "notcontains"}
_OPERATOR_LABELS = {
    "gt": ">", "lt": "<", "eq": "==", "ne": "!=", "gte": ">=", "lte": "<=",
    "contains": "contains", "notcontains": "not contains",
}


def filter_rows(
    df: pd.DataFrame,
    column: str,
    operator: str,
    value: float | str,
) -> tuple[pd.DataFrame, dict]:
    """Remove rows where the condition column OP value is True.

    For numeric columns: gt/lt/eq/ne/gte/lte
    For string columns: contains/notcontains/eq/ne

    Example: column="quantity", operator="lt", value=0  → removes negative quantities.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")
    if operator not in _VALID_OPERATORS:
        raise ValueError(f"Unknown operator '{operator}'. Valid: {sorted(_VALID_OPERATORS)}")

    before = len(df)
    col = df[column]

    if operator == "gt":
        mask = col > float(value)
    elif operator == "lt":
        mask = col < float(value)
    elif operator == "gte":
        mask = col >= float(value)
    elif operator == "lte":
        mask = col <= float(value)
    elif operator == "eq":
        mask = col == value
    elif operator == "ne":
        mask = col != value
    elif operator == "contains":
        mask = col.astype(str).str.contains(str(value), case=False, na=False)
    else:  # notcontains
        mask = ~col.astype(str).str.contains(str(value), case=False, na=False)

    # Remove rows WHERE the condition is true (filter OUT matching rows)
    cleaned = df[~mask].reset_index(drop=True)
    after = len(cleaned)
    removed = before - after

    op_label = _OPERATOR_LABELS[operator]
    return cleaned, {
        "operation": "filter_rows",
        "column": column,
        "operator": operator,
        "value": value,
        "before_rows": before,
        "after_rows": after,
        "modified_count": removed,
        "summary": (
            f"Removed {removed} row(s) where '{column}' {op_label} {value!r}. "
            f"{after} rows remain."
            if removed > 0
            else f"No rows matched '{column}' {op_label} {value!r} — dataset unchanged."
        ),
    }


def cap_outliers(
    df: pd.DataFrame,
    column: str,
    percentile: float = 99.0,
) -> tuple[pd.DataFrame, dict]:
    """Clip *column* values at [1-percentile, percentile] using percentile bounds.

    Values above the upper bound are capped; values below the lower bound are
    raised to the lower bound.  Returns a clipped DataFrame (no rows removed).

    percentile: upper percentile, e.g. 99.0 means cap at 99th / 1st percentile.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"cap_outliers requires a numeric column; '{column}' is not numeric.")

    percentile = float(max(50.0, min(99.9, percentile)))
    lower_pct = 100.0 - percentile
    upper = float(np.percentile(df[column].dropna(), percentile))
    lower = float(np.percentile(df[column].dropna(), lower_pct))

    cleaned = df.copy()
    before_vals = cleaned[column].copy()
    cleaned[column] = cleaned[column].clip(lower=lower, upper=upper)
    changed = int((cleaned[column] != before_vals).sum())

    return cleaned, {
        "operation": "cap_outliers",
        "column": column,
        "percentile": percentile,
        "lower_bound": round(lower, 4),
        "upper_bound": round(upper, 4),
        "before_rows": len(df),
        "after_rows": len(cleaned),
        "modified_count": changed,
        "summary": (
            f"Capped {changed} value(s) in '{column}' to [{lower:.4g}, {upper:.4g}] "
            f"({lower_pct:.1f}th–{percentile:.1f}th percentile)."
            if changed > 0
            else f"No values in '{column}' were outside the [{lower:.4g}, {upper:.4g}] range."
        ),
    }


def drop_column(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, dict]:
    """Remove a column entirely from the DataFrame."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset.")

    cleaned = df.drop(columns=[column])
    return cleaned, {
        "operation": "drop_column",
        "column": column,
        "before_rows": len(df),
        "after_rows": len(cleaned),
        "before_columns": len(df.columns),
        "after_columns": len(cleaned.columns),
        "modified_count": len(cleaned),
        "summary": f"Dropped column '{column}'. Dataset now has {len(cleaned.columns)} columns.",
    }
