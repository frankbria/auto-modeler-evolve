"""Computed column operations — add derived columns through conversation.

Users can say "add a column called margin = revenue / cost" and this module
evaluates the expression safely using pd.DataFrame.eval(), which restricts
evaluation to arithmetic/comparison operations on column names.

Security: pd.eval() does NOT allow imports, attribute access, or arbitrary
Python execution — only arithmetic, comparisons, and limited math functions.
"""

from __future__ import annotations

import re

import pandas as pd

# Valid column name pattern: letters, digits, underscore, must start with letter/underscore
_VALID_COL_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def add_computed_column(
    df: pd.DataFrame,
    name: str,
    expression: str,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate *expression* and add the result as a new column named *name*.

    Uses pd.DataFrame.eval() for safe expression evaluation — only arithmetic
    and comparison operations on existing column references are supported.

    Returns:
        (updated_df, result_dict)

    Raises:
        ValueError: if name is invalid, expression errors, or name collides
                    with a non-overwrite scenario.
    """
    # Validate column name
    if not _VALID_COL_NAME.match(name):
        raise ValueError(
            f"Invalid column name '{name}'. "
            "Use letters, digits, and underscores only (must start with a letter or '_')."
        )

    # Validate expression is not empty
    expression = expression.strip()
    if not expression:
        raise ValueError("Expression cannot be empty.")

    # Evaluate expression using pandas eval (safe subset of Python expressions)
    try:
        computed_series = df.eval(expression)
    except Exception as exc:
        raise ValueError(f"Invalid expression '{expression}': {exc}") from exc

    # Ensure result is a Series (not a scalar)
    if not isinstance(computed_series, pd.Series):
        # Scalar result — broadcast to all rows
        computed_series = pd.Series([computed_series] * len(df), index=df.index)

    is_new = name not in df.columns
    result_df = df.copy()
    result_df[name] = computed_series

    # Build sample preview (first 5 non-null values)
    sample = [
        v if pd.notna(v) else None
        for v in computed_series.head(5).tolist()
    ]

    action = "added" if is_new else "updated"
    return result_df, {
        "column_name": name,
        "expression": expression,
        "dtype": str(computed_series.dtype),
        "sample_values": sample,
        "row_count": len(result_df),
        "column_count": len(result_df.columns),
        "action": action,
        "summary": (
            f"{'Added new column' if is_new else 'Updated column'} '{name}' "
            f"= {expression}. "
            f"First values: {', '.join(str(v) for v in sample[:3])}."
        ),
    }


def preview_computed_column(
    df: pd.DataFrame,
    name: str,
    expression: str,
    n: int = 5,
) -> dict:
    """Evaluate expression and return a preview without modifying the DataFrame.

    Raises ValueError on invalid expression or name.
    """
    if not _VALID_COL_NAME.match(name):
        raise ValueError(f"Invalid column name '{name}'.")

    expression = expression.strip()
    if not expression:
        raise ValueError("Expression cannot be empty.")

    try:
        computed_series = df.eval(expression)
    except Exception as exc:
        raise ValueError(f"Invalid expression '{expression}': {exc}") from exc

    if not isinstance(computed_series, pd.Series):
        computed_series = pd.Series([computed_series] * len(df), index=df.index)

    sample = [
        v if pd.notna(v) else None
        for v in computed_series.head(n).tolist()
    ]

    return {
        "column_name": name,
        "expression": expression,
        "dtype": str(computed_series.dtype),
        "sample_values": sample,
    }
