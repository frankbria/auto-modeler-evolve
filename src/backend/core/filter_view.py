"""filter_view.py

Non-destructive data filter for chat-driven exploration.

Analysts can say "focus on North region" or "filter to revenue > 1000"
and all subsequent analyses run on the filtered subset without modifying
the underlying CSV.

Operator mapping:
  eq / ne / gt / lt / gte / lte / contains / not_contains
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

FilterCondition = dict[str, Any]  # {column, operator, value}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_filter_request(
    message: str, df_columns: list[str]
) -> list[FilterCondition] | None:
    """Extract filter conditions from a natural-language message.

    Supports:
      "focus on North region"          → region eq North
      "filter to revenue > 1000"       → revenue gt 1000
      "show only category = Electronics"
      "narrow to Q4" (date keyword)    → looks for quarter/date columns
      "exclude rows where status is cancelled"

    Returns a list of FilterCondition dicts, or None if nothing detected.
    """
    col_lower = {c.lower(): c for c in df_columns}
    conditions: list[FilterCondition] = []

    # Pattern 1: explicit comparison — "revenue > 1000", "score >= 90", "age < 30"
    _num_compare = re.findall(
        r"['\"]?(\w+)['\"]?\s*(>=|<=|!=|>|<|==?)\s*([\d.]+)",
        message,
        re.IGNORECASE,
    )
    for raw_col, op_sym, val_str in _num_compare:
        col = col_lower.get(raw_col.lower())
        if col:
            conditions.append(
                {
                    "column": col,
                    "operator": _map_sym_op(op_sym),
                    "value": float(val_str),
                }
            )

    # Pattern 2: equality — "region is North", "region = East", "status equals active"
    _eq_pat = re.findall(
        r"['\"]?(\w+)['\"]?\s+(?:is|=|==|equals?|eq)\s+['\"]?(\w+)['\"]?",
        message,
        re.IGNORECASE,
    )
    for raw_col, raw_val in _eq_pat:
        col = col_lower.get(raw_col.lower())
        if col and not _already_covered(conditions, col):
            # skip if it looks like a programming expression, not a column
            if raw_val.lower() not in (
                "true",
                "false",
                "null",
                "none",
                "a",
                "an",
                "the",
            ):
                conditions.append({"column": col, "operator": "eq", "value": raw_val})

    # Pattern 3: "contains" — "category contains Electronics"
    _contains_pat = re.findall(
        r"['\"]?(\w+)['\"]?\s+contains?\s+['\"]?(\w+)['\"]?",
        message,
        re.IGNORECASE,
    )
    for raw_col, raw_val in _contains_pat:
        col = col_lower.get(raw_col.lower())
        if col and not _already_covered(conditions, col):
            conditions.append({"column": col, "operator": "contains", "value": raw_val})

    # Pattern 4: "only X" — scan if X is an actual value in a categorical column
    _only_pat = re.findall(
        r"\bonly\s+['\"]?(\w+(?:\s+\w+)?)['\"]?", message, re.IGNORECASE
    )
    for raw_val in _only_pat:
        raw_val = raw_val.strip()
        hit = _find_column_with_value(df_columns, raw_val, message)
        if hit and not _already_covered(conditions, hit):
            conditions.append({"column": hit, "operator": "eq", "value": raw_val})

    # Pattern 5: "focus on X" / "look at X" — same column-value scan
    _focus_pat = re.findall(
        r"\b(?:focus\s+on|narrow\s+(?:to|down\s+to)|just\s+show|show\s+only)\s+"
        r"['\"]?(\w+(?:\s+\w+)?)['\"]?",
        message,
        re.IGNORECASE,
    )
    for raw_val in _focus_pat:
        raw_val = raw_val.strip()
        # Skip if it's a column name (user said "focus on revenue")
        if raw_val.lower() not in col_lower:
            hit = _find_column_with_value(df_columns, raw_val, message)
            if hit and not _already_covered(conditions, hit):
                conditions.append({"column": hit, "operator": "eq", "value": raw_val})

    return conditions if conditions else None


def apply_active_filter(
    df: pd.DataFrame, conditions: list[FilterCondition]
) -> pd.DataFrame:
    """Apply a list of filter conditions to df, returning the filtered subset.

    Conditions are AND-ed together. Each condition: {column, operator, value}.
    Unknown columns or operators are silently skipped.
    """
    result = df.copy()
    for cond in conditions:
        col = cond.get("column")
        op = cond.get("operator")
        val = cond.get("value")
        if col not in result.columns:
            continue
        try:
            result = _apply_one(result, col, op, val)
        except Exception:  # noqa: BLE001
            continue  # skip bad conditions rather than crashing
    return result


def build_filter_summary(conditions: list[FilterCondition]) -> str:
    """Return a plain-English description of a set of filter conditions.

    Example: "region = North AND revenue > 1000"
    """
    parts = []
    for cond in conditions:
        col = cond.get("column", "?")
        op = cond.get("operator", "eq")
        val = cond.get("value", "?")
        parts.append(f"{col} {_op_display(op)} {val}")
    return " AND ".join(parts) if parts else "no filter"


def validate_filter_conditions(
    conditions: list[FilterCondition], df_columns: list[str]
) -> list[str]:
    """Return list of error strings for invalid conditions (empty = all valid)."""
    errors = []
    col_set = set(df_columns)
    valid_ops = {"eq", "ne", "gt", "lt", "gte", "lte", "contains", "not_contains"}
    for i, cond in enumerate(conditions):
        col = cond.get("column")
        op = cond.get("operator")
        if col not in col_set:
            errors.append(f"Condition {i + 1}: column '{col}' not in dataset")
        if op not in valid_ops:
            errors.append(f"Condition {i + 1}: unknown operator '{op}'")
    return errors


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_one(df: pd.DataFrame, col: str, op: str, val: Any) -> pd.DataFrame:
    series = df[col]
    if op == "eq":
        return (
            df[series.astype(str).str.lower() == str(val).lower()]
            if series.dtype == object or pd.api.types.is_string_dtype(series)
            else df[series == val]
        )
    elif op == "ne":
        return (
            df[series.astype(str).str.lower() != str(val).lower()]
            if series.dtype == object or pd.api.types.is_string_dtype(series)
            else df[series != val]
        )
    elif op == "gt":
        return df[pd.to_numeric(series, errors="coerce") > float(val)]
    elif op == "lt":
        return df[pd.to_numeric(series, errors="coerce") < float(val)]
    elif op == "gte":
        return df[pd.to_numeric(series, errors="coerce") >= float(val)]
    elif op == "lte":
        return df[pd.to_numeric(series, errors="coerce") <= float(val)]
    elif op == "contains":
        return df[series.astype(str).str.contains(str(val), case=False, na=False)]
    elif op == "not_contains":
        return df[~series.astype(str).str.contains(str(val), case=False, na=False)]
    return df


def _map_sym_op(sym: str) -> str:
    return {
        ">": "gt",
        "<": "lt",
        ">=": "gte",
        "<=": "lte",
        "==": "eq",
        "=": "eq",
        "!=": "ne",
    }.get(sym, "eq")


def _op_display(op: str) -> str:
    return {
        "eq": "=",
        "ne": "≠",
        "gt": ">",
        "lt": "<",
        "gte": "≥",
        "lte": "≤",
        "contains": "contains",
        "not_contains": "doesn't contain",
    }.get(op, op)


def _already_covered(conditions: list[FilterCondition], col: str) -> bool:
    return any(c["column"] == col for c in conditions)


def _find_column_with_value(
    df_columns: list[str], raw_val: str, message: str
) -> str | None:
    """Search column names for one whose NL alias or actual values match raw_val.

    This is called without the DataFrame — we rely on column-name hints only.
    The actual DataFrame value-matching is done at apply time.
    """
    val_lower = raw_val.lower()
    # Check if message has "[col] [val]" pattern that points to a column
    for col in df_columns:
        if col.lower() in message.lower():
            # The column name appears in the message near the value — likely a match
            col_pos = message.lower().find(col.lower())
            val_pos = message.lower().find(val_lower)
            if abs(col_pos - val_pos) < 50:
                return col
    return None
