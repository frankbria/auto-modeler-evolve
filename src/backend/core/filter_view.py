"""filter_view.py

Non-destructive data filter for chat-driven exploration.

Analysts can say "focus on North region" or "filter to revenue > 1000"
and all subsequent analyses run on the filtered subset without modifying
the underlying CSV.

Operator mapping:
  eq / ne / gt / lt / gte / lte / contains / not_contains / date_range
"""

from __future__ import annotations

import re
from datetime import date, timedelta
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
        if op == "date_range" and isinstance(val, dict):
            start = val.get("start", "?")
            end = val.get("end", "?")
            parts.append(f"{col} between {start} and {end}")
        else:
            parts.append(f"{col} {_op_display(op)} {val}")
    return " AND ".join(parts) if parts else "no filter"


def validate_filter_conditions(
    conditions: list[FilterCondition], df_columns: list[str]
) -> list[str]:
    """Return list of error strings for invalid conditions (empty = all valid)."""
    errors = []
    col_set = set(df_columns)
    valid_ops = {"eq", "ne", "gt", "lt", "gte", "lte", "contains", "not_contains", "date_range"}
    for i, cond in enumerate(conditions):
        col = cond.get("column")
        op = cond.get("operator")
        if col not in col_set:
            errors.append(f"Condition {i + 1}: column '{col}' not in dataset")
        if op not in valid_ops:
            errors.append(f"Condition {i + 1}: unknown operator '{op}'")
    return errors


# ---------------------------------------------------------------------------
# Date range filter
# ---------------------------------------------------------------------------

# Quarter month bounds: Q1=(1,3), Q2=(4,6), Q3=(7,9), Q4=(10,12)
_QUARTER_MONTHS: dict[int, tuple[int, int]] = {
    1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)
}

_MONTH_NAMES: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

_QUARTER_WORDS: dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "1st": 1, "2nd": 2, "3rd": 3, "4th": 4,
}

# Patterns (compiled once)
_RE_QUARTER_NUM = re.compile(
    r"\bq([1-4])\s*(?:of\s*)?(?:fy\s*)?(\d{4})?\b", re.IGNORECASE
)
_RE_QUARTER_WORD = re.compile(
    r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s*(?:of\s*)?(\d{4})?\b",
    re.IGNORECASE,
)
_RE_YEAR_ONLY = re.compile(r"\b(20\d{2})\s+(?:data|year|records?)?\b", re.IGNORECASE)
_RE_LAST_N = re.compile(
    r"\blast\s+(\d+)\s+(day|week|month|year)s?\b", re.IGNORECASE
)
_RE_THIS_LAST_PERIOD = re.compile(
    r"\b(this|last)\s+(year|month|quarter)\b", re.IGNORECASE
)
_RE_MONTH_RANGE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    r"\s+(?:through|to|-)\s+"
    r"(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    r"(?:\s+(\d{4}))?",
    re.IGNORECASE,
)
_RE_SINGLE_MONTH = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december"
    r"|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    r"(?:\s+(\d{4}))?\b",
    re.IGNORECASE,
)


def _detect_date_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that appear to contain date/datetime values."""
    date_cols = []
    for col in df.columns:
        col_lower = col.lower()
        # Name hint: common date-related column names
        if any(
            kw in col_lower
            for kw in ("date", "time", "year", "month", "day", "period", "quarter", "week")
        ):
            date_cols.append(col)
            continue
        # Value hint: try parsing a sample of non-null string values as dates.
        # Only consider object/string columns to avoid false-positive on numeric columns.
        if not (df[col].dtype == object or pd.api.types.is_string_dtype(df[col])):
            continue
        sample = df[col].dropna().head(5)
        if len(sample) == 0:
            continue
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() >= min(3, len(sample)):
                date_cols.append(col)
        except Exception:
            pass
    return date_cols


def _make_date_condition(col: str, start: date, end: date) -> FilterCondition:
    return {
        "column": col,
        "operator": "date_range",
        "value": {"start": start.isoformat(), "end": end.isoformat()},
    }


def _last_day_of_month(year: int, month: int) -> int:
    """Return the last calendar day of the given month."""
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - timedelta(days=1)).day


def parse_date_filter_request(
    message: str, df: pd.DataFrame
) -> list[FilterCondition] | None:
    """Extract date-range filter conditions from a natural-language message.

    Supports:
      "show Q4 2023 data"           → date_col between 2023-10-01 and 2023-12-31
      "filter to Q1"                → date_col between CURRENT_YEAR-01-01 and CURRENT_YEAR-03-31
      "last 6 months"               → date_col >= today-180 days
      "this year"                   → date_col between YEAR-01-01 and YEAR-12-31
      "last year"                   → date_col between (YEAR-1)-01-01 and (YEAR-1)-12-31
      "show 2024 data"              → date_col between 2024-01-01 and 2024-12-31
      "January through March 2023"  → date_col between 2023-01-01 and 2023-03-31
      "filter to March"             → date_col between YEAR-03-01 and YEAR-03-31

    Returns a list with one FilterCondition (date_range operator), or None if no
    date pattern is recognised or no date column is found.
    """
    date_cols = _detect_date_columns(df)
    if not date_cols:
        return None

    today = date.today()
    current_year = today.year
    col = date_cols[0]  # use the first detected date column

    # --- Pattern 1: "last N days/weeks/months/years" ---
    m = _RE_LAST_N.search(message)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit == "day":
            start = today - timedelta(days=n)
        elif unit == "week":
            start = today - timedelta(weeks=n)
        elif unit == "month":
            start = today - timedelta(days=n * 30)
        else:  # year
            start = today - timedelta(days=n * 365)
        return [_make_date_condition(col, start, today)]

    # --- Pattern 2: "this/last year/month/quarter" ---
    m = _RE_THIS_LAST_PERIOD.search(message)
    if m:
        qualifier = m.group(1).lower()  # "this" or "last"
        period = m.group(2).lower()     # "year", "month", "quarter"
        if period == "year":
            yr = current_year if qualifier == "this" else current_year - 1
            return [_make_date_condition(col, date(yr, 1, 1), date(yr, 12, 31))]
        if period == "month":
            mn = today.month if qualifier == "this" else (today.month - 1 or 12)
            yr = current_year if qualifier == "this" or today.month > 1 else current_year - 1
            last_day = _last_day_of_month(yr, mn)
            return [_make_date_condition(col, date(yr, mn, 1), date(yr, mn, last_day))]
        if period == "quarter":
            q_num = (today.month - 1) // 3 + 1
            if qualifier == "last":
                q_num = q_num - 1 if q_num > 1 else 4
                yr = current_year if q_num != 4 else current_year - 1
            else:
                yr = current_year
            start_month, end_month = _QUARTER_MONTHS[q_num]
            last_day = _last_day_of_month(yr, end_month)
            return [_make_date_condition(col, date(yr, start_month, 1), date(yr, end_month, last_day))]

    # --- Pattern 3: "Q1", "Q4 2023", "first quarter 2024" ---
    m = _RE_QUARTER_NUM.search(message)
    if m:
        q_num = int(m.group(1))
        yr = int(m.group(2)) if m.group(2) else current_year
        start_month, end_month = _QUARTER_MONTHS[q_num]
        last_day = _last_day_of_month(yr, end_month)
        return [_make_date_condition(col, date(yr, start_month, 1), date(yr, end_month, last_day))]

    m = _RE_QUARTER_WORD.search(message)
    if m:
        q_num = _QUARTER_WORDS.get(m.group(1).lower(), 1)
        yr = int(m.group(2)) if m.group(2) else current_year
        start_month, end_month = _QUARTER_MONTHS[q_num]
        last_day = _last_day_of_month(yr, end_month)
        return [_make_date_condition(col, date(yr, start_month, 1), date(yr, end_month, last_day))]

    # --- Pattern 4: "January through March 2023", "Jan to Mar" ---
    m = _RE_MONTH_RANGE.search(message)
    if m:
        start_month = _MONTH_NAMES.get(m.group(1).lower(), 1)
        end_month = _MONTH_NAMES.get(m.group(2).lower(), 12)
        yr = int(m.group(3)) if m.group(3) else current_year
        last_day = _last_day_of_month(yr, end_month)
        return [_make_date_condition(col, date(yr, start_month, 1), date(yr, end_month, last_day))]

    # --- Pattern 5: Year only: "2023 data", "show 2024" ---
    m = _RE_YEAR_ONLY.search(message)
    if m:
        yr = int(m.group(1))
        return [_make_date_condition(col, date(yr, 1, 1), date(yr, 12, 31))]

    # --- Pattern 6: Single month name: "filter to March" / "March 2024" ---
    m = _RE_SINGLE_MONTH.search(message)
    if m:
        mn = _MONTH_NAMES.get(m.group(1).lower(), 1)
        yr = int(m.group(2)) if m.group(2) else current_year
        last_day = _last_day_of_month(yr, mn)
        return [_make_date_condition(col, date(yr, mn, 1), date(yr, mn, last_day))]

    return None


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
    elif op == "date_range":
        try:
            dates = pd.to_datetime(series, errors="coerce")
            start = pd.Timestamp(val["start"])
            end = pd.Timestamp(val["end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            return df[(dates >= start) & (dates <= end)]
        except Exception:
            return df
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
