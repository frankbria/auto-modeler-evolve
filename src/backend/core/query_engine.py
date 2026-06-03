"""query_engine.py

Natural-language data query pipeline.

Flow:
  user question
    → Claude parses it into a structured QuerySpec (JSON)
    → execute_query_spec() runs it against the real dataframe
    → chart_from_query_result() picks the best chart type
    → returns QueryResult(text, chart_spec)

QuerySpec is intentionally simple — no arbitrary code execution. This keeps
the system safe and testable while covering 90% of analyst questions.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import anthropic
import pandas as pd

from core.chart_builder import chart_from_query_result

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Outcome of running a natural-language query against a dataframe."""

    text: str  # Human-readable answer
    chart_spec: dict[str, Any] | None  # Recharts config, or None
    result_rows: list[dict] = field(default_factory=list)  # Raw data rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_nl_query(
    question: str,
    df: pd.DataFrame,
    column_info: list[dict],
) -> QueryResult:
    """Parse a natural-language question and execute it against `df`.

    Returns a QueryResult with a plain-English answer and optional chart spec.
    """
    # Step 1: Ask Claude to parse the question into a QuerySpec
    spec = _parse_question_to_spec(question, df, column_info)
    if spec is None:
        return QueryResult(
            text="I couldn't understand that question well enough to query the data. "
            "Try asking something like 'which region has the highest revenue?' "
            "or 'show me the distribution of sales'.",
            chart_spec=None,
        )

    # Step 2: Execute the structured query against the dataframe
    try:
        result_df, answer_text = _execute_spec(spec, df)
    except Exception as exc:  # noqa: BLE001
        return QueryResult(
            text=f"I understood the question but hit an error running it: {exc}. "
            "Try rephrasing or asking about a specific column.",
            chart_spec=None,
        )

    if result_df is None or result_df.empty:
        return QueryResult(
            text=answer_text or "No data matched that query.", chart_spec=None
        )

    # Step 3: Build chart spec
    x_col = spec.get("x_col")
    y_col = spec.get("y_col")
    chart = chart_from_query_result(result_df, question, x_col=x_col, y_col=y_col)

    return QueryResult(
        text=answer_text,
        chart_spec=chart,
        result_rows=_safe_rows(result_df.head(20)),
    )


def generate_chart_for_message(
    message: str,
    df: pd.DataFrame,
    column_info: list[dict],
    assistant_response: str,
) -> dict[str, Any] | None:
    """Opportunistically generate a chart to accompany a chat response.

    Called after the text stream completes. Returns a Recharts spec or None.
    """
    # Only bother if the assistant response contains numbers/comparisons
    data_keywords = [
        "highest",
        "lowest",
        "most",
        "least",
        "average",
        "mean",
        "total",
        "distribution",
        "trend",
        "pattern",
        "correlation",
        "compare",
        "comparison",
        "percent",
        "percentage",
        "top",
        "bottom",
        "increase",
        "decrease",
        "over time",
    ]
    response_lower = assistant_response.lower()
    question_lower = message.lower()

    has_data_intent = any(
        kw in response_lower or kw in question_lower for kw in data_keywords
    )

    if not has_data_intent:
        return None

    result = run_nl_query(message, df, column_info)
    return result.chart_spec


# ---------------------------------------------------------------------------
# Internal: LLM parsing
# ---------------------------------------------------------------------------

_PARSE_SYSTEM = """You are a data query parser. Given a natural-language question about a dataset,
output a JSON object describing the query. Only return valid JSON — no prose.

Schema:
{
  "operation": "groupby" | "distribution" | "filter" | "timeseries" | "top_n" | "correlation",
  "columns": ["col1", "col2"],   // columns involved
  "group_by": ["col"] | null,
  "aggregate": {"col": "sum" | "mean" | "count" | "min" | "max"} | null,
  "filter": {"column": "col", "operator": ">" | "<" | "==" | "contains", "value": any} | null,
  "sort_ascending": true | false,
  "limit": 10 | null,
  "x_col": "col" | null,   // which column is the x-axis
  "y_col": "col" | null    // which column is the y-axis (the metric)
}

Rules:
- For "which X has highest Y?" use operation=top_n, group_by=[X], aggregate={Y: "sum"}, sort_ascending=false, limit=10
- For "distribution of X" use operation=distribution, columns=[X]
- For "trend over time" use operation=timeseries, x_col=date_column, y_col=metric_column
- For "correlation between X and Y" use operation=correlation, columns=[X, Y]
- Only use column names that exist in the dataset schema provided.
- If the question cannot be answered with data, return null.
"""


def _parse_question_to_spec(
    question: str,
    df: pd.DataFrame,
    column_info: list[dict],
) -> dict | None:
    """Ask Claude to convert the question into a QuerySpec dict."""
    schema_lines = [f"- {c['name']} ({c['dtype']})" for c in column_info]
    schema_str = "\n".join(schema_lines)
    prompt = f"Dataset columns:\n{schema_str}\n\nQuestion: {question}"

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=os.environ.get("QUERY_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=512,
            system=_PARSE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        if raw.lower() == "null":
            return None
        return json.loads(raw)
    except Exception:  # noqa: BLE001 — catch auth errors, network errors, parse errors
        return None


# ---------------------------------------------------------------------------
# Internal: Query execution
# ---------------------------------------------------------------------------


def _execute_spec(spec: dict, df: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    """Execute a QuerySpec against a dataframe. Returns (result_df, text_summary)."""
    op = spec.get("operation", "")

    if op == "distribution":
        columns = spec.get("columns", [])
        col = _find_col(columns, df)
        if col is None:
            return None, "Column not found."
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            text = (
                f"Distribution of **{col}**:\n"
                f"- Min: {desc['min']:.4g}, Max: {desc['max']:.4g}\n"
                f"- Mean: {desc['mean']:.4g}, Std: {desc['std']:.4g}\n"
                f"- 25th pct: {desc['25%']:.4g}, Median: {desc['50%']:.4g}, 75th pct: {desc['75%']:.4g}"
            )
            return df[[col]].dropna(), text
        else:
            counts = df[col].value_counts().head(10)
            lines = [f"- {k}: {v}" for k, v in counts.items()]
            text = f"Value distribution for **{col}**:\n" + "\n".join(lines)
            result_df = counts.reset_index()
            result_df.columns = [col, "count"]
            return result_df, text

    elif op in ("groupby", "top_n"):
        group_by = spec.get("group_by") or []
        aggregate = spec.get("aggregate") or {}
        sort_asc = spec.get("sort_ascending", False)
        limit = spec.get("limit") or 10

        valid_groups = [c for c in group_by if c in df.columns]
        valid_agg = {c: f for c, f in aggregate.items() if c in df.columns}

        if not valid_groups or not valid_agg:
            return None, "Couldn't identify the grouping or metric columns."

        result = df.groupby(valid_groups).agg(valid_agg).reset_index()
        result.columns = [
            "_".join(filter(None, [g if isinstance(g, str) else g[0]]))
            for g in result.columns
        ]

        # Sort and limit
        metric_cols = [c for c in result.columns if c not in valid_groups]
        if metric_cols:
            result = result.sort_values(metric_cols[0], ascending=sort_asc).head(limit)

        group_str = " and ".join(valid_groups)
        metric_str = ", ".join(f"{fn}({col})" for col, fn in valid_agg.items())
        direction = "lowest" if sort_asc else "highest"
        text = (
            f"Top {limit} results for **{metric_str}** by **{group_str}** ({direction} first):\n"
            + _df_to_text(result.head(10))
        )
        return result, text

    elif op == "timeseries":
        x_col = spec.get("x_col") or (spec.get("columns") or [None])[0]
        y_col = spec.get("y_col") or (spec.get("columns") or [None, None])[1]

        if not x_col or x_col not in df.columns:
            return None, "Date column not found."
        if not y_col or y_col not in df.columns:
            return None, "Metric column not found."

        ts_df = df[[x_col, y_col]].dropna().sort_values(x_col)
        text = (
            f"Trend of **{y_col}** over **{x_col}** — showing {len(ts_df)} data points."
        )
        return ts_df, text

    elif op == "correlation":
        columns = spec.get("columns", [])
        valid = [
            c
            for c in columns
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(valid) < 2:
            return None, "Need two numeric columns to show correlation."
        c1, c2 = valid[0], valid[1]
        corr_val = df[c1].corr(df[c2])
        strength = (
            "strong"
            if abs(corr_val) >= 0.7
            else "moderate" if abs(corr_val) >= 0.4 else "weak"
        )
        direction = "positive" if corr_val >= 0 else "negative"
        text = (
            f"**{c1}** and **{c2}** have a {strength} {direction} correlation "
            f"(r = {corr_val:.3f})."
        )
        return df[[c1, c2]].dropna(), text

    elif op == "filter":
        filt = spec.get("filter") or {}
        col = filt.get("column")
        operator = filt.get("operator")
        value = filt.get("value")

        if not col or col not in df.columns:
            return None, "Filter column not found."

        result = _apply_filter(df, col, operator, value)
        text = f"Found {len(result)} rows where **{col}** {operator} {value}."
        return result.head(20), text

    return None, "I'm not sure how to answer that with the data available."


def _find_col(columns: list[str], df: pd.DataFrame) -> str | None:
    """Return the first column name that exists in df."""
    for c in columns:
        if c in df.columns:
            return c
    return None


def _apply_filter(
    df: pd.DataFrame, col: str, operator: str, value: Any
) -> pd.DataFrame:
    series = df[col]
    if operator == ">":
        return df[series > value]
    elif operator == "<":
        return df[series < value]
    elif operator == ">=":
        return df[series >= value]
    elif operator == "<=":
        return df[series <= value]
    elif operator in ("==", "="):
        return df[series == value]
    elif operator == "contains":
        return df[series.astype(str).str.contains(str(value), case=False, na=False)]
    return df


def _df_to_text(df: pd.DataFrame) -> str:
    """Render a small dataframe as Markdown-ish bullet list."""
    lines = []
    for _, row in df.iterrows():
        parts = [f"{k}={v}" for k, v in row.items()]
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def _safe_rows(df: pd.DataFrame) -> list[dict]:
    """Convert dataframe rows to JSON-safe dicts."""
    import numpy as np

    rows = []
    for record in df.to_dict(orient="records"):
        safe = {}
        for k, v in record.items():
            if isinstance(v, float) and (v != v):  # NaN check
                safe[k] = None
            elif isinstance(v, np.generic):
                safe[k] = v.item()
            else:
                safe[k] = v
        rows.append(safe)
    return rows
