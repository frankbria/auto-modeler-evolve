"""AI-powered data dictionary generator.

Classifies each column by its likely semantic role (ID, metric, dimension,
date, flag, text) and generates a plain-English description that a business
analyst can understand without needing to know the data schema.

When ANTHROPIC_API_KEY is available, Claude generates rich, context-aware
descriptions for the whole dataset in one shot. Otherwise a deterministic
rule-based fallback provides useful descriptions based on column names,
types, and statistics alone.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

# ---------------------------------------------------------------------------
# Column type classification
# ---------------------------------------------------------------------------

_DATE_HINTS = re.compile(
    r"(date|time|at$|_at|created|updated|modified|timestamp|day|month|year|period|week|quarter)",
    re.IGNORECASE,
)
_ID_HINTS = re.compile(
    r"(^id$|_id$|^id_|_key$|^key$|_code$|^code$|_num$|^num$|_no$|^no$|_ref$|^ref$|uuid|guid)",
    re.IGNORECASE,
)
_METRIC_HINTS = re.compile(
    r"(amount|revenue|cost|price|sales|quantity|qty|count|rate|score|value|total|sum|avg|mean|"
    r"profit|loss|margin|spend|budget|volume|size|weight|height|width|length|duration|age|"
    r"salary|income|balance|fee|charge|payment|units|inventory|stock|forecast)",
    re.IGNORECASE,
)
_DIMENSION_HINTS = re.compile(
    r"(region|country|state|city|location|area|zone|category|type|status|group|segment|tier|"
    r"channel|source|medium|campaign|product|brand|model|version|level|grade|class|team|dept|"
    r"department|division|unit|sector|industry|market|label|tag|flag|gender|role|plan|tier)",
    re.IGNORECASE,
)


def classify_column_type(
    col_name: str,
    dtype: str,
    unique_count: int,
    row_count: int,
    sample_values: list[Any],
    null_pct: float = 0.0,
) -> str:
    """Return one of: id | metric | dimension | date | flag | text | unknown."""
    if row_count == 0:
        return "unknown"

    unique_ratio = unique_count / row_count
    lower = col_name.lower()

    # Date / timestamp
    if "datetime" in dtype or "date" in dtype or _DATE_HINTS.search(lower):
        return "date"

    # Boolean / flag (2-valued or bool dtype)
    if dtype == "bool" or unique_count <= 2:
        return "flag"

    # String / object columns — check text length first (before ID check)
    if dtype == "object":
        str_samples = [str(v) for v in sample_values if v is not None]
        avg_len = (
            sum(len(s) for s in str_samples) / len(str_samples) if str_samples else 0
        )
        if avg_len > 60:
            return "text"

    # ID column — high cardinality + name hints, or object dtype with > 80% unique
    if _ID_HINTS.search(lower) or (
        unique_ratio > 0.85 and dtype in ("object", "int64", "int32", "float64")
    ):
        # Only classify as ID if not clearly a metric
        if not _METRIC_HINTS.search(lower):
            return "id"

    # Numeric metric
    if "int" in dtype or "float" in dtype:
        if _METRIC_HINTS.search(lower):
            return "metric"
        # High-cardinality numerics are likely metrics even without name hints
        if unique_ratio > 0.1:
            return "metric"
        # Low-cardinality numeric — treat as dimension (e.g., "store_tier = 1/2/3")
        return "dimension"

    # String / object columns (text case already handled above)
    if dtype == "object":
        # Dimension hints
        if _DIMENSION_HINTS.search(lower):
            return "dimension"
        # Low cardinality → dimension
        if unique_ratio < 0.15:
            return "dimension"
        # Medium cardinality → could be ID or dimension
        if unique_ratio > 0.6 and not _DIMENSION_HINTS.search(lower):
            return "id"
        return "dimension"

    return "unknown"


# ---------------------------------------------------------------------------
# Static description generator (no Claude)
# ---------------------------------------------------------------------------

_TYPE_TEMPLATES: dict[str, str] = {
    "id": (
        "A unique identifier — each value appears in at most one row. "
        "Typically not useful as a predictor, but useful for joining other datasets."
    ),
    "metric": (
        "A numeric measurement or quantity. "
        "Strong candidate as a prediction target or input feature."
    ),
    "dimension": (
        "A categorical grouping or label. "
        "Useful for segmentation; will be encoded before model training."
    ),
    "date": (
        "A date or time value. "
        "Can be decomposed into day-of-week, month, or seasonality features."
    ),
    "flag": (
        "A binary indicator (yes/no, true/false, or 0/1). "
        "Used directly as a feature or as a prediction target for binary classification."
    ),
    "text": (
        "Free-form text content. "
        "Not directly usable by most models; consider extracting keywords or sentiment."
    ),
    "unknown": "A column whose type could not be determined automatically.",
}


def _static_description(col_name: str, col_type: str, stats: dict[str, Any]) -> str:
    """Build a deterministic description from type + statistics."""
    base = _TYPE_TEMPLATES.get(col_type, _TYPE_TEMPLATES["unknown"])
    parts = [base]

    null_pct = stats.get("null_pct", 0)
    if null_pct > 0:
        parts.append(f"{null_pct:.1f}% of values are missing.")

    if col_type == "metric":
        mn = stats.get("min")
        mx = stats.get("max")
        mean = stats.get("mean")
        if mn is not None and mx is not None:
            parts.append(
                f"Range: {mn:.2g} – {mx:.2g}"
                + (f" (avg {mean:.2g})." if mean is not None else ".")
            )

    if col_type == "dimension":
        uc = stats.get("unique_count")
        if uc is not None:
            parts.append(f"{uc} distinct value{'s' if uc != 1 else ''}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Claude-powered description generator
# ---------------------------------------------------------------------------

_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
_CLAUDE_MAX_TOKENS = 1024


def _call_claude_for_dictionary(
    filename: str,
    row_count: int,
    col_summaries: list[dict[str, Any]],
) -> dict[str, str] | None:
    """Call Claude to generate descriptions for all columns in one shot.

    Returns a dict {col_name: description} or None if the call fails / no key.
    """
    if not (
        os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    ):
        return None

    col_lines = []
    for c in col_summaries:
        stats_str = f"type={c['col_type']}, dtype={c['dtype']}, unique={c['unique_count']}, null_pct={c['null_pct']:.1f}%"
        if c.get("min") is not None:
            stats_str += f", range={c['min']:.2g}–{c['max']:.2g}"
        if c.get("sample_values"):
            sample_str = ", ".join(str(v) for v in c["sample_values"][:3])
            stats_str += f", samples=[{sample_str}]"
        col_lines.append(f"- {c['name']}: {stats_str}")

    prompt = (
        f"You are analysing a dataset called '{filename}' with {row_count:,} rows.\n\n"
        "Columns:\n"
        + "\n".join(col_lines)
        + "\n\nFor each column, write a 1-2 sentence plain-English description a business analyst "
        "would understand. Focus on what the column represents and its practical use. "
        "Keep each description under 120 characters.\n\n"
        "Respond with ONLY a JSON object mapping column name to description string. "
        'Example: {"column_name": "Description here.", "other_col": "Another description."}'
    )

    try:
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=_CLAUDE_MODEL,
            max_tokens=_CLAUDE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()  # type: ignore[index]
        # Strip markdown fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_dictionary(
    columns_data: list[dict[str, Any]],
    filename: str = "dataset",
    row_count: int = 0,
) -> list[dict[str, Any]]:
    """Generate a data dictionary for the given column stats list.

    Each item in the returned list is the original column dict extended with:
    - ``col_type``: semantic role ("id" | "metric" | "dimension" | "date" | "flag" | "text")
    - ``description``: plain-English description

    Args:
        columns_data: List of column stat dicts from ``analyze_dataframe()``.
        filename: Dataset filename (used in Claude prompt).
        row_count: Total row count (used for unique-ratio computation).

    Returns:
        Extended list of column dicts.
    """
    # Step 1 — classify every column
    enriched = []
    for col in columns_data:
        col_type = classify_column_type(
            col_name=col["name"],
            dtype=col["dtype"],
            unique_count=col["unique_count"],
            row_count=row_count or 1,
            sample_values=col.get("sample_values", []),
            null_pct=col.get("null_pct", 0.0),
        )
        enriched.append({**col, "col_type": col_type})

    # Step 2 — try Claude for all descriptions in one call
    claude_descriptions = _call_claude_for_dictionary(filename, row_count, enriched)

    # Step 3 — apply descriptions (Claude if available, fallback otherwise)
    result = []
    for col in enriched:
        if claude_descriptions and col["name"] in claude_descriptions:
            description = claude_descriptions[col["name"]]
        else:
            description = _static_description(col["name"], col["col_type"], col)
        result.append({**col, "description": description})

    return result
