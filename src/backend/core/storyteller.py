"""storyteller.py

Automated Data Story — orchestrates multiple analysis modules into a single
cohesive narrative for business analysts.

When a user asks "analyze my data", "walk me through this", or "what's
interesting here?", this module runs all relevant analyses and returns a
structured DataStory with key findings and a recommended next step.

Design:
- Pure orchestration: calls existing analysis functions, no new ML
- Each section is independently optional (graceful fallback if analysis fails)
- Returns structured dict suitable for SSE + frontend card rendering
- Fast enough for synchronous chat handler use (<500ms on typical datasets)
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def generate_data_story(
    df: pd.DataFrame,
    dataset_id: str,
    target_col: str | None = None,
    dataset_filename: str | None = None,
) -> dict[str, Any]:
    """Orchestrate a comprehensive data analysis narrative.

    Runs up to 4 analysis modules and combines results into a DataStory:
      1. Data readiness (always)
      2. Top group-by breakdown on best categorical column (if any exist)
      3. Target correlations (if target_col is provided)
      4. Anomaly count (if numeric columns exist)

    Returns a dict with:
      dataset_id, filename, row_count, col_count,
      readiness_score, readiness_grade,
      sections (list), summary (str), recommended_next_step (str)
    """
    row_count = len(df)
    col_count = len(df.columns)
    sections: list[dict] = []

    # ------------------------------------------------------------------ #
    # Section 1: Data readiness                                           #
    # ------------------------------------------------------------------ #
    readiness_score = 0
    readiness_grade = "F"
    readiness_status = "not_ready"
    try:
        from core.readiness import compute_data_readiness

        rd = compute_data_readiness(df, target_col=target_col)
        readiness_score = rd["score"]
        readiness_grade = rd["grade"]
        readiness_status = rd["status"]
        rec_text = rd["recommendations"][0] if rd["recommendations"] else ""
        sections.append(
            {
                "type": "readiness",
                "title": "Data Quality",
                "insight": f"Grade {readiness_grade} ({readiness_score}/100). {rec_text}",
                "data": rd,
            }
        )
    except Exception:  # noqa: BLE001
        pass

    # ------------------------------------------------------------------ #
    # Section 2: Group-by breakdown on best categorical column            #
    # ------------------------------------------------------------------ #
    try:
        from core.analyzer import compute_group_stats

        cat_cols = [
            c
            for c in df.columns
            if (
                df[c].dtype == object
                or pd.api.types.is_string_dtype(df[c])
            )
            and 2 <= df[c].nunique() <= 20
        ]
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if cat_cols and numeric_cols:
            # Pick the categorical column with the most interesting spread
            # (column with moderate unique count relative to row count)
            best_cat = min(
                cat_cols,
                key=lambda c: abs(df[c].nunique() - min(10, row_count // 10)),
            )
            value_col = numeric_cols[0] if numeric_cols else None
            grp_result = compute_group_stats(df, best_cat, value_cols=[value_col] if value_col else None)
            if not grp_result.get("error"):
                sections.append(
                    {
                        "type": "group_by",
                        "title": f"Breakdown by {best_cat}",
                        "insight": grp_result["summary"],
                        "data": grp_result,
                    }
                )
    except Exception:  # noqa: BLE001
        pass

    # ------------------------------------------------------------------ #
    # Section 3: Target correlations (if target known)                    #
    # ------------------------------------------------------------------ #
    if target_col:
        try:
            from core.analyzer import analyze_target_correlations

            corr_result = analyze_target_correlations(df, target_col, top_n=5)
            if not corr_result.get("error") and corr_result.get("correlations"):
                sections.append(
                    {
                        "type": "correlations",
                        "title": f"What Drives {target_col}",
                        "insight": corr_result["summary"],
                        "data": corr_result,
                    }
                )
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ #
    # Section 4: Anomaly count                                            #
    # ------------------------------------------------------------------ #
    anomaly_count = 0
    try:
        from core.anomaly import detect_anomalies

        numeric_cols = df.select_dtypes(include="number").columns.tolist()[:8]
        if numeric_cols and row_count >= 10:
            anom_result = detect_anomalies(df, features=numeric_cols, contamination=0.05, n_top=5)
            anomaly_count = anom_result.get("anomaly_count", 0)
            if anomaly_count > 0:
                sections.append(
                    {
                        "type": "anomalies",
                        "title": "Anomaly Scan",
                        "insight": anom_result["summary"],
                        "data": anom_result,
                    }
                )
    except Exception:  # noqa: BLE001
        pass

    # ------------------------------------------------------------------ #
    # Narrative summary + recommended next step                           #
    # ------------------------------------------------------------------ #
    summary = _build_summary(
        row_count, col_count, readiness_score, readiness_grade,
        anomaly_count, target_col, sections
    )
    recommended_next_step = _recommend_next_step(readiness_status, target_col, sections)

    return {
        "dataset_id": dataset_id,
        "filename": dataset_filename or "dataset",
        "row_count": row_count,
        "col_count": col_count,
        "readiness_score": readiness_score,
        "readiness_grade": readiness_grade,
        "sections": sections,
        "summary": summary,
        "recommended_next_step": recommended_next_step,
    }


def _build_summary(
    row_count: int,
    col_count: int,
    readiness_score: int,
    readiness_grade: str,
    anomaly_count: int,
    target_col: str | None,
    sections: list[dict],
) -> str:
    """Compose a 2–3 sentence plain-English summary."""
    parts = [
        f"Your dataset has {row_count:,} rows and {col_count} columns "
        f"with a data quality grade of {readiness_grade} ({readiness_score}/100)."
    ]

    # Group-by insight
    grp = next((s for s in sections if s["type"] == "group_by"), None)
    if grp:
        parts.append(grp["insight"])

    # Correlations insight
    corr = next((s for s in sections if s["type"] == "correlations"), None)
    if corr:
        parts.append(corr["insight"])
    elif not target_col:
        parts.append(
            "Set a target column in the Features tab to see what's driving your key metrics."
        )

    # Anomaly note
    if anomaly_count > 0:
        parts.append(
            f"I also found {anomaly_count} potentially anomalous row(s) worth reviewing."
        )

    return " ".join(parts)


def _recommend_next_step(
    readiness_status: str,
    target_col: str | None,
    sections: list[dict],
) -> str:
    """Return a single recommended next action."""
    if readiness_status == "not_ready":
        rd = next((s for s in sections if s["type"] == "readiness"), None)
        if rd and rd["data"].get("recommendations"):
            return f"Fix data quality first: {rd['data']['recommendations'][0]}"
        return "Improve data quality before training — check the readiness details above."

    if not target_col:
        return (
            "Set a target column in the Features tab — tell me what you want to predict "
            "(e.g. 'I want to predict revenue') and I'll set it up automatically."
        )

    if readiness_status in ("ready", "needs_attention"):
        return (
            "Your data looks ready to model! Say 'train a model to predict "
            + target_col
            + "' and I'll kick off training immediately."
        )

    return "Explore your data further — ask about specific patterns or try 'what drives " + (target_col or "my metrics") + "'."
