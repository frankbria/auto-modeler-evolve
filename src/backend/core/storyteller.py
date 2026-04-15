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
            if (df[c].dtype == object or pd.api.types.is_string_dtype(df[c]))
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
            grp_result = compute_group_stats(
                df, best_cat, value_cols=[value_col] if value_col else None
            )
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
            anom_result = detect_anomalies(
                df, features=numeric_cols, contamination=0.05, n_top=5
            )
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
        row_count,
        col_count,
        readiness_score,
        readiness_grade,
        anomaly_count,
        target_col,
        sections,
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
        return (
            "Improve data quality before training — check the readiness details above."
        )

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

    return (
        "Explore your data further — ask about specific patterns or try 'what drives "
        + (target_col or "my metrics")
        + "'."
    )


# ---------------------------------------------------------------------------
# Executive Briefing Generator
# ---------------------------------------------------------------------------


def generate_executive_briefing(
    project_name: str,
    dataset_filename: str | None,
    row_count: int | None,
    col_count: int | None,
    target_column: str | None,
    problem_type: str | None,
    algorithm: str | None,
    primary_metric_name: str | None,
    primary_metric_value: float | None,
    deployment_url: str | None,
    request_count: int | None,
    conversation_snippet: str | None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    """Generate a structured executive briefing for sharing with a VP or stakeholder.

    Returns a dict with:
      project_name, target_column, problem_type,
      sections: list of {heading, body} dicts (plain-English narrative),
      summary (one-sentence headline),
      action_items (list of recommended actions for the stakeholder),
      prediction_url (if deployed)
    """
    algo_plain = _plain_algo_name(algorithm)
    metric_label, metric_explanation = _metric_explanation(
        primary_metric_name, primary_metric_value
    )

    sections: list[dict[str, str]] = []

    # Section 1: What we analyzed
    if dataset_filename or row_count:
        data_desc = f"{row_count:,} rows" if row_count else "your data"
        if col_count:
            data_desc += f" across {col_count} columns"
        if dataset_filename:
            data_desc = f'"{dataset_filename}" ({data_desc})'
        sections.append(
            {
                "heading": "What We Analyzed",
                "body": (
                    f"We analyzed {data_desc} to build a predictive model. "
                    + (
                        f"The goal was to predict **{target_column}** "
                        f"using the patterns found in the historical data."
                        if target_column
                        else "The dataset was explored for patterns and prediction opportunities."
                    )
                ),
            }
        )

    # Section 2: The model
    if algorithm:
        sections.append(
            {
                "heading": "The Prediction Model",
                "body": (
                    f"We used a **{algo_plain}** algorithm — "
                    + _algo_description(algorithm)
                    + (
                        f" The model predicts **{target_column}**."
                        if target_column
                        else ""
                    )
                ),
            }
        )

    # Section 3: How accurate is it?
    if primary_metric_value is not None:
        sections.append(
            {
                "heading": "How Accurate Is It?",
                "body": metric_explanation,
            }
        )

    # Section 4: What was found (key insights)
    if target_column and primary_metric_value is not None:
        sections.append(
            {
                "heading": "Key Findings",
                "body": (
                    f"The model has identified the key patterns in your data that "
                    f"drive **{target_column}**. These patterns were learned from "
                    f"historical records and can now be applied to new inputs for "
                    f"instant predictions. "
                    + (
                        f"The model has been used {request_count:,} time{'s' if request_count != 1 else ''} so far."
                        if request_count
                        else "The model is ready for use."
                    )
                ),
            }
        )

    # Section 5: Limitations / caveats
    lim_list = limitations or []
    if not lim_list:
        # Generate sensible default limitations
        if problem_type == "regression":
            lim_list = [
                "Predictions are most reliable within the range of values seen in training data.",
                "Significant changes in business conditions may reduce accuracy over time — consider retraining quarterly.",
            ]
        elif problem_type == "classification":
            lim_list = [
                "Confidence scores indicate probability, not certainty — always review borderline predictions.",
                "New customer segments or product categories not seen in training may produce less reliable results.",
            ]
        else:
            lim_list = [
                "Predictions are based on historical patterns and should be reviewed by a domain expert.",
            ]
    if lim_list:
        bullet_lines = "\n".join(f"• {item}" for item in lim_list)
        sections.append(
            {
                "heading": "What To Watch Out For",
                "body": bullet_lines,
            }
        )

    # Section 6: Recommended actions
    action_items: list[str] = []
    if deployment_url:
        action_items.append(
            "Use the prediction dashboard at the link below to run scenarios and share results with your team."
        )
    if primary_metric_value is not None:
        if problem_type == "regression" and primary_metric_value < 0.70:
            action_items.append(
                "Consider collecting more data or additional features to improve model accuracy before relying on it for key decisions."
            )
        elif problem_type == "classification" and primary_metric_value < 0.75:
            action_items.append(
                "Accuracy is moderate — validate predictions against known outcomes before using for high-stakes decisions."
            )
    action_items.append(
        "Schedule a quarterly model review to check whether accuracy has changed as new data arrives."
    )

    # One-sentence summary
    summary = _build_briefing_summary(
        project_name, target_column, algo_plain, metric_label, deployment_url
    )

    return {
        "project_name": project_name,
        "target_column": target_column,
        "problem_type": problem_type,
        "algorithm": algo_plain,
        "metric_label": metric_label,
        "metric_value": primary_metric_value,
        "sections": sections,
        "summary": summary,
        "action_items": action_items,
        "prediction_url": deployment_url,
    }


def _plain_algo_name(algorithm: str | None) -> str:
    """Convert a raw algorithm ID to a plain-English name."""
    _MAP = {
        "linear_regression": "Linear Regression",
        "ridge": "Ridge Regression",
        "lasso": "Lasso Regression",
        "random_forest_regressor": "Random Forest",
        "random_forest_classifier": "Random Forest",
        "gradient_boosting_regressor": "Gradient Boosting",
        "gradient_boosting_classifier": "Gradient Boosting",
        "logistic_regression": "Logistic Regression",
        "decision_tree_regressor": "Decision Tree",
        "decision_tree_classifier": "Decision Tree",
        "xgboost_regressor": "XGBoost",
        "xgboost_classifier": "XGBoost",
        "lgbm_regressor": "LightGBM",
        "lgbm_classifier": "LightGBM",
        "mlp_regressor": "Neural Network",
        "mlp_classifier": "Neural Network",
        "voting_regressor": "Voting Ensemble",
        "voting_classifier": "Voting Ensemble",
        "stacking_regressor": "Stacking Ensemble",
        "stacking_classifier": "Stacking Ensemble",
    }
    if not algorithm:
        return "Machine Learning Model"
    return _MAP.get(algorithm.lower(), algorithm.replace("_", " ").title())


def _algo_description(algorithm: str | None) -> str:
    """Return a one-sentence plain-English description of the algorithm."""
    _DESC = {
        "linear_regression": "it finds the best straight-line relationship between your inputs and the outcome.",
        "ridge": "it finds a linear relationship between your inputs and the outcome, with built-in protection against overfitting.",
        "lasso": "it finds a linear relationship and automatically removes features that don't contribute.",
        "random_forest_regressor": "it combines hundreds of decision trees and averages their predictions for a reliable result.",
        "random_forest_classifier": "it combines hundreds of decision trees and takes a majority vote to classify each record.",
        "gradient_boosting_regressor": "it builds trees sequentially, each correcting the mistakes of the previous one, for high accuracy.",
        "gradient_boosting_classifier": "it builds trees sequentially, each correcting the mistakes of the previous one, for high accuracy.",
        "logistic_regression": "it measures the probability of each outcome category based on your input features.",
        "decision_tree_regressor": "it learns a set of if/then rules from your data to produce a prediction.",
        "decision_tree_classifier": "it learns a set of if/then rules from your data to classify each record.",
        "xgboost_regressor": "it uses an advanced boosting technique that typically achieves the highest accuracy.",
        "xgboost_classifier": "it uses an advanced boosting technique that typically achieves the highest accuracy.",
        "lgbm_regressor": "it uses a fast gradient boosting technique optimised for large datasets.",
        "lgbm_classifier": "it uses a fast gradient boosting technique optimised for large datasets.",
        "mlp_regressor": "it uses a neural network to learn complex non-linear patterns in the data.",
        "mlp_classifier": "it uses a neural network to learn complex non-linear patterns in the data.",
        "voting_regressor": "it combines multiple models and averages their predictions to reduce error.",
        "voting_classifier": "it combines multiple models and takes a majority vote — often the most robust approach.",
        "stacking_regressor": "it uses a 'meta-learner' that learns how to best combine the predictions of several base models.",
        "stacking_classifier": "it uses a 'meta-learner' that learns how to best combine the predictions of several base models.",
    }
    if not algorithm:
        return ""
    return _DESC.get(algorithm.lower(), "It learned patterns from your historical data to make predictions.")


def _metric_explanation(
    metric_name: str | None, metric_value: float | None
) -> tuple[str, str]:
    """Return (short label, plain-English explanation) for a model metric."""
    if metric_name is None or metric_value is None:
        return "N/A", "Model accuracy metrics are not yet available."

    if metric_name == "r2":
        pct = round(metric_value * 100, 1)
        if metric_value >= 0.85:
            quality = "excellent"
        elif metric_value >= 0.70:
            quality = "good"
        elif metric_value >= 0.50:
            quality = "moderate"
        else:
            quality = "limited"
        label = f"R² = {metric_value:.3f}"
        explanation = (
            f"The model achieves an **R² of {metric_value:.3f}** — it explains "
            f"**{pct}% of the variation** in the outcome. "
            f"This is considered **{quality} accuracy** for this type of prediction. "
            + (
                "The model is ready for production use."
                if quality in ("excellent", "good")
                else "Some improvement may be possible with more data or feature engineering."
            )
        )
        return label, explanation

    if metric_name == "accuracy":
        pct = round(metric_value * 100, 1)
        if metric_value >= 0.90:
            quality = "excellent"
        elif metric_value >= 0.80:
            quality = "good"
        elif metric_value >= 0.70:
            quality = "moderate"
        else:
            quality = "limited"
        label = f"Accuracy = {pct}%"
        explanation = (
            f"The model correctly classifies **{pct}% of cases** in testing. "
            f"This is considered **{quality} accuracy** for this type of problem. "
            + (
                "Reliable for production use."
                if quality in ("excellent", "good")
                else "Consider validating predictions manually for high-stakes decisions."
            )
        )
        return label, explanation

    label = f"{metric_name.upper()} = {metric_value:.4f}"
    explanation = f"The model's {metric_name.upper()} score is {metric_value:.4f}."
    return label, explanation


def _build_briefing_summary(
    project_name: str,
    target_column: str | None,
    algo_plain: str,
    metric_label: str,
    deployment_url: str | None,
) -> str:
    """Build a one-sentence executive headline."""
    parts = [f'AutoModeler built a {algo_plain} model for "{project_name}"']
    if target_column:
        parts[0] += f" to predict {target_column}"
    if metric_label and metric_label != "N/A":
        parts.append(f"achieving {metric_label}")
    if deployment_url:
        parts.append("now live as a shareable prediction dashboard")
    return ", ".join(parts) + "."
