"""Prompt templates and content library for the AutoModeler chat system.

Provides:
- ALGORITHM_INTROS: Plain-English descriptions for each supported ML algorithm
- METRIC_GLOSSARY: Plain-English definitions and thresholds for ML metrics
- Stage-specific conversation starters for proactive guidance
- Helper functions to format metrics and build focused Claude prompts
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Algorithm descriptions (plain English, non-technical)
# ---------------------------------------------------------------------------

ALGORITHM_INTROS: dict[str, str] = {
    # Regression
    "LinearRegression": (
        "Linear Regression draws a straight line through your data. "
        "It's the simplest predictor — transparent, fast, and easy to explain to your team. "
        "Best when the relationship between inputs and your target is roughly linear."
    ),
    "RandomForestRegressor": (
        "Random Forest asks 100 independent decision trees and averages their answers. "
        "More accurate than a single tree, handles complex patterns well, and tells you "
        "which inputs matter most. A great all-around choice."
    ),
    "GradientBoostingRegressor": (
        "Gradient Boosting builds trees one at a time, each one correcting the mistakes "
        "of the last. Usually the most accurate option, but takes longer to train. "
        "Good when you need maximum precision."
    ),
    # Classification
    "LogisticRegression": (
        "Logistic Regression predicts which category something belongs to by finding "
        "the best boundary between groups. Fast, interpretable, and a solid baseline — "
        "works well when your categories are reasonably separable."
    ),
    "RandomForestClassifier": (
        "Random Forest Classifier votes across 100 independent trees to decide which "
        "category fits best. Handles messy, complex patterns and gives you a "
        "confidence score alongside each prediction."
    ),
    "GradientBoostingClassifier": (
        "Gradient Boosting Classifier builds trees sequentially, each focused on the "
        "hardest cases from the round before. Typically the most accurate classifier, "
        "ideal when every correct prediction matters."
    ),
}

# Short tagline for each algorithm (used in comparison summaries)
ALGORITHM_TAGLINES: dict[str, str] = {
    "LinearRegression": "Simple & transparent",
    "RandomForestRegressor": "Accurate & reliable",
    "GradientBoostingRegressor": "Maximum accuracy",
    "LogisticRegression": "Fast & interpretable",
    "RandomForestClassifier": "Accurate & robust",
    "GradientBoostingClassifier": "Highest precision",
}


# ---------------------------------------------------------------------------
# Metric glossary
# ---------------------------------------------------------------------------

METRIC_GLOSSARY: dict[str, dict] = {
    # Regression metrics
    "r2": {
        "name": "R² (R-squared)",
        "plain": "how well the model explains variation in your data",
        "scale": "0 to 1 — higher is better. 0.8+ is good; 0.9+ is excellent.",
        "threshold_good": 0.80,
        "threshold_excellent": 0.90,
        "higher_is_better": True,
    },
    "mae": {
        "name": "MAE (Mean Absolute Error)",
        "plain": "average gap between predicted and actual values",
        "scale": "lower is better — in the same units as your target column",
        "higher_is_better": False,
    },
    "rmse": {
        "name": "RMSE (Root Mean Square Error)",
        "plain": "like MAE but penalises large errors more heavily",
        "scale": "lower is better — in the same units as your target column",
        "higher_is_better": False,
    },
    # Classification metrics
    "accuracy": {
        "name": "Accuracy",
        "plain": "percentage of predictions that were correct",
        "scale": "0 to 1 — higher is better. 0.85+ is good; 0.95+ is excellent.",
        "threshold_good": 0.85,
        "threshold_excellent": 0.95,
        "higher_is_better": True,
    },
    "f1": {
        "name": "F1 Score",
        "plain": "balance between catching all positives and avoiding false alarms",
        "scale": "0 to 1 — higher is better. Useful when classes are imbalanced.",
        "threshold_good": 0.80,
        "threshold_excellent": 0.90,
        "higher_is_better": True,
    },
    "precision": {
        "name": "Precision",
        "plain": "when the model says 'yes', how often is it right?",
        "scale": "0 to 1 — higher is better",
        "higher_is_better": True,
    },
    "recall": {
        "name": "Recall",
        "plain": "what fraction of real positives did the model catch?",
        "scale": "0 to 1 — higher is better",
        "higher_is_better": True,
    },
}


def format_metric(name: str, value: float | None) -> str:
    """Return a human-readable metric string with plain-English context.

    Example:
        format_metric("r2", 0.87)
        → "R² = 0.87 — how well the model explains variation in your data"
    """
    if value is None:
        return f"{name} = N/A"

    info = METRIC_GLOSSARY.get(name.lower(), {})
    display_name = info.get("name", name.upper())
    plain = info.get("plain", "")

    formatted_value = f"{value:.1%}" if name.lower() in ("accuracy", "f1", "precision", "recall") else f"{value:.3f}"

    result = f"{display_name} = {formatted_value}"
    if plain:
        result += f" — {plain}"

    # Add qualitative label for metrics with thresholds
    if info.get("higher_is_better") and info.get("threshold_excellent"):
        if value >= info["threshold_excellent"]:
            result += " ✓ (excellent)"
        elif value >= info.get("threshold_good", 0):
            result += " ✓ (good)"
        elif value < 0.5:
            result += " (needs improvement)"

    return result


def summarise_metrics(metrics: dict, problem_type: str) -> str:
    """Return a short multi-line metrics summary with plain-English context."""
    lines = []
    if problem_type == "regression":
        for key in ("r2", "mae", "rmse"):
            if key in metrics:
                lines.append(f"  • {format_metric(key, metrics[key])}")
    else:
        for key in ("accuracy", "f1", "precision", "recall"):
            if key in metrics:
                lines.append(f"  • {format_metric(key, metrics[key])}")
    return "\n".join(lines) if lines else "  • No metrics available"


# ---------------------------------------------------------------------------
# Proactive insight prompt builder
# ---------------------------------------------------------------------------

def build_proactive_insight_prompt(
    dataset_summary: str,
    profile_highlights: str,
    n_rows: int,
    n_cols: int,
) -> str:
    """Build a Claude prompt to surface 2-3 interesting, non-obvious data insights.

    The response from this prompt should be short (3-5 sentences) and written
    in plain English as if narrated by a friendly data analyst colleague.
    """
    return (
        f"You are a friendly data analyst looking at a dataset with {n_rows:,} rows "
        f"and {n_cols} columns.\n\n"
        f"Dataset summary:\n{dataset_summary}\n\n"
        f"Profile highlights:\n{profile_highlights}\n\n"
        "In 2-4 sentences, share the most interesting or surprising pattern you notice "
        "in this data. Speak directly to the user (use 'your data', 'I noticed'). "
        "Be specific — mention actual column names and numbers. "
        "Don't use bullet points. Write in plain English, no jargon. "
        "End with one concrete question to guide next steps."
    )


def build_model_comparison_narrative_prompt(
    models: list[dict],
    problem_type: str,
    target_column: str,
) -> str:
    """Build a Claude prompt to narrate a model comparison in plain English.

    Args:
        models: List of model run dicts with 'algorithm', 'metrics', 'summary' keys.
        problem_type: 'regression' or 'classification'
        target_column: The column being predicted.
    """
    model_lines = []
    for m in models:
        algo = m.get("algorithm", "?")
        tagline = ALGORITHM_TAGLINES.get(algo, "")
        metrics = m.get("metrics") or {}
        metrics_str = summarise_metrics(metrics, problem_type)
        model_lines.append(f"  {algo} ({tagline}):\n{metrics_str}")

    models_block = "\n\n".join(model_lines)

    primary_metric = "R²" if problem_type == "regression" else "accuracy"

    return (
        f"You trained {len(models)} models to predict '{target_column}' "
        f"({problem_type}).\n\n"
        f"{models_block}\n\n"
        f"In 3-5 sentences, explain what these results mean in plain English. "
        f"Which model performed best on {primary_metric} and why might someone choose it? "
        f"Is there a trade-off worth noting (e.g. accuracy vs. explainability)? "
        f"Speak directly to the user. End by recommending one model and why."
    )


# ---------------------------------------------------------------------------
# Stage-specific opening messages (used when a new stage begins)
# ---------------------------------------------------------------------------

STAGE_OPENERS: dict[str, str] = {
    "upload": (
        "Hi! I'm your AutoModeler assistant. To get started, upload a CSV file "
        "using the drop zone on the right — or click **Load sample data** to try "
        "with a built-in sales dataset. What kind of data are you working with today?"
    ),
    "explore": (
        "Your data is loaded and I've run a quick analysis. "
        "Ask me anything about it — patterns, trends, which columns matter, "
        "comparisons across groups. When you're ready, we'll move on to building a model."
    ),
    "shape": (
        "Time to shape your features! I've suggested some transformations that "
        "might help the model learn better patterns. Review them in the Features tab, "
        "approve or reject each one, and tell me which column you want to predict."
    ),
    "model": (
        "Features are set — let's train some models. Head to the Models tab to see "
        "my algorithm recommendations, or just ask me to pick the best ones for you. "
        "Training usually takes under a minute."
    ),
    "validate": (
        "Your models are trained! Now let's understand how well they perform and "
        "where they might struggle. Check the Validate tab for detailed results, "
        "or ask me to explain the numbers in plain English."
    ),
    "deploy": (
        "Your model is live! You can share the prediction dashboard link or "
        "connect the API to your own tools. Ask me anything about how to use it."
    ),
}
