"""Chat orchestrator — builds context-aware system prompts for Claude.

The orchestrator detects where the user is in the workflow and tailors the
assistant's guidance to their current stage:

  upload → explore → shape → model → validate → deploy
"""
import json
from typing import Optional

from models.dataset import Dataset
from models.deployment import Deployment
from models.feature_set import FeatureSet
from models.model_run import ModelRun
from models.project import Project


# ---------------------------------------------------------------------------
# State detection
# ---------------------------------------------------------------------------

def detect_state(
    dataset: Optional[Dataset],
    feature_set: Optional[FeatureSet],
    model_runs: list[ModelRun],
    deployment: Optional[Deployment],
) -> str:
    """Infer the conversation state from the project's current artefacts.

    Returns one of: upload | explore | shape | model | validate | deploy
    """
    if not dataset:
        return "upload"
    if deployment and deployment.is_active:
        return "deploy"
    completed_runs = [mr for mr in model_runs if mr.status == "done"]
    if completed_runs:
        return "validate"
    if feature_set and feature_set.target_column:
        return "model"
    return "explore"


# ---------------------------------------------------------------------------
# Stage-specific guidance snippets
# ---------------------------------------------------------------------------

_STATE_GUIDANCE: dict[str, str] = {
    "upload": (
        "The user hasn't uploaded a dataset yet. Your immediate goal is to help them "
        "get started. Encourage them to upload a CSV file using the drop zone on the right, "
        "or suggest loading the built-in sample dataset. Once a file is uploaded, "
        "you will automatically profile it and share what you find."
    ),
    "explore": (
        "A dataset has been loaded. Your role now is an exploratory data analyst. "
        "Help the user understand their data: patterns, distributions, anomalies, correlations. "
        "Answer questions in plain English with specifics from the data. "
        "Encourage them to ask things like 'which region has the highest sales?' or "
        "'are there any seasonal trends?'. "
        "When they are ready to build a model, gently suggest moving to the Features tab "
        "to pick a target column and prepare features."
    ),
    "shape": (
        "The user is in feature engineering. Help them understand what transformations are "
        "suggested and why each one might improve model performance. Explain concepts simply: "
        "e.g. 'one-hot encoding turns categories into separate yes/no columns that algorithms "
        "can understand'. When they've approved features and chosen a target column, "
        "suggest moving to the Models tab to start training."
    ),
    "model": (
        "The user has prepared features and a target column. Your role is to guide them "
        "through model selection and training. Explain the algorithm options in plain language. "
        "After training, help them interpret metrics: what R² or accuracy means in practice, "
        "which model to prefer and why. Suggest moving to the Validate tab once a model "
        "is selected."
    ),
    "validate": (
        "One or more models have been trained. Help the user understand validation results: "
        "cross-validation scores, error patterns, feature importance. Be honest about "
        "limitations — if accuracy is low, explain what that means and suggest improvements. "
        "When they're satisfied, guide them to the Deploy tab to publish the model."
    ),
    "deploy": (
        "The model is live! Help the user understand how to use their deployment: "
        "share the prediction dashboard link, explain how the API works for developers, "
        "and suggest how to interpret prediction outputs. If they want to improve the model, "
        "guide them back to the Features or Models tabs."
    ),
}


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    project: Project,
    dataset: Optional[Dataset] = None,
    feature_set: Optional[FeatureSet] = None,
    model_runs: Optional[list[ModelRun]] = None,
    deployment: Optional[Deployment] = None,
) -> str:
    """Build a Claude system prompt with full project context and stage guidance.

    The prompt positions the assistant as a helpful data analyst colleague
    who speaks in plain English and explains technical terms when used.
    It also includes state-specific guidance so Claude proactively guides
    users through the UPLOAD → EXPLORE → SHAPE → MODEL → VALIDATE → DEPLOY flow.
    """
    if model_runs is None:
        model_runs = []

    state = detect_state(dataset, feature_set, model_runs, deployment)

    parts = [
        "You are a friendly, expert data modeling assistant working inside AutoModeler. "
        f'You are helping with the project "{project.name}".',
    ]

    if project.description:
        parts.append(f"Project description: {project.description}")

    parts.append(
        "Always speak in plain English. When you use a technical term, briefly explain "
        "what it means in parentheses. Keep responses focused and actionable — avoid walls "
        "of text. Use short paragraphs or bullet points. Celebrate the user's progress. "
        "Never show stack traces or raw error objects."
    )

    # Stage guidance
    guidance = _STATE_GUIDANCE.get(state, "")
    if guidance:
        parts.append(f"\n## Current Stage: {state.upper()}\n{guidance}")

    # Dataset context
    if dataset:
        parts.append(f"\n## Loaded Dataset: {dataset.filename}")
        parts.append(f"Rows: {dataset.row_count:,} | Columns: {dataset.column_count}")

        if dataset.columns:
            try:
                columns = json.loads(dataset.columns)
                col_descriptions = []
                for col in columns:
                    desc = f"  - {col['name']} ({col['dtype']})"
                    if "mean" in col and col["mean"] is not None:
                        desc += f", mean={col['mean']:.2f}"
                    if col.get("null_pct", 0) > 0:
                        desc += f", {col['null_pct']}% missing"
                    col_descriptions.append(desc)
                parts.append("Columns:\n" + "\n".join(col_descriptions))
            except (json.JSONDecodeError, KeyError):
                pass

        if dataset.profile:
            try:
                profile = json.loads(dataset.profile)
                # Include a concise subset to avoid prompt bloat
                summary_keys = ["patterns", "warnings", "correlations"]
                summary = {k: profile[k] for k in summary_keys if k in profile}
                if summary:
                    parts.append(
                        f"Data profile highlights: {json.dumps(summary, default=str)}"
                    )
            except json.JSONDecodeError:
                pass
    else:
        parts.append(
            "\nNo dataset has been uploaded yet. Help the user get started by "
            "asking them to upload a CSV file, or suggest the sample dataset."
        )

    # Feature set context
    if feature_set and feature_set.target_column:
        parts.append(
            f"\n## Feature Engineering\n"
            f"Target column: **{feature_set.target_column}** "
            f"(problem type: {feature_set.problem_type or 'auto-detect'})"
        )
        if feature_set.transformations:
            try:
                transforms = json.loads(feature_set.transformations)
                if transforms:
                    names = [t.get("name", t.get("type", "?")) for t in transforms[:5]]
                    parts.append(
                        f"Active transformations: {', '.join(names)}"
                        + (" (+ more)" if len(transforms) > 5 else "")
                    )
            except (json.JSONDecodeError, TypeError):
                pass

    # Model run context
    completed_runs = [mr for mr in model_runs if mr.status == "done"]
    if completed_runs:
        run_summaries = []
        for mr in completed_runs[:3]:
            metrics_str = ""
            if mr.metrics:
                try:
                    m = json.loads(mr.metrics)
                    if "r2" in m:
                        metrics_str = f"R²={m['r2']:.3f}, MAE={m.get('mae', '?'):.3f}"
                    elif "accuracy" in m:
                        metrics_str = (
                            f"accuracy={m['accuracy']:.1%}, "
                            f"F1={m.get('f1', '?'):.3f}"
                        )
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
            selected = " ✓ selected" if mr.is_selected else ""
            run_summaries.append(f"  - {mr.algorithm}: {metrics_str}{selected}")
        parts.append("\n## Trained Models\n" + "\n".join(run_summaries))

    # Deployment context
    if deployment and deployment.is_active:
        parts.append(
            f"\n## Deployment\n"
            f"Model is LIVE at {deployment.dashboard_url}\n"
            f"API endpoint: {deployment.endpoint_path}\n"
            f"Predictions served: {deployment.request_count:,}"
        )

    return "\n\n".join(parts)
