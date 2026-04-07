"""Guided onboarding wizard state computation.

A pure function that maps project progress signals (booleans/counts) to a
step-by-step onboarding state object.  No database access; all callers supply
the relevant flags derived from their own queries.
"""

from __future__ import annotations

_STEPS = [
    {
        "name": "upload",
        "title": "Upload your data",
        "description": "Drag and drop a CSV file, or click the upload button in the Data tab.",
        "hint": "Any CSV works — sales data, customer records, operational metrics. Even a small sample is fine to start.",
        "suggested_action": "Upload a CSV",
        "suggested_tab": "data",
        "icon": "📂",
    },
    {
        "name": "explore",
        "title": "Explore your data",
        "description": "Ask a question about your data in the chat window.",
        "hint": "Try: 'What are the trends in my data?' or 'Show me the top 5 regions by revenue.'",
        "suggested_action": "Ask a question",
        "suggested_tab": None,
        "icon": "💬",
    },
    {
        "name": "target",
        "title": "Set a prediction target",
        "description": "Tell AutoModeler what you want to predict.",
        "hint": "Try: 'I want to predict revenue' or 'Help me predict customer churn.' AutoModeler will guide you through feature setup.",
        "suggested_action": "Set prediction target",
        "suggested_tab": "features",
        "icon": "🎯",
    },
    {
        "name": "train",
        "title": "Train a model",
        "description": "Click Train in the Models tab, or ask AutoModeler to train a model for you.",
        "hint": "AutoModeler automatically selects the best algorithm. Training usually takes 10–30 seconds.",
        "suggested_action": "Go to Models",
        "suggested_tab": "models",
        "icon": "🤖",
    },
    {
        "name": "validate",
        "title": "Review your results",
        "description": "Check accuracy and validation metrics in the Validation tab.",
        "hint": "Look for R² > 0.7 (regression) or accuracy > 70% (classification) as a healthy starting point.",
        "suggested_action": "View Validation",
        "suggested_tab": "validation",
        "icon": "✅",
    },
    {
        "name": "deploy",
        "title": "Deploy your model",
        "description": "Click Deploy in the Models tab to create a live prediction API and shareable dashboard.",
        "hint": "You'll get a link you can share with your VP or developer — no code required.",
        "suggested_action": "Deploy model",
        "suggested_tab": "models",
        "icon": "🚀",
    },
]


def compute_onboarding_state(
    has_dataset: bool,
    message_count: int,
    has_target: bool,
    has_model_run: bool,
    has_cross_val: bool,
    has_deployment: bool,
) -> dict:
    """Compute the analyst's guided onboarding progress.

    Parameters
    ----------
    has_dataset:
        True when at least one dataset has been uploaded to the project.
    message_count:
        Number of user (or assistant) messages in the conversation.  A count
        ≥ 2 means the analyst has actively engaged (first message is the
        auto-narration on upload).
    has_target:
        True when an active FeatureSet with a non-empty target_column exists.
    has_model_run:
        True when at least one completed ModelRun exists for the project.
    has_cross_val:
        True when the best ModelRun's metrics include cross-validation results
        (``cv_r2_mean`` or ``cv_accuracy_mean``).
    has_deployment:
        True when at least one active Deployment exists for the project.

    Returns
    -------
    dict with keys:
        step_index       - 0-based index of the current (incomplete) step
        total_steps      - total number of steps (6)
        completion_pct   - 0-100 integer
        steps            - list of step dicts, each with is_done / is_current
        current_step     - the active step dict (or None if complete)
        is_complete      - True when all steps are done
        summary          - one-line plain-English progress summary
    """
    done_flags = [
        has_dataset,
        has_dataset and message_count >= 2,
        has_target,
        has_model_run,
        has_cross_val,  # validation = having cross-validation results
        has_deployment,
    ]

    # Find the first incomplete step
    current_idx = next(
        (i for i, done in enumerate(done_flags) if not done),
        len(_STEPS),  # all done → past the end
    )
    is_complete = current_idx == len(_STEPS)
    completion_pct = round(current_idx / len(_STEPS) * 100)

    steps = []
    for i, step in enumerate(_STEPS):
        steps.append(
            {
                **step,
                "is_done": done_flags[i],
                "is_current": i == current_idx and not is_complete,
            }
        )

    current_step = None if is_complete else steps[current_idx]

    if is_complete:
        summary = "All steps complete! Your model is deployed and ready to use."
    elif current_idx == 0:
        summary = "Get started by uploading a CSV dataset."
    else:
        done_count = current_idx
        summary = (
            f"{done_count} of {len(_STEPS)} steps complete — "
            f"next: {steps[current_idx]['title'].lower()}."
        )

    return {
        "step_index": current_idx,
        "total_steps": len(_STEPS),
        "completion_pct": completion_pct,
        "steps": steps,
        "current_step": current_step,
        "is_complete": is_complete,
        "summary": summary,
    }
