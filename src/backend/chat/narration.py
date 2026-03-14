"""Plain-English narrators for AutoModeler events.

These functions generate conversational messages that are injected into the
project's chat history automatically after key events — making the assistant
feel like a proactive colleague who keeps the user informed.

Usage:
    from chat.narration import narrate_upload, narrate_training_complete
    msg = narrate_upload("sales_q1.csv", 1200, 8, insights=[...])
    _append_bot_message(project_id, msg, session)
"""
from __future__ import annotations

import json
import os


# ---------------------------------------------------------------------------
# Claude API helper (best-effort, always falls back to static text)
# ---------------------------------------------------------------------------

_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
_CLAUDE_MAX_TOKENS = 512


def _call_claude(prompt: str, fallback: str) -> str:
    """Call Claude synchronously and return the text response.

    Returns *fallback* immediately if:
    - ANTHROPIC_API_KEY is not set in the environment
    - The API call fails for any reason (network, quota, rate limit, etc.)

    This ensures narration never blocks or crashes event flows.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return fallback
    try:
        import anthropic

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=_CLAUDE_MODEL,
            max_tokens=_CLAUDE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()  # type: ignore[index]
    except Exception:  # noqa: BLE001
        return fallback


# ---------------------------------------------------------------------------
# Upload narration
# ---------------------------------------------------------------------------

def narrate_upload(
    filename: str,
    row_count: int,
    col_count: int,
    insights: list[str] | None = None,
    column_names: list[str] | None = None,
) -> str:
    """Generate a proactive welcome message after a CSV is uploaded.

    Example output:
        "Great — I've loaded your sales_q1.csv (1,200 rows, 8 columns)!
        I can already see: region, product, revenue, date...
        Here's what caught my eye: ..."
    """
    lines: list[str] = [
        f"I've loaded **{filename}** — {row_count:,} rows, {col_count} columns. "
    ]

    if column_names:
        shown = column_names[:6]
        extra = len(column_names) - len(shown)
        col_list = ", ".join(f"`{c}`" for c in shown)
        if extra > 0:
            col_list += f", and {extra} more"
        lines.append(f"Columns: {col_list}.")

    if insights:
        lines.append("\nHere's what I noticed right away:")
        for insight in insights[:3]:
            lines.append(f"  • {insight}")

    lines.append(
        "\nFeel free to ask me anything about the data — "
        "\"what's the distribution of X?\", \"are there any outliers?\", "
        "\"which columns have missing values?\" — or just say **'explore'** "
        "and I'll highlight the most interesting patterns."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AI-powered proactive data insight (second message after upload)
# ---------------------------------------------------------------------------

def narrate_data_insights_ai(
    dataset_summary: str,
    profile_highlights: str,
    n_rows: int,
    n_cols: int,
) -> str | None:
    """Call Claude to generate a proactive, AI-authored insight about the data.

    Returns None if Claude is unavailable — callers should skip injecting
    the message in that case rather than injecting an empty string.

    This supplements the static `narrate_upload` message with a genuinely
    intelligent observation ("I noticed that your top 3 regions account for
    80% of revenue — that's unusual concentration, worth investigating").
    """
    from chat.prompts import build_proactive_insight_prompt

    prompt = build_proactive_insight_prompt(
        dataset_summary=dataset_summary,
        profile_highlights=profile_highlights,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    result = _call_claude(prompt, fallback="")
    return result if result else None


# ---------------------------------------------------------------------------
# Profile insight narration
# ---------------------------------------------------------------------------

def narrate_profile_highlights(profile: dict) -> str | None:
    """Generate a short proactive message from data profile results.

    Returns None if no interesting highlights are found.
    """
    insights: list[str] = []

    # Patterns from the profile
    patterns = profile.get("patterns", [])
    if isinstance(patterns, list):
        insights.extend(str(p) for p in patterns[:2])

    # Warnings (missing values, duplicates, etc.)
    warnings = profile.get("warnings", [])
    if isinstance(warnings, list):
        for w in warnings[:2]:
            if isinstance(w, dict):
                msg = w.get("message", "")
                if msg:
                    insights.append(msg)
            elif isinstance(w, str):
                insights.append(w)

    # Correlations
    correlations = profile.get("correlations", {})
    if isinstance(correlations, dict) and correlations:
        # Find the strongest correlation
        strongest_pair = None
        strongest_val = 0.0
        for pair_key, val in correlations.items():
            try:
                abs_val = abs(float(val))
                if abs_val > strongest_val and abs_val < 1.0:
                    strongest_val = abs_val
                    strongest_pair = pair_key
            except (TypeError, ValueError):
                continue
        if strongest_pair and strongest_val >= 0.7:
            direction = "positively" if float(correlations[strongest_pair]) > 0 else "negatively"
            insights.append(
                f"Strong correlation: {strongest_pair} columns are {direction} correlated "
                f"(r = {strongest_val:.2f})"
            )

    if not insights:
        return None

    lines = ["I found a few things worth knowing about your data:"]
    for insight in insights[:3]:
        lines.append(f"  • {insight}")
    lines.append(
        "\nWant me to dive into any of these? Or ask me a question like "
        "\"show me the distribution of [column name]\"."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training completion narration
# ---------------------------------------------------------------------------

def narrate_training_complete(
    runs: list[dict],
    problem_type: str,
    target_column: str,
) -> str:
    """Generate a post-training narrative comparing the trained models.

    Args:
        runs: List of completed run dicts with 'algorithm', 'metrics', 'summary' keys.
        problem_type: 'regression' or 'classification'
        target_column: The column being predicted.
    """
    completed = [r for r in runs if r.get("status") == "done"]
    failed = [r for r in runs if r.get("status") == "failed"]

    if not completed:
        if failed:
            return (
                "Unfortunately all training runs failed. "
                "This can happen with very small datasets or incompatible column types. "
                "Check the Models tab for error details, or ask me to help troubleshoot."
            )
        return "Training is still in progress. I'll let you know when models are ready!"

    lines: list[str] = []

    if len(completed) == 1:
        run = completed[0]
        algo = run.get("algorithm", "the model")
        summary = run.get("summary", "")
        metrics = run.get("metrics") or {}
        lines.append(f"**{algo}** finished training!")
        if summary:
            lines.append(summary)
        elif metrics:
            primary = "r2" if problem_type == "regression" else "accuracy"
            label = "R²" if problem_type == "regression" else "accuracy"
            val = metrics.get(primary)
            if val is not None:
                val_str = f"{val:.1%}" if primary == "accuracy" else f"{val:.3f}"
                lines.append(f"Performance: {label} = {val_str}")
    else:
        lines.append(f"All {len(completed)} models finished training! Here's a quick summary:")
        primary = "r2" if problem_type == "regression" else "accuracy"
        label = "R²" if problem_type == "regression" else "accuracy"

        ranked = sorted(
            completed,
            key=lambda r: (r.get("metrics") or {}).get(primary, -1),
            reverse=True,
        )

        for i, run in enumerate(ranked):
            algo = run.get("algorithm", "?")
            metrics = run.get("metrics") or {}
            val = metrics.get(primary)
            val_str = f"{val:.1%}" if primary == "accuracy" and val is not None else (f"{val:.3f}" if val is not None else "N/A")
            marker = " 🏆" if i == 0 else ""
            lines.append(f"  • **{algo}**: {label} = {val_str}{marker}")

    if failed:
        lines.append(
            f"\n({len(failed)} model(s) failed to train — check the Models tab for details.)"
        )

    lines.append(
        "\nHead to the **Validate** tab to see detailed performance metrics, "
        "or ask me to explain which model you should pick and why."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AI-powered training completion narration (multi-turn model trade-off reasoning)
# ---------------------------------------------------------------------------

def narrate_training_with_ai(
    runs: list[dict],
    problem_type: str,
    target_column: str,
) -> str:
    """Generate a rich, Claude-authored training completion narrative.

    Uses `build_model_comparison_narrative_prompt` to ask Claude to reason
    about trade-offs between the trained models (accuracy vs. explainability,
    overfitting risk, dataset size fit). Falls back to `narrate_training_complete`
    if Claude is unavailable.

    This is the "multi-turn reasoning about model selection trade-offs" spec item —
    Claude produces a narrative that references the specific models, metrics, and
    what they mean for the user's prediction goal.
    """
    from chat.prompts import build_model_comparison_narrative_prompt

    completed = [r for r in runs if r.get("status") == "done"]
    static_fallback = narrate_training_complete(runs, problem_type, target_column)

    if len(completed) < 2:
        # Single model or all failed — static narration is sufficient
        return static_fallback

    prompt = build_model_comparison_narrative_prompt(
        models=completed,
        problem_type=problem_type,
        target_column=target_column,
    )
    ai_text = _call_claude(prompt, fallback="")
    if not ai_text:
        return static_fallback

    # Append the standard CTA so users know what to do next
    return (
        ai_text
        + "\n\nHead to the **Validate** tab to see detailed performance metrics, "
        "or ask me to explain which model you should pick and why."
    )


# ---------------------------------------------------------------------------
# Model selection narration
# ---------------------------------------------------------------------------

def narrate_model_selected(algorithm: str, metrics: dict, problem_type: str) -> str:
    """Generate a confirmation message when the user selects a model."""
    from chat.prompts import ALGORITHM_INTROS, summarise_metrics

    intro = ALGORITHM_INTROS.get(algorithm, "")
    metrics_summary = summarise_metrics(metrics, problem_type)

    lines = [f"**{algorithm}** is now your selected model."]
    if intro:
        lines.append(intro)
    if metrics_summary:
        lines.append(f"\nPerformance summary:\n{metrics_summary}")
    lines.append(
        "\nWhen you're ready, head to the **Deploy** tab to publish this model "
        "as a live API and shareable prediction dashboard."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deployment narration
# ---------------------------------------------------------------------------

def narrate_deployment(
    algorithm: str,
    dashboard_url: str,
    endpoint_path: str,
) -> str:
    """Generate a deployment success message."""
    return (
        f"Your **{algorithm}** model is now live!\n\n"
        f"**Prediction dashboard:** {dashboard_url}  \n"
        f"Share this link and anyone can enter new data to get a prediction.\n\n"
        f"**API endpoint:** `{endpoint_path}`  \n"
        f"Developers can POST JSON to this endpoint to get predictions programmatically.\n\n"
        "Ask me anything about how to use your deployed model, or return here "
        "anytime to retrain with new data."
    )


# ---------------------------------------------------------------------------
# Shared utility: append a bot message to a project conversation
# ---------------------------------------------------------------------------

def append_bot_message_to_conversation(
    project_id: str,
    message: str,
    session,  # sqlmodel.Session — no import to avoid circular deps
) -> None:
    """Append an assistant message to the project's conversation history.

    Creates the Conversation record if it doesn't yet exist. Uses the passed
    session so callers control transaction lifetime.
    """
    from datetime import UTC, datetime

    from sqlmodel import select

    from models.conversation import Conversation

    def _utcnow():
        return datetime.now(UTC).replace(tzinfo=None)

    statement = select(Conversation).where(Conversation.project_id == project_id)
    conv = session.exec(statement).first()
    if not conv:
        conv = Conversation(project_id=project_id)
        session.add(conv)
        session.commit()
        session.refresh(conv)

    messages = json.loads(conv.messages)
    messages.append(
        {
            "role": "assistant",
            "content": message,
            "timestamp": _utcnow().isoformat(),
        }
    )
    conv.messages = json.dumps(messages)
    conv.updated_at = _utcnow()
    session.add(conv)
    session.commit()
