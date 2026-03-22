"""PDF report generator for AutoModeler model runs.

Generates a clean, professional PDF report summarising:
- Project metadata
- Dataset overview
- Model algorithm and training metrics
- Plain-English summary and confidence assessment
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_model_report(
    *,
    project_name: str,
    dataset_filename: str,
    dataset_rows: int,
    dataset_columns: int,
    algorithm: str,
    problem_type: str,
    metrics: dict[str, Any],
    summary: str | None,
    training_duration_ms: int | None,
    feature_importances: list[dict[str, Any]] | None = None,
    confidence_assessment: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> bytes:
    """Return a PDF as raw bytes.

    Parameters
    ----------
    project_name:           Human-readable project name.
    dataset_filename:       Original CSV filename.
    dataset_rows/columns:   Shape of the uploaded dataset.
    algorithm:              Algorithm name (e.g. "RandomForest").
    problem_type:           "regression" | "classification".
    metrics:                Dict of metric name → value.
    summary:                Plain-English metric summary from trainer.
    training_duration_ms:   Training wall-clock time.
    feature_importances:    Optional list of {feature, importance, rank} dicts.
    confidence_assessment:  Optional {confidence_level, strengths, limitations} dict.
    created_at:             Model creation timestamp.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    # Custom styles
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=20,
        textColor=colors.HexColor("#111827"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#6B7280"),
        spaceAfter=16,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1D4ED8"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#374151"),
        leading=15,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#9CA3AF"),
        spaceAfter=4,
    )

    elements = []

    # ── Header ──────────────────────────────────────────────────────────────
    elements.append(Paragraph("AutoModeler", caption_style))
    elements.append(Paragraph(f"Model Report — {project_name}", title_style))
    report_date = (created_at or datetime.now(UTC).replace(tzinfo=None)).strftime(
        "%B %d, %Y"
    )
    elements.append(Paragraph(f"Generated {report_date}", subtitle_style))
    elements.append(
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E5E7EB"))
    )
    elements.append(Spacer(1, 0.4 * cm))

    # ── Overview table ───────────────────────────────────────────────────────
    elements.append(Paragraph("Overview", section_style))

    algo_label = _format_algorithm(algorithm)
    duration_str = (
        f"{training_duration_ms / 1000:.1f}s" if training_duration_ms else "—"
    )
    overview_data = [
        ["Project", project_name],
        ["Dataset", dataset_filename],
        ["Rows", f"{dataset_rows:,}"],
        ["Columns", f"{dataset_columns}"],
        ["Problem Type", problem_type.title()],
        ["Algorithm", algo_label],
        ["Training Time", duration_str],
    ]
    elements.append(_make_table(overview_data))
    elements.append(Spacer(1, 0.3 * cm))

    # ── Metrics ──────────────────────────────────────────────────────────────
    elements.append(Paragraph("Performance Metrics", section_style))

    if metrics:
        metric_rows = []
        metric_display = _format_metrics(metrics, problem_type)
        for name, value, plain in metric_display:
            metric_rows.append([name, value, plain])

        if metric_rows:
            metric_table_data = [["Metric", "Value", "What it means"]] + metric_rows
            elements.append(_make_metric_table(metric_table_data))

    if summary:
        elements.append(Spacer(1, 0.3 * cm))
        elements.append(Paragraph("Summary", section_style))
        elements.append(Paragraph(summary, body_style))

    # ── Feature Importance ───────────────────────────────────────────────────
    if feature_importances:
        elements.append(Paragraph("Top Features", section_style))
        elements.append(
            Paragraph(
                "The features below had the most influence on the model's predictions.",
                body_style,
            )
        )
        elements.append(Spacer(1, 0.2 * cm))
        fi_data = [["Rank", "Feature", "Importance"]]
        for entry in feature_importances[:15]:
            fi_data.append(
                [
                    str(entry.get("rank", "")),
                    str(entry.get("feature", "")),
                    f"{float(entry.get('importance', 0)):.4f}",
                ]
            )
        elements.append(_make_table(fi_data, header=True))

    # ── Confidence Assessment ────────────────────────────────────────────────
    if confidence_assessment:
        elements.append(Paragraph("Confidence & Limitations", section_style))
        level = confidence_assessment.get("confidence_level", "")
        if level:
            elements.append(Paragraph(f"<b>Confidence level:</b> {level}", body_style))
        strengths = confidence_assessment.get("strengths", [])
        if strengths:
            elements.append(Spacer(1, 0.15 * cm))
            elements.append(Paragraph("<b>Strengths</b>", body_style))
            for s in strengths:
                elements.append(Paragraph(f"• {s}", body_style))
        limitations = confidence_assessment.get("limitations", [])
        if limitations:
            elements.append(Spacer(1, 0.15 * cm))
            elements.append(Paragraph("<b>Limitations</b>", body_style))
            for lim in limitations:
                elements.append(Paragraph(f"• {lim}", body_style))

    # ── Footer ───────────────────────────────────────────────────────────────
    elements.append(Spacer(1, 0.8 * cm))
    elements.append(
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E5E7EB"))
    )
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(
        Paragraph(
            "Generated by AutoModeler — AI-powered conversational data modeling.",
            caption_style,
        )
    )

    doc.build(elements)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_algorithm(algo: str) -> str:
    names = {
        "LinearRegression": "Linear Regression",
        "RandomForest": "Random Forest",
        "GradientBoosting": "Gradient Boosting",
        "LogisticRegression": "Logistic Regression",
        "RandomForestClassifier": "Random Forest (Classification)",
        "GradientBoostingClassifier": "Gradient Boosting (Classification)",
    }
    return names.get(algo, algo)


def _format_metrics(metrics: dict, problem_type: str) -> list[tuple[str, str, str]]:
    """Return list of (metric_name, formatted_value, plain_english) tuples."""
    rows = []
    plain = {
        "r2": "How well the model explains variance (0–1, higher is better)",
        "mae": "Average absolute prediction error (lower is better)",
        "rmse": "Root mean squared error — penalises large mistakes (lower is better)",
        "mape": "Mean absolute percentage error (lower is better)",
        "accuracy": "Fraction of correct predictions (0–1, higher is better)",
        "f1": "Balance of precision and recall (0–1, higher is better)",
        "precision": "When model says positive, how often it's right",
        "recall": "How many actual positives the model caught",
        "roc_auc": "Ability to distinguish between classes (0–1, higher is better)",
    }
    for key, val in metrics.items():
        if val is None:
            continue
        label = key.upper().replace("_", " ")
        if isinstance(val, float):
            formatted = f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}"
        else:
            formatted = str(val)
        rows.append((label, formatted, plain.get(key.lower(), "")))
    return rows


def _make_table(data: list[list], header: bool = False) -> Table:
    col_widths = [4.5 * cm, None]
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#6B7280")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#111827")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        (
            "ROWBACKGROUNDS",
            (0, 0),
            (-1, -1),
            [colors.HexColor("#F9FAFB"), colors.white],
        ),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]
    if header:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFF6FF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1D4ED8")),
        ]
    t.setStyle(TableStyle(style))
    return t


def _make_metric_table(data: list[list]) -> Table:
    col_widths = [4 * cm, 3 * cm, None]
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFF6FF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1D4ED8")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.HexColor("#F9FAFB"), colors.white],
                ),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7EB")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                ("TEXTCOLOR", (2, 1), (2, -1), colors.HexColor("#6B7280")),
            ]
        )
    )
    return t
