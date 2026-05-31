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


# ---------------------------------------------------------------------------
# Model Card HTML export
# ---------------------------------------------------------------------------


def generate_model_card_html(
    *,
    project_name: str,
    algorithm_plain: str,
    problem_type: str,
    target_column: str,
    metric_name: str,
    metric_display: str,
    metric_plain_english: str,
    row_count: int,
    feature_count: int,
    top_features: list[dict],
    limitations: list[str],
    trained_at: str,
    is_deployed: bool = False,
    deployment_endpoint: str | None = None,
    calibration_note: str | None = None,
    brier_score: float | None = None,
    summary: str = "",
) -> str:
    """Return a self-contained HTML model card document as a string.

    Follows the standard ML Model Card format: overview, intended use,
    performance, features, limitations, and deployment info.
    """
    generated_at = datetime.now(UTC).strftime("%B %d, %Y at %H:%M UTC")

    # Build feature importance rows
    feat_rows_html = ""
    if top_features:
        rows = []
        max_imp = max((f.get("importance", 0) for f in top_features), default=1) or 1
        for f in top_features[:8]:
            imp = f.get("importance", 0)
            pct = round(imp * 100, 1)
            bar_pct = round((imp / max_imp) * 100)
            rows.append(f"""<tr>
                  <td style="padding:6px 8px;font-size:0.875rem">{f.get("feature", "")}</td>
                  <td style="padding:6px 8px;font-size:0.875rem">
                    <div style="background:#e0e7ff;border-radius:3px;height:12px;width:100%">
                      <div style="background:#4f46e5;border-radius:3px;height:12px;width:{bar_pct}%"></div>
                    </div>
                  </td>
                  <td style="padding:6px 8px;font-size:0.875rem;text-align:right">{pct}%</td>
                </tr>""")
        feat_rows_html = "".join(rows)
    feat_section = (
        f"""<h2 style="font-size:1rem;font-weight:600;color:#1e40af;margin:1.5rem 0 0.5rem">
            Top Features
        </h2>
        <table style="width:100%;border-collapse:collapse;border:1px solid #e0e7ff;border-radius:6px">
          <thead>
            <tr style="background:#eff6ff">
              <th style="padding:6px 8px;text-align:left;font-size:0.75rem;color:#3730a3">Feature</th>
              <th style="padding:6px 8px;text-align:left;font-size:0.75rem;color:#3730a3">Importance</th>
              <th style="padding:6px 8px;text-align:right;font-size:0.75rem;color:#3730a3">%</th>
            </tr>
          </thead>
          <tbody>{feat_rows_html}</tbody>
        </table>"""
        if feat_rows_html
        else ""
    )

    # Limitations
    lim_items = "".join(
        f'<li style="margin-bottom:4px">{html_escape(lim)}</li>' for lim in limitations
    )

    # Calibration section (classification only)
    cal_html = ""
    if brier_score is not None and calibration_note:
        score_color = (
            "#059669"
            if brier_score < 0.1
            else "#d97706" if brier_score < 0.2 else "#dc2626"
        )
        cal_html = f"""<h2 style="font-size:1rem;font-weight:600;color:#1e40af;margin:1.5rem 0 0.5rem">
            Calibration
        </h2>
        <p style="font-size:0.875rem;color:#374151">
            Brier Score: <strong style="color:{score_color}">{brier_score:.4f}</strong>
            &nbsp;·&nbsp; {html_escape(calibration_note)}
        </p>
        <p style="font-size:0.75rem;color:#6b7280;margin-top:4px">
            A lower Brier score is better (0 = perfect, 0.25 = random). A well-calibrated
            model's confidence percentages reflect the true probability of being correct.
        </p>"""

    # Deployment section
    deploy_html = ""
    if is_deployed and deployment_endpoint:
        deploy_html = f"""<h2 style="font-size:1rem;font-weight:600;color:#1e40af;margin:1.5rem 0 0.5rem">
            Deployment
        </h2>
        <p style="font-size:0.875rem;color:#374151">
            <strong>Status:</strong> <span style="color:#059669">&#9679; Live</span>
        </p>
        <p style="font-size:0.875rem;color:#374151;margin-top:4px">
            <strong>Endpoint:</strong>
            <code style="background:#f3f4f6;padding:2px 6px;border-radius:3px;font-size:0.8rem">
                {html_escape(deployment_endpoint)}
            </code>
        </p>"""
    elif is_deployed:
        deploy_html = """<h2 style="font-size:1rem;font-weight:600;color:#1e40af;margin:1.5rem 0 0.5rem">
            Deployment
        </h2>
        <p style="font-size:0.875rem;color:#059669">&#9679; This model is deployed as a live prediction API.</p>"""

    problem_badge_color = "#1d4ed8" if problem_type == "regression" else "#7c3aed"
    problem_label = "Regression" if problem_type == "regression" else "Classification"
    safe_project_name = html_escape(project_name)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Model Card — {safe_project_name}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f3ff; color: #111827; line-height: 1.6; padding: 2rem; }}
    .container {{ max-width: 760px; margin: 0 auto; background: #fff;
                  border-radius: 10px; padding: 2rem; box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .header {{ border-bottom: 2px solid #4f46e5; padding-bottom: 1rem; margin-bottom: 1.25rem; }}
    .badge {{ display: inline-block; padding: 2px 10px; border-radius: 9999px;
              font-size: 0.72rem; font-weight: 600; margin-right: 6px; }}
    .footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e5e7eb;
               text-align: center; font-size: 0.72rem; color: #9ca3af; }}
    @media print {{ body {{ background: #fff; padding: 0.5rem; }} .container {{ box-shadow: none; }} }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <p style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.05em;
                color:#6b7280;margin-bottom:4px">AutoModeler · Model Card</p>
      <h1 style="font-size:1.5rem;font-weight:700;color:#111827;margin-bottom:6px">
        {safe_project_name}
      </h1>
      <span class="badge" style="background:#ede9fe;color:{problem_badge_color}">{problem_label}</span>
      <span class="badge" style="background:#f0fdf4;color:#059669">Target: {html_escape(target_column)}</span>
      <span class="badge" style="background:#f3f4f6;color:#374151">
        Trained {html_escape(trained_at)}
      </span>
      {('<span class="badge" style="background:#dcfce7;color:#059669">&#9679; Deployed</span>' if is_deployed else "")}
    </div>

    <h2 style="font-size:1rem;font-weight:600;color:#1e40af;margin-bottom:0.5rem">Overview</h2>
    <table style="width:100%;border-collapse:collapse;font-size:0.875rem">
      <tr>
        <td style="padding:5px 0;color:#6b7280;width:180px">Algorithm</td>
        <td style="padding:5px 0;font-weight:500">{html_escape(algorithm_plain)}</td>
      </tr>
      <tr>
        <td style="padding:5px 0;color:#6b7280">Prediction target</td>
        <td style="padding:5px 0;font-weight:500">{html_escape(target_column)}</td>
      </tr>
      <tr>
        <td style="padding:5px 0;color:#6b7280">Problem type</td>
        <td style="padding:5px 0;font-weight:500">{problem_label}</td>
      </tr>
      <tr>
        <td style="padding:5px 0;color:#6b7280">Training rows</td>
        <td style="padding:5px 0;font-weight:500">{row_count:,}</td>
      </tr>
      <tr>
        <td style="padding:5px 0;color:#6b7280">Input features</td>
        <td style="padding:5px 0;font-weight:500">{feature_count}</td>
      </tr>
    </table>

    <h2 style="font-size:1rem;font-weight:600;color:#1e40af;margin:1.5rem 0 0.5rem">
      Performance
    </h2>
    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:1rem">
      <p style="font-size:1.25rem;font-weight:700;color:#1e40af">
        {html_escape(metric_name)}: {html_escape(metric_display)}
      </p>
      <p style="font-size:0.875rem;color:#374151;margin-top:4px">
        {html_escape(metric_plain_english)}
      </p>
    </div>
    {f'<p style="font-size:0.875rem;color:#374151;margin-top:0.75rem">{html_escape(summary)}</p>' if summary else ""}

    {feat_section}
    {cal_html}

    <h2 style="font-size:1rem;font-weight:600;color:#1e40af;margin:1.5rem 0 0.5rem">
      Intended Use
    </h2>
    <p style="font-size:0.875rem;color:#374151">
      This model was trained to predict <strong>{html_escape(target_column)}</strong> from
      {feature_count} input features. It is intended for use by business analysts to support
      data-driven decision making. Predictions should be reviewed alongside domain expertise
      before being acted upon.
    </p>

    <h2 style="font-size:1rem;font-weight:600;color:#1e40af;margin:1.5rem 0 0.5rem">
      Known Limitations
    </h2>
    <ul style="font-size:0.875rem;color:#374151;padding-left:1.2rem">
      {lim_items}
    </ul>

    {deploy_html}

    <div class="footer">
      Generated {generated_at} &nbsp;·&nbsp; <strong>AutoModeler</strong>
      — AI-powered data modeling for business analysts
    </div>
  </div>
</body>
</html>"""


def html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
