"""Template projects API.

Templates are pre-configured project setups for common use cases:
- sales_forecast: Revenue prediction from sales data
- customer_churn: Binary classification to predict customer churn
- demand_forecast: Predicting units sold based on pricing and promotions

Each template ships with sample data and a guided conversation starter.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from db import get_session
from models.dataset import Dataset
from models.project import Project
from core.analyzer import analyze_dataframe, compute_full_profile
import pandas as pd
import json

router = APIRouter(prefix="/api/templates", tags=["templates"])

# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample"
UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"

TEMPLATES: dict[str, dict] = {
    "sales_forecast": {
        "id": "sales_forecast",
        "name": "Sales Revenue Forecast",
        "description": (
            "Predict monthly revenue using historical sales data. "
            "Explore which products, regions, and time patterns drive revenue."
        ),
        "use_case": "Revenue prediction",
        "target_column": "revenue",
        "problem_type": "regression",
        "sample_file": "sample_sales.csv",
        "suggested_algorithms": [
            "random_forest_regressor",
            "gradient_boosting_regressor",
            "xgboost_regressor",
        ],
        "conversation_starter": (
            "I've loaded your sales forecast template with 200 rows of sales data "
            "covering dates, products, regions, and revenue. "
            "I can see patterns across 3 product lines and 4 regions.\n\n"
            "**Your goal:** Predict revenue based on product, region, and timing.\n\n"
            "Try asking:\n"
            '- "Which region has the highest revenue?"\n'
            '- "Are there any seasonal patterns?"\n'
            '- "What\'s driving revenue most?"'
        ),
        "tags": ["regression", "sales", "time-series"],
        "difficulty": "beginner",
    },
    "customer_churn": {
        "id": "customer_churn",
        "name": "Customer Churn Prediction",
        "description": (
            "Predict which customers are likely to cancel their subscription. "
            "Understand the factors that drive churn: contract type, support calls, pricing."
        ),
        "use_case": "Binary classification",
        "target_column": "churn",
        "problem_type": "classification",
        "sample_file": "customer_churn.csv",
        "suggested_algorithms": [
            "random_forest_classifier",
            "xgboost_classifier",
            "logistic_regression",
        ],
        "conversation_starter": (
            "I've loaded the customer churn template with 300 customer records. "
            "About 29% of customers have churned — this is a binary classification problem.\n\n"
            "**Your goal:** Predict whether a customer will churn ('Yes'/'No') "
            "based on their tenure, charges, usage, and support history.\n\n"
            "Try asking:\n"
            '- "What does a typical churning customer look like?"\n'
            '- "Does monthly charge affect churn rate?"\n'
            '- "Which contract type has the lowest churn?"'
        ),
        "tags": ["classification", "churn", "customer"],
        "difficulty": "intermediate",
    },
    "demand_forecast": {
        "id": "demand_forecast",
        "name": "Product Demand Forecasting",
        "description": (
            "Predict weekly product demand based on price, promotions, "
            "competitor pricing, and temperature. A classic retail ML problem."
        ),
        "use_case": "Demand prediction",
        "target_column": "units_sold",
        "problem_type": "regression",
        "sample_file": "demand_forecast.csv",
        "suggested_algorithms": [
            "gradient_boosting_regressor",
            "random_forest_regressor",
            "linear_regression",
        ],
        "conversation_starter": (
            "I've loaded the demand forecasting template with 250 weeks of data "
            "across 4 products. Key drivers include price, promotions, and temperature.\n\n"
            "**Your goal:** Predict units sold based on pricing and market conditions.\n\n"
            "Try asking:\n"
            '- "How much does a promotion boost sales?"\n'
            '- "What\'s the price elasticity?"\n'
            '- "Which product has the most stable demand?"'
        ),
        "tags": ["regression", "demand", "retail"],
        "difficulty": "intermediate",
    },
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("")
def list_templates():
    """Return all available project templates."""
    return {
        "templates": [
            {
                "id": t["id"],
                "name": t["name"],
                "description": t["description"],
                "use_case": t["use_case"],
                "target_column": t["target_column"],
                "problem_type": t["problem_type"],
                "tags": t["tags"],
                "difficulty": t["difficulty"],
            }
            for t in TEMPLATES.values()
        ]
    }


@router.get("/{template_id}")
def get_template(template_id: str):
    """Return full template metadata including the conversation starter."""
    tpl = TEMPLATES.get(template_id)
    if not tpl:
        raise HTTPException(
            status_code=404, detail=f"Template '{template_id}' not found"
        )
    return tpl


@router.post("/{template_id}/apply", status_code=201)
def apply_template(template_id: str, session: Session = Depends(get_session)):
    """Create a new project from a template.

    Creates a Project + Dataset record, copies sample data to the upload directory,
    and returns the project_id, dataset_id, and a conversation starter message.
    """
    tpl = TEMPLATES.get(template_id)
    if not tpl:
        raise HTTPException(
            status_code=404, detail=f"Template '{template_id}' not found"
        )

    # Locate the sample file
    sample_path = SAMPLE_DIR / tpl["sample_file"]
    if not sample_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Template sample file not found: {tpl['sample_file']}",
        )

    # Create a Project
    project = Project(
        name=tpl["name"],
        description=tpl["description"],
        status="exploring",
    )
    session.add(project)
    session.commit()
    session.refresh(project)

    # Copy sample file into upload dir
    project_upload_dir = UPLOAD_DIR / project.id
    project_upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = project_upload_dir / tpl["sample_file"]
    shutil.copy2(sample_path, dest_path)

    # Parse and profile the CSV
    df = pd.read_csv(dest_path)
    analysis = analyze_dataframe(df)
    profile = compute_full_profile(df)

    # Create Dataset record
    dataset = Dataset(
        project_id=project.id,
        filename=tpl["sample_file"],
        file_path=str(dest_path),
        row_count=len(df),
        column_count=len(df.columns),
        columns=json.dumps(analysis["columns"]),
        profile=json.dumps(profile),
        size_bytes=dest_path.stat().st_size,
    )
    session.add(dataset)

    project.updated_at = datetime.now(UTC).replace(tzinfo=None)
    session.add(project)
    session.commit()
    session.refresh(dataset)

    return {
        "project_id": project.id,
        "dataset_id": dataset.id,
        "template_id": template_id,
        "name": tpl["name"],
        "target_column": tpl["target_column"],
        "problem_type": tpl["problem_type"],
        "suggested_algorithms": tpl["suggested_algorithms"],
        "conversation_starter": tpl["conversation_starter"],
        "row_count": len(df),
        "columns": [c["name"] for c in analysis["columns"]],
        "message": (
            f"Created project '{tpl['name']}' from template. "
            f"Dataset: {len(df)} rows × {len(df.columns)} columns. "
            f"Goal: predict '{tpl['target_column']}' ({tpl['problem_type']})."
        ),
    }
