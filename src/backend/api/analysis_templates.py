"""Analysis Templates API.

Saved sets of chat queries that analysts can replay on any dataset.
Enables monthly-reporting workflows: save the analysis you did on Q3 data
and run it again on Q4 data with one click.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from db import get_session
from models.analysis_template import AnalysisTemplate

router = APIRouter(prefix="/api/projects", tags=["analysis-templates"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class CreateTemplateRequest(BaseModel):
    name: str
    queries: list[str]
    description: str | None = None


class TemplateOut(BaseModel):
    id: str
    project_id: str
    name: str
    queries: list[str]
    description: str | None
    created_at: str


def _to_out(t: AnalysisTemplate) -> TemplateOut:
    return TemplateOut(
        id=t.id,
        project_id=t.project_id,
        name=t.name,
        queries=json.loads(t.queries) if t.queries else [],
        description=t.description,
        created_at=t.created_at.isoformat(),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/{project_id}/analysis-templates")
def list_templates(
    project_id: str,
    session: Session = Depends(get_session),
) -> list[TemplateOut]:
    """Return all saved analysis templates for a project, newest first."""
    templates = session.exec(
        select(AnalysisTemplate)
        .where(AnalysisTemplate.project_id == project_id)
        .order_by(AnalysisTemplate.created_at.desc())
    ).all()
    return [_to_out(t) for t in templates]


@router.post("/{project_id}/analysis-templates", status_code=201)
def create_template(
    project_id: str,
    body: CreateTemplateRequest,
    session: Session = Depends(get_session),
) -> TemplateOut:
    """Save a new analysis template for a project."""
    if not body.name.strip():
        raise HTTPException(status_code=422, detail="Template name cannot be empty.")
    if not body.queries:
        raise HTTPException(status_code=422, detail="Template must contain at least one query.")

    template = AnalysisTemplate(
        project_id=project_id,
        name=body.name.strip(),
        queries=json.dumps(body.queries),
        description=body.description,
    )
    session.add(template)
    session.commit()
    session.refresh(template)
    return _to_out(template)


@router.delete("/{project_id}/analysis-templates/{template_id}", status_code=204)
def delete_template(
    project_id: str,
    template_id: str,
    session: Session = Depends(get_session),
) -> None:
    """Delete a saved analysis template."""
    template = session.exec(
        select(AnalysisTemplate)
        .where(AnalysisTemplate.id == template_id)
        .where(AnalysisTemplate.project_id == project_id)
    ).first()
    if not template:
        raise HTTPException(status_code=404, detail="Template not found.")
    session.delete(template)
    session.commit()
