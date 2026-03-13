from datetime import UTC, datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from db import get_session
from models.dataset import Dataset
from models.model_run import ModelRun
from models.project import Project

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    status: str


class ProjectWithStats(ProjectResponse):
    dataset_filename: Optional[str] = None
    dataset_rows: Optional[int] = None
    model_count: int = 0
    has_deployment: bool = False


@router.post("", response_model=ProjectResponse, status_code=201)
def create_project(body: ProjectCreate, session: Session = Depends(get_session)):
    project = Project(name=body.name, description=body.description)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@router.get("", response_model=list[ProjectWithStats])
def list_projects(session: Session = Depends(get_session)):
    projects = session.exec(select(Project)).all()
    result = []
    for project in projects:
        dataset = session.exec(
            select(Dataset).where(Dataset.project_id == project.id)
        ).first()
        model_count = len(
            session.exec(
                select(ModelRun).where(
                    ModelRun.project_id == project.id,
                    ModelRun.status == "done",
                )
            ).all()
        )
        has_deployment = any(
            mr.is_deployed
            for mr in session.exec(
                select(ModelRun).where(ModelRun.project_id == project.id)
            ).all()
        )
        result.append(
            ProjectWithStats(
                id=project.id,
                name=project.name,
                description=project.description,
                created_at=project.created_at,
                updated_at=project.updated_at,
                status=project.status,
                dataset_filename=dataset.filename if dataset else None,
                dataset_rows=dataset.row_count if dataset else None,
                model_count=model_count,
                has_deployment=has_deployment,
            )
        )
    return result


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: str,
    body: ProjectUpdate,
    session: Session = Depends(get_session),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if body.name is not None:
        project.name = body.name
    if body.description is not None:
        project.description = body.description
    project.updated_at = datetime.now(UTC).replace(tzinfo=None)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@router.post("/{project_id}/duplicate", response_model=ProjectResponse, status_code=201)
def duplicate_project(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    new_project = Project(
        name=f"{project.name} (copy)",
        description=project.description,
        status="exploring",
    )
    session.add(new_project)
    session.commit()
    session.refresh(new_project)
    return new_project


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    session.delete(project)
    session.commit()
