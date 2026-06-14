"""Owner/tenant authorization helpers.

Ownership lives on ``Project.owner_id``. Every other resource is scoped by
cascading through its owning project: a dataset/model-run/deployment belongs to
the user who owns its project. Cross-tenant access returns **404** (not 403) so
that resource existence is not leaked to other tenants.
"""

from fastapi import HTTPException
from sqlmodel import Session

from models.dataset import Dataset
from models.deployment import Deployment
from models.feature_set import FeatureSet
from models.model_run import ModelRun
from models.project import Project
from models.user import User


def _not_found(resource: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{resource} not found")


def assert_owns_project(
    project: Project | None, user: User, resource: str = "Project"
) -> Project:
    """Raise 404 unless ``project`` exists and is owned by ``user``."""
    if project is None or project.owner_id != user.id:
        raise _not_found(resource)
    return project


def get_owned_project(project_id: str, user: User, session: Session) -> Project:
    return assert_owns_project(session.get(Project, project_id), user)


def assert_project_not_foreign(project_id: str, user: User, session: Session):
    """For create-under-project endpoints (e.g. CSV upload).

    Blocks writing into a project owned by ANOTHER tenant (404), but tolerates a
    project_id that does not exist yet — historically the upload endpoint created
    a dataset without requiring a Project row. Returns the Project if present.
    """
    project = session.get(Project, project_id)
    if project is not None and project.owner_id != user.id:
        raise _not_found("Project")
    return project


def get_owned_dataset(dataset_id: str, user: User, session: Session) -> Dataset:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise _not_found("Dataset")
    assert_owns_project(session.get(Project, dataset.project_id), user, "Dataset")
    return dataset


def get_owned_model_run(run_id: str, user: User, session: Session) -> ModelRun:
    run = session.get(ModelRun, run_id)
    if run is None:
        raise _not_found("Model")
    assert_owns_project(session.get(Project, run.project_id), user, "Model")
    return run


def get_owned_deployment(
    deployment_id: str, user: User, session: Session
) -> Deployment:
    deployment = session.get(Deployment, deployment_id)
    if deployment is None:
        raise _not_found("Deployment")
    assert_owns_project(session.get(Project, deployment.project_id), user, "Deployment")
    return deployment


def get_owned_feature_set(
    feature_set_id: str, user: User, session: Session
) -> FeatureSet:
    feature_set = session.get(FeatureSet, feature_set_id)
    if feature_set is None:
        raise _not_found("Feature set")
    get_owned_dataset(feature_set.dataset_id, user, session)
    return feature_set


def _project_owner_id(project_id, session: Session):
    project = session.get(Project, project_id) if project_id else None
    return project.owner_id if project else None


def authorize_path_params(path_params: dict, user: User, session: Session) -> None:
    """Block cross-tenant access to the EXISTING path resource (404).

    Only resources that *exist* and resolve to another tenant's project are
    rejected here. A missing resource is deliberately left to the handler's own
    existence handling — this preserves legacy 404 messages and "safe to call"
    endpoints (e.g. empty scorecards / chat history) while still closing IDOR.

    Deeper child ids (schedule_id, webhook_id, …) always co-occur with their
    parent ``deployment_id`` in the route path, so authorizing the parent covers
    them.
    """

    def block(owner_id) -> None:
        if owner_id is not None and owner_id != user.id:
            raise _not_found("Resource")

    pp = path_params
    if "project_id" in pp:
        block(_project_owner_id(pp["project_id"], session))
    if "dataset_id" in pp:
        ds = session.get(Dataset, pp["dataset_id"])
        if ds is not None:
            block(_project_owner_id(ds.project_id, session))
    if "feature_set_id" in pp:
        fs = session.get(FeatureSet, pp["feature_set_id"])
        if fs is not None:
            ds = session.get(Dataset, fs.dataset_id)
            if ds is not None:
                block(_project_owner_id(ds.project_id, session))
    for key in ("model_run_id", "run_id"):
        if key in pp:
            run = session.get(ModelRun, pp[key])
            if run is not None:
                block(_project_owner_id(run.project_id, session))
    if "deployment_id" in pp:
        dep = session.get(Deployment, pp["deployment_id"])
        if dep is not None:
            block(_project_owner_id(dep.project_id, session))
