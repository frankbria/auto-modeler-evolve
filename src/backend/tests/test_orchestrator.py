"""Tests for chat orchestrator — state detection and system prompt generation."""

import pytest
from unittest.mock import MagicMock


def make_dataset(filename="sales.csv", rows=200, cols=5):
    ds = MagicMock()
    ds.filename = filename
    ds.row_count = rows
    ds.column_count = cols
    ds.columns = None
    ds.profile = None
    return ds


def make_feature_set(target="revenue", problem_type="regression", transformations=None):
    fs = MagicMock()
    fs.target_column = target
    fs.problem_type = problem_type
    fs.transformations = transformations
    return fs


def make_model_run(algorithm="RandomForest", status="done", selected=False, metrics=None):
    mr = MagicMock()
    mr.algorithm = algorithm
    mr.status = status
    mr.is_selected = selected
    mr.metrics = metrics
    return mr


def make_deployment(active=True, url="/predict/abc", endpoint="/api/predict/abc", count=10):
    dep = MagicMock()
    dep.is_active = active
    dep.dashboard_url = url
    dep.endpoint_path = endpoint
    dep.request_count = count
    return dep


def make_project(name="Test Project", description=None):
    p = MagicMock()
    p.name = name
    p.description = description
    return p


class TestDetectState:
    def test_no_dataset_returns_upload(self):
        from chat.orchestrator import detect_state
        assert detect_state(None, None, [], None) == "upload"

    def test_dataset_only_returns_explore(self):
        from chat.orchestrator import detect_state
        ds = make_dataset()
        assert detect_state(ds, None, [], None) == "explore"

    def test_feature_set_with_target_returns_model(self):
        from chat.orchestrator import detect_state
        ds = make_dataset()
        fs = make_feature_set()
        assert detect_state(ds, fs, [], None) == "model"

    def test_feature_set_no_target_returns_explore(self):
        from chat.orchestrator import detect_state
        ds = make_dataset()
        fs = MagicMock()
        fs.target_column = None
        assert detect_state(ds, fs, [], None) == "explore"

    def test_completed_model_run_returns_validate(self):
        from chat.orchestrator import detect_state
        ds = make_dataset()
        fs = make_feature_set()
        mr = make_model_run(status="done")
        assert detect_state(ds, fs, [mr], None) == "validate"

    def test_only_pending_runs_returns_model(self):
        from chat.orchestrator import detect_state
        ds = make_dataset()
        fs = make_feature_set()
        mr = make_model_run(status="training")
        assert detect_state(ds, fs, [mr], None) == "model"

    def test_active_deployment_returns_deploy(self):
        from chat.orchestrator import detect_state
        ds = make_dataset()
        fs = make_feature_set()
        mr = make_model_run(status="done")
        dep = make_deployment(active=True)
        assert detect_state(ds, fs, [mr], dep) == "deploy"

    def test_inactive_deployment_falls_back_to_validate(self):
        from chat.orchestrator import detect_state
        ds = make_dataset()
        fs = make_feature_set()
        mr = make_model_run(status="done")
        dep = make_deployment(active=False)
        assert detect_state(ds, fs, [mr], dep) == "validate"


class TestBuildSystemPrompt:
    def test_no_dataset_includes_upload_guidance(self):
        from chat.orchestrator import build_system_prompt
        project = make_project()
        prompt = build_system_prompt(project)
        assert "upload" in prompt.lower()
        assert "CSV" in prompt or "csv" in prompt.lower()

    def test_dataset_context_in_prompt(self):
        from chat.orchestrator import build_system_prompt
        import json
        project = make_project("Sales Analysis")
        ds = make_dataset("q1_sales.csv", rows=500, cols=8)
        ds.columns = json.dumps([
            {"name": "revenue", "dtype": "float64", "mean": 1200.0, "null_pct": 0},
            {"name": "region", "dtype": "object", "null_pct": 2.0},
        ])
        ds.profile = None
        prompt = build_system_prompt(project, dataset=ds)
        assert "q1_sales.csv" in prompt
        assert "500" in prompt
        assert "revenue" in prompt
        assert "region" in prompt

    def test_explore_stage_guidance(self):
        from chat.orchestrator import build_system_prompt
        project = make_project()
        ds = make_dataset()
        ds.columns = None
        ds.profile = None
        prompt = build_system_prompt(project, dataset=ds)
        assert "EXPLORE" in prompt

    def test_model_stage_shows_target_column(self):
        from chat.orchestrator import build_system_prompt
        project = make_project()
        ds = make_dataset()
        ds.columns = None
        ds.profile = None
        fs = make_feature_set(target="revenue", problem_type="regression")
        fs.transformations = None
        prompt = build_system_prompt(project, dataset=ds, feature_set=fs)
        assert "revenue" in prompt
        assert "MODEL" in prompt

    def test_validate_stage_shows_trained_models(self):
        from chat.orchestrator import build_system_prompt
        import json
        project = make_project()
        ds = make_dataset()
        ds.columns = None
        ds.profile = None
        fs = make_feature_set()
        fs.transformations = None
        mr = make_model_run(
            algorithm="RandomForest",
            status="done",
            selected=True,
            metrics=json.dumps({"r2": 0.87, "mae": 150.0}),
        )
        prompt = build_system_prompt(project, dataset=ds, feature_set=fs, model_runs=[mr])
        assert "RandomForest" in prompt
        assert "VALIDATE" in prompt

    def test_deploy_stage_shows_endpoint(self):
        from chat.orchestrator import build_system_prompt
        import json
        project = make_project()
        ds = make_dataset()
        ds.columns = None
        ds.profile = None
        fs = make_feature_set()
        fs.transformations = None
        mr = make_model_run(status="done", metrics=json.dumps({"accuracy": 0.92, "f1": 0.91}))
        dep = make_deployment(active=True, url="/predict/xyz", endpoint="/api/predict/xyz", count=42)
        prompt = build_system_prompt(
            project, dataset=ds, feature_set=fs, model_runs=[mr], deployment=dep
        )
        assert "/predict/xyz" in prompt
        assert "DEPLOY" in prompt
        assert "42" in prompt

    def test_project_description_included(self):
        from chat.orchestrator import build_system_prompt
        project = make_project(description="Monthly revenue forecasting project")
        prompt = build_system_prompt(project)
        assert "Monthly revenue forecasting" in prompt

    def test_default_model_runs_is_empty_list(self):
        from chat.orchestrator import build_system_prompt
        project = make_project()
        # Should not raise even with no model_runs arg
        prompt = build_system_prompt(project)
        assert isinstance(prompt, str)
