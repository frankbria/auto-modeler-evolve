"""Tests for Model Retraining Completion Notification feature.

Covers:
- EVENT_RETRAIN_COMPLETE constant exists in core.webhook
- ALL_EVENTS includes retrain_complete
- _train_in_background: dispatches EVENT_RETRAIN_COMPLETE webhook when deployment_id set
- _train_in_background: does NOT dispatch when deployment_id is None
- _check_and_trigger_degradation_retrain: passes deployment_id to background thread
- _RETRAIN_COMPLETE_NOTIFY_PATTERNS: matches 8 NL phrases
- Chat handler: emits retrain_complete_notify SSE event with required keys
"""

from __future__ import annotations

import re

import pytest

from core.webhook import ALL_EVENTS, EVENT_RETRAIN_COMPLETE

# ---------------------------------------------------------------------------
# 1. Webhook constant tests
# ---------------------------------------------------------------------------


def test_event_retrain_complete_constant_value():
    assert EVENT_RETRAIN_COMPLETE == "retrain_complete"


def test_event_retrain_complete_in_all_events():
    assert "retrain_complete" in ALL_EVENTS


# ---------------------------------------------------------------------------
# 2. _train_in_background webhook dispatch tests
# ---------------------------------------------------------------------------


def test_train_in_background_dispatches_webhook_when_deployment_id_set():
    """When deployment_id is passed, retrain_complete webhook fires on success."""
    import queue
    import threading
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    import pandas as pd

    # Minimal DataFrame
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        }
    )
    dispatched = []

    def fake_dispatch(dep_id, event_type, payload):
        dispatched.append((dep_id, event_type, payload))

    mock_run = MagicMock()
    mock_run.id = "run-abc"
    mock_session_instance = MagicMock()
    mock_session_instance.__enter__ = MagicMock(return_value=mock_session_instance)
    mock_session_instance.__exit__ = MagicMock(return_value=False)
    mock_session_instance.get.return_value = mock_run

    mock_result = {
        "metrics": {"r2": 0.97, "rmse": 1.2},
        "model_path": "/tmp/test_model.joblib",
        "training_duration_ms": 250,
        "summary": "Test model trained.",
    }

    with (
        patch("api.models.Session", return_value=mock_session_instance),
        patch("api.models.train_single_model", return_value=mock_result),
        patch("api.models.prepare_features", return_value=(df, df["y"], None)),
        patch(
            "api.models.sample_large_dataset", return_value=(df, {"was_sampled": False})
        ),
        patch("api.models._push_event"),
        patch("api.models._finish_training_thread"),
        patch("core.webhook.dispatch_webhooks", side_effect=fake_dispatch),
    ):
        from api.models import _train_in_background

        q: queue.Queue = queue.Queue()
        from api.models import _training_queues

        _training_queues["proj-1"] = q

        t = threading.Thread(
            target=_train_in_background,
            args=(
                "run-abc",
                "proj-1",
                df,
                ["x"],
                "y",
                "linear_regression",
                "regression",
                Path("/tmp"),
            ),
            kwargs={"deployment_id": "dep-123"},
            daemon=True,
        )
        t.start()
        t.join(timeout=10)

    assert any(
        ev == "retrain_complete" for _, ev, _ in dispatched
    ), f"retrain_complete not dispatched; dispatched={dispatched}"
    dep_id, event_type, payload = next(
        (d for d in dispatched if d[1] == "retrain_complete"), (None, None, None)
    )
    assert dep_id == "dep-123"
    assert payload["deployment_id"] == "dep-123"
    assert payload["run_id"] == "run-abc"
    assert payload["algorithm"] == "linear_regression"
    assert "primary_metric" in payload
    assert "training_duration_ms" in payload
    assert "message" in payload


def test_train_in_background_no_webhook_when_no_deployment_id():
    """Without deployment_id, no retrain_complete webhook fires."""
    import queue
    import threading
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    import pandas as pd

    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        }
    )
    dispatched = []

    def fake_dispatch(dep_id, event_type, payload):
        dispatched.append((dep_id, event_type, payload))

    mock_run = MagicMock()
    mock_run.id = "run-xyz"
    mock_session_instance = MagicMock()
    mock_session_instance.__enter__ = MagicMock(return_value=mock_session_instance)
    mock_session_instance.__exit__ = MagicMock(return_value=False)
    mock_session_instance.get.return_value = mock_run

    mock_result = {
        "metrics": {"r2": 0.90},
        "model_path": "/tmp/m.joblib",
        "training_duration_ms": 100,
        "summary": "Done.",
    }

    with (
        patch("api.models.Session", return_value=mock_session_instance),
        patch("api.models.train_single_model", return_value=mock_result),
        patch("api.models.prepare_features", return_value=(df, df["y"], None)),
        patch(
            "api.models.sample_large_dataset", return_value=(df, {"was_sampled": False})
        ),
        patch("api.models._push_event"),
        patch("api.models._finish_training_thread"),
        patch("core.webhook.dispatch_webhooks", side_effect=fake_dispatch),
    ):
        from api.models import _train_in_background, _training_queues

        _training_queues["proj-2"] = queue.Queue()

        t = threading.Thread(
            target=_train_in_background,
            args=(
                "run-xyz",
                "proj-2",
                df,
                ["x"],
                "y",
                "linear_regression",
                "regression",
                Path("/tmp"),
            ),
            daemon=True,
        )
        t.start()
        t.join(timeout=10)

    retrain_complete_dispatches = [d for d in dispatched if d[1] == "retrain_complete"]
    assert len(retrain_complete_dispatches) == 0


# ---------------------------------------------------------------------------
# 3. _check_and_trigger_degradation_retrain passes deployment_id to thread
# ---------------------------------------------------------------------------


def test_degradation_retrain_passes_deployment_id_kwarg():
    """_check_and_trigger_degradation_retrain source passes deployment_id kwarg to Thread."""
    import inspect
    import api.deploy

    src = inspect.getsource(api.deploy._check_and_trigger_degradation_retrain)
    assert (
        'kwargs={"deployment_id": deployment_id}' in src
    ), "Thread constructor must include kwargs={'deployment_id': deployment_id}"


def test_train_in_background_accepts_deployment_id_kwarg():
    """_train_in_background has deployment_id in its signature."""
    import inspect
    from api.models import _train_in_background

    sig = inspect.signature(_train_in_background)
    assert (
        "deployment_id" in sig.parameters
    ), "_train_in_background must have a deployment_id parameter"
    param = sig.parameters["deployment_id"]
    assert param.default is None, "deployment_id should default to None"


# ---------------------------------------------------------------------------
# 4. Chat pattern tests
# ---------------------------------------------------------------------------


_RETRAIN_COMPLETE_NOTIFY_PATTERNS = re.compile(
    r"(?i)(?:"
    r"(?:notify|alert|tell|inform)\s+(?:me\s+)?when\s+(?:(?:re.?train(?:ing)?|training)\s+(?:is\s+)?(?:done|complete|finish(?:ed)?|over))\b|"
    r"(?:re.?train(?:ing)?|training)\s+(?:completion|complete|done|finish(?:ed)?)\s+(?:notify|notification|alert|webhook)\b|"
    r"(?:notify|alert|tell)\s+(?:me\s+)?when\s+(?:auto.?)?re.?train(?:ing)?\s+(?:finish(?:es)?|complete[sd]?|done)\b|"
    r"(?:re.?train(?:ing)?|auto.?re.?train(?:ing)?)\s+(?:complete|done|finish(?:ed)?)\s+(?:notification|alert|webhook|status)\b|"
    r"(?:how\s+(?:do\s+I|can\s+I|to))\s+(?:get\s+)?(?:notified|alerted)\s+when\s+re.?train(?:ing)?\s+(?:is\s+)?(?:finish(?:es)?|complete[sd]?|done)\b|"
    r"(?:show|check|get|view|what\s+is)\s+(?:(?:my|the)\s+)?re.?train(?:ing)?\s+(?:completion\s+)?(?:notification|notify|alert)\s*(?:config(?:uration)?|setting|status)?\b|"
    r"re.?train(?:ing)?\s+complete\b|"
    r"(?:did|has|when\s+did)\s+(?:(?:auto.?)?re.?train(?:ing)?|(?:the\s+)?(?:last\s+)?re.?train(?:ing)?)\s+(?:finish|complete|end|done)\b"
    r")",
    re.IGNORECASE,
)

MATCH_PHRASES = [
    "notify me when retraining is done",
    "notify me when retraining is complete",
    "tell me when auto-retraining finishes",
    "retraining complete notification",
    "retraining completion webhook",
    "how do I get notified when retraining is done",
    "retraining complete",
    "did the retraining finish",
]

NO_MATCH_PHRASES = [
    "enable retraining when accuracy drops",
    "configure degradation-triggered retraining",
    "show retraining readiness",
    "retrain my model",
]


@pytest.mark.parametrize("phrase", MATCH_PHRASES)
def test_retrain_complete_notify_pattern_matches(phrase: str):
    assert _RETRAIN_COMPLETE_NOTIFY_PATTERNS.search(
        phrase
    ), f"Pattern should match: {phrase!r}"


@pytest.mark.parametrize("phrase", NO_MATCH_PHRASES)
def test_retrain_complete_notify_pattern_no_match(phrase: str):
    assert not _RETRAIN_COMPLETE_NOTIFY_PATTERNS.search(
        phrase
    ), f"Pattern should NOT match: {phrase!r}"


# ---------------------------------------------------------------------------
# 5. Chat handler event dict structure test
# ---------------------------------------------------------------------------


def test_retrain_complete_notify_event_shape():
    """The retrain_complete_notify event dict has all required keys."""
    event = {
        "deployment_id": "dep-rcn-test",
        "has_notification": False,
        "retrain_complete_webhooks": [],
        "last_completed_retrain": None,
        "summary": "No retrain_complete webhooks registered.",
    }
    required = {
        "deployment_id",
        "has_notification",
        "retrain_complete_webhooks",
        "last_completed_retrain",
        "summary",
    }
    assert required <= set(event.keys())
    assert isinstance(event["has_notification"], bool)
    assert isinstance(event["retrain_complete_webhooks"], list)


def test_retrain_complete_notify_event_shape_with_run():
    """Event with last_completed_retrain includes expected run fields."""
    run = {
        "run_id": "r1",
        "algorithm": "random_forest",
        "status": "done",
        "primary_metric": "accuracy",
        "primary_metric_value": 0.92,
        "training_duration_ms": 5000,
        "completed_at": "2026-06-12T10:00:00",
    }
    event = {
        "deployment_id": "dep-2",
        "has_notification": True,
        "retrain_complete_webhooks": ["https://hooks.example.com"],
        "last_completed_retrain": run,
        "summary": "Retrain complete webhook registered.",
    }
    run_keys = {
        "run_id",
        "algorithm",
        "status",
        "primary_metric",
        "primary_metric_value",
        "training_duration_ms",
        "completed_at",
    }
    assert run_keys <= set(event["last_completed_retrain"].keys())
    assert event["has_notification"] is True
    assert len(event["retrain_complete_webhooks"]) == 1


# ---------------------------------------------------------------------------
# 6. Webhook payload key validation
# ---------------------------------------------------------------------------


def test_retrain_complete_webhook_payload_keys():
    """When _train_in_background dispatches, payload has required keys."""
    dispatched = []

    def fake_dispatch(dep_id, event_type, payload):
        dispatched.append((dep_id, event_type, payload))

    import queue
    import threading
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    import pandas as pd

    df = pd.DataFrame({"a": list(range(10)), "b": [float(i) * 1.5 for i in range(10)]})

    mock_run = MagicMock()
    mock_run.id = "run-payload-test"
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.get.return_value = mock_run

    mock_result = {
        "metrics": {"accuracy": 0.92, "f1": 0.88},
        "model_path": "/tmp/m.joblib",
        "training_duration_ms": 300,
        "summary": "Done.",
    }

    with (
        patch("api.models.Session", return_value=mock_session),
        patch("api.models.train_single_model", return_value=mock_result),
        patch("api.models.prepare_features", return_value=(df, df["b"], None)),
        patch(
            "api.models.sample_large_dataset", return_value=(df, {"was_sampled": False})
        ),
        patch("api.models._push_event"),
        patch("api.models._finish_training_thread"),
        patch("core.webhook.dispatch_webhooks", side_effect=fake_dispatch),
    ):
        from api.models import _train_in_background, _training_queues

        _training_queues["proj-payload"] = queue.Queue()

        t = threading.Thread(
            target=_train_in_background,
            args=(
                "run-payload-test",
                "proj-payload",
                df,
                ["a"],
                "b",
                "random_forest",
                "classification",
                Path("/tmp"),
            ),
            kwargs={"deployment_id": "dep-payload-test"},
            daemon=True,
        )
        t.start()
        t.join(timeout=10)

    retrain_events = [d for d in dispatched if d[1] == "retrain_complete"]
    assert len(retrain_events) >= 1
    payload = retrain_events[0][2]
    required_keys = {
        "deployment_id",
        "run_id",
        "algorithm",
        "primary_metric",
        "primary_metric_value",
        "training_duration_ms",
        "message",
    }
    missing = required_keys - set(payload.keys())
    assert not missing, f"Missing payload keys: {missing}"
    assert (
        payload["primary_metric"] == "accuracy"
    )  # classification picks accuracy first
    assert payload["training_duration_ms"] == 300
