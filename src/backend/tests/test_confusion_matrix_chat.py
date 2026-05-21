"""Tests for confusion matrix chat card — pattern matching and integration."""

import numpy as np
import pytest

from core.validator import compute_confusion_matrix

# ---------------------------------------------------------------------------
# Unit tests for enhanced compute_confusion_matrix
# ---------------------------------------------------------------------------


class TestComputeConfusionMatrix:
    def test_binary_basic(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        result = compute_confusion_matrix(y_true, y_pred)
        assert result["matrix"] == [[2, 1], [1, 2]]
        assert result["total"] == 6
        assert result["correct"] == 4
        assert result["accuracy"] == pytest.approx(4 / 6, abs=0.001)
        assert result["labels"] == ["0", "1"]

    def test_binary_per_class_metrics(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        result = compute_confusion_matrix(y_true, y_pred)
        metrics = result["per_class_metrics"]
        assert len(metrics) == 2
        # Class 0: TP=2, FP=1, FN=1
        m0 = metrics[0]
        assert m0["label"] == "0"
        assert m0["precision"] == pytest.approx(2 / 3, abs=0.01)
        assert m0["recall"] == pytest.approx(2 / 3, abs=0.01)
        assert m0["support"] == 3

    def test_binary_most_confused_pair(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 1, 0])  # 0→1 occurs twice, 1→0 once
        result = compute_confusion_matrix(y_true, y_pred)
        pair = result["most_confused_pair"]
        assert pair is not None
        assert pair["actual"] == "0"
        assert pair["predicted"] == "1"
        assert pair["count"] == 2

    def test_perfect_predictions_no_confused_pair(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        result = compute_confusion_matrix(y_true, y_pred)
        assert result["accuracy"] == 1.0
        assert result["most_confused_pair"] is None

    def test_custom_class_labels(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        result = compute_confusion_matrix(y_true, y_pred, class_labels=["cat", "dog"])
        assert result["labels"] == ["cat", "dog"]
        assert result["per_class_metrics"][0]["label"] == "cat"
        assert result["per_class_metrics"][1]["label"] == "dog"

    def test_multiclass_per_class_metrics(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 0])
        result = compute_confusion_matrix(y_true, y_pred)
        assert len(result["per_class_metrics"]) == 3
        # Class 2: TP=1, FP=1 (0 pred as 2), FN=1 (2 pred as 0)
        m2 = result["per_class_metrics"][2]
        assert m2["label"] == "2"
        assert m2["precision"] == pytest.approx(0.5, abs=0.01)
        assert m2["recall"] == pytest.approx(0.5, abs=0.01)

    def test_f1_computed_correctly(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])  # precision=recall=0.5 for class 1
        result = compute_confusion_matrix(y_true, y_pred)
        m1 = result["per_class_metrics"][1]
        assert m1["f1"] == pytest.approx(0.5, abs=0.01)

    def test_summary_binary_present(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = compute_confusion_matrix(y_true, y_pred)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_summary_multiclass_names_worst(self):
        # Class "b" should have worst recall (0/2 correct)
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 0, 0, 2, 2])  # class 1 has 0% recall
        result = compute_confusion_matrix(y_true, y_pred, class_labels=["a", "b", "c"])
        assert "b" in result["summary"]

    def test_return_fields_complete(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 0])
        result = compute_confusion_matrix(y_true, y_pred)
        for field in (
            "matrix",
            "labels",
            "total",
            "correct",
            "accuracy",
            "per_class_metrics",
            "most_confused_pair",
            "summary",
        ):
            assert field in result


# ---------------------------------------------------------------------------
# Pattern matching tests
# ---------------------------------------------------------------------------


class TestConfusionMatrixPatterns:
    @pytest.fixture
    def pattern(self):
        from api.chat import _CONFUSION_MATRIX_PATTERNS

        return _CONFUSION_MATRIX_PATTERNS

    @pytest.mark.parametrize(
        "msg",
        [
            "show me the confusion matrix",
            "confusion matrix",
            "display the confusion matrix",
            "where does my model make mistakes",
            "how does my model make errors",
            "true positives",
            "false negatives",
            "classification accuracy by class",
            "precision per class",
            "recall per class",
            "f1 per class",
            "model classification breakdown",
            "model classification errors",
        ],
    )
    def test_positive_matches(self, pattern, msg):
        assert pattern.search(msg), f"Should match: {msg!r}"

    @pytest.mark.parametrize(
        "msg",
        [
            "train a model",
            "what is my accuracy",
            "show me the data",
            "how consistent is my model",
            "cross validation scores",
        ],
    )
    def test_negative_no_match(self, pattern, msg):
        assert not pattern.search(msg), f"Should not match: {msg!r}"
