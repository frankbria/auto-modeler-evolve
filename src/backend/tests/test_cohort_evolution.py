"""Tests for compute_cohort_evolution() — Track B: Predictive Cohort Monitoring."""

import math
import re
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------

_COHORT_EVOLUTION_PATTERNS = re.compile(
    r"(?i)(?:"
    r"how\s+ha(?:s|ve)\s+(?:my\s+)?(?:top|at.risk|highest.scoring|predicted)\s*(?:\d+\s+)?(?:cohort|predictions?|customers?|records?|group)\s+changed\b|"
    r"(?:track|monitor|watch)\s+(?:my\s+)?(?:top|at.risk)\s*(?:\d+\s+)?(?:cohort|predictions?|customers?|records?)\s+over\s+time\b|"
    r"cohort\s+(?:evolution|monitoring|tracking|trend|drift|shift|changes?)\b|"
    r"(?:which|what)\s+(?:segments?|groups?|categories?)\s+(?:are\s+)?(?:rising|falling|trending|shifting|growing|declining)\s+(?:in|among|within)\s+(?:my\s+)?(?:top|highest|at.risk)\b|"
    r"(?:how\s+(?:has|have)|what\s+changed\s+(?:about|in))\s+(?:my\s+)?(?:top|at.risk|highest)\s*(?:\d+\s+)?(?:predictions?|customers?|records?|cohort)\b|"
    r"predictive\s+cohort\s+(?:monitoring|tracking|analysis|report)\b|"
    r"(?:who|what)\s+(?:is|are)\s+(?:rising|falling|trending|shifting)\s+(?:into|out\s+of)\s+(?:my\s+)?(?:top|highest|at.risk|predicted)\b|"
    r"(?:trend|change|shift|evolution)\s+(?:in|of)\s+(?:my\s+)?(?:top|highest|at.risk)\s*(?:\d+\s+)?(?:cohort|predictions?|customers?|records?)\b|"
    r"cohort\s+(?:profile|composition)\s+(?:over\s+time|across\s+(?:periods?|uploads?|datasets?))\b"
    r")",
    re.IGNORECASE,
)


class TestCohortEvolutionRegex:
    """Test that the regex matches expected natural-language phrases."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "how has my top cohort changed",
            "how have my top 20 customers changed",
            "how has my highest-scoring group changed",
            "track my top cohort over time",
            "monitor my at-risk predictions over time",
            "cohort evolution report",
            "cohort monitoring",
            "cohort tracking",
            "cohort trend",
            "cohort drift",
            "cohort shift",
            "what categories are rising in my top",
            "which segments are falling within my highest",
            "which groups are growing in my at-risk",
            "how has my top 10 cohort changed",
            "what changed about my top predictions",
            "what changed in my at-risk cohort",
            "predictive cohort monitoring",
            "predictive cohort tracking",
            "predictive cohort analysis",
            "who is rising into my top",
            "what are shifting out of my predicted",
            "trend in my top 20 cohort",
            "shift of my highest predictions",
            "evolution in my top records",
            "cohort profile over time",
            "cohort composition across uploads",
            "cohort composition across periods",
        ],
    )
    def test_matches(self, phrase):
        assert _COHORT_EVOLUTION_PATTERNS.search(
            phrase
        ), f"Expected match for: {phrase!r}"

    @pytest.mark.parametrize(
        "phrase",
        [
            "show me the data",
            "train a model",
            "what are the top features",
            "cohort",  # bare word — no qualifying noun after it
        ],
    )
    def test_no_false_positives(self, phrase):
        assert not _COHORT_EVOLUTION_PATTERNS.search(
            phrase
        ), f"Unexpected match for: {phrase!r}"


# ---------------------------------------------------------------------------
# Pure function unit tests
# ---------------------------------------------------------------------------


def _make_minimal_pipeline(
    target_column: str = "score", problem_type: str = "regression"
):
    """Return a minimal mock pipeline object."""
    pipeline = MagicMock()
    pipeline.target_column = target_column
    pipeline.problem_type = problem_type
    return pipeline


def _make_cohort_result(
    cat_profile: list[dict] | None = None,
    num_profile: list[dict] | None = None,
    target: str = "score",
) -> dict:
    """Return a minimal compute_prediction_cohort() result."""
    return {
        "target_column": target,
        "problem_type": "regression",
        "n": 20,
        "direction": "highest",
        "categorical_profile": cat_profile or [],
        "numeric_profile": num_profile or [],
        "summary": "Test cohort",
    }


def _make_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age_group": rng.choice(["18-25", "26-35", "36+"], size=n),
            "region": rng.choice(["north", "south"], size=n),
            "score": rng.uniform(0, 1, size=n),
        }
    )


class TestComputeCohortEvolution:
    """Unit tests for compute_cohort_evolution() via mocked dependencies."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self, tmp_path):
        """Patch load_pipeline, joblib.load, and compute_prediction_cohort."""
        fake_pipeline = _make_minimal_pipeline()

        # Build two distinct categorical profiles so shifts can be computed.
        cat_profile_a = [
            {
                "column": "age_group",
                "categories": [
                    {
                        "value": "18-25",
                        "top_pct": 60.0,
                        "overall_pct": 30.0,
                        "ratio": 2.0,
                    }
                ],
                "dominant": "18-25",
                "dominant_top_pct": 60.0,
            }
        ]
        cat_profile_b = [
            {
                "column": "age_group",
                "categories": [
                    {
                        "value": "18-25",
                        "top_pct": 40.0,
                        "overall_pct": 30.0,
                        "ratio": 1.33,
                    }
                ],
                "dominant": "18-25",
                "dominant_top_pct": 40.0,
            }
        ]

        cohort_results = [
            _make_cohort_result(cat_profile=cat_profile_a),
            _make_cohort_result(cat_profile=cat_profile_b),
        ]
        self._call_count = 0

        def _side_effect(*args, **kwargs):
            result = cohort_results[min(self._call_count, len(cohort_results) - 1)]
            self._call_count += 1
            return result

        self._model_path = str(tmp_path / "model.pkl")
        self._pipeline_path = str(tmp_path / "pipeline.pkl")

        # Touch the files so Path.exists() returns True inside the function
        open(self._model_path, "w").close()
        open(self._pipeline_path, "w").close()

        with (
            patch("core.deployer.load_pipeline", return_value=fake_pipeline),
            patch("core.deployer.joblib.load", return_value=MagicMock()),
            patch("core.deployer.compute_prediction_cohort", side_effect=_side_effect),
        ):
            yield

    def _run(self, dframes=None, n: int = 20, direction: str = "highest"):
        from core.deployer import compute_cohort_evolution

        if dframes is None:
            df_a = _make_df(100)
            df_b = _make_df(100)
            dframes = [(df_a, "ds1", "Upload 1"), (df_b, "ds2", "Upload 2")]
        return compute_cohort_evolution(
            self._pipeline_path, self._model_path, dframes, n=n, direction=direction
        )

    # -- Structure --

    def test_returns_required_keys(self):
        result = self._run()
        for key in (
            "n",
            "direction",
            "target_column",
            "problem_type",
            "n_periods",
            "periods",
            "shifts",
            "summary",
        ):
            assert key in result, f"Missing key: {key}"

    def test_n_periods_matches_periods_list(self):
        result = self._run()
        assert result["n_periods"] == len(result["periods"])

    def test_shifts_length_is_periods_minus_one(self):
        result = self._run()
        assert len(result["shifts"]) == result["n_periods"] - 1

    def test_period_has_required_keys(self):
        result = self._run()
        for period in result["periods"]:
            for key in (
                "period_label",
                "dataset_id",
                "total_rows",
                "categorical_profile",
                "numeric_profile",
            ):
                assert key in period, f"Period missing key: {key}"

    def test_shift_has_required_keys(self):
        result = self._run()
        for shift in result["shifts"]:
            for key in ("from_period", "to_period", "categorical_shifts", "summary"):
                assert key in shift, f"Shift missing key: {key}"

    def test_categorical_shift_has_required_keys(self):
        result = self._run()
        for shift in result["shifts"]:
            for cs in shift["categorical_shifts"]:
                for key in (
                    "column",
                    "category",
                    "from_pct",
                    "to_pct",
                    "change",
                    "direction",
                ):
                    assert key in cs, f"CategoricalShift missing key: {key}"

    # -- Values --

    def test_direction_values(self):
        result = self._run(direction="highest")
        assert result["direction"] == "highest"

    def test_n_value_passed_through(self):
        result = self._run(n=50)
        assert result["n"] == 50

    def test_shift_direction_field(self):
        result = self._run()
        for shift in result["shifts"]:
            for cs in shift["categorical_shifts"]:
                assert cs["direction"] in ("rising", "falling")

    def test_shifts_only_surfaced_when_at_least_5pp(self):
        """All surfaced shifts must have abs(change) >= 5.0."""
        result = self._run()
        for shift in result["shifts"]:
            for cs in shift["categorical_shifts"]:
                assert (
                    abs(cs["change"]) >= 5.0
                ), f"Small shift leaked: column={cs['column']}, change={cs['change']}"

    def test_shift_change_calculated_correctly(self):
        result = self._run()
        for shift in result["shifts"]:
            for cs in shift["categorical_shifts"]:
                expected = round(cs["to_pct"] - cs["from_pct"], 1)
                assert math.isclose(cs["change"], expected, abs_tol=0.15)

    def test_summary_is_nonempty_string(self):
        result = self._run()
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    # -- Edge cases --

    def test_raises_with_fewer_than_two_dframes(self):
        from core.deployer import compute_cohort_evolution

        with pytest.raises(ValueError, match="At least 2"):
            compute_cohort_evolution(
                self._pipeline_path,
                self._model_path,
                [(_make_df(), "ds1", "Upload 1")],
            )

    def test_caps_at_six_most_recent_periods(self):
        """If more than 6 frames are passed, only the last 6 are used."""
        dframes = [(_make_df(), f"ds{i}", f"Upload {i}") for i in range(8)]
        result = self._run(dframes=dframes)
        assert result["n_periods"] <= 6

    def test_empty_dataframes_skipped(self):
        """Empty DataFrames should be silently skipped; compute still works if ≥2 remain."""
        df_empty = pd.DataFrame()
        df_a = _make_df(100)
        df_b = _make_df(100)
        dframes = [
            (df_empty, "ds0", "Empty"),
            (df_a, "ds1", "Upload 1"),
            (df_b, "ds2", "Upload 2"),
        ]
        result = self._run(dframes=dframes)
        # Empty frame is skipped — should still have at least 2 periods
        assert result["n_periods"] >= 2
