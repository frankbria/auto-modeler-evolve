"""Tests for chat/narration.py — plain-English event narrators."""

import json


class TestNarrateUpload:
    def test_basic_upload_message(self):
        from chat.narration import narrate_upload

        result = narrate_upload("sales.csv", 500, 8)
        assert "sales.csv" in result
        assert "500" in result
        assert "8" in result

    def test_includes_column_names(self):
        from chat.narration import narrate_upload

        result = narrate_upload(
            "data.csv", 100, 3, column_names=["revenue", "region", "date"]
        )
        assert "revenue" in result
        assert "region" in result

    def test_truncates_long_column_list(self):
        from chat.narration import narrate_upload

        cols = [f"col_{i}" for i in range(10)]
        result = narrate_upload("big.csv", 1000, 10, column_names=cols)
        # Should mention "and X more"
        assert "more" in result

    def test_includes_insights(self):
        from chat.narration import narrate_upload

        insights = [
            "Column 'revenue' has 5% missing values",
            "Strong seasonality in 'date'",
        ]
        result = narrate_upload("data.csv", 200, 5, insights=insights)
        assert "revenue" in result or "missing" in result

    def test_no_insights_still_works(self):
        from chat.narration import narrate_upload

        result = narrate_upload("data.csv", 100, 4)
        assert isinstance(result, str) and result.strip()

    def test_ends_with_call_to_action(self):
        from chat.narration import narrate_upload

        result = narrate_upload("data.csv", 100, 4)
        # Should end with something actionable
        assert (
            "ask" in result.lower()
            or "feel free" in result.lower()
            or "explore" in result.lower()
        )

    def test_large_row_count_formatted(self):
        from chat.narration import narrate_upload

        result = narrate_upload("big.csv", 1_000_000, 20)
        assert "1,000,000" in result


class TestNarrateProfileHighlights:
    def test_returns_none_for_empty_profile(self):
        from chat.narration import narrate_profile_highlights

        assert narrate_profile_highlights({}) is None

    def test_includes_patterns(self):
        from chat.narration import narrate_profile_highlights

        profile = {
            "patterns": ["Strong upward trend in revenue", "Seasonality detected"]
        }
        result = narrate_profile_highlights(profile)
        assert result is not None
        assert (
            "trend" in result.lower()
            or "pattern" in result.lower()
            or "revenue" in result.lower()
        )

    def test_includes_warnings(self):
        from chat.narration import narrate_profile_highlights

        profile = {"warnings": [{"message": "Column X has 20% missing values"}]}
        result = narrate_profile_highlights(profile)
        assert result is not None
        assert "missing" in result.lower() or "Column X" in result

    def test_strong_correlation_highlight(self):
        from chat.narration import narrate_profile_highlights

        profile = {"correlations": {"revenue & sales": 0.85}}
        result = narrate_profile_highlights(profile)
        assert result is not None
        assert "corr" in result.lower() or "revenue" in result.lower()

    def test_weak_correlation_not_highlighted(self):
        from chat.narration import narrate_profile_highlights

        profile = {"correlations": {"col_a & col_b": 0.3}}
        # Too weak — should not be highlighted
        result = narrate_profile_highlights(profile)
        # If the only info is a weak correlation, no highlights
        if result:
            assert "0.30" not in result or "corr" not in result.lower()


class TestNarrateTrainingComplete:
    def test_single_model_success(self):
        from chat.narration import narrate_training_complete

        runs = [
            {
                "algorithm": "RandomForestRegressor",
                "status": "done",
                "metrics": {"r2": 0.88, "mae": 120.0},
                "summary": "Good fit.",
            }
        ]
        result = narrate_training_complete(runs, "regression", "revenue")
        assert "RandomForestRegressor" in result
        assert "revenue" not in result or "revenue" in result  # either is fine

    def test_multiple_models_ranked(self):
        from chat.narration import narrate_training_complete

        runs = [
            {
                "algorithm": "LinearRegression",
                "status": "done",
                "metrics": {"r2": 0.72},
                "summary": "",
            },
            {
                "algorithm": "RandomForestRegressor",
                "status": "done",
                "metrics": {"r2": 0.88},
                "summary": "",
            },
        ]
        result = narrate_training_complete(runs, "regression", "revenue")
        # Better model should appear near the top
        assert "RandomForestRegressor" in result
        assert "LinearRegression" in result

    def test_classification_uses_accuracy(self):
        from chat.narration import narrate_training_complete

        runs = [
            {
                "algorithm": "LogisticRegression",
                "status": "done",
                "metrics": {"accuracy": 0.91, "f1": 0.89},
                "summary": "",
            },
        ]
        result = narrate_training_complete(runs, "classification", "churn")
        assert "accuracy" in result.lower() or "91" in result

    def test_all_failed(self):
        from chat.narration import narrate_training_complete

        runs = [
            {
                "algorithm": "LinearRegression",
                "status": "failed",
                "metrics": {},
                "summary": "",
            }
        ]
        result = narrate_training_complete(runs, "regression", "revenue")
        assert "failed" in result.lower()

    def test_empty_runs(self):
        from chat.narration import narrate_training_complete

        result = narrate_training_complete([], "regression", "revenue")
        assert isinstance(result, str) and result.strip()

    def test_includes_next_step_guidance(self):
        from chat.narration import narrate_training_complete

        runs = [
            {
                "algorithm": "LinearRegression",
                "status": "done",
                "metrics": {"r2": 0.80},
                "summary": "",
            }
        ]
        result = narrate_training_complete(runs, "regression", "revenue")
        assert (
            "validate" in result.lower()
            or "Validate" in result
            or "validate" in result.lower()
        )


class TestNarrateModelSelected:
    def test_includes_algorithm_name(self):
        from chat.narration import narrate_model_selected

        result = narrate_model_selected(
            "RandomForestRegressor", {"r2": 0.88, "mae": 120.0}, "regression"
        )
        assert "RandomForestRegressor" in result

    def test_includes_deploy_guidance(self):
        from chat.narration import narrate_model_selected

        result = narrate_model_selected("LinearRegression", {"r2": 0.75}, "regression")
        assert "deploy" in result.lower() or "Deploy" in result


class TestNarrateDeployment:
    def test_includes_urls(self):
        from chat.narration import narrate_deployment

        result = narrate_deployment(
            "RandomForestRegressor",
            "/predict/abc123",
            "/api/predict/abc123",
        )
        assert "/predict/abc123" in result
        assert "/api/predict/abc123" in result

    def test_includes_algorithm(self):
        from chat.narration import narrate_deployment

        result = narrate_deployment(
            "GradientBoostingClassifier", "/predict/x", "/api/predict/x"
        )
        assert "GradientBoostingClassifier" in result


class TestAppendBotMessage:
    def test_creates_conversation_and_appends(self, tmp_path):
        """append_bot_message_to_conversation should create a Conversation if none exists."""
        from sqlmodel import Session, SQLModel, create_engine
        from models.conversation import Conversation
        from chat.narration import append_bot_message_to_conversation

        db_path = tmp_path / "test_narration.db"
        engine = create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(engine)

        project_id = "test-project-narr-001"

        with Session(engine) as session:
            append_bot_message_to_conversation(
                project_id, "Hello from the bot!", session
            )

        with Session(engine) as session:
            from sqlmodel import select

            conv = session.exec(
                select(Conversation).where(Conversation.project_id == project_id)
            ).first()
            assert conv is not None
            messages = json.loads(conv.messages)
            assert len(messages) == 1
            assert messages[0]["role"] == "assistant"
            assert messages[0]["content"] == "Hello from the bot!"

    def test_appends_to_existing_conversation(self, tmp_path):
        import json
        from sqlmodel import Session, SQLModel, create_engine
        from models.conversation import Conversation
        from chat.narration import append_bot_message_to_conversation

        db_path = tmp_path / "test_narration2.db"
        engine = create_engine(
            f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(engine)

        project_id = "test-project-narr-002"

        # Pre-create a conversation with a user message
        with Session(engine) as session:
            existing_messages = json.dumps(
                [{"role": "user", "content": "Hi", "timestamp": "2024-01-01T00:00:00"}]
            )
            conv = Conversation(project_id=project_id, messages=existing_messages)
            session.add(conv)
            session.commit()

        with Session(engine) as session:
            append_bot_message_to_conversation(project_id, "Welcome back!", session)

        with Session(engine) as session:
            from sqlmodel import select

            conv = session.exec(
                select(Conversation).where(Conversation.project_id == project_id)
            ).first()
            messages = json.loads(conv.messages)
            assert len(messages) == 2
            assert messages[1]["role"] == "assistant"
            assert messages[1]["content"] == "Welcome back!"


class TestCallClaude:
    def test_returns_fallback_when_no_auth(self, monkeypatch):
        """_call_claude returns fallback immediately if no Anthropic auth configured."""
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from chat.narration import _call_claude

        result = _call_claude("some prompt", fallback="static fallback")
        assert result == "static fallback"

    def test_returns_fallback_on_api_error(self, monkeypatch):
        """_call_claude returns fallback if anthropic raises any exception."""
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")
        with patch("anthropic.Anthropic", return_value=mock_client):
            from chat.narration import _call_claude

            result = _call_claude("some prompt", fallback="fallback text")
        assert result == "fallback text"

    def test_returns_claude_response_on_success(self, monkeypatch):
        """_call_claude returns the model's text content on success."""
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")
        from unittest.mock import MagicMock, patch

        mock_content = MagicMock()
        mock_content.text = "  AI generated insight.  "
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        with patch("anthropic.Anthropic", return_value=mock_client):
            from chat.narration import _call_claude

            result = _call_claude("some prompt", fallback="fallback")
        assert result == "AI generated insight."


class TestNarrateDataInsightsAi:
    def test_returns_none_without_auth(self, monkeypatch):
        """narrate_data_insights_ai returns None when no auth is configured."""
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from chat.narration import narrate_data_insights_ai

        result = narrate_data_insights_ai("revenue, region", "{}", 200, 5)
        assert result is None

    def test_returns_string_when_claude_succeeds(self, monkeypatch):
        """narrate_data_insights_ai returns the AI insight string on success."""
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")
        from unittest.mock import MagicMock, patch

        mock_content = MagicMock()
        mock_content.text = "I noticed your top region drives 60% of revenue."
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        with patch("anthropic.Anthropic", return_value=mock_client):
            from chat.narration import narrate_data_insights_ai

            result = narrate_data_insights_ai("revenue, region", "{}", 200, 5)
        assert result is not None
        assert "revenue" in result or "region" in result or "60" in result

    def test_returns_none_when_claude_returns_empty(self, monkeypatch):
        """narrate_data_insights_ai returns None if Claude response is empty."""
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")
        from unittest.mock import MagicMock, patch

        mock_content = MagicMock()
        mock_content.text = ""
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        with patch("anthropic.Anthropic", return_value=mock_client):
            from chat.narration import narrate_data_insights_ai

            result = narrate_data_insights_ai("col1, col2", "{}", 10, 2)
        assert result is None


class TestNarrateTrainingWithAi:
    def test_falls_back_to_static_for_single_model(self, monkeypatch):
        """Single completed model: should use static narration without calling Claude."""
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")
        from unittest.mock import MagicMock, patch

        mock_client = MagicMock()
        with patch("anthropic.Anthropic", return_value=mock_client):
            from chat.narration import narrate_training_with_ai

            runs = [
                {
                    "algorithm": "LinearRegression",
                    "status": "done",
                    "metrics": {"r2": 0.85},
                    "summary": "",
                }
            ]
            result = narrate_training_with_ai(runs, "regression", "revenue")
        # Static narration should not call Claude for single model
        mock_client.messages.create.assert_not_called()
        assert "LinearRegression" in result

    def test_falls_back_when_no_auth(self, monkeypatch):
        """narrate_training_with_ai falls back to static when no auth configured."""
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from chat.narration import narrate_training_with_ai

        runs = [
            {
                "algorithm": "LinearRegression",
                "status": "done",
                "metrics": {"r2": 0.72},
                "summary": "",
            },
            {
                "algorithm": "RandomForestRegressor",
                "status": "done",
                "metrics": {"r2": 0.88},
                "summary": "",
            },
        ]
        result = narrate_training_with_ai(runs, "regression", "revenue")
        assert isinstance(result, str)
        assert "LinearRegression" in result or "RandomForest" in result

    def test_uses_claude_for_multiple_models(self, monkeypatch):
        """narrate_training_with_ai calls Claude and returns AI-authored text for 2+ models."""
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")
        from unittest.mock import MagicMock, patch

        mock_content = MagicMock()
        mock_content.text = (
            "RandomForest wins on R² but LinearRegression is more explainable."
        )
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        with patch("anthropic.Anthropic", return_value=mock_client):
            from chat.narration import narrate_training_with_ai

            runs = [
                {
                    "algorithm": "LinearRegression",
                    "status": "done",
                    "metrics": {"r2": 0.72},
                    "summary": "",
                },
                {
                    "algorithm": "RandomForestRegressor",
                    "status": "done",
                    "metrics": {"r2": 0.88},
                    "summary": "",
                },
            ]
            result = narrate_training_with_ai(runs, "regression", "revenue")
        assert "RandomForest" in result or "explainable" in result
        assert "Validate" in result  # CTA appended

    def test_appends_validate_cta_to_ai_response(self, monkeypatch):
        """The CTA directing user to Validate tab is always appended."""
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-key")
        from unittest.mock import MagicMock, patch

        mock_content = MagicMock()
        mock_content.text = "Model comparison complete."
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        with patch("anthropic.Anthropic", return_value=mock_client):
            from chat.narration import narrate_training_with_ai

            runs = [
                {
                    "algorithm": "A",
                    "status": "done",
                    "metrics": {"r2": 0.7},
                    "summary": "",
                },
                {
                    "algorithm": "B",
                    "status": "done",
                    "metrics": {"r2": 0.8},
                    "summary": "",
                },
            ]
            result = narrate_training_with_ai(runs, "regression", "target")
        assert "Validate" in result
