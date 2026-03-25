"""Smoke-тесты: импорты, конфигурация, модели."""

from app.models.schemas import (
    ConfidenceLevel,
    compute_confidence_label,
    compute_confidence_level,
)


class TestImports:
    """Проверяем что основные модули импортируются без ошибок."""

    def test_import_config(self):
        from app.config import Settings

        assert Settings is not None

    def test_import_schemas(self):
        from app.models.schemas import ChatRequest, ChatResponse, HealthResponse

        assert ChatRequest is not None
        assert ChatResponse is not None
        assert HealthResponse is not None

    def test_import_reason_schemas(self):
        from app.models.reason_schemas import ContactReason, ContactReasonsData

        assert ContactReason is not None
        assert ContactReasonsData is not None

    def test_import_classifier(self):
        from app.classifier.reason_classifier import ClassificationCandidate, L1Result

        assert ClassificationCandidate is not None
        assert L1Result is not None

    def test_import_section_classifier(self):
        from app.classifier.section_classifier import L2Result

        assert L2Result is not None

    def test_import_reason_store(self):
        from app.database.reason_store import load_reasons, save_reasons

        assert load_reasons is not None
        assert save_reasons is not None


class TestConfidenceLevels:
    """Тесты бизнес-логики уровней уверенности."""

    def test_high_confidence(self):
        assert compute_confidence_level(0.9) == ConfidenceLevel.high
        assert compute_confidence_level(0.8) == ConfidenceLevel.high

    def test_acceptable_confidence(self):
        assert compute_confidence_level(0.7) == ConfidenceLevel.acceptable
        assert compute_confidence_level(0.6) == ConfidenceLevel.acceptable

    def test_partial_confidence(self):
        assert compute_confidence_level(0.5) == ConfidenceLevel.partial
        assert compute_confidence_level(0.3) == ConfidenceLevel.partial

    def test_escalation_confidence(self):
        assert compute_confidence_level(0.2) == ConfidenceLevel.escalation
        assert compute_confidence_level(0.0) == ConfidenceLevel.escalation

    def test_confidence_labels(self):
        label = compute_confidence_label(0.9)
        assert "Уверенный" in label

        label = compute_confidence_label(0.1)
        assert "эскалация" in label.lower()


class TestPydanticModels:
    """Тесты валидации Pydantic-моделей."""

    def test_health_response(self):
        from app.models.schemas import HealthResponse

        resp = HealthResponse(
            status="ok",
            version="2.0.0",
            knowledge_base_ready=False,
            total_articles=0,
            total_chunks=0,
            support_tickets_count=0,
        )
        assert resp.status == "ok"
        assert resp.version == "2.0.0"

    def test_chat_request_minimal(self):
        from app.models.schemas import ChatRequest

        req = ChatRequest(message="Привет")
        assert req.message == "Привет"

    def test_contact_reason_model(self):
        from app.models.reason_schemas import ContactReason, Markers

        reason = ContactReason(
            id="test-1",
            name="Тест",
            markers=Markers(
                nouns=["тест"],
                verbs=["тестировать"],
            ),
        )
        assert reason.id == "test-1"
        assert "тест" in reason.markers.nouns
