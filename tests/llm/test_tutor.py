"""Tests for Language Model tutor module.

This module tests the LLM-based language tutor functionality
including conversation management and response generation.
"""

import pytest

from polyglott.constants import AgeGroup, TargetLanguage
from polyglott.llm.tutor import (
    LanguageTutor,
    Message,
    TutorConfig,
    TutorResponse,
    check_ollama_available,
    create_tutor,
)


class TestMessage:
    """Tests for Message dataclass."""

    def test_creation(self) -> None:
        """Test message can be created."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_to_dict(self) -> None:
        """Test message converts to dict."""
        msg = Message(role="assistant", content="Hi there!")
        d = msg.to_dict()
        assert d == {"role": "assistant", "content": "Hi there!"}


class TestTutorResponse:
    """Tests for TutorResponse dataclass."""

    def test_creation_default(self) -> None:
        """Test response with defaults."""
        response = TutorResponse(text="Hello!")
        assert response.text == "Hello!"
        assert response.is_complete is True
        assert response.tokens_generated is None

    def test_creation_incomplete(self) -> None:
        """Test incomplete response."""
        response = TutorResponse(text="Error", is_complete=False)
        assert response.is_complete is False


class TestTutorConfig:
    """Tests for TutorConfig dataclass."""

    def test_defaults(self) -> None:
        """Test config has sensible defaults."""
        config = TutorConfig()
        assert config.model == "qwen2.5:7b"
        assert config.target_language == TargetLanguage.SPANISH
        assert config.age_group == AgeGroup.EARLY_PRIMARY

    def test_custom_values(self) -> None:
        """Test config accepts custom values."""
        config = TutorConfig(
            target_language=TargetLanguage.JAPANESE,
            native_language="German",
            age_group=AgeGroup.PRESCHOOL,
        )
        assert config.target_language == TargetLanguage.JAPANESE
        assert config.native_language == "German"
        assert config.age_group == AgeGroup.PRESCHOOL


class TestLanguageTutor:
    """Tests for LanguageTutor class."""

    def test_initialization(self) -> None:
        """Test tutor initializes correctly."""
        tutor = LanguageTutor()
        assert tutor.config is not None
        assert tutor.conversation_length == 0

    def test_initialization_with_config(self) -> None:
        """Test tutor accepts custom config."""
        config = TutorConfig(target_language=TargetLanguage.GERMAN)
        tutor = LanguageTutor(config)
        assert tutor.config.target_language == TargetLanguage.GERMAN

    def test_system_prompt_generation(self) -> None:
        """Test system prompt is generated correctly."""
        config = TutorConfig(
            target_language=TargetLanguage.SPANISH,
            native_language="English",
            lesson_focus="greetings",
        )
        tutor = LanguageTutor(config)
        prompt = tutor.system_prompt

        assert "Spanish" in prompt
        assert "English" in prompt
        assert "greetings" in prompt

    def test_reset_conversation(self) -> None:
        """Test conversation reset returns greeting."""
        tutor = LanguageTutor()
        greeting = tutor.reset_conversation()

        assert isinstance(greeting, str)
        assert len(greeting) > 0

    def test_get_available_lessons(self) -> None:
        """Test getting available lessons."""
        tutor = LanguageTutor()
        lessons = tutor.get_available_lessons()

        assert isinstance(lessons, list)
        assert len(lessons) > 0

    def test_set_lesson_focus(self) -> None:
        """Test changing lesson focus."""
        tutor = LanguageTutor()
        tutor.set_lesson_focus("numbers and counting")
        assert tutor.config.lesson_focus == "numbers and counting"


class TestCreateTutor:
    """Tests for create_tutor factory function."""

    def test_creates_tutor(self) -> None:
        """Test factory creates valid tutor."""
        tutor = create_tutor()
        assert isinstance(tutor, LanguageTutor)

    def test_with_parameters(self) -> None:
        """Test factory accepts parameters."""
        tutor = create_tutor(
            target_language=TargetLanguage.MANDARIN,
            native_language="German",
            age_group=AgeGroup.LATE_PRIMARY,
        )
        assert tutor.config.target_language == TargetLanguage.MANDARIN
        assert tutor.config.native_language == "German"
        assert tutor.config.age_group == AgeGroup.LATE_PRIMARY


class TestCheckOllamaAvailable:
    """Tests for check_ollama_available function."""

    def test_returns_tuple(self) -> None:
        """Test function returns proper tuple."""
        available, message = check_ollama_available()
        assert isinstance(available, bool)
        assert isinstance(message, str)
