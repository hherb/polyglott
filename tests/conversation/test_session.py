"""Tests for conversation session module.

This module tests session management and tracking functionality.
"""

import pytest

from polyglott.constants import AgeGroup, TargetLanguage
from polyglott.conversation.session import (
    ConversationSession,
    SessionConfig,
    create_session,
)


class TestSessionConfig:
    """Tests for SessionConfig dataclass."""

    def test_defaults(self) -> None:
        """Test config has sensible defaults."""
        config = SessionConfig()
        assert config.student_name == "Student"
        assert config.native_language == "English"
        assert config.target_language == TargetLanguage.SPANISH

    def test_custom_values(self) -> None:
        """Test config accepts custom values."""
        config = SessionConfig(
            student_name="Emma",
            native_language="German",
            target_language=TargetLanguage.JAPANESE,
        )
        assert config.student_name == "Emma"
        assert config.native_language == "German"
        assert config.target_language == TargetLanguage.JAPANESE


class TestConversationSession:
    """Tests for ConversationSession class."""

    def test_creation(self) -> None:
        """Test session can be created."""
        session = ConversationSession()
        assert session.session_id is not None
        assert session.turn_count == 0
        assert len(session.words_learned) == 0

    def test_record_turn(self) -> None:
        """Test recording a conversation turn."""
        session = ConversationSession()
        session.record_turn("Hello", "Hi there!")
        assert session.turn_count == 1

    def test_add_word_learned(self) -> None:
        """Test adding learned words."""
        session = ConversationSession()
        session.add_word_learned("hola")
        session.add_word_learned("adiós")
        session.add_word_learned("hola")  # Duplicate

        assert len(session.words_learned) == 2
        assert "hola" in session.words_learned
        assert "adiós" in session.words_learned

    def test_duration_minutes(self) -> None:
        """Test duration calculation."""
        session = ConversationSession()
        duration = session.duration_minutes
        assert duration >= 0

    def test_get_summary(self) -> None:
        """Test summary generation."""
        config = SessionConfig(student_name="Emma")
        session = ConversationSession(config=config)
        session.record_turn("Hi", "Hello!")

        summary = session.get_summary()

        assert "Emma" in summary
        assert "Session" in summary
        assert "1" in summary  # turn count


class TestCreateSession:
    """Tests for create_session factory function."""

    def test_creates_session(self) -> None:
        """Test factory creates valid session."""
        session = create_session()
        assert isinstance(session, ConversationSession)

    def test_with_parameters(self) -> None:
        """Test factory accepts parameters."""
        session = create_session(
            student_name="Max",
            target_language=TargetLanguage.GERMAN,
            native_language="English",
            age_group=AgeGroup.PRESCHOOL,
        )
        assert session.config.student_name == "Max"
        assert session.config.target_language == TargetLanguage.GERMAN
        assert session.config.age_group == AgeGroup.PRESCHOOL
