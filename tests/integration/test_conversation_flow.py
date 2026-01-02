"""Integration tests for conversation flow.

These tests verify the complete conversation flow works
correctly across all components.
"""

import pytest

from polyglott.constants import AgeGroup, TargetLanguage
from polyglott.conversation.manager import ConversationManager, create_manager
from polyglott.conversation.session import ConversationSession


class TestConversationFlowIntegration:
    """Integration tests for conversation flow."""

    def test_manager_creation(self) -> None:
        """Test manager can be created."""
        manager = create_manager(
            student_name="Test Student",
            target_language=TargetLanguage.SPANISH,
        )
        assert isinstance(manager, ConversationManager)
        assert manager.student_name == "Test Student"
        assert manager.target_language == TargetLanguage.SPANISH

    def test_session_lifecycle(self) -> None:
        """Test complete session lifecycle."""
        manager = ConversationManager(
            student_name="Emma",
            target_language=TargetLanguage.GERMAN,
            native_language="English",
            age_group=AgeGroup.EARLY_PRIMARY,
        )

        # Start session
        greeting = manager.start()
        assert isinstance(greeting, str)
        assert len(greeting) > 0
        assert manager.is_running is True
        assert manager.session is not None

        # Check session properties
        session = manager.session
        assert session.config.student_name == "Emma"
        assert session.config.target_language == TargetLanguage.GERMAN
        assert session.turn_count == 0

    def test_lesson_focus_management(self) -> None:
        """Test lesson focus can be changed."""
        manager = ConversationManager(
            student_name="Max",
            target_language=TargetLanguage.SPANISH,
            age_group=AgeGroup.PRESCHOOL,
        )
        manager.start()

        # Get available lessons
        lessons = manager.get_available_lessons()
        assert isinstance(lessons, list)
        assert len(lessons) > 0

        # Change lesson focus
        new_focus = lessons[0]
        manager.set_lesson_focus(new_focus)
        assert manager.session.config.lesson_focus == new_focus

    def test_stop_during_session(self) -> None:
        """Test stopping during active session."""
        manager = ConversationManager(
            student_name="Test",
            target_language=TargetLanguage.ENGLISH,
        )
        manager.start()
        assert manager.is_running is True

        manager.stop()
        assert manager.is_running is False


class TestLanguageConfiguration:
    """Tests for language configuration across flow."""

    @pytest.mark.parametrize("language", [
        TargetLanguage.ENGLISH,
        TargetLanguage.GERMAN,
        TargetLanguage.SPANISH,
        TargetLanguage.JAPANESE,
        TargetLanguage.MANDARIN,
    ])
    def test_all_languages_supported(self, language: TargetLanguage) -> None:
        """Test manager can be created for all target languages."""
        manager = ConversationManager(
            student_name="Student",
            target_language=language,
        )
        manager.start()
        assert manager.target_language == language
        manager.stop()

    @pytest.mark.parametrize("age_group", [
        AgeGroup.PRESCHOOL,
        AgeGroup.EARLY_PRIMARY,
        AgeGroup.LATE_PRIMARY,
    ])
    def test_all_age_groups_supported(self, age_group: AgeGroup) -> None:
        """Test manager can be created for all age groups."""
        manager = ConversationManager(
            student_name="Student",
            target_language=TargetLanguage.SPANISH,
            age_group=age_group,
        )
        manager.start()
        assert manager.age_group == age_group

        # Verify lessons are appropriate for age
        lessons = manager.get_available_lessons()
        assert len(lessons) > 0
        manager.stop()
