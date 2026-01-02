"""Conversation session management.

This module provides session handling for tutoring conversations,
including progress tracking and session persistence.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import uuid4

from polyglott.constants import AgeGroup, TargetLanguage


@dataclass
class SessionConfig:
    """Configuration for a tutoring session.

    Attributes:
        student_name: Name of the student.
        native_language: Student's native language.
        target_language: Language being learned.
        age_group: Student's age group.
        lesson_focus: Current lesson topic.
    """

    student_name: str = "Student"
    native_language: str = "English"
    target_language: TargetLanguage = TargetLanguage.SPANISH
    age_group: AgeGroup = AgeGroup.EARLY_PRIMARY
    lesson_focus: str = "greetings and saying hello/goodbye"


@dataclass
class ConversationSession:
    """A tutoring conversation session.

    Tracks the conversation history and session metadata.

    Attributes:
        session_id: Unique session identifier.
        config: Session configuration.
        started_at: Session start timestamp.
        turn_count: Number of conversation turns.
        words_learned: Words introduced in this session.
    """

    session_id: str = field(default_factory=lambda: str(uuid4())[:8])
    config: SessionConfig = field(default_factory=SessionConfig)
    started_at: datetime = field(default_factory=datetime.now)
    turn_count: int = 0
    words_learned: list[str] = field(default_factory=list)

    def record_turn(self, user_text: str, tutor_text: str) -> None:
        """Record a conversation turn.

        Args:
            user_text: What the student said.
            tutor_text: Tutor's response.
        """
        self.turn_count += 1

    def add_word_learned(self, word: str) -> None:
        """Add a word to the learned vocabulary.

        Args:
            word: Word that was learned.
        """
        if word not in self.words_learned:
            self.words_learned.append(word)

    @property
    def duration_minutes(self) -> float:
        """Get session duration in minutes.

        Returns:
            Duration since session start.
        """
        delta = datetime.now() - self.started_at
        return delta.total_seconds() / 60

    def get_summary(self) -> str:
        """Get a summary of the session.

        Returns:
            Human-readable session summary.
        """
        return (
            f"Session {self.session_id}\n"
            f"Student: {self.config.student_name}\n"
            f"Learning: {self.config.target_language.value}\n"
            f"Duration: {self.duration_minutes:.1f} minutes\n"
            f"Turns: {self.turn_count}\n"
            f"Words learned: {len(self.words_learned)}"
        )


def create_session(
    student_name: str = "Student",
    target_language: TargetLanguage = TargetLanguage.SPANISH,
    native_language: str = "English",
    age_group: AgeGroup = AgeGroup.EARLY_PRIMARY,
) -> ConversationSession:
    """Create a new tutoring session.

    Args:
        student_name: Name of the student.
        target_language: Language to learn.
        native_language: Student's native language.
        age_group: Student's age group.

    Returns:
        New ConversationSession instance.
    """
    config = SessionConfig(
        student_name=student_name,
        native_language=native_language,
        target_language=target_language,
        age_group=age_group,
    )
    return ConversationSession(config=config)
