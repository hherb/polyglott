"""Conversation session management.

This module provides session handling for tutoring conversations,
including progress tracking and session persistence with full
conversation history.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from polyglott.constants import AgeGroup, TargetLanguage


@dataclass
class SessionMessage:
    """A single message in the conversation history.

    Attributes:
        role: Message role (user or tutor).
        content: Message text content.
        timestamp: When the message was sent.
    """

    role: str  # "user" or "tutor"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMessage":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )


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
    Supports saving and loading for session continuity.

    Attributes:
        session_id: Unique session identifier.
        config: Session configuration.
        started_at: Session start timestamp.
        turn_count: Number of conversation turns.
        words_learned: Words introduced in this session.
        messages: Full conversation history.
        title: Optional session title for display.
    """

    session_id: str = field(default_factory=lambda: str(uuid4())[:8])
    config: SessionConfig = field(default_factory=SessionConfig)
    started_at: datetime = field(default_factory=datetime.now)
    turn_count: int = 0
    words_learned: list[str] = field(default_factory=list)
    messages: list[SessionMessage] = field(default_factory=list)
    title: Optional[str] = None

    def record_turn(self, user_text: str, tutor_text: str) -> None:
        """Record a conversation turn.

        Args:
            user_text: What the student said.
            tutor_text: Tutor's response.
        """
        self.turn_count += 1

        # Add messages to history
        self.messages.append(SessionMessage(role="user", content=user_text))
        self.messages.append(SessionMessage(role="tutor", content=tutor_text))

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

    def to_dict(self) -> dict:
        """Convert session to dictionary for serialization.

        Returns:
            Dictionary representation of session.
        """
        return {
            "session_id": self.session_id,
            "title": self.title or f"Session {self.session_id}",
            "config": {
                "student_name": self.config.student_name,
                "native_language": self.config.native_language,
                "target_language": self.config.target_language.value,
                "age_group": self.config.age_group.value,
                "lesson_focus": self.config.lesson_focus,
            },
            "started_at": self.started_at.isoformat(),
            "turn_count": self.turn_count,
            "words_learned": self.words_learned,
            "messages": [msg.to_dict() for msg in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationSession":
        """Create session from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            ConversationSession instance.
        """
        config_data = data.get("config", {})

        # Parse target language
        target_lang_str = config_data.get("target_language", "es")
        target_language = TargetLanguage.SPANISH
        for lang in TargetLanguage:
            if lang.value == target_lang_str:
                target_language = lang
                break

        # Parse age group
        age_group_str = config_data.get("age_group", "early_primary")
        age_group = AgeGroup.EARLY_PRIMARY
        for group in AgeGroup:
            if group.value == age_group_str:
                age_group = group
                break

        config = SessionConfig(
            student_name=config_data.get("student_name", "Student"),
            native_language=config_data.get("native_language", "English"),
            target_language=target_language,
            age_group=age_group,
            lesson_focus=config_data.get("lesson_focus", ""),
        )

        # Parse messages
        messages = [
            SessionMessage.from_dict(msg)
            for msg in data.get("messages", [])
        ]

        return cls(
            session_id=data.get("session_id", str(uuid4())[:8]),
            config=config,
            started_at=datetime.fromisoformat(data.get("started_at", datetime.now().isoformat())),
            turn_count=data.get("turn_count", 0),
            words_learned=data.get("words_learned", []),
            messages=messages,
            title=data.get("title"),
        )

    def save(self, path: Optional[Path] = None) -> Path:
        """Save session to disk.

        Args:
            path: Path to save to, or None for default location.

        Returns:
            Path where session was saved.
        """
        if path is None:
            sessions_dir = get_sessions_dir()
            path = sessions_dir / f"{self.session_id}.json"

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return path

    @classmethod
    def load(cls, path: Path) -> Optional["ConversationSession"]:
        """Load session from disk.

        Args:
            path: Path to load from.

        Returns:
            ConversationSession if loaded successfully, None otherwise.
        """
        try:
            with open(path) as f:
                data = json.load(f)
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError, FileNotFoundError):
            return None

    def get_last_messages(self, count: int = 10) -> list[SessionMessage]:
        """Get the most recent messages.

        Args:
            count: Number of messages to return.

        Returns:
            List of recent messages.
        """
        return self.messages[-count:] if self.messages else []


def get_sessions_dir() -> Path:
    """Get the directory for session storage.

    Returns:
        Path to ~/.polyglott/sessions/ directory.
    """
    sessions_dir = Path.home() / ".polyglott" / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir


def list_sessions(student_name: Optional[str] = None) -> list[dict]:
    """List all saved sessions.

    Args:
        student_name: Optional filter by student name.

    Returns:
        List of session metadata dictionaries.
    """
    sessions_dir = get_sessions_dir()
    sessions = []

    for session_file in sessions_dir.glob("*.json"):
        try:
            with open(session_file) as f:
                data = json.load(f)
            # Apply filter if specified
            if student_name is not None:
                config = data.get("config", {})
                if config.get("student_name") != student_name:
                    continue
            sessions.append({
                "session_id": data.get("session_id"),
                "title": data.get("title"),
                "student_name": data.get("config", {}).get("student_name"),
                "target_language": data.get("config", {}).get("target_language"),
                "started_at": data.get("started_at"),
                "turn_count": data.get("turn_count", 0),
                "path": session_file,
            })
        except (json.JSONDecodeError, KeyError):
            continue

    # Sort by start time, most recent first
    sessions.sort(key=lambda s: s.get("started_at", ""), reverse=True)
    return sessions


def load_session(session_id: str) -> Optional[ConversationSession]:
    """Load a session by ID.

    Args:
        session_id: Session ID to load.

    Returns:
        ConversationSession if found, None otherwise.
    """
    sessions_dir = get_sessions_dir()
    path = sessions_dir / f"{session_id}.json"

    if path.exists():
        return ConversationSession.load(path)
    return None


def delete_session(session_id: str) -> bool:
    """Delete a session by ID.

    Args:
        session_id: Session ID to delete.

    Returns:
        True if deleted, False if not found.
    """
    sessions_dir = get_sessions_dir()
    path = sessions_dir / f"{session_id}.json"

    if path.exists():
        path.unlink()
        return True
    return False


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
