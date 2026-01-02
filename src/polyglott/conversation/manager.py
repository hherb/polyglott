"""Conversation flow management.

This module provides high-level conversation management,
coordinating between the audio pipeline and session tracking.
"""

from typing import Callable, Optional

from polyglott.audio.pipeline import AudioPipeline, ConversationTurn, PipelineState
from polyglott.constants import AgeGroup, TargetLanguage
from polyglott.conversation.session import ConversationSession, create_session


# Callback types
TurnCallback = Callable[[ConversationTurn], None]
StateCallback = Callable[[PipelineState], None]


class ConversationManager:
    """High-level conversation manager.

    Coordinates the audio pipeline with session tracking
    to provide a complete tutoring experience.

    Example:
        >>> manager = ConversationManager(
        ...     student_name="Emma",
        ...     target_language=TargetLanguage.SPANISH
        ... )
        >>> manager.start()
        >>> manager.run_conversation_loop()
    """

    def __init__(
        self,
        student_name: str = "Student",
        target_language: TargetLanguage = TargetLanguage.SPANISH,
        native_language: str = "English",
        age_group: AgeGroup = AgeGroup.EARLY_PRIMARY,
        on_turn_complete: Optional[TurnCallback] = None,
        on_state_change: Optional[StateCallback] = None,
    ) -> None:
        """Initialize the conversation manager.

        Args:
            student_name: Name of the student.
            target_language: Language being learned.
            native_language: Student's native language.
            age_group: Student's age group.
            on_turn_complete: Callback after each turn.
            on_state_change: Callback for state changes.
        """
        self.student_name = student_name
        self.target_language = target_language
        self.native_language = native_language
        self.age_group = age_group
        self.on_turn_complete = on_turn_complete
        self.on_state_change = on_state_change

        self._session: Optional[ConversationSession] = None
        self._pipeline: Optional[AudioPipeline] = None
        self._is_running = False

    def start(self) -> str:
        """Start a new conversation session.

        Returns:
            Initial greeting from the tutor.
        """
        # Create session
        self._session = create_session(
            student_name=self.student_name,
            target_language=self.target_language,
            native_language=self.native_language,
            age_group=self.age_group,
        )

        # Create pipeline
        self._pipeline = AudioPipeline(
            target_language=self.target_language,
            native_language=self.native_language,
            age_group=self.age_group,
            on_state_change=self.on_state_change,
        )

        # Start session and get greeting
        greeting = self._pipeline.start_session()
        self._is_running = True

        return greeting

    def process_turn(self) -> ConversationTurn:
        """Process a single conversation turn.

        Returns:
            ConversationTurn with turn data.

        Raises:
            RuntimeError: If conversation not started.
        """
        if not self._is_running or self._pipeline is None:
            raise RuntimeError("Conversation not started. Call start() first.")

        turn = self._pipeline.process_turn()

        # Record in session
        if self._session and turn.user_text:
            self._session.record_turn(turn.user_text, turn.tutor_text)

        # Notify callback
        if self.on_turn_complete and turn.user_text:
            self.on_turn_complete(turn)

        return turn

    def speak_greeting(self) -> None:
        """Speak the initial greeting.

        Raises:
            RuntimeError: If conversation not started.
        """
        if not self._pipeline:
            raise RuntimeError("Conversation not started. Call start() first.")

        greeting = self._pipeline.start_session()
        self._pipeline.speak(greeting, language="en")

    def run_conversation_loop(
        self,
        max_turns: int = 100,
        exit_phrases: Optional[list[str]] = None,
    ) -> None:
        """Run the main conversation loop.

        Args:
            max_turns: Maximum number of turns before ending.
            exit_phrases: Phrases that end the conversation.
        """
        exit_phrases = exit_phrases or ["goodbye", "bye", "quit", "exit", "stop"]

        for _ in range(max_turns):
            if not self._is_running:
                break

            turn = self.process_turn()

            # Check for exit
            if turn.user_text:
                user_lower = turn.user_text.lower()
                if any(phrase in user_lower for phrase in exit_phrases):
                    self.end_session()
                    break

    def end_session(self, say_goodbye: bool = True) -> str:
        """End the current session.

        Args:
            say_goodbye: Whether to speak a goodbye message.

        Returns:
            Session summary.
        """
        self._is_running = False

        summary = ""
        if self._session:
            summary = self._session.get_summary()

        if say_goodbye and self._pipeline:
            self._pipeline.speak("Great job today! See you next time!", language="en")

        if self._pipeline:
            self._pipeline.stop()

        return summary

    def stop(self) -> None:
        """Immediately stop the conversation."""
        self._is_running = False
        if self._pipeline:
            self._pipeline.stop()

    @property
    def is_running(self) -> bool:
        """Check if conversation is running.

        Returns:
            True if conversation is active.
        """
        return self._is_running

    @property
    def session(self) -> Optional[ConversationSession]:
        """Get the current session.

        Returns:
            Current session or None.
        """
        return self._session

    def set_lesson_focus(self, focus: str) -> None:
        """Change the lesson focus.

        Args:
            focus: New lesson topic.
        """
        if self._pipeline:
            self._pipeline.set_lesson_focus(focus)
        if self._session:
            self._session.config.lesson_focus = focus

    def get_available_lessons(self) -> list[str]:
        """Get available lesson topics.

        Returns:
            List of lesson topics.
        """
        if self._pipeline:
            return self._pipeline.get_available_lessons()
        return []


def create_manager(
    student_name: str = "Student",
    target_language: TargetLanguage = TargetLanguage.SPANISH,
) -> ConversationManager:
    """Factory function to create a conversation manager.

    Args:
        student_name: Name of the student.
        target_language: Language to learn.

    Returns:
        Configured ConversationManager instance.
    """
    return ConversationManager(
        student_name=student_name,
        target_language=target_language,
    )
