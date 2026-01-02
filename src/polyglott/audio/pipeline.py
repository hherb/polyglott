"""Audio processing pipeline for the language tutor.

This module orchestrates the complete audio pipeline:
Record -> Transcribe -> Generate Response -> Synthesize -> Play
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np

from polyglott.audio.player import AudioPlayer
from polyglott.audio.recorder import AudioRecorder, RecordingResult
from polyglott.constants import (
    AUDIO_SAMPLE_RATE,
    AgeGroup,
    TargetLanguage,
)
from polyglott.llm.tutor import LanguageTutor, TutorConfig, TutorResponse
from polyglott.stt.transcriber import SpeechTranscriber, TranscriptionResult
from polyglott.tts.synthesizer import SpeechSynthesizer, SynthesisResult


class PipelineState(str, Enum):
    """Current state of the audio pipeline."""

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass
class ConversationTurn:
    """A single turn in the conversation.

    Attributes:
        user_audio: Recorded user audio.
        user_text: Transcribed user speech.
        tutor_text: Generated tutor response.
        tutor_audio: Synthesized tutor speech.
        state: Final state of the turn.
    """

    user_audio: Optional[np.ndarray] = None
    user_text: str = ""
    tutor_text: str = ""
    tutor_audio: Optional[np.ndarray] = None
    state: PipelineState = PipelineState.IDLE


# Type alias for state change callbacks
StateCallback = Callable[[PipelineState], None]


class AudioPipeline:
    """Complete audio processing pipeline for conversation.

    This class orchestrates all components to enable
    voice-based conversation with the language tutor.

    Example:
        >>> pipeline = AudioPipeline(
        ...     target_language=TargetLanguage.SPANISH,
        ...     native_language="English"
        ... )
        >>> pipeline.start_session()
        >>> while True:
        ...     turn = pipeline.process_turn()
        ...     if not turn.user_text:
        ...         break
    """

    def __init__(
        self,
        target_language: TargetLanguage = TargetLanguage.SPANISH,
        native_language: str = "English",
        age_group: AgeGroup = AgeGroup.EARLY_PRIMARY,
        on_state_change: Optional[StateCallback] = None,
    ) -> None:
        """Initialize the audio pipeline.

        Args:
            target_language: Language the child is learning.
            native_language: Child's native language.
            age_group: Child's age group for difficulty.
            on_state_change: Callback for state changes.
        """
        self.target_language = target_language
        self.native_language = native_language
        self.age_group = age_group
        self.on_state_change = on_state_change

        self._state = PipelineState.IDLE
        self._recorder: Optional[AudioRecorder] = None
        self._transcriber: Optional[SpeechTranscriber] = None
        self._tutor: Optional[LanguageTutor] = None
        self._synthesizer: Optional[SpeechSynthesizer] = None
        self._player: Optional[AudioPlayer] = None

    def _set_state(self, state: PipelineState) -> None:
        """Update pipeline state and notify callback.

        Args:
            state: New pipeline state.
        """
        self._state = state
        if self.on_state_change:
            self.on_state_change(state)

    @property
    def state(self) -> PipelineState:
        """Get current pipeline state.

        Returns:
            Current state.
        """
        return self._state

    def _ensure_components(self) -> None:
        """Ensure all pipeline components are initialized."""
        if self._recorder is None:
            self._recorder = AudioRecorder()

        if self._transcriber is None:
            self._transcriber = SpeechTranscriber()

        if self._tutor is None:
            config = TutorConfig(
                target_language=self.target_language,
                native_language=self.native_language,
                age_group=self.age_group,
            )
            self._tutor = LanguageTutor(config)

        if self._synthesizer is None:
            self._synthesizer = SpeechSynthesizer()

        if self._player is None:
            self._player = AudioPlayer()

    def start_session(self) -> str:
        """Start a new tutoring session.

        Returns:
            Initial greeting from the tutor.
        """
        self._ensure_components()
        greeting = self._tutor.reset_conversation()
        return greeting

    def process_turn(self) -> ConversationTurn:
        """Process a single conversation turn.

        This method:
        1. Records user speech
        2. Transcribes the audio
        3. Generates a tutor response
        4. Synthesizes and plays the response

        Returns:
            ConversationTurn with all data from the turn.
        """
        self._ensure_components()
        turn = ConversationTurn()

        # Step 1: Record user speech
        self._set_state(PipelineState.LISTENING)
        recording = self._recorder.record_utterance(
            on_speech_start=lambda: None,
            on_speech_end=lambda: None,
        )

        if not recording.was_speech_detected:
            self._set_state(PipelineState.IDLE)
            return turn

        turn.user_audio = recording.audio

        # Step 2: Transcribe
        self._set_state(PipelineState.TRANSCRIBING)
        transcription = self._transcriber.transcribe(
            recording.audio,
            language=self.target_language.value,
        )
        turn.user_text = transcription.text

        if not turn.user_text.strip():
            self._set_state(PipelineState.IDLE)
            return turn

        # Step 3: Generate response
        self._set_state(PipelineState.THINKING)
        response = self._tutor.respond(turn.user_text)
        turn.tutor_text = response.text

        # Step 4: Synthesize and play
        self._set_state(PipelineState.SPEAKING)
        synthesis = self._synthesizer.synthesize(
            turn.tutor_text,
            language=self.target_language.value,
        )
        turn.tutor_audio = synthesis.audio

        self._player.play(synthesis.audio, synthesis.sample_rate, blocking=True)

        self._set_state(PipelineState.IDLE)
        turn.state = PipelineState.IDLE

        return turn

    def speak(self, text: str, language: Optional[str] = None) -> None:
        """Synthesize and speak text.

        Args:
            text: Text to speak.
            language: Language code, or None for target language.
        """
        self._ensure_components()
        lang = language or self.target_language.value

        self._set_state(PipelineState.SPEAKING)
        synthesis = self._synthesizer.synthesize(text, language=lang)
        self._player.play(synthesis.audio, synthesis.sample_rate, blocking=True)
        self._set_state(PipelineState.IDLE)

    def stop(self) -> None:
        """Stop all pipeline activity."""
        if self._recorder:
            self._recorder.stop()
        if self._player:
            self._player.stop()
        self._set_state(PipelineState.IDLE)

    def set_lesson_focus(self, focus: str) -> None:
        """Change the current lesson focus.

        Args:
            focus: New lesson topic.
        """
        if self._tutor:
            self._tutor.set_lesson_focus(focus)

    def get_available_lessons(self) -> list[str]:
        """Get available lesson topics.

        Returns:
            List of lesson topics.
        """
        self._ensure_components()
        return self._tutor.get_available_lessons()


def create_pipeline(
    target_language: TargetLanguage = TargetLanguage.SPANISH,
    native_language: str = "English",
    age_group: AgeGroup = AgeGroup.EARLY_PRIMARY,
) -> AudioPipeline:
    """Factory function to create a configured pipeline.

    Args:
        target_language: Language to learn.
        native_language: Child's native language.
        age_group: Child's age group.

    Returns:
        Configured AudioPipeline instance.
    """
    return AudioPipeline(
        target_language=target_language,
        native_language=native_language,
        age_group=age_group,
    )
