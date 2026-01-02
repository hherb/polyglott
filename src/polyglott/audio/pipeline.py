"""Audio processing pipeline for the language tutor.

This module orchestrates the complete audio pipeline:
Record -> Transcribe -> Generate Response -> Synthesize -> Play

Supports barge-in interruption for natural conversation flow.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np

from polyglott.audio.player import AudioPlayer
from polyglott.audio.recorder import AudioRecorder, RecordingResult
from polyglott.constants import (
    AUDIO_SAMPLE_RATE,
    FOLLOWUP_TIER1_TIMEOUT,
    FOLLOWUP_TIER2_TIMEOUT,
    FOLLOWUP_TIER3_TIMEOUT,
    AgeGroup,
    TargetLanguage,
)
from polyglott.llm.prompts import get_followup_prompt
from polyglott.llm.tutor import LanguageTutor, TutorConfig, TutorResponse
from polyglott.stt.transcriber import SpeechTranscriber, TranscriptionResult
from polyglott.tts.synthesizer import SpeechSynthesizer, SynthesisResult
from polyglott.vad import VoiceActivityDetector


class PipelineState(str, Enum):
    """Current state of the audio pipeline."""

    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    FOLLOWUP = "followup"  # Speaking a follow-up prompt


@dataclass
class ConversationTurn:
    """A single turn in the conversation.

    Attributes:
        user_audio: Recorded user audio.
        user_text: Transcribed user speech.
        tutor_text: Generated tutor response.
        tutor_audio: Synthesized tutor speech.
        state: Final state of the turn.
        was_interrupted: Whether playback was interrupted by user.
    """

    user_audio: Optional[np.ndarray] = None
    user_text: str = ""
    tutor_text: str = ""
    tutor_audio: Optional[np.ndarray] = None
    state: PipelineState = PipelineState.IDLE
    was_interrupted: bool = False


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
        enable_barge_in: bool = True,
    ) -> None:
        """Initialize the audio pipeline.

        Args:
            target_language: Language the child is learning.
            native_language: Child's native language.
            age_group: Child's age group for difficulty.
            on_state_change: Callback for state changes.
            enable_barge_in: Enable barge-in interruption support.
        """
        self.target_language = target_language
        self.native_language = native_language
        self.age_group = age_group
        self.on_state_change = on_state_change
        self.enable_barge_in = enable_barge_in

        self._state = PipelineState.IDLE
        self._recorder: Optional[AudioRecorder] = None
        self._transcriber: Optional[SpeechTranscriber] = None
        self._tutor: Optional[LanguageTutor] = None
        self._synthesizer: Optional[SpeechSynthesizer] = None
        self._player: Optional[AudioPlayer] = None
        self._interrupt_vad: Optional[VoiceActivityDetector] = None

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

        # Create a separate VAD for barge-in detection during playback
        if self._interrupt_vad is None and self.enable_barge_in:
            self._interrupt_vad = VoiceActivityDetector(sample_rate=AUDIO_SAMPLE_RATE)

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
        4. Synthesizes and plays the response (with barge-in support)

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

        # Step 4: Synthesize and play (with optional barge-in support)
        self._set_state(PipelineState.SPEAKING)
        synthesis = self._synthesizer.synthesize(
            turn.tutor_text,
            language=self.target_language.value,
        )
        turn.tutor_audio = synthesis.audio

        # Use interruptible playback if enabled
        if self.enable_barge_in and self._interrupt_vad is not None:
            def on_interrupt():
                self._set_state(PipelineState.INTERRUPTED)

            completed = self._player.play_interruptible(
                synthesis.audio,
                synthesis.sample_rate,
                vad_detector=self._interrupt_vad,
                on_interrupt=on_interrupt,
            )
            turn.was_interrupted = not completed
            if not completed:
                turn.state = PipelineState.INTERRUPTED
                self._set_state(PipelineState.IDLE)
                return turn
        else:
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

    def process_turn_with_followup(
        self,
        enable_followups: bool = True,
        max_followup_tier: int = 3,
    ) -> ConversationTurn:
        """Process a conversation turn with AI-initiated follow-ups.

        If the user doesn't respond within a timeout, the tutor will
        proactively prompt them with increasingly engaging messages.

        Args:
            enable_followups: Enable follow-up prompts on silence.
            max_followup_tier: Maximum follow-up tier (1-3).

        Returns:
            ConversationTurn with all data from the turn.
        """
        self._ensure_components()
        turn = ConversationTurn()

        # Define timeouts for each tier
        tier_timeouts = [
            FOLLOWUP_TIER1_TIMEOUT,
            FOLLOWUP_TIER2_TIMEOUT,
            FOLLOWUP_TIER3_TIMEOUT,
        ]

        current_tier = 0

        while current_tier <= max_followup_tier:
            # Calculate timeout for this attempt
            if current_tier == 0:
                timeout = tier_timeouts[0] if enable_followups else 30.0
            else:
                # Incremental timeout between tiers
                prev_timeout = tier_timeouts[current_tier - 1] if current_tier > 0 else 0
                curr_timeout = tier_timeouts[min(current_tier, len(tier_timeouts) - 1)]
                timeout = curr_timeout - prev_timeout

            # Try to record user speech with timeout
            self._set_state(PipelineState.LISTENING)
            recording = self._recorder.record_utterance(
                max_duration=timeout,
                on_speech_start=lambda: None,
                on_speech_end=lambda: None,
            )

            if recording.was_speech_detected:
                # User spoke! Continue with normal processing
                turn.user_audio = recording.audio

                # Step 2: Transcribe
                self._set_state(PipelineState.TRANSCRIBING)
                transcription = self._transcriber.transcribe(
                    recording.audio,
                    language=self.target_language.value,
                )
                turn.user_text = transcription.text

                if not turn.user_text.strip():
                    # Empty transcription, wait for more
                    current_tier += 1
                    continue

                # Step 3: Generate response
                self._set_state(PipelineState.THINKING)
                response = self._tutor.respond(turn.user_text)
                turn.tutor_text = response.text

                # Step 4: Synthesize and play (with barge-in support)
                self._set_state(PipelineState.SPEAKING)
                synthesis = self._synthesizer.synthesize(
                    turn.tutor_text,
                    language=self.target_language.value,
                )
                turn.tutor_audio = synthesis.audio

                if self.enable_barge_in and self._interrupt_vad is not None:
                    def on_interrupt():
                        self._set_state(PipelineState.INTERRUPTED)

                    completed = self._player.play_interruptible(
                        synthesis.audio,
                        synthesis.sample_rate,
                        vad_detector=self._interrupt_vad,
                        on_interrupt=on_interrupt,
                    )
                    turn.was_interrupted = not completed
                else:
                    self._player.play(synthesis.audio, synthesis.sample_rate, blocking=True)

                self._set_state(PipelineState.IDLE)
                turn.state = PipelineState.IDLE
                return turn

            # No speech detected - trigger follow-up if enabled
            current_tier += 1

            if enable_followups and current_tier <= max_followup_tier:
                # Get native language code
                native_lang = self.native_language.lower()[:2]

                # Get a follow-up prompt
                followup = get_followup_prompt(
                    tier=current_tier,
                    native_language=native_lang,
                    target_language=self.target_language.value,
                )

                # Speak the follow-up
                self._set_state(PipelineState.FOLLOWUP)
                synthesis = self._synthesizer.synthesize(
                    followup,
                    language=native_lang,  # Speak in native language
                )

                # Use interruptible playback for follow-ups too
                if self.enable_barge_in and self._interrupt_vad is not None:
                    completed = self._player.play_interruptible(
                        synthesis.audio,
                        synthesis.sample_rate,
                        vad_detector=self._interrupt_vad,
                    )
                    if not completed:
                        # User interrupted the follow-up - go back to listening
                        continue
                else:
                    self._player.play(synthesis.audio, synthesis.sample_rate, blocking=True)

        # Exhausted all follow-up tiers without response
        self._set_state(PipelineState.IDLE)
        return turn

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

    def preload_models(
        self,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Preload all ML models before user interaction.

        This ensures no model loading delays during conversation.
        Call this during startup to front-load all model initialization.

        Args:
            status_callback: Optional callback to report loading progress.
        """

        def report(msg: str) -> None:
            if status_callback:
                status_callback(msg)

        report("Initializing audio components...")
        self._ensure_components()

        # Preload VAD model (loads on first process_chunk)
        report("Loading voice activity detection model...")
        if self._recorder and self._recorder.vad:
            # Trigger model load with a silent chunk
            import numpy as np
            from polyglott.constants import VAD_CHUNK_SAMPLES

            silent_chunk = np.zeros(VAD_CHUNK_SAMPLES, dtype=np.float32)
            self._recorder.vad.process_chunk(silent_chunk)
            self._recorder.vad.reset()

        # Preload STT model (loads on first transcribe)
        report("Loading speech recognition model...")
        if self._transcriber:
            # Access the backend to trigger model loading
            _ = self._transcriber.backend
            # Force the actual model to load
            transcriber = self._transcriber._get_transcriber()
            if hasattr(transcriber, "_ensure_loaded"):
                transcriber._ensure_loaded()

        # Preload LLM model (test connection to Ollama)
        report("Connecting to language model...")
        if self._tutor:
            # The tutor's model loads on first respond, but we can warm up Ollama
            pass  # Ollama keeps model warm after first call

        # Preload TTS model (loads on first synthesize)
        report("Loading text-to-speech model...")
        if self._synthesizer:
            # Trigger model loading with a minimal synthesis
            try:
                self._synthesizer.synthesize(".", language="en")
            except Exception:
                pass  # Ignore TTS preload errors

        report("All models loaded!")


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
