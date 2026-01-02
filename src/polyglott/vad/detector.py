"""Voice Activity Detection using Silero VAD.

This module provides real-time voice activity detection for
identifying when a user is speaking during conversation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import torch

from polyglott.constants import (
    AUDIO_SAMPLE_RATE,
    VAD_CHUNK_DURATION_MS,
    VAD_CHUNK_SAMPLES,
    VAD_SILENCE_PAD_FRAMES,
    VAD_SPEECH_PAD_FRAMES,
    VAD_SPEECH_THRESHOLD,
)


class SpeechState(str, Enum):
    """Current state of speech detection."""

    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEAKING = "speaking"
    SPEECH_END = "speech_end"


@dataclass
class VADResult:
    """Result from processing an audio chunk.

    Attributes:
        speech_probability: Probability of speech in the chunk (0.0-1.0).
        is_speech: Whether the chunk contains speech above threshold.
        state: Current speech state in the conversation.
    """

    speech_probability: float
    is_speech: bool
    state: SpeechState


class VoiceActivityDetector:
    """Real-time voice activity detector using Silero VAD.

    This class provides streaming voice activity detection for
    identifying speech segments in real-time audio input.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        threshold: Speech probability threshold for detection.
        model: Loaded Silero VAD model.

    Example:
        >>> detector = VoiceActivityDetector()
        >>> for chunk in audio_stream:
        ...     result = detector.process_chunk(chunk)
        ...     if result.state == SpeechState.SPEECH_END:
        ...         # Process complete utterance
        ...         pass
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        threshold: float = VAD_SPEECH_THRESHOLD,
        speech_pad_frames: int = VAD_SPEECH_PAD_FRAMES,
        silence_pad_frames: int = VAD_SILENCE_PAD_FRAMES,
    ) -> None:
        """Initialize the voice activity detector.

        Args:
            sample_rate: Audio sample rate in Hz (8000 or 16000).
            threshold: Speech probability threshold (0.0-1.0).
            speech_pad_frames: Consecutive speech frames to confirm speech start.
            silence_pad_frames: Consecutive silence frames to confirm speech end.

        Raises:
            ValueError: If sample_rate is not 8000 or 16000.
        """
        if sample_rate not in (8000, 16000):
            raise ValueError(f"Sample rate must be 8000 or 16000, got {sample_rate}")

        self.sample_rate = sample_rate
        self.threshold = threshold
        self.speech_pad_frames = speech_pad_frames
        self.silence_pad_frames = silence_pad_frames

        self._model: Optional[torch.jit.ScriptModule] = None
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._is_speaking = False

    @property
    def model(self) -> torch.jit.ScriptModule:
        """Lazy-load the Silero VAD model.

        Returns:
            Loaded Silero VAD model.
        """
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> torch.jit.ScriptModule:
        """Load the Silero VAD model.

        Returns:
            Loaded Silero VAD model.
        """
        # Set single thread for consistent performance
        torch.set_num_threads(1)

        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        return model

    def reset(self) -> None:
        """Reset the detector state for a new utterance."""
        if self._model is not None:
            self._model.reset_states()
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._is_speaking = False

    def process_chunk(self, audio_chunk: np.ndarray) -> VADResult:
        """Process an audio chunk and detect voice activity.

        Args:
            audio_chunk: Audio samples as numpy array. Should be
                VAD_CHUNK_SAMPLES long (30ms at 16kHz = 480 samples).

        Returns:
            VADResult with speech probability, detection flag, and state.

        Raises:
            ValueError: If audio_chunk has wrong shape.
        """
        expected_samples = int(self.sample_rate * VAD_CHUNK_DURATION_MS / 1000)
        if len(audio_chunk) != expected_samples:
            raise ValueError(
                f"Expected {expected_samples} samples, got {len(audio_chunk)}"
            )

        # Convert to tensor
        audio_tensor = self._prepare_audio(audio_chunk)

        # Get speech probability
        speech_prob = self.model(audio_tensor, self.sample_rate).item()
        is_speech = speech_prob >= self.threshold

        # Determine state based on frame counts
        state = self._update_state(is_speech)

        return VADResult(
            speech_probability=speech_prob,
            is_speech=is_speech,
            state=state,
        )

    def _prepare_audio(self, audio_chunk: np.ndarray) -> torch.Tensor:
        """Prepare audio chunk for the model.

        Args:
            audio_chunk: Raw audio samples.

        Returns:
            Normalized torch tensor.
        """
        # Ensure float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Normalize to [-1, 1] if needed
        max_val = np.abs(audio_chunk).max()
        if max_val > 1.0:
            audio_chunk = audio_chunk / max_val

        return torch.from_numpy(audio_chunk)

    def _update_state(self, is_speech: bool) -> SpeechState:
        """Update internal state and return current speech state.

        Args:
            is_speech: Whether current chunk contains speech.

        Returns:
            Current speech state.
        """
        if is_speech:
            self._speech_frame_count += 1
            self._silence_frame_count = 0
        else:
            self._silence_frame_count += 1
            self._speech_frame_count = 0

        # Determine state transitions
        if not self._is_speaking:
            if self._speech_frame_count >= self.speech_pad_frames:
                self._is_speaking = True
                return SpeechState.SPEECH_START
            return SpeechState.SILENCE
        else:
            if self._silence_frame_count >= self.silence_pad_frames:
                self._is_speaking = False
                return SpeechState.SPEECH_END
            return SpeechState.SPEAKING

    @property
    def is_speaking(self) -> bool:
        """Check if currently in speaking state.

        Returns:
            True if user is currently speaking.
        """
        return self._is_speaking

    @staticmethod
    def get_chunk_samples(sample_rate: int = AUDIO_SAMPLE_RATE) -> int:
        """Get the number of samples per VAD chunk.

        Args:
            sample_rate: Audio sample rate in Hz.

        Returns:
            Number of samples per 30ms chunk.
        """
        return int(sample_rate * VAD_CHUNK_DURATION_MS / 1000)


def create_vad_detector(
    threshold: float = VAD_SPEECH_THRESHOLD,
) -> VoiceActivityDetector:
    """Factory function to create a configured VAD detector.

    Args:
        threshold: Speech probability threshold (0.0-1.0).

    Returns:
        Configured VoiceActivityDetector instance.
    """
    return VoiceActivityDetector(threshold=threshold)
