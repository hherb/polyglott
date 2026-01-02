"""Base protocol for Voice Activity Detection backends.

This module defines the interface that all VAD implementations must follow,
enabling easy swapping between different VAD engines (Silero, TEN VAD, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np


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


@runtime_checkable
class VADBackend(Protocol):
    """Protocol defining the interface for VAD backends.

    All VAD implementations must support these methods to be
    compatible with the audio pipeline.
    """

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        ...

    @property
    def threshold(self) -> float:
        """Speech probability threshold (0.0-1.0)."""
        ...

    @property
    def is_speaking(self) -> bool:
        """Check if currently in speaking state."""
        ...

    def process_chunk(self, audio_chunk: np.ndarray) -> VADResult:
        """Process an audio chunk and detect voice activity.

        Args:
            audio_chunk: Audio samples as numpy array.

        Returns:
            VADResult with speech probability, detection flag, and state.
        """
        ...

    def reset(self) -> None:
        """Reset the detector state for a new utterance."""
        ...

    @staticmethod
    def get_chunk_samples(sample_rate: int) -> int:
        """Get the number of samples per VAD chunk.

        Args:
            sample_rate: Audio sample rate in Hz.

        Returns:
            Number of samples per chunk.
        """
        ...


class BaseVADDetector(ABC):
    """Abstract base class for VAD implementations.

    Provides common state machine logic for speech detection
    that all backends can inherit from.
    """

    def __init__(
        self,
        sample_rate: int,
        threshold: float,
        speech_pad_frames: int,
        silence_pad_frames: int,
    ) -> None:
        """Initialize the base detector.

        Args:
            sample_rate: Audio sample rate in Hz.
            threshold: Speech probability threshold (0.0-1.0).
            speech_pad_frames: Consecutive speech frames to confirm speech start.
            silence_pad_frames: Consecutive silence frames to confirm speech end.
        """
        self._sample_rate = sample_rate
        self._threshold = threshold
        self.speech_pad_frames = speech_pad_frames
        self.silence_pad_frames = silence_pad_frames

        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._is_speaking = False

    @property
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        return self._sample_rate

    @property
    def threshold(self) -> float:
        """Speech probability threshold (0.0-1.0)."""
        return self._threshold

    @property
    def is_speaking(self) -> bool:
        """Check if currently in speaking state."""
        return self._is_speaking

    def reset(self) -> None:
        """Reset the detector state for a new utterance."""
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._is_speaking = False
        self._reset_model()

    @abstractmethod
    def _reset_model(self) -> None:
        """Reset backend-specific model state."""
        ...

    @abstractmethod
    def _get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """Get speech probability from the backend model.

        Args:
            audio_chunk: Audio samples as numpy array.

        Returns:
            Speech probability (0.0-1.0).
        """
        ...

    @abstractmethod
    def _get_expected_samples(self) -> int:
        """Get expected number of samples per chunk for this backend."""
        ...

    def process_chunk(self, audio_chunk: np.ndarray) -> VADResult:
        """Process an audio chunk and detect voice activity.

        Args:
            audio_chunk: Audio samples as numpy array.

        Returns:
            VADResult with speech probability, detection flag, and state.

        Raises:
            ValueError: If audio_chunk has wrong shape.
        """
        expected_samples = self._get_expected_samples()
        if len(audio_chunk) != expected_samples:
            raise ValueError(
                f"Expected {expected_samples} samples, got {len(audio_chunk)}"
            )

        # Get speech probability from backend
        speech_prob = self._get_speech_probability(audio_chunk)
        is_speech = speech_prob >= self._threshold

        # Determine state based on frame counts
        state = self._update_state(is_speech)

        return VADResult(
            speech_probability=speech_prob,
            is_speech=is_speech,
            state=state,
        )

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
