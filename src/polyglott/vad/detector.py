"""Voice Activity Detection using Silero VAD.

This module provides real-time voice activity detection for
identifying when a user is speaking during conversation.
"""

from typing import Optional

import numpy as np
import torch

from polyglott.constants import (
    AUDIO_SAMPLE_RATE,
    VAD_CHUNK_DURATION_MS,
    VAD_SILENCE_PAD_FRAMES,
    VAD_SPEECH_PAD_FRAMES,
    VAD_SPEECH_THRESHOLD,
)
from polyglott.vad.base import BaseVADDetector, SpeechState, VADResult


class VoiceActivityDetector(BaseVADDetector):
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

    # Silero VAD v6+ requires minimum 32ms chunks
    CHUNK_DURATION_MS = 32

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

        super().__init__(
            sample_rate=sample_rate,
            threshold=threshold,
            speech_pad_frames=speech_pad_frames,
            silence_pad_frames=silence_pad_frames,
        )

        self._model: Optional[torch.jit.ScriptModule] = None

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

    def _reset_model(self) -> None:
        """Reset Silero model state."""
        if self._model is not None:
            self._model.reset_states()

    def _get_expected_samples(self) -> int:
        """Get expected samples for Silero VAD (32ms chunks)."""
        return int(self._sample_rate * self.CHUNK_DURATION_MS / 1000)

    def _get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """Get speech probability from Silero VAD.

        Args:
            audio_chunk: Audio samples as numpy array.

        Returns:
            Speech probability (0.0-1.0).
        """
        audio_tensor = self._prepare_audio(audio_chunk)
        return self.model(audio_tensor, self._sample_rate).item()

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

    @staticmethod
    def get_chunk_samples(sample_rate: int = AUDIO_SAMPLE_RATE) -> int:
        """Get the number of samples per VAD chunk.

        Args:
            sample_rate: Audio sample rate in Hz.

        Returns:
            Number of samples per 32ms chunk.
        """
        return int(sample_rate * VAD_CHUNK_DURATION_MS / 1000)


# Alias for backward compatibility
SileroVadDetector = VoiceActivityDetector


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
