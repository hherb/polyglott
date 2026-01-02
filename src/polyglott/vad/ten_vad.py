"""Voice Activity Detection using TEN VAD.

This module provides real-time voice activity detection using
TEN VAD (TEN-framework/ten-vad), a high-performance, low-latency
VAD engine with ONNX Runtime backend.

TEN VAD offers:
- Faster speech-to-silence transitions than Silero VAD
- Smaller library footprint (~300KB vs ~2.2MB)
- No PyTorch dependency (uses ONNX Runtime)

Installation:
    TEN VAD requires building from source. See:
    https://github.com/TEN-framework/ten-vad

    # Clone and build for macOS/Linux:
    git clone https://github.com/TEN-framework/ten-vad.git
    cd ten-vad/examples_onnx/python
    ./build-and-deploy-macos.sh  # or build-and-deploy-linux.sh
"""

from typing import Optional

import numpy as np

from polyglott.constants import (
    AUDIO_SAMPLE_RATE,
    VAD_SILENCE_PAD_FRAMES,
    VAD_SPEECH_PAD_FRAMES,
    VAD_SPEECH_THRESHOLD,
)
from polyglott.vad.base import BaseVADDetector

# Try to import TEN VAD - it requires building from source
try:
    import ten_vad_python

    TEN_VAD_AVAILABLE = True
except ImportError:
    TEN_VAD_AVAILABLE = False
    ten_vad_python = None


class TenVadNotInstalledError(ImportError):
    """Raised when TEN VAD is not installed."""

    def __init__(self) -> None:
        super().__init__(
            "TEN VAD is not installed. To use TEN VAD:\n"
            "1. Clone: git clone https://github.com/TEN-framework/ten-vad.git\n"
            "2. Build: cd ten-vad/examples_onnx/python && ./build-and-deploy-macos.sh\n"
            "3. Install: pip install -e .\n"
            "See: https://github.com/TEN-framework/ten-vad"
        )


class TenVadDetector(BaseVADDetector):
    """Real-time voice activity detector using TEN VAD.

    TEN VAD provides lower latency speech detection compared to Silero VAD,
    with faster speech-to-silence transitions and a smaller memory footprint.

    Attributes:
        sample_rate: Audio sample rate in Hz (must be 16000).
        threshold: Speech probability threshold for detection.
        hop_size: Number of samples per frame (160 or 256).

    Example:
        >>> from polyglott.vad.ten_vad import TenVadDetector, TEN_VAD_AVAILABLE
        >>> if TEN_VAD_AVAILABLE:
        ...     detector = TenVadDetector()
        ...     for chunk in audio_stream:
        ...         result = detector.process_chunk(chunk)
        ...         if result.state == SpeechState.SPEECH_END:
        ...             # Process complete utterance
        ...             pass
    """

    # TEN VAD supported hop sizes (samples at 16kHz)
    # 160 samples = 10ms, 256 samples = 16ms
    SUPPORTED_HOP_SIZES = (160, 256)
    DEFAULT_HOP_SIZE = 256  # 16ms - good balance of latency and stability

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        threshold: float = VAD_SPEECH_THRESHOLD,
        speech_pad_frames: int = VAD_SPEECH_PAD_FRAMES,
        silence_pad_frames: int = VAD_SILENCE_PAD_FRAMES,
        hop_size: int = DEFAULT_HOP_SIZE,
    ) -> None:
        """Initialize the TEN VAD detector.

        Args:
            sample_rate: Audio sample rate in Hz (must be 16000).
            threshold: Speech probability threshold (0.0-1.0).
            speech_pad_frames: Consecutive speech frames to confirm speech start.
            silence_pad_frames: Consecutive silence frames to confirm speech end.
            hop_size: Number of samples per frame (160 or 256).

        Raises:
            TenVadNotInstalledError: If TEN VAD is not installed.
            ValueError: If sample_rate is not 16000 or hop_size is invalid.
        """
        if not TEN_VAD_AVAILABLE:
            raise TenVadNotInstalledError()

        if sample_rate != 16000:
            raise ValueError(
                f"TEN VAD only supports 16kHz sample rate, got {sample_rate}"
            )

        if hop_size not in self.SUPPORTED_HOP_SIZES:
            raise ValueError(
                f"hop_size must be one of {self.SUPPORTED_HOP_SIZES}, got {hop_size}"
            )

        super().__init__(
            sample_rate=sample_rate,
            threshold=threshold,
            speech_pad_frames=speech_pad_frames,
            silence_pad_frames=silence_pad_frames,
        )

        self.hop_size = hop_size
        self._model: Optional["ten_vad_python.VAD"] = None

    @property
    def model(self) -> "ten_vad_python.VAD":
        """Lazy-load the TEN VAD model.

        Returns:
            Loaded TEN VAD model.
        """
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> "ten_vad_python.VAD":
        """Load the TEN VAD model.

        Returns:
            Loaded TEN VAD model.
        """
        return ten_vad_python.VAD(
            hop_size=self.hop_size,
            threshold=self._threshold,
        )

    def _reset_model(self) -> None:
        """Reset TEN VAD model state.

        Note: TEN VAD may not have explicit state reset - recreate if needed.
        """
        # TEN VAD appears to be stateless per-frame, but we recreate
        # the model to ensure clean state if needed
        if self._model is not None:
            self._model = None  # Will be recreated on next use

    def _get_expected_samples(self) -> int:
        """Get expected samples for TEN VAD."""
        return self.hop_size

    def _get_speech_probability(self, audio_chunk: np.ndarray) -> float:
        """Get speech probability from TEN VAD.

        Args:
            audio_chunk: Audio samples as numpy array.

        Returns:
            Speech probability (0.0-1.0).
        """
        audio_chunk = self._prepare_audio(audio_chunk)
        prob, _ = self.model.process(audio_chunk)
        return float(prob)

    def _prepare_audio(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Prepare audio chunk for the model.

        Args:
            audio_chunk: Raw audio samples.

        Returns:
            Normalized float32 numpy array.
        """
        # Ensure float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Normalize to [-1, 1] if needed
        max_val = np.abs(audio_chunk).max()
        if max_val > 1.0:
            audio_chunk = audio_chunk / max_val

        return audio_chunk

    @staticmethod
    def get_chunk_samples(sample_rate: int = AUDIO_SAMPLE_RATE) -> int:
        """Get the number of samples per VAD chunk.

        Args:
            sample_rate: Audio sample rate in Hz.

        Returns:
            Number of samples per chunk (default hop size).
        """
        # TEN VAD uses fixed hop sizes, return default
        return TenVadDetector.DEFAULT_HOP_SIZE

    @property
    def chunk_duration_ms(self) -> float:
        """Get chunk duration in milliseconds."""
        return (self.hop_size / self._sample_rate) * 1000


def is_ten_vad_available() -> bool:
    """Check if TEN VAD is installed and available.

    Returns:
        True if TEN VAD can be imported, False otherwise.
    """
    return TEN_VAD_AVAILABLE


def create_ten_vad_detector(
    threshold: float = VAD_SPEECH_THRESHOLD,
    hop_size: int = TenVadDetector.DEFAULT_HOP_SIZE,
) -> TenVadDetector:
    """Factory function to create a configured TEN VAD detector.

    Args:
        threshold: Speech probability threshold (0.0-1.0).
        hop_size: Number of samples per frame (160 or 256).

    Returns:
        Configured TenVadDetector instance.

    Raises:
        TenVadNotInstalledError: If TEN VAD is not installed.
    """
    return TenVadDetector(threshold=threshold, hop_size=hop_size)
