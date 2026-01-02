"""Speech-to-Text transcription using Moonshine ASR.

This module provides speech recognition functionality optimized
for real-time conversational applications with multilingual support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol, Union

import numpy as np

from polyglott.constants import (
    AUDIO_SAMPLE_RATE,
    MAX_TRANSCRIPTION_AUDIO_SECONDS,
    MOONSHINE_MODEL_SIZE,
    TargetLanguage,
)


class TranscriberBackend(str, Enum):
    """Available transcription backends."""

    MOONSHINE = "moonshine"
    WHISPER_MLX = "whisper_mlx"


@dataclass
class TranscriptionResult:
    """Result from speech transcription.

    Attributes:
        text: Transcribed text.
        language: Detected or specified language code.
        confidence: Confidence score if available (0.0-1.0).
        duration_seconds: Duration of processed audio.
    """

    text: str
    language: str
    confidence: Optional[float] = None
    duration_seconds: Optional[float] = None


class TranscriberProtocol(Protocol):
    """Protocol for transcription backends."""

    def transcribe(
        self,
        audio: Union[np.ndarray, Path, str],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text."""
        ...


# Mapping of languages to best Moonshine models
MOONSHINE_LANGUAGE_MODELS: dict[str, str] = {
    "en": "moonshine/base",
    "de": "moonshine/base",  # Use base for German (multilingual)
    "es": "moonshine/base",  # Spanish base model
    "ja": "moonshine/tiny",  # Japanese specialized
    "zh": "moonshine/tiny",  # Chinese specialized
}


class MoonshineTranscriber:
    """Speech transcriber using Moonshine ASR.

    Moonshine provides fast, accurate speech recognition optimized
    for edge devices. It's 5-15x faster than Whisper.

    Attributes:
        model_size: Size of Moonshine model to use.
        sample_rate: Expected audio sample rate.

    Example:
        >>> transcriber = MoonshineTranscriber()
        >>> result = transcriber.transcribe(audio_array, language="en")
        >>> print(result.text)
    """

    def __init__(
        self,
        model_size: str = MOONSHINE_MODEL_SIZE,
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> None:
        """Initialize the Moonshine transcriber.

        Args:
            model_size: Model size ("tiny" or "base").
            sample_rate: Expected audio sample rate in Hz.
        """
        self.model_size = model_size
        self.sample_rate = sample_rate
        self._moonshine = None

    def _ensure_loaded(self) -> None:
        """Ensure moonshine_onnx is imported and available."""
        if self._moonshine is None:
            try:
                import moonshine_onnx

                self._moonshine = moonshine_onnx
            except ImportError as e:
                raise ImportError(
                    "Moonshine ONNX not installed. "
                    "Install with: uv add useful-moonshine-onnx"
                ) from e

    def _get_model_for_language(self, language: str) -> str:
        """Get the appropriate model for a language.

        Args:
            language: ISO language code (e.g., "en", "ja").

        Returns:
            Moonshine model identifier.
        """
        return MOONSHINE_LANGUAGE_MODELS.get(language, f"moonshine/{self.model_size}")

    def transcribe(
        self,
        audio: Union[np.ndarray, Path, str],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio data as numpy array, or path to audio file.
            language: Optional language code for model selection.

        Returns:
            TranscriptionResult with transcribed text.

        Raises:
            ValueError: If audio is too long or invalid.
        """
        self._ensure_loaded()

        # Handle file path input
        if isinstance(audio, (Path, str)):
            audio_path = Path(audio)
            if not audio_path.exists():
                raise ValueError(f"Audio file not found: {audio_path}")
            result = self._transcribe_file(str(audio_path), language)
            return result

        # Handle numpy array input
        return self._transcribe_array(audio, language)

    def _transcribe_file(
        self,
        audio_path: str,
        language: Optional[str],
    ) -> TranscriptionResult:
        """Transcribe an audio file.

        Args:
            audio_path: Path to audio file.
            language: Optional language code.

        Returns:
            TranscriptionResult with transcribed text.
        """
        model = self._get_model_for_language(language or "en")
        result = self._moonshine.transcribe(audio_path, model)

        text = result[0] if result else ""
        return TranscriptionResult(
            text=text,
            language=language or "en",
        )

    def _transcribe_array(
        self,
        audio: np.ndarray,
        language: Optional[str],
    ) -> TranscriptionResult:
        """Transcribe a numpy audio array.

        Args:
            audio: Audio samples as numpy array.
            language: Optional language code.

        Returns:
            TranscriptionResult with transcribed text.

        Raises:
            ValueError: If audio is too long.
        """
        # Check duration
        duration = len(audio) / self.sample_rate
        if duration > MAX_TRANSCRIPTION_AUDIO_SECONDS:
            raise ValueError(
                f"Audio too long: {duration:.1f}s > {MAX_TRANSCRIPTION_AUDIO_SECONDS}s"
            )

        # Ensure correct format
        audio = self._prepare_audio(audio)

        model = self._get_model_for_language(language or "en")
        result = self._moonshine.transcribe(audio, model)

        text = result[0] if result else ""
        return TranscriptionResult(
            text=text,
            language=language or "en",
            duration_seconds=duration,
        )

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """Prepare audio array for transcription.

        Args:
            audio: Raw audio samples.

        Returns:
            Properly formatted audio array.
        """
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        return audio


class WhisperMLXTranscriber:
    """Speech transcriber using Whisper via MLX.

    This transcriber uses Apple's MLX framework for optimized
    Whisper inference on Apple Silicon Macs.

    Note:
        Only available on macOS with Apple Silicon.
    """

    def __init__(
        self,
        model_size: str = "base",
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> None:
        """Initialize the Whisper MLX transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large).
            sample_rate: Expected audio sample rate in Hz.
        """
        self.model_size = model_size
        self.sample_rate = sample_rate
        self._mlx_whisper = None

    def _ensure_loaded(self) -> None:
        """Ensure mlx_whisper is imported and available."""
        if self._mlx_whisper is None:
            try:
                import mlx_whisper

                self._mlx_whisper = mlx_whisper
            except ImportError as e:
                raise ImportError(
                    "mlx-whisper not installed. "
                    "Install with: uv add mlx-whisper (macOS only)"
                ) from e

    def transcribe(
        self,
        audio: Union[np.ndarray, Path, str],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text using Whisper.

        Args:
            audio: Audio data as numpy array, or path to audio file.
            language: Optional language code.

        Returns:
            TranscriptionResult with transcribed text.
        """
        self._ensure_loaded()

        # Handle file path
        if isinstance(audio, (Path, str)):
            audio_path = str(audio)
        else:
            # For array input, we need to save temporarily
            # This is a limitation of mlx-whisper
            import tempfile
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, self.sample_rate)
                audio_path = f.name

        result = self._mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=f"mlx-community/whisper-{self.model_size}-mlx",
            language=language,
        )

        return TranscriptionResult(
            text=result.get("text", "").strip(),
            language=result.get("language", language or "en"),
        )


class SpeechTranscriber:
    """Unified speech transcriber with backend selection.

    This class provides a unified interface for speech transcription,
    automatically selecting the best available backend.

    Example:
        >>> transcriber = SpeechTranscriber()
        >>> result = transcriber.transcribe(audio_data, language="en")
        >>> print(result.text)
    """

    def __init__(
        self,
        backend: Optional[TranscriberBackend] = None,
        model_size: str = MOONSHINE_MODEL_SIZE,
    ) -> None:
        """Initialize the speech transcriber.

        Args:
            backend: Specific backend to use, or None for auto-selection.
            model_size: Model size for the selected backend.
        """
        self.model_size = model_size
        self._backend = backend
        self._transcriber: Optional[TranscriberProtocol] = None

    def _get_transcriber(self) -> TranscriberProtocol:
        """Get or create the transcriber backend.

        Returns:
            Transcriber instance.
        """
        if self._transcriber is not None:
            return self._transcriber

        # Auto-select backend if not specified
        if self._backend is None:
            self._backend = self._detect_best_backend()

        if self._backend == TranscriberBackend.MOONSHINE:
            self._transcriber = MoonshineTranscriber(model_size=self.model_size)
        else:
            self._transcriber = WhisperMLXTranscriber(model_size=self.model_size)

        return self._transcriber

    def _detect_best_backend(self) -> TranscriberBackend:
        """Detect the best available transcription backend.

        Returns:
            Best available backend.
        """
        # Try Moonshine first (preferred for speed)
        try:
            import moonshine_onnx  # noqa: F401

            return TranscriberBackend.MOONSHINE
        except ImportError:
            pass

        # Try MLX Whisper (macOS only)
        try:
            import mlx_whisper  # noqa: F401

            return TranscriberBackend.WHISPER_MLX
        except ImportError:
            pass

        # Default to Moonshine (will fail with helpful message if not installed)
        return TranscriberBackend.MOONSHINE

    def transcribe(
        self,
        audio: Union[np.ndarray, Path, str],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: Audio data as numpy array, or path to audio file.
            language: Optional language code (e.g., "en", "de", "ja").

        Returns:
            TranscriptionResult with transcribed text and metadata.
        """
        transcriber = self._get_transcriber()
        return transcriber.transcribe(audio, language)

    @property
    def backend(self) -> TranscriberBackend:
        """Get the current transcription backend.

        Returns:
            Current backend enum value.
        """
        if self._backend is None:
            self._backend = self._detect_best_backend()
        return self._backend


def create_transcriber(
    language: Optional[str] = None,
    prefer_speed: bool = True,
) -> SpeechTranscriber:
    """Factory function to create a configured transcriber.

    Args:
        language: Primary language for transcription.
        prefer_speed: If True, prefer faster models over accuracy.

    Returns:
        Configured SpeechTranscriber instance.
    """
    model_size = "tiny" if prefer_speed else "base"

    return SpeechTranscriber(model_size=model_size)
