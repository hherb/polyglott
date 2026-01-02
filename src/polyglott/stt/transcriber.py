"""Speech-to-Text transcription with multiple backend support.

This module provides speech recognition functionality optimized
for real-time conversational applications with multilingual support.

Backends:
- Whisper (faster-whisper): Best for multilingual/code-switching, cross-platform
- Whisper (MLX): Optimized for Apple Silicon Macs
- Moonshine: Fast, lightweight, good for single-language

Supports both batch and streaming transcription modes for
optimal latency in different scenarios.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generator, Optional, Protocol, Union

import numpy as np

from polyglott.constants import (
    AUDIO_SAMPLE_RATE,
    MAX_TRANSCRIPTION_AUDIO_SECONDS,
    MOONSHINE_MODEL_SIZE,
    WHISPER_MODEL_SIZE,
    TargetLanguage,
)


class TranscriberBackend(str, Enum):
    """Available transcription backends."""

    MOONSHINE = "moonshine"
    WHISPER_MLX = "whisper_mlx"
    FASTER_WHISPER = "faster_whisper"


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


@dataclass
class StreamingTranscriptionChunk:
    """A chunk of streaming transcription output.

    Attributes:
        text: Partial transcribed text so far.
        is_final: Whether this is the final result.
        audio_position_seconds: Position in audio stream.
    """

    text: str
    is_final: bool = False
    audio_position_seconds: float = 0.0


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

    def transcribe_streaming(
        self,
        audio_generator: Generator[np.ndarray, None, None],
        language: Optional[str] = None,
        chunk_duration_seconds: float = 1.0,
        overlap_seconds: float = 0.3,
    ) -> Generator[StreamingTranscriptionChunk, None, TranscriptionResult]:
        """Transcribe audio in a streaming fashion.

        Processes audio chunks as they arrive, yielding partial
        transcription results for lower perceived latency.

        Args:
            audio_generator: Generator yielding audio chunks.
            language: Language code for transcription.
            chunk_duration_seconds: Duration of each processing window.
            overlap_seconds: Overlap between windows for continuity.

        Yields:
            StreamingTranscriptionChunk with partial results.

        Returns:
            Final TranscriptionResult with complete text.
        """
        self._ensure_loaded()

        # Collect audio chunks
        audio_buffer = []
        chunk_samples = int(chunk_duration_seconds * self.sample_rate)
        overlap_samples = int(overlap_seconds * self.sample_rate)
        last_text = ""
        total_duration = 0.0

        for audio_chunk in audio_generator:
            audio_buffer.append(audio_chunk)
            total_samples = sum(len(c) for c in audio_buffer)

            # Process when we have enough audio
            if total_samples >= chunk_samples:
                # Concatenate buffer
                full_audio = np.concatenate(audio_buffer)
                audio_to_process = self._prepare_audio(full_audio)

                # Transcribe current buffer
                try:
                    model = self._get_model_for_language(language or "en")
                    result = self._moonshine.transcribe(audio_to_process, model)
                    current_text = result[0] if result else ""

                    if current_text != last_text:
                        last_text = current_text
                        total_duration = len(full_audio) / self.sample_rate

                        yield StreamingTranscriptionChunk(
                            text=current_text,
                            is_final=False,
                            audio_position_seconds=total_duration,
                        )
                except Exception:
                    # Continue on errors
                    pass

                # Keep overlap for continuity (trim buffer)
                if len(full_audio) > overlap_samples:
                    trimmed = full_audio[-overlap_samples:]
                    audio_buffer = [trimmed]

        # Final transcription of complete audio
        if audio_buffer:
            full_audio = np.concatenate(audio_buffer)
            audio_to_process = self._prepare_audio(full_audio)

            try:
                model = self._get_model_for_language(language or "en")
                result = self._moonshine.transcribe(audio_to_process, model)
                final_text = result[0] if result else ""
            except Exception:
                final_text = last_text

            total_duration = len(full_audio) / self.sample_rate

            # Yield final chunk
            yield StreamingTranscriptionChunk(
                text=final_text,
                is_final=True,
                audio_position_seconds=total_duration,
            )

            return TranscriptionResult(
                text=final_text,
                language=language or "en",
                duration_seconds=total_duration,
            )

        return TranscriptionResult(
            text=last_text,
            language=language or "en",
            duration_seconds=total_duration,
        )


class WhisperMLXTranscriber:
    """Speech transcriber using Whisper via MLX.

    This transcriber uses Apple's MLX framework for optimized
    Whisper inference on Apple Silicon Macs.

    Note:
        Only available on macOS with Apple Silicon.
    """

    # Model name mappings for MLX community models
    # Some models don't follow the simple whisper-{size}-mlx pattern
    MLX_MODEL_REPOS: dict[str, str] = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large": "mlx-community/whisper-large-mlx",
        "large-v2": "mlx-community/whisper-large-v2-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    }

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> None:
        """Initialize the Whisper MLX transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large,
                       large-v2, large-v3, large-v3-turbo).
            sample_rate: Expected audio sample rate in Hz.
        """
        self.model_size = model_size
        self.sample_rate = sample_rate
        self._mlx_whisper = None

    def _get_model_repo(self) -> str:
        """Get the HuggingFace repo path for the model.

        Returns:
            HuggingFace model repository path.
        """
        if self.model_size in self.MLX_MODEL_REPOS:
            return self.MLX_MODEL_REPOS[self.model_size]
        # Fallback for unknown sizes
        return f"mlx-community/whisper-{self.model_size}-mlx"

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
            language: Optional language code (None for auto-detection).

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
            path_or_hf_repo=self._get_model_repo(),
            language=language,  # None enables auto-detection
        )

        return TranscriptionResult(
            text=result.get("text", "").strip(),
            language=result.get("language", language or "en"),
        )


class FasterWhisperTranscriber:
    """Speech transcriber using faster-whisper.

    This transcriber uses CTranslate2-optimized Whisper models for
    fast inference on CPU and GPU (CUDA). Works on Linux, Windows, and macOS.

    Note:
        Requires faster-whisper package: uv add faster-whisper
    """

    # Model name mappings for Hugging Face models
    MODEL_NAMES: dict[str, str] = {
        "tiny": "tiny",
        "base": "base",
        "small": "small",
        "medium": "medium",
        "large": "large-v2",
        "large-v2": "large-v2",
        "large-v3": "large-v3",
        "large-v3-turbo": "large-v3-turbo",
    }

    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        sample_rate: int = AUDIO_SAMPLE_RATE,
        device: str = "auto",
        compute_type: str = "auto",
    ) -> None:
        """Initialize the faster-whisper transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large,
                       large-v2, large-v3, large-v3-turbo).
            sample_rate: Expected audio sample rate in Hz.
            device: Device to use ('auto', 'cpu', 'cuda').
            compute_type: Compute type ('auto', 'int8', 'float16', 'float32').
        """
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _get_model_name(self) -> str:
        """Get the model name for faster-whisper.

        Returns:
            Model name string.
        """
        return self.MODEL_NAMES.get(self.model_size, self.model_size)

    def _ensure_loaded(self) -> None:
        """Ensure faster-whisper model is loaded."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel

                model_name = self._get_model_name()
                self._model = WhisperModel(
                    model_name,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except ImportError as e:
                raise ImportError(
                    "faster-whisper not installed. "
                    "Install with: uv add faster-whisper"
                ) from e

    def transcribe(
        self,
        audio: Union[np.ndarray, Path, str],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text using faster-whisper.

        Args:
            audio: Audio data as numpy array, or path to audio file.
            language: Optional language code (None for auto-detection).

        Returns:
            TranscriptionResult with transcribed text.
        """
        self._ensure_loaded()

        # Handle file path
        if isinstance(audio, (Path, str)):
            audio_path = str(audio)
            segments, info = self._model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
            )
        else:
            # faster-whisper can handle numpy arrays directly
            segments, info = self._model.transcribe(
                audio,
                language=language,
                beam_size=5,
            )

        # Collect all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text)

        full_text = " ".join(text_parts).strip()

        return TranscriptionResult(
            text=full_text,
            language=info.language if info.language else (language or "en"),
            confidence=info.language_probability if hasattr(info, 'language_probability') else None,
            duration_seconds=info.duration if hasattr(info, 'duration') else None,
        )


class SpeechTranscriber:
    """Unified speech transcriber with backend selection.

    This class provides a unified interface for speech transcription,
    automatically selecting the best available backend.

    For multilingual/code-switching support, Whisper backends are preferred.
    For single-language speed, Moonshine may be faster.

    Example:
        >>> transcriber = SpeechTranscriber()
        >>> result = transcriber.transcribe(audio_data, language=None)  # Auto-detect
        >>> print(result.text)
    """

    def __init__(
        self,
        backend: Optional[TranscriberBackend] = None,
        model_size: Optional[str] = None,
    ) -> None:
        """Initialize the speech transcriber.

        Args:
            backend: Specific backend to use, or None for auto-selection.
            model_size: Model size for the selected backend. If None, uses
                       WHISPER_MODEL_SIZE for Whisper backends or
                       MOONSHINE_MODEL_SIZE for Moonshine.
        """
        self._requested_model_size = model_size
        self._backend = backend
        self._transcriber: Optional[TranscriberProtocol] = None

    @property
    def model_size(self) -> str:
        """Get the effective model size based on backend."""
        if self._requested_model_size:
            return self._requested_model_size
        # Use appropriate default based on backend
        if self._backend == TranscriberBackend.MOONSHINE:
            return MOONSHINE_MODEL_SIZE
        return WHISPER_MODEL_SIZE

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
        elif self._backend == TranscriberBackend.WHISPER_MLX:
            self._transcriber = WhisperMLXTranscriber(model_size=self.model_size)
        elif self._backend == TranscriberBackend.FASTER_WHISPER:
            self._transcriber = FasterWhisperTranscriber(model_size=self.model_size)
        else:
            # Fallback to faster-whisper
            self._transcriber = FasterWhisperTranscriber(model_size=self.model_size)

        return self._transcriber

    def _detect_best_backend(self) -> TranscriberBackend:
        """Detect the best available transcription backend.

        Prefers Whisper backends for better multilingual/code-switching support.
        On Apple Silicon, prefers MLX Whisper for hardware optimization.

        Returns:
            Best available backend.
        """
        import platform
        import sys

        # Check if running on Apple Silicon
        is_apple_silicon = (
            sys.platform == "darwin" and platform.machine() == "arm64"
        )

        # On Apple Silicon, prefer MLX Whisper (hardware optimized)
        if is_apple_silicon:
            try:
                import mlx_whisper  # noqa: F401

                return TranscriberBackend.WHISPER_MLX
            except ImportError:
                pass

        # Try faster-whisper (cross-platform, good multilingual)
        try:
            from faster_whisper import WhisperModel  # noqa: F401

            return TranscriberBackend.FASTER_WHISPER
        except ImportError:
            pass

        # Try MLX Whisper as fallback on non-Apple Silicon Macs
        if not is_apple_silicon:
            try:
                import mlx_whisper  # noqa: F401

                return TranscriberBackend.WHISPER_MLX
            except ImportError:
                pass

        # Fall back to Moonshine (fast but single-language focused)
        try:
            import moonshine_onnx  # noqa: F401

            return TranscriberBackend.MOONSHINE
        except ImportError:
            pass

        # Default to faster-whisper (will fail with helpful message if not installed)
        return TranscriberBackend.FASTER_WHISPER

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

    def transcribe_streaming(
        self,
        audio_generator: Generator[np.ndarray, None, None],
        language: Optional[str] = None,
        chunk_duration_seconds: float = 1.0,
    ) -> Generator[StreamingTranscriptionChunk, None, TranscriptionResult]:
        """Transcribe audio in a streaming fashion.

        Processes audio chunks as they arrive, yielding partial
        results for lower perceived latency.

        Args:
            audio_generator: Generator yielding audio chunks.
            language: Language code for transcription.
            chunk_duration_seconds: Duration of each processing window.

        Yields:
            StreamingTranscriptionChunk with partial results.

        Returns:
            Final TranscriptionResult with complete text.
        """
        transcriber = self._get_transcriber()

        # Only Moonshine supports streaming currently
        if isinstance(transcriber, MoonshineTranscriber):
            return transcriber.transcribe_streaming(
                audio_generator,
                language=language,
                chunk_duration_seconds=chunk_duration_seconds,
            )
        else:
            # Fall back to collecting all audio and transcribing at once
            audio_chunks = list(audio_generator)
            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                result = transcriber.transcribe(full_audio, language)

                yield StreamingTranscriptionChunk(
                    text=result.text,
                    is_final=True,
                    audio_position_seconds=result.duration_seconds or 0.0,
                )

                return result
            else:
                return TranscriptionResult(text="", language=language or "en")

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
