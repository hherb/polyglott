"""Text-to-Speech synthesis using Kokoro and Piper TTS.

This module provides speech synthesis for multiple languages,
using Kokoro TTS for most languages and Piper TTS for German.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Protocol, Union

import numpy as np

from polyglott.constants import (
    TTS_CHILDREN_SPEED,
    TTS_DEFAULT_SPEED,
    TTS_LANGUAGE_CODES,
    TTS_SAMPLE_RATE,
    TargetLanguage,
)


class TTSBackend(str, Enum):
    """Available TTS backends."""

    KOKORO = "kokoro"
    PIPER = "piper"


@dataclass
class SynthesisResult:
    """Result from speech synthesis.

    Attributes:
        audio: Synthesized audio as numpy array.
        sample_rate: Audio sample rate in Hz.
        duration_seconds: Duration of audio in seconds.
    """

    audio: np.ndarray
    sample_rate: int
    duration_seconds: float


class SynthesizerProtocol(Protocol):
    """Protocol for TTS synthesizer backends."""

    def synthesize(
        self,
        text: str,
        language: str,
        speed: float = 1.0,
    ) -> SynthesisResult:
        """Synthesize text to speech."""
        ...


# Kokoro language code mapping
KOKORO_LANG_CODES: dict[str, str] = {
    "en": "a",  # American English
    "es": "e",  # Spanish
    "ja": "j",  # Japanese
    "zh": "z",  # Mandarin Chinese
}

# Default Kokoro voices per language
KOKORO_VOICES: dict[str, str] = {
    "en": "af_heart",  # Warm female voice
    "es": "ef_dora",   # Spanish female
    "ja": "jf_alpha",  # Japanese female
    "zh": "zf_xiaobei", # Chinese female
}


class KokoroSynthesizer:
    """Speech synthesizer using Kokoro TTS.

    Kokoro is a lightweight (82M parameters) TTS model with
    high quality output at 24kHz for multiple languages.

    Note:
        Does not support German. Use PiperSynthesizer for German.

    Example:
        >>> synth = KokoroSynthesizer()
        >>> result = synth.synthesize("Hello!", language="en")
        >>> play_audio(result.audio, result.sample_rate)
    """

    def __init__(self, sample_rate: int = TTS_SAMPLE_RATE) -> None:
        """Initialize the Kokoro synthesizer.

        Args:
            sample_rate: Output sample rate (Kokoro outputs 24kHz).
        """
        self.sample_rate = sample_rate
        self._pipeline_cache: dict[str, object] = {}
        self._kokoro = None

    def _ensure_loaded(self) -> None:
        """Ensure Kokoro is imported and available."""
        if self._kokoro is None:
            try:
                import kokoro

                self._kokoro = kokoro
            except ImportError as e:
                raise ImportError(
                    "Kokoro not installed. Install with: uv add kokoro"
                ) from e

    def _get_pipeline(self, language: str) -> object:
        """Get or create a Kokoro pipeline for a language.

        Args:
            language: ISO language code.

        Returns:
            Kokoro KPipeline instance.
        """
        self._ensure_loaded()

        if language not in self._pipeline_cache:
            lang_code = KOKORO_LANG_CODES.get(language)
            if lang_code is None:
                raise ValueError(
                    f"Language '{language}' not supported by Kokoro. "
                    f"Supported: {list(KOKORO_LANG_CODES.keys())}"
                )
            self._pipeline_cache[language] = self._kokoro.KPipeline(lang_code=lang_code)

        return self._pipeline_cache[language]

    def synthesize(
        self,
        text: str,
        language: str = "en",
        speed: float = TTS_DEFAULT_SPEED,
        voice: Optional[str] = None,
    ) -> SynthesisResult:
        """Synthesize text to speech.

        Args:
            text: Text to synthesize.
            language: ISO language code (en, es, ja, zh).
            speed: Speech speed multiplier (0.5-2.0).
            voice: Optional specific voice name.

        Returns:
            SynthesisResult with audio data.

        Raises:
            ValueError: If language not supported.
        """
        if not text.strip():
            # Return silence for empty text
            return SynthesisResult(
                audio=np.zeros(int(self.sample_rate * 0.1), dtype=np.float32),
                sample_rate=self.sample_rate,
                duration_seconds=0.1,
            )

        pipeline = self._get_pipeline(language)
        voice = voice or KOKORO_VOICES.get(language, "af_heart")

        audio_chunks = []
        for _, _, audio in pipeline(text, voice=voice, speed=speed):
            audio_chunks.append(audio)

        if not audio_chunks:
            # Return silence if nothing generated
            return SynthesisResult(
                audio=np.zeros(int(self.sample_rate * 0.1), dtype=np.float32),
                sample_rate=self.sample_rate,
                duration_seconds=0.1,
            )

        audio = np.concatenate(audio_chunks)
        duration = len(audio) / self.sample_rate

        return SynthesisResult(
            audio=audio.astype(np.float32),
            sample_rate=self.sample_rate,
            duration_seconds=duration,
        )

    @property
    def supported_languages(self) -> list[str]:
        """Get list of supported language codes.

        Returns:
            List of ISO language codes.
        """
        return list(KOKORO_LANG_CODES.keys())


class PiperSynthesizer:
    """Speech synthesizer using Piper TTS.

    Piper is a fast, local neural TTS system that supports
    German and many other languages.

    Example:
        >>> synth = PiperSynthesizer()
        >>> result = synth.synthesize("Hallo!", language="de")
        >>> play_audio(result.audio, result.sample_rate)
    """

    # Piper voice models for each language
    PIPER_VOICES: dict[str, str] = {
        "de": "de_DE-thorsten-high",
        "en": "en_US-lessac-high",
        "es": "es_ES-davefx-medium",
    }

    def __init__(self, sample_rate: int = 22050) -> None:
        """Initialize the Piper synthesizer.

        Args:
            sample_rate: Output sample rate (Piper typically uses 22050).
        """
        self.sample_rate = sample_rate
        self._piper = None
        self._voice_cache: dict[str, object] = {}

    def _ensure_loaded(self) -> None:
        """Ensure Piper is imported and available."""
        if self._piper is None:
            try:
                import piper

                self._piper = piper
            except ImportError as e:
                raise ImportError(
                    "Piper TTS not installed. Install with: uv add piper-tts"
                ) from e

    def synthesize(
        self,
        text: str,
        language: str = "de",
        speed: float = TTS_DEFAULT_SPEED,
        voice: Optional[str] = None,
    ) -> SynthesisResult:
        """Synthesize text to speech.

        Args:
            text: Text to synthesize.
            language: ISO language code.
            speed: Speech speed multiplier.
            voice: Optional specific voice model name.

        Returns:
            SynthesisResult with audio data.
        """
        self._ensure_loaded()

        if not text.strip():
            return SynthesisResult(
                audio=np.zeros(int(self.sample_rate * 0.1), dtype=np.float32),
                sample_rate=self.sample_rate,
                duration_seconds=0.1,
            )

        voice_name = voice or self.PIPER_VOICES.get(language, self.PIPER_VOICES["de"])

        # Piper synthesis (simplified - actual implementation may vary)
        try:
            from piper import PiperVoice

            # Get or create voice
            if voice_name not in self._voice_cache:
                self._voice_cache[voice_name] = PiperVoice.load(voice_name)

            piper_voice = self._voice_cache[voice_name]
            audio_data = piper_voice.synthesize(text, speed=speed)

            audio = np.array(audio_data, dtype=np.float32)
            duration = len(audio) / self.sample_rate

            return SynthesisResult(
                audio=audio,
                sample_rate=self.sample_rate,
                duration_seconds=duration,
            )
        except Exception as e:
            # Fallback: return silence with error logged
            print(f"Piper synthesis failed: {e}")
            return SynthesisResult(
                audio=np.zeros(int(self.sample_rate * 0.5), dtype=np.float32),
                sample_rate=self.sample_rate,
                duration_seconds=0.5,
            )

    @property
    def supported_languages(self) -> list[str]:
        """Get list of supported language codes.

        Returns:
            List of ISO language codes.
        """
        return list(self.PIPER_VOICES.keys())


class SpeechSynthesizer:
    """Unified speech synthesizer with automatic backend selection.

    This class provides a unified interface for TTS, automatically
    selecting the appropriate backend based on the target language.

    Example:
        >>> synth = SpeechSynthesizer()
        >>> # Uses Kokoro for Spanish
        >>> result = synth.synthesize("¡Hola!", language="es")
        >>> # Uses Piper for German
        >>> result = synth.synthesize("Guten Tag!", language="de")
    """

    def __init__(
        self,
        speed: float = TTS_CHILDREN_SPEED,
    ) -> None:
        """Initialize the unified synthesizer.

        Args:
            speed: Default speech speed (slower for children).
        """
        self.speed = speed
        self._kokoro: Optional[KokoroSynthesizer] = None
        self._piper: Optional[PiperSynthesizer] = None

    def _get_backend(self, language: str) -> tuple[SynthesizerProtocol, TTSBackend]:
        """Get appropriate backend for a language.

        Args:
            language: ISO language code.

        Returns:
            Tuple of (synthesizer, backend_type).
        """
        # Use Piper for German
        if language == "de":
            if self._piper is None:
                self._piper = PiperSynthesizer()
            return self._piper, TTSBackend.PIPER

        # Use Kokoro for everything else
        if self._kokoro is None:
            self._kokoro = KokoroSynthesizer()
        return self._kokoro, TTSBackend.KOKORO

    def synthesize(
        self,
        text: str,
        language: str = "en",
        speed: Optional[float] = None,
    ) -> SynthesisResult:
        """Synthesize text to speech.

        Args:
            text: Text to synthesize.
            language: ISO language code (en, de, es, ja, zh).
            speed: Speech speed, or None for default.

        Returns:
            SynthesisResult with audio data.
        """
        speed = speed or self.speed
        backend, _ = self._get_backend(language)
        return backend.synthesize(text, language=language, speed=speed)

    def synthesize_to_file(
        self,
        text: str,
        output_path: Union[str, Path],
        language: str = "en",
        speed: Optional[float] = None,
    ) -> Path:
        """Synthesize text and save to audio file.

        Args:
            text: Text to synthesize.
            output_path: Path to save audio file.
            language: ISO language code.
            speed: Speech speed, or None for default.

        Returns:
            Path to saved audio file.
        """
        import soundfile as sf

        result = self.synthesize(text, language=language, speed=speed)
        output_path = Path(output_path)

        sf.write(str(output_path), result.audio, result.sample_rate)
        return output_path

    @property
    def supported_languages(self) -> list[str]:
        """Get all supported language codes.

        Returns:
            List of ISO language codes.
        """
        return ["en", "de", "es", "ja", "zh"]


def create_synthesizer(
    for_children: bool = True,
) -> SpeechSynthesizer:
    """Factory function to create a configured synthesizer.

    Args:
        for_children: If True, use slower speed for children.

    Returns:
        Configured SpeechSynthesizer instance.
    """
    speed = TTS_CHILDREN_SPEED if for_children else TTS_DEFAULT_SPEED
    return SpeechSynthesizer(speed=speed)
