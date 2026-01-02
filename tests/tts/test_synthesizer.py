"""Tests for Text-to-Speech synthesizer module.

This module tests the TTS synthesis functionality including
backend selection and audio output.
"""

import numpy as np
import pytest

from polyglott.constants import TTS_CHILDREN_SPEED, TTS_DEFAULT_SPEED
from polyglott.tts.synthesizer import (
    KOKORO_LANG_CODES,
    KOKORO_VOICES,
    SpeechSynthesizer,
    SynthesisResult,
    TTSBackend,
    create_synthesizer,
)


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    def test_creation(self) -> None:
        """Test result can be created."""
        audio = np.zeros(24000, dtype=np.float32)
        result = SynthesisResult(
            audio=audio,
            sample_rate=24000,
            duration_seconds=1.0,
        )
        assert len(result.audio) == 24000
        assert result.sample_rate == 24000
        assert result.duration_seconds == 1.0


class TestTTSBackend:
    """Tests for TTSBackend enum."""

    def test_all_backends_exist(self) -> None:
        """Test all expected backends are defined."""
        assert TTSBackend.KOKORO.value == "kokoro"
        assert TTSBackend.PIPER.value == "piper"


class TestKokoroLangCodes:
    """Tests for Kokoro language code mapping."""

    def test_supported_languages(self) -> None:
        """Test expected languages are supported."""
        assert "en" in KOKORO_LANG_CODES
        assert "es" in KOKORO_LANG_CODES
        assert "ja" in KOKORO_LANG_CODES
        assert "zh" in KOKORO_LANG_CODES

    def test_german_not_supported(self) -> None:
        """Test German is not in Kokoro (uses Piper)."""
        assert "de" not in KOKORO_LANG_CODES


class TestKokoroVoices:
    """Tests for Kokoro voice mapping."""

    def test_voices_for_languages(self) -> None:
        """Test each language has a default voice."""
        for lang in KOKORO_LANG_CODES:
            assert lang in KOKORO_VOICES


class TestSpeechSynthesizer:
    """Tests for SpeechSynthesizer class."""

    def test_initialization(self) -> None:
        """Test synthesizer initializes correctly."""
        synth = SpeechSynthesizer()
        assert synth.speed == TTS_CHILDREN_SPEED

    def test_initialization_custom_speed(self) -> None:
        """Test synthesizer accepts custom speed."""
        synth = SpeechSynthesizer(speed=1.0)
        assert synth.speed == 1.0

    def test_supported_languages(self) -> None:
        """Test all target languages are supported."""
        synth = SpeechSynthesizer()
        languages = synth.supported_languages

        assert "en" in languages
        assert "de" in languages
        assert "es" in languages
        assert "ja" in languages
        assert "zh" in languages


class TestCreateSynthesizer:
    """Tests for create_synthesizer factory function."""

    def test_creates_synthesizer(self) -> None:
        """Test factory creates valid synthesizer."""
        synth = create_synthesizer()
        assert isinstance(synth, SpeechSynthesizer)

    def test_for_children(self) -> None:
        """Test children mode uses slower speed."""
        synth = create_synthesizer(for_children=True)
        assert synth.speed == TTS_CHILDREN_SPEED

    def test_for_adults(self) -> None:
        """Test adult mode uses normal speed."""
        synth = create_synthesizer(for_children=False)
        assert synth.speed == TTS_DEFAULT_SPEED
