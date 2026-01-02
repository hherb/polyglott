"""Tests for Text-to-Speech synthesizer module.

This module tests the TTS synthesis functionality including
backend selection, audio output, and multilingual synthesis.
"""

import numpy as np
import pytest

from polyglott.constants import TTS_CHILDREN_SPEED, TTS_DEFAULT_SPEED, TTS_SAMPLE_RATE
from polyglott.tts.synthesizer import (
    KOKORO_LANG_CODES,
    KOKORO_VOICES,
    PiperSynthesizer,
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


class TestPiperSynthesizer:
    """Tests for PiperSynthesizer class."""

    def test_initialization(self) -> None:
        """Test Piper synthesizer initializes correctly."""
        synth = PiperSynthesizer()
        assert synth.sample_rate == 22050
        assert synth._cache_dir.exists()

    def test_supported_languages(self) -> None:
        """Test Piper supports expected languages."""
        synth = PiperSynthesizer()
        languages = synth.supported_languages
        assert "de" in languages
        assert "en" in languages
        assert "es" in languages

    def test_voice_configs(self) -> None:
        """Test voice configurations are properly structured."""
        for lang, config in PiperSynthesizer.PIPER_VOICES.items():
            assert "name" in config
            assert "path" in config
            assert config["name"].startswith(lang[:2])

    def test_cache_dir_created(self) -> None:
        """Test cache directory is created."""
        synth = PiperSynthesizer()
        assert synth._cache_dir.exists()
        assert synth._cache_dir.is_dir()

    def test_empty_text_returns_silence(self) -> None:
        """Test empty text returns silent audio."""
        synth = PiperSynthesizer()
        result = synth.synthesize("", language="de")
        assert isinstance(result, SynthesisResult)
        assert result.duration_seconds == pytest.approx(0.1, rel=0.1)


class TestSynthesizeMultilingual:
    """Tests for synthesize_multilingual method."""

    def test_no_tags_uses_default_language(self) -> None:
        """Test text without tags uses default language."""
        synth = SpeechSynthesizer()
        result = synth.synthesize_multilingual(
            "Hello, how are you?",
            default_language="en",
        )
        assert isinstance(result, SynthesisResult)
        assert result.sample_rate == TTS_SAMPLE_RATE
        assert len(result.audio) > 0

    def test_returns_valid_result(self) -> None:
        """Test multilingual synthesis returns valid audio."""
        synth = SpeechSynthesizer()
        result = synth.synthesize_multilingual(
            "The word is <lang:de>Hund</lang>.",
            default_language="en",
        )
        assert isinstance(result, SynthesisResult)
        assert result.audio is not None
        assert len(result.audio) > 0
        assert result.duration_seconds > 0

    def test_empty_text_returns_silence(self) -> None:
        """Test empty text returns short silence."""
        synth = SpeechSynthesizer()
        result = synth.synthesize_multilingual(
            "",
            default_language="en",
        )
        assert isinstance(result, SynthesisResult)
        assert result.duration_seconds == pytest.approx(0.1, rel=0.1)

    def test_only_whitespace_returns_silence(self) -> None:
        """Test whitespace-only text returns silence."""
        synth = SpeechSynthesizer()
        result = synth.synthesize_multilingual(
            "   ",
            default_language="en",
        )
        assert isinstance(result, SynthesisResult)
        assert result.duration_seconds == pytest.approx(0.1, rel=0.1)

    def test_respects_speed_parameter(self) -> None:
        """Test speed parameter affects synthesis."""
        synth = SpeechSynthesizer()

        # Synthesize same text at different speeds
        fast_result = synth.synthesize_multilingual(
            "Hello world",
            default_language="en",
            speed=1.2,
        )
        slow_result = synth.synthesize_multilingual(
            "Hello world",
            default_language="en",
            speed=0.8,
        )

        # Slower speech should have longer duration
        assert slow_result.duration_seconds > fast_result.duration_seconds

    def test_sample_rate_consistent(self) -> None:
        """Test output sample rate is consistent across segments."""
        synth = SpeechSynthesizer()
        result = synth.synthesize_multilingual(
            "Say <lang:de>Hallo</lang> and <lang:es>Hola</lang>!",
            default_language="en",
        )
        assert result.sample_rate == TTS_SAMPLE_RATE

    def test_handles_adjacent_tags(self) -> None:
        """Test adjacent language tags are handled."""
        synth = SpeechSynthesizer()
        result = synth.synthesize_multilingual(
            "<lang:de>Eins</lang><lang:de>zwei</lang><lang:de>drei</lang>",
            default_language="en",
        )
        assert isinstance(result, SynthesisResult)
        assert len(result.audio) > 0


class TestResample:
    """Tests for the _resample method."""

    def test_same_rate_unchanged(self) -> None:
        """Test resampling at same rate returns original."""
        synth = SpeechSynthesizer()
        audio = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
        result = synth._resample(audio, 24000, 24000)
        np.testing.assert_array_almost_equal(audio, result)

    def test_upsample(self) -> None:
        """Test upsampling produces more samples."""
        synth = SpeechSynthesizer()
        audio = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
        result = synth._resample(audio, 22050, 24000)
        expected_length = int(1000 * 24000 / 22050)
        assert len(result) == expected_length

    def test_downsample(self) -> None:
        """Test downsampling produces fewer samples."""
        synth = SpeechSynthesizer()
        audio = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
        result = synth._resample(audio, 24000, 22050)
        expected_length = int(1000 * 22050 / 24000)
        assert len(result) == expected_length

    def test_output_is_float32(self) -> None:
        """Test resampled audio is float32."""
        synth = SpeechSynthesizer()
        audio = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
        result = synth._resample(audio, 22050, 24000)
        assert result.dtype == np.float32
