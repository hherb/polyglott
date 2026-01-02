"""Tests for Speech-to-Text transcriber module.

This module tests the speech transcription functionality
including backend selection and result handling.
"""

import numpy as np
import pytest

from polyglott.constants import AUDIO_SAMPLE_RATE
from polyglott.stt.transcriber import (
    MOONSHINE_LANGUAGE_MODELS,
    SpeechTranscriber,
    TranscriberBackend,
    TranscriptionResult,
    create_transcriber,
)


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_creation_minimal(self) -> None:
        """Test result can be created with minimal fields."""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
        )
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.confidence is None
        assert result.duration_seconds is None

    def test_creation_full(self) -> None:
        """Test result can be created with all fields."""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            confidence=0.95,
            duration_seconds=2.5,
        )
        assert result.confidence == 0.95
        assert result.duration_seconds == 2.5


class TestTranscriberBackend:
    """Tests for TranscriberBackend enum."""

    def test_all_backends_exist(self) -> None:
        """Test all expected backends are defined."""
        assert TranscriberBackend.MOONSHINE.value == "moonshine"
        assert TranscriberBackend.WHISPER_MLX.value == "whisper_mlx"


class TestMoonshineLanguageModels:
    """Tests for Moonshine language model mapping."""

    def test_all_languages_have_models(self) -> None:
        """Test all target languages have model mappings."""
        required_languages = ["en", "de", "es", "ja", "zh"]
        for lang in required_languages:
            assert lang in MOONSHINE_LANGUAGE_MODELS

    def test_models_are_valid(self) -> None:
        """Test model names are properly formatted."""
        for lang, model in MOONSHINE_LANGUAGE_MODELS.items():
            assert model.startswith("moonshine/")


class TestSpeechTranscriber:
    """Tests for SpeechTranscriber class."""

    def test_initialization_default(self) -> None:
        """Test transcriber initializes with defaults."""
        transcriber = SpeechTranscriber()
        assert transcriber.model_size == "base"
        assert transcriber._backend is None

    def test_initialization_with_backend(self) -> None:
        """Test transcriber initializes with specific backend."""
        transcriber = SpeechTranscriber(backend=TranscriberBackend.MOONSHINE)
        assert transcriber._backend == TranscriberBackend.MOONSHINE

    def test_supported_backends(self) -> None:
        """Test backend detection does not raise."""
        transcriber = SpeechTranscriber()
        # Should not raise even if backends not installed
        backend = transcriber.backend
        assert backend in [TranscriberBackend.MOONSHINE, TranscriberBackend.WHISPER_MLX]


class TestCreateTranscriber:
    """Tests for create_transcriber factory function."""

    def test_creates_transcriber(self) -> None:
        """Test factory creates valid transcriber."""
        transcriber = create_transcriber()
        assert isinstance(transcriber, SpeechTranscriber)

    def test_prefer_speed(self) -> None:
        """Test prefer_speed affects model size."""
        fast = create_transcriber(prefer_speed=True)
        accurate = create_transcriber(prefer_speed=False)

        assert fast.model_size == "tiny"
        assert accurate.model_size == "base"
