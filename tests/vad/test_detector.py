"""Tests for Voice Activity Detection module.

This module tests the VAD detector functionality including
state transitions and speech probability calculation.
"""

import numpy as np
import pytest

from polyglott.constants import AUDIO_SAMPLE_RATE, VAD_CHUNK_SAMPLES
from polyglott.vad.detector import (
    SpeechState,
    VADResult,
    VoiceActivityDetector,
    create_vad_detector,
)


class TestVADResult:
    """Tests for VADResult dataclass."""

    def test_creation(self) -> None:
        """Test VADResult can be created with required fields."""
        result = VADResult(
            speech_probability=0.8,
            is_speech=True,
            state=SpeechState.SPEAKING,
        )
        assert result.speech_probability == 0.8
        assert result.is_speech is True
        assert result.state == SpeechState.SPEAKING


class TestSpeechState:
    """Tests for SpeechState enum."""

    def test_all_states_exist(self) -> None:
        """Test all expected speech states are defined."""
        assert SpeechState.SILENCE.value == "silence"
        assert SpeechState.SPEECH_START.value == "speech_start"
        assert SpeechState.SPEAKING.value == "speaking"
        assert SpeechState.SPEECH_END.value == "speech_end"


class TestVoiceActivityDetector:
    """Tests for VoiceActivityDetector class."""

    def test_initialization_default(self) -> None:
        """Test detector initializes with default values."""
        detector = VoiceActivityDetector()
        assert detector.sample_rate == AUDIO_SAMPLE_RATE
        assert detector.threshold == 0.5
        assert detector.is_speaking is False

    def test_initialization_custom(self) -> None:
        """Test detector initializes with custom values."""
        detector = VoiceActivityDetector(
            sample_rate=8000,
            threshold=0.7,
        )
        assert detector.sample_rate == 8000
        assert detector.threshold == 0.7

    def test_invalid_sample_rate(self) -> None:
        """Test detector rejects invalid sample rates."""
        with pytest.raises(ValueError, match="Sample rate must be"):
            VoiceActivityDetector(sample_rate=44100)

    def test_reset(self) -> None:
        """Test detector reset clears state."""
        detector = VoiceActivityDetector()
        detector._is_speaking = True
        detector._speech_frame_count = 10

        detector.reset()

        assert detector._is_speaking is False
        assert detector._speech_frame_count == 0

    def test_process_chunk_wrong_size(self) -> None:
        """Test process_chunk rejects wrong chunk size."""
        detector = VoiceActivityDetector()
        wrong_chunk = np.zeros(100, dtype=np.float32)

        with pytest.raises(ValueError, match="Expected .* samples"):
            detector.process_chunk(wrong_chunk)

    def test_process_silence_chunk(self, silence_chunk: np.ndarray) -> None:
        """Test processing a silence chunk returns low probability."""
        detector = VoiceActivityDetector()
        result = detector.process_chunk(silence_chunk)

        assert isinstance(result, VADResult)
        assert 0.0 <= result.speech_probability <= 1.0
        assert result.state == SpeechState.SILENCE

    def test_get_chunk_samples(self) -> None:
        """Test get_chunk_samples returns correct value."""
        samples_16k = VoiceActivityDetector.get_chunk_samples(16000)
        samples_8k = VoiceActivityDetector.get_chunk_samples(8000)

        assert samples_16k == 480  # 30ms at 16kHz
        assert samples_8k == 240   # 30ms at 8kHz


class TestCreateVadDetector:
    """Tests for create_vad_detector factory function."""

    def test_creates_detector(self) -> None:
        """Test factory creates valid detector."""
        detector = create_vad_detector()
        assert isinstance(detector, VoiceActivityDetector)

    def test_custom_threshold(self) -> None:
        """Test factory accepts custom threshold."""
        detector = create_vad_detector(threshold=0.8)
        assert detector.threshold == 0.8
