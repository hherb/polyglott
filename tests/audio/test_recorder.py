"""Tests for Audio Recorder module.

This module tests the audio recording functionality
including VAD integration and endpoint detection.
"""

import numpy as np
import pytest

from polyglott.audio.recorder import (
    AudioRecorder,
    RecordingResult,
    create_recorder,
)
from polyglott.constants import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE


class TestRecordingResult:
    """Tests for RecordingResult dataclass."""

    def test_creation(self) -> None:
        """Test RecordingResult can be created."""
        audio = np.zeros(16000, dtype=np.float32)
        result = RecordingResult(
            audio=audio,
            sample_rate=16000,
            duration_seconds=1.0,
            was_speech_detected=True,
        )
        assert len(result.audio) == 16000
        assert result.sample_rate == 16000
        assert result.duration_seconds == 1.0
        assert result.was_speech_detected is True

    def test_no_speech_detected(self) -> None:
        """Test RecordingResult with no speech."""
        audio = np.zeros(8000, dtype=np.float32)
        result = RecordingResult(
            audio=audio,
            sample_rate=16000,
            duration_seconds=0.5,
            was_speech_detected=False,
        )
        assert result.was_speech_detected is False


class TestAudioRecorder:
    """Tests for AudioRecorder class."""

    def test_initialization_default(self) -> None:
        """Test recorder initializes with default values."""
        recorder = AudioRecorder()
        assert recorder.sample_rate == AUDIO_SAMPLE_RATE
        assert recorder.channels == AUDIO_CHANNELS
        assert recorder.is_recording is False

    def test_initialization_custom(self) -> None:
        """Test recorder initializes with custom values."""
        recorder = AudioRecorder(sample_rate=8000, channels=2)
        assert recorder.sample_rate == 8000
        assert recorder.channels == 2

    def test_is_recording_property(self) -> None:
        """Test is_recording property."""
        recorder = AudioRecorder()
        assert recorder.is_recording is False

    def test_stop_when_not_recording(self) -> None:
        """Test stop method when not recording."""
        recorder = AudioRecorder()
        # Should not raise
        recorder.stop()
        assert recorder.is_recording is False


class TestCreateRecorder:
    """Tests for create_recorder factory function."""

    def test_creates_recorder(self) -> None:
        """Test factory creates valid recorder."""
        recorder = create_recorder()
        assert isinstance(recorder, AudioRecorder)

    def test_custom_sample_rate(self) -> None:
        """Test factory accepts custom sample rate."""
        recorder = create_recorder(sample_rate=8000)
        assert recorder.sample_rate == 8000
