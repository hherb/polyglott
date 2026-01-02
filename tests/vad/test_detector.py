"""Tests for Voice Activity Detection module.

This module tests the VAD detector functionality including
state transitions, speech probability calculation, and
backend selection.
"""

import numpy as np
import pytest

from polyglott.constants import AUDIO_SAMPLE_RATE, VAD_CHUNK_SAMPLES, VADBackendType
from polyglott.vad import (
    BaseVADDetector,
    SpeechState,
    TEN_VAD_AVAILABLE,
    TenVadDetector,
    TenVadNotInstalledError,
    VADResult,
    VoiceActivityDetector,
    create_vad_detector,
    get_available_backends,
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
    """Tests for VoiceActivityDetector (Silero VAD) class."""

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

        assert samples_16k == 512  # 32ms at 16kHz (min required by Silero)
        assert samples_8k == 256   # 32ms at 8kHz

    def test_inherits_base_detector(self) -> None:
        """Test VoiceActivityDetector inherits from BaseVADDetector."""
        detector = VoiceActivityDetector()
        assert isinstance(detector, BaseVADDetector)


class TestTenVadDetector:
    """Tests for TenVadDetector class."""

    def test_not_installed_error(self) -> None:
        """Test TenVadNotInstalledError has helpful message."""
        error = TenVadNotInstalledError()
        assert "TEN VAD is not installed" in str(error)
        assert "git clone" in str(error)

    @pytest.mark.skipif(
        not TEN_VAD_AVAILABLE,
        reason="TEN VAD not installed"
    )
    def test_initialization_default(self) -> None:
        """Test TEN VAD detector initializes with default values."""
        detector = TenVadDetector()
        assert detector.sample_rate == AUDIO_SAMPLE_RATE
        assert detector.threshold == 0.5
        assert detector.hop_size == 256
        assert detector.is_speaking is False

    @pytest.mark.skipif(
        not TEN_VAD_AVAILABLE,
        reason="TEN VAD not installed"
    )
    def test_initialization_custom_hop_size(self) -> None:
        """Test TEN VAD detector accepts custom hop size."""
        detector = TenVadDetector(hop_size=160)
        assert detector.hop_size == 160

    @pytest.mark.skipif(
        TEN_VAD_AVAILABLE,
        reason="TEN VAD is installed"
    )
    def test_raises_when_not_installed(self) -> None:
        """Test TenVadDetector raises error when not installed."""
        with pytest.raises(TenVadNotInstalledError):
            TenVadDetector()

    def test_invalid_sample_rate(self) -> None:
        """Test TEN VAD rejects non-16kHz sample rates."""
        if not TEN_VAD_AVAILABLE:
            pytest.skip("TEN VAD not installed")

        with pytest.raises(ValueError, match="only supports 16kHz"):
            TenVadDetector(sample_rate=8000)

    def test_invalid_hop_size(self) -> None:
        """Test TEN VAD rejects invalid hop sizes."""
        if not TEN_VAD_AVAILABLE:
            pytest.skip("TEN VAD not installed")

        with pytest.raises(ValueError, match="hop_size must be"):
            TenVadDetector(hop_size=512)

    @pytest.mark.skipif(
        not TEN_VAD_AVAILABLE,
        reason="TEN VAD not installed"
    )
    def test_chunk_duration_ms(self) -> None:
        """Test chunk duration calculation."""
        detector_256 = TenVadDetector(hop_size=256)
        detector_160 = TenVadDetector(hop_size=160)

        assert detector_256.chunk_duration_ms == 16.0  # 256/16000 * 1000
        assert detector_160.chunk_duration_ms == 10.0  # 160/16000 * 1000

    @pytest.mark.skipif(
        not TEN_VAD_AVAILABLE,
        reason="TEN VAD not installed"
    )
    def test_inherits_base_detector(self) -> None:
        """Test TenVadDetector inherits from BaseVADDetector."""
        detector = TenVadDetector()
        assert isinstance(detector, BaseVADDetector)


class TestGetAvailableBackends:
    """Tests for get_available_backends function."""

    def test_silero_always_available(self) -> None:
        """Test Silero VAD is always in available backends."""
        backends = get_available_backends()
        assert "silero" in backends

    def test_returns_list(self) -> None:
        """Test function returns a list."""
        backends = get_available_backends()
        assert isinstance(backends, list)

    def test_ten_vad_conditional(self) -> None:
        """Test TEN VAD availability matches import status."""
        backends = get_available_backends()
        if TEN_VAD_AVAILABLE:
            assert "ten_vad" in backends
        else:
            assert "ten_vad" not in backends


class TestCreateVadDetector:
    """Tests for create_vad_detector factory function."""

    def test_creates_silero_by_default(self) -> None:
        """Test factory creates Silero detector by default."""
        detector = create_vad_detector()
        assert isinstance(detector, VoiceActivityDetector)

    def test_creates_silero_explicitly(self) -> None:
        """Test factory creates Silero detector when specified."""
        detector = create_vad_detector(backend="silero")
        assert isinstance(detector, VoiceActivityDetector)

    def test_custom_threshold(self) -> None:
        """Test factory accepts custom threshold."""
        detector = create_vad_detector(threshold=0.8)
        assert detector.threshold == 0.8

    def test_using_enum(self) -> None:
        """Test factory accepts VADBackendType enum."""
        detector = create_vad_detector(backend=VADBackendType.SILERO)
        assert isinstance(detector, VoiceActivityDetector)

    def test_invalid_backend(self) -> None:
        """Test factory raises error for unknown backend."""
        with pytest.raises(ValueError, match="Unknown VAD backend"):
            create_vad_detector(backend="unknown_vad")

    @pytest.mark.skipif(
        not TEN_VAD_AVAILABLE,
        reason="TEN VAD not installed"
    )
    def test_creates_ten_vad(self) -> None:
        """Test factory creates TEN VAD detector when specified."""
        detector = create_vad_detector(backend="ten_vad")
        assert isinstance(detector, TenVadDetector)

    @pytest.mark.skipif(
        not TEN_VAD_AVAILABLE,
        reason="TEN VAD not installed"
    )
    def test_ten_vad_with_hop_size(self) -> None:
        """Test factory passes hop_size to TEN VAD."""
        detector = create_vad_detector(backend="ten_vad", hop_size=160)
        assert isinstance(detector, TenVadDetector)
        assert detector.hop_size == 160

    @pytest.mark.skipif(
        TEN_VAD_AVAILABLE,
        reason="TEN VAD is installed"
    )
    def test_ten_vad_raises_when_not_installed(self) -> None:
        """Test factory raises error when TEN VAD requested but not installed."""
        with pytest.raises(TenVadNotInstalledError):
            create_vad_detector(backend="ten_vad")
