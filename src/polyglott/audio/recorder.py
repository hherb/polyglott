"""Audio recording with voice activity detection.

This module provides real-time audio recording that automatically
detects when the user starts and stops speaking.
"""

from collections import deque
from dataclasses import dataclass
from threading import Event, Thread
from typing import Callable, Optional

import numpy as np

from polyglott.constants import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    MAX_RECORDING_DURATION_SECONDS,
    MIN_SPEECH_DURATION_SECONDS,
    SILENCE_THRESHOLD_SECONDS,
    VAD_CHUNK_SAMPLES,
)
from polyglott.vad import VoiceActivityDetector
from polyglott.vad.detector import SpeechState


@dataclass
class RecordingResult:
    """Result from audio recording.

    Attributes:
        audio: Recorded audio as numpy array.
        sample_rate: Audio sample rate in Hz.
        duration_seconds: Duration of recording.
        was_speech_detected: Whether speech was detected.
    """

    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    was_speech_detected: bool


class AudioRecorder:
    """Real-time audio recorder with VAD-based endpoint detection.

    This class records audio from the microphone and automatically
    stops when the user finishes speaking.

    Example:
        >>> recorder = AudioRecorder()
        >>> print("Speak now...")
        >>> result = recorder.record_utterance()
        >>> print(f"Recorded {result.duration_seconds:.1f}s of audio")
    """

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        vad_detector: Optional[VoiceActivityDetector] = None,
    ) -> None:
        """Initialize the audio recorder.

        Args:
            sample_rate: Audio sample rate in Hz.
            channels: Number of audio channels (1 for mono).
            vad_detector: VAD detector instance, or None to create one.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.vad = vad_detector or VoiceActivityDetector(sample_rate=sample_rate)

        self._sounddevice = None
        self._is_recording = False
        self._stop_event = Event()

    def _ensure_sounddevice(self) -> None:
        """Ensure sounddevice is imported."""
        if self._sounddevice is None:
            try:
                import sounddevice as sd

                self._sounddevice = sd
            except ImportError as e:
                raise ImportError(
                    "sounddevice not installed. Install with: uv add sounddevice"
                ) from e

    def record_utterance(
        self,
        max_duration: float = MAX_RECORDING_DURATION_SECONDS,
        silence_timeout: float = SILENCE_THRESHOLD_SECONDS,
        min_duration: float = MIN_SPEECH_DURATION_SECONDS,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
    ) -> RecordingResult:
        """Record a single utterance with automatic endpoint detection.

        Records audio until the user stops speaking (detected by VAD)
        or maximum duration is reached.

        Args:
            max_duration: Maximum recording duration in seconds.
            silence_timeout: Seconds of silence to end recording.
            min_duration: Minimum speech duration to be valid.
            on_speech_start: Callback when speech starts.
            on_speech_end: Callback when speech ends.

        Returns:
            RecordingResult with recorded audio.
        """
        self._ensure_sounddevice()
        self.vad.reset()

        audio_buffer: list[np.ndarray] = []
        speech_buffer: list[np.ndarray] = []
        speech_detected = False
        speech_started = False

        chunk_samples = VAD_CHUNK_SAMPLES
        max_chunks = int(max_duration * self.sample_rate / chunk_samples)

        self._is_recording = True
        self._stop_event.clear()

        try:
            with self._sounddevice.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                blocksize=chunk_samples,
            ) as stream:
                for _ in range(max_chunks):
                    if self._stop_event.is_set():
                        break

                    # Read audio chunk
                    chunk, overflowed = stream.read(chunk_samples)
                    if overflowed:
                        continue

                    # Flatten to mono if needed
                    if chunk.ndim > 1:
                        chunk = chunk[:, 0]

                    # Process with VAD
                    vad_result = self.vad.process_chunk(chunk)

                    # Handle state transitions
                    if vad_result.state == SpeechState.SPEECH_START:
                        speech_detected = True
                        speech_started = True
                        if on_speech_start:
                            on_speech_start()
                        # Include pre-speech buffer
                        speech_buffer.extend(audio_buffer[-10:])  # ~300ms lookback
                        speech_buffer.append(chunk)

                    elif vad_result.state == SpeechState.SPEAKING:
                        if speech_started:
                            speech_buffer.append(chunk)

                    elif vad_result.state == SpeechState.SPEECH_END:
                        if speech_started:
                            speech_buffer.append(chunk)
                            if on_speech_end:
                                on_speech_end()
                            break

                    else:  # SILENCE
                        audio_buffer.append(chunk)
                        # Keep buffer limited
                        if len(audio_buffer) > 20:
                            audio_buffer.pop(0)

        finally:
            self._is_recording = False

        # Combine speech buffer
        if speech_buffer:
            audio = np.concatenate(speech_buffer)
        else:
            audio = np.zeros(int(self.sample_rate * 0.1), dtype=np.float32)

        duration = len(audio) / self.sample_rate

        # Check minimum duration
        valid_speech = speech_detected and duration >= min_duration

        return RecordingResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            was_speech_detected=valid_speech,
        )

    def record_fixed_duration(self, duration: float) -> RecordingResult:
        """Record audio for a fixed duration.

        Args:
            duration: Recording duration in seconds.

        Returns:
            RecordingResult with recorded audio.
        """
        self._ensure_sounddevice()

        samples = int(duration * self.sample_rate)
        audio = self._sounddevice.rec(
            samples,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
        )
        self._sounddevice.wait()

        # Flatten to mono
        if audio.ndim > 1:
            audio = audio[:, 0]

        return RecordingResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration_seconds=duration,
            was_speech_detected=True,
        )

    def stop(self) -> None:
        """Stop current recording."""
        self._stop_event.set()

    @property
    def is_recording(self) -> bool:
        """Check if currently recording.

        Returns:
            True if recording is in progress.
        """
        return self._is_recording


def create_recorder(
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> AudioRecorder:
    """Factory function to create an audio recorder.

    Args:
        sample_rate: Audio sample rate in Hz.

    Returns:
        Configured AudioRecorder instance.
    """
    return AudioRecorder(sample_rate=sample_rate)
