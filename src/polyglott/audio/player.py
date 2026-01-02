"""Audio playback for synthesized speech.

This module provides audio playback functionality for
playing synthesized speech responses with support for
barge-in interruption.
"""

from __future__ import annotations

from threading import Event, Thread
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from polyglott.constants import TTS_SAMPLE_RATE, VAD_CHUNK_SAMPLES

if TYPE_CHECKING:
    from polyglott.vad import VoiceActivityDetector


class AudioPlayer:
    """Audio player for speech playback.

    This class handles playback of synthesized speech audio
    with support for interruption.

    Example:
        >>> player = AudioPlayer()
        >>> player.play(audio_array, sample_rate=24000)
        >>> # Wait for playback to finish
        >>> player.wait()
    """

    def __init__(self) -> None:
        """Initialize the audio player."""
        self._sounddevice = None
        self._is_playing = False
        self._was_interrupted = False
        self._stop_event = Event()
        self._playback_thread: Optional[Thread] = None

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

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int = TTS_SAMPLE_RATE,
        blocking: bool = False,
    ) -> None:
        """Play audio.

        Args:
            audio: Audio samples as numpy array.
            sample_rate: Audio sample rate in Hz.
            blocking: If True, wait for playback to complete.
        """
        self._ensure_sounddevice()

        # Stop any current playback
        self.stop()

        self._stop_event.clear()
        self._is_playing = True

        if blocking:
            self._play_blocking(audio, sample_rate)
        else:
            self._playback_thread = Thread(
                target=self._play_blocking,
                args=(audio, sample_rate),
                daemon=True,
            )
            self._playback_thread.start()

    def _play_blocking(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play audio and block until complete.

        Args:
            audio: Audio samples.
            sample_rate: Sample rate in Hz.
        """
        try:
            # Ensure float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize if needed
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val

            self._sounddevice.play(audio, sample_rate)
            self._sounddevice.wait()

        except Exception as e:
            print(f"Playback error: {e}")

        finally:
            self._is_playing = False

    def play_interruptible(
        self,
        audio: np.ndarray,
        sample_rate: int = TTS_SAMPLE_RATE,
        vad_detector: Optional["VoiceActivityDetector"] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        interrupt_threshold: float = 0.6,
    ) -> bool:
        """Play audio with barge-in interruption support.

        Plays audio while monitoring the microphone for speech.
        If the user starts speaking, playback is interrupted.

        Args:
            audio: Audio samples to play.
            sample_rate: Audio sample rate in Hz.
            vad_detector: VAD detector for speech detection.
            on_interrupt: Callback when interrupted.
            interrupt_threshold: Speech probability threshold (0.0-1.0).

        Returns:
            True if playback completed, False if interrupted.
        """
        self._ensure_sounddevice()

        # Stop any current playback
        self.stop()
        self._was_interrupted = False
        self._stop_event.clear()
        self._is_playing = True

        # Prepare audio
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val

        # If no VAD provided, just play normally
        if vad_detector is None:
            self._play_blocking(audio, sample_rate)
            return True

        # Import here to avoid circular imports
        from polyglott.constants import AUDIO_SAMPLE_RATE

        # Calculate chunk size for monitoring (use 32ms chunks)
        monitor_chunk_samples = int(AUDIO_SAMPLE_RATE * 0.032)

        # Start playback in a non-blocking way
        try:
            # Create output and input streams
            output_stream = self._sounddevice.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
            )
            input_stream = self._sounddevice.InputStream(
                samplerate=AUDIO_SAMPLE_RATE,
                channels=1,
                dtype=np.float32,
                blocksize=monitor_chunk_samples,
            )

            vad_detector.reset()
            audio_pos = 0
            output_chunk_size = int(sample_rate * 0.032)  # 32ms output chunks
            # Skip first few VAD checks to avoid false positives from audio bleed
            warmup_chunks = 5
            chunks_processed = 0

            with output_stream, input_stream:
                while audio_pos < len(audio):
                    if self._stop_event.is_set():
                        self._was_interrupted = True
                        break

                    # Write audio chunk to output
                    chunk_end = min(audio_pos + output_chunk_size, len(audio))
                    output_chunk = audio[audio_pos:chunk_end]
                    output_stream.write(output_chunk.reshape(-1, 1))
                    audio_pos = chunk_end

                    # Read and check microphone input for barge-in
                    try:
                        mic_data, overflowed = input_stream.read(monitor_chunk_samples)
                        if not overflowed and len(mic_data) > 0:
                            mic_chunk = mic_data[:, 0] if mic_data.ndim > 1 else mic_data.flatten()
                            chunks_processed += 1

                            # Skip warmup period to avoid false positives
                            if chunks_processed > warmup_chunks:
                                vad_result = vad_detector.process_chunk(mic_chunk)

                                # Check for speech with higher threshold for interruption
                                if vad_result.speech_probability > interrupt_threshold:
                                    # User is speaking - interrupt!
                                    self._was_interrupted = True
                                    if on_interrupt:
                                        on_interrupt()
                                    break
                    except Exception:
                        # Ignore mic read errors
                        pass

            # Drain output buffer if not interrupted
            if not self._was_interrupted:
                output_stream.stop()

        except Exception as e:
            print(f"Interruptible playback error: {e}")
            # Fall back to regular playback
            self._play_blocking(audio, sample_rate)
            return True

        finally:
            self._is_playing = False

        return not self._was_interrupted

    def stop(self) -> None:
        """Stop current playback."""
        self._stop_event.set()
        if self._sounddevice is not None:
            try:
                self._sounddevice.stop()
            except Exception:
                pass
        self._is_playing = False

    def wait(self) -> None:
        """Wait for current playback to complete."""
        if self._playback_thread is not None:
            self._playback_thread.join()

    @property
    def is_playing(self) -> bool:
        """Check if audio is currently playing.

        Returns:
            True if playback is in progress.
        """
        return self._is_playing

    @property
    def was_interrupted(self) -> bool:
        """Check if last playback was interrupted.

        Returns:
            True if last playback was interrupted by user speech.
        """
        return self._was_interrupted


def create_player() -> AudioPlayer:
    """Factory function to create an audio player.

    Returns:
        Configured AudioPlayer instance.
    """
    return AudioPlayer()
