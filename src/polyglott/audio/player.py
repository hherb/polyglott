"""Audio playback for synthesized speech.

This module provides audio playback functionality for
playing synthesized speech responses.
"""

from threading import Event, Thread
from typing import Optional

import numpy as np

from polyglott.constants import TTS_SAMPLE_RATE


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


def create_player() -> AudioPlayer:
    """Factory function to create an audio player.

    Returns:
        Configured AudioPlayer instance.
    """
    return AudioPlayer()
