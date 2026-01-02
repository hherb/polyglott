"""Tests for Audio Player module.

This module tests the audio playback functionality.
"""

import numpy as np
import pytest

from polyglott.audio.player import AudioPlayer, create_player
from polyglott.constants import TTS_SAMPLE_RATE


class TestAudioPlayer:
    """Tests for AudioPlayer class."""

    def test_initialization(self) -> None:
        """Test player initializes correctly."""
        player = AudioPlayer()
        assert player.is_playing is False

    def test_is_playing_property(self) -> None:
        """Test is_playing property."""
        player = AudioPlayer()
        assert player.is_playing is False

    def test_stop_when_not_playing(self) -> None:
        """Test stop method when not playing."""
        player = AudioPlayer()
        # Should not raise
        player.stop()
        assert player.is_playing is False

    def test_wait_when_not_playing(self) -> None:
        """Test wait method when not playing."""
        player = AudioPlayer()
        # Should not block or raise
        player.wait()


class TestCreatePlayer:
    """Tests for create_player factory function."""

    def test_creates_player(self) -> None:
        """Test factory creates valid player."""
        player = create_player()
        assert isinstance(player, AudioPlayer)
