"""Pytest configuration and shared fixtures.

This module provides common test fixtures and configuration
used across all test modules.
"""

import numpy as np
import pytest

from polyglott.constants import AUDIO_SAMPLE_RATE, VAD_CHUNK_SAMPLES


@pytest.fixture
def sample_rate() -> int:
    """Provide standard sample rate for tests.

    Returns:
        Sample rate in Hz.
    """
    return AUDIO_SAMPLE_RATE


@pytest.fixture
def silence_chunk() -> np.ndarray:
    """Provide a chunk of silence audio.

    Returns:
        Numpy array of zeros.
    """
    return np.zeros(VAD_CHUNK_SAMPLES, dtype=np.float32)


@pytest.fixture
def speech_chunk() -> np.ndarray:
    """Provide a chunk of simulated speech audio.

    Returns:
        Numpy array with sine wave.
    """
    # Generate a simple sine wave at 440Hz
    t = np.linspace(0, VAD_CHUNK_SAMPLES / AUDIO_SAMPLE_RATE, VAD_CHUNK_SAMPLES)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


@pytest.fixture
def sample_audio_1s() -> np.ndarray:
    """Provide 1 second of sample audio.

    Returns:
        Numpy array with 1 second of audio.
    """
    # Mix of frequencies to simulate speech-like audio
    t = np.linspace(0, 1.0, AUDIO_SAMPLE_RATE)
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.1 * np.sin(2 * np.pi * 800 * t)
    )
    return audio.astype(np.float32)


@pytest.fixture
def sample_text_en() -> str:
    """Provide sample English text for testing.

    Returns:
        English text string.
    """
    return "Hello, how are you today?"


@pytest.fixture
def sample_text_es() -> str:
    """Provide sample Spanish text for testing.

    Returns:
        Spanish text string.
    """
    return "Hola, ¿cómo estás hoy?"


@pytest.fixture
def sample_text_de() -> str:
    """Provide sample German text for testing.

    Returns:
        German text string.
    """
    return "Hallo, wie geht es dir heute?"
