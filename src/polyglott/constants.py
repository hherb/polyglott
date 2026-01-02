"""Global constants for the Polyglott language tutor.

This module defines all magic numbers and configuration values
used throughout the application. Following Golden Rule #3:
No magic numbers - always use helpfully named constants.
"""

from enum import Enum
from typing import Final

# =============================================================================
# Audio Configuration
# =============================================================================

# Standard sample rate for speech recognition (Hz)
AUDIO_SAMPLE_RATE: Final[int] = 16000

# Bit depth for audio samples
AUDIO_BIT_DEPTH: Final[int] = 16

# Number of audio channels (mono for speech)
AUDIO_CHANNELS: Final[int] = 1

# Duration of audio chunks for VAD processing (milliseconds)
# Note: Silero VAD v6+ requires minimum 32ms chunks (512 samples at 16kHz)
VAD_CHUNK_DURATION_MS: Final[int] = 32

# Number of samples per VAD chunk
VAD_CHUNK_SAMPLES: Final[int] = int(AUDIO_SAMPLE_RATE * VAD_CHUNK_DURATION_MS / 1000)

# Maximum audio recording duration before auto-stop (seconds)
MAX_RECORDING_DURATION_SECONDS: Final[float] = 30.0

# Silence duration to consider speech ended (seconds)
SILENCE_THRESHOLD_SECONDS: Final[float] = 1.5

# Minimum speech duration to process (seconds)
MIN_SPEECH_DURATION_SECONDS: Final[float] = 0.3

# Follow-up timeouts for AI-initiated prompts (seconds)
FOLLOWUP_TIER1_TIMEOUT: Final[float] = 10.0  # Gentle check-in
FOLLOWUP_TIER2_TIMEOUT: Final[float] = 20.0  # Encouraging prompt
FOLLOWUP_TIER3_TIMEOUT: Final[float] = 30.0  # Re-engagement activity


# =============================================================================
# VAD (Voice Activity Detection) Configuration
# =============================================================================

# Speech probability threshold (0.0 to 1.0) - used by all VAD backends
VAD_SPEECH_THRESHOLD: Final[float] = 0.5

# Number of consecutive speech frames to confirm speech start
VAD_SPEECH_PAD_FRAMES: Final[int] = 3

# Number of consecutive silence frames to confirm speech end
VAD_SILENCE_PAD_FRAMES: Final[int] = 10


class VADBackendType(str, Enum):
    """Available VAD backend implementations."""

    SILERO = "silero"  # Silero VAD (PyTorch-based, 32ms chunks)
    TEN_VAD = "ten_vad"  # TEN VAD (ONNX-based, 10/16ms chunks)


# Default VAD backend to use
DEFAULT_VAD_BACKEND: Final[str] = VADBackendType.SILERO

# TEN VAD specific configuration
TEN_VAD_HOP_SIZE: Final[int] = 256  # 16ms at 16kHz (options: 160=10ms, 256=16ms)


# =============================================================================
# STT (Speech-to-Text) Configuration
# =============================================================================

# Default Moonshine model size
MOONSHINE_MODEL_SIZE: Final[str] = "base"

# Default Whisper model size
WHISPER_MODEL_SIZE: Final[str] = "base"

# Maximum audio length for transcription (seconds)
MAX_TRANSCRIPTION_AUDIO_SECONDS: Final[float] = 30.0


# =============================================================================
# LLM Configuration
# =============================================================================

# Default Ollama model for conversation
DEFAULT_LLM_MODEL: Final[str] = "gpt-oss:20b"

# Maximum tokens in LLM response
MAX_LLM_RESPONSE_TOKENS: Final[int] = 256

# LLM temperature for response generation (lower = more focused)
LLM_TEMPERATURE: Final[float] = 0.7

# LLM request timeout (seconds)
LLM_TIMEOUT_SECONDS: Final[float] = 30.0

# Maximum conversation history to maintain (number of turns)
MAX_CONVERSATION_HISTORY: Final[int] = 20


# =============================================================================
# TTS (Text-to-Speech) Configuration
# =============================================================================

# TTS output sample rate (Hz)
TTS_SAMPLE_RATE: Final[int] = 24000

# Default speaking speed (1.0 = normal)
TTS_DEFAULT_SPEED: Final[float] = 0.9

# Speed adjustment for children (slightly slower)
TTS_CHILDREN_SPEED: Final[float] = 0.85


# =============================================================================
# Supported Languages
# =============================================================================

class StudentLanguage(str, Enum):
    """Languages the student can use as their native language."""

    ENGLISH = "en"
    GERMAN = "de"


class TargetLanguage(str, Enum):
    """Languages available for learning."""

    ENGLISH = "en"
    GERMAN = "de"
    SPANISH = "es"
    JAPANESE = "ja"
    MANDARIN = "zh"


# Language display names for UI
LANGUAGE_NAMES: Final[dict[str, str]] = {
    "en": "English",
    "de": "German / Deutsch",
    "es": "Spanish / Español",
    "ja": "Japanese / 日本語",
    "zh": "Mandarin / 中文",
}

# ISO language codes for TTS engines
TTS_LANGUAGE_CODES: Final[dict[str, str]] = {
    "en": "en-us",
    "de": "de-de",
    "es": "es-es",
    "ja": "ja-jp",
    "zh": "zh-cn",
}


# =============================================================================
# Application Configuration
# =============================================================================

# Application name
APP_NAME: Final[str] = "Polyglott"

# Application version (should match pyproject.toml)
APP_VERSION: Final[str] = "0.1.0"

# Default age group for difficulty adjustment
class AgeGroup(str, Enum):
    """Age groups for difficulty adjustment."""

    PRESCHOOL = "preschool"  # 3-5 years
    EARLY_PRIMARY = "early_primary"  # 6-8 years
    LATE_PRIMARY = "late_primary"  # 9-12 years


# Difficulty settings per age group
AGE_GROUP_SETTINGS: Final[dict[str, dict[str, int | float]]] = {
    AgeGroup.PRESCHOOL: {
        "max_words_per_sentence": 5,
        "vocabulary_level": 1,
        "tts_speed": 0.8,
    },
    AgeGroup.EARLY_PRIMARY: {
        "max_words_per_sentence": 10,
        "vocabulary_level": 2,
        "tts_speed": 0.85,
    },
    AgeGroup.LATE_PRIMARY: {
        "max_words_per_sentence": 15,
        "vocabulary_level": 3,
        "tts_speed": 0.9,
    },
}
