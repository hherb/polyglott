"""Configuration management for Polyglott.

This module provides configuration loading from environment variables
and .env files, making it easy to customize the application.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from polyglott.constants import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_LLM_MODEL,
    DEFAULT_VAD_BACKEND,
    FOLLOWUP_TIER1_TIMEOUT,
    FOLLOWUP_TIER2_TIMEOUT,
    FOLLOWUP_TIER3_TIMEOUT,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    MAX_CONVERSATION_HISTORY,
    MAX_LLM_RESPONSE_TOKENS,
    MOONSHINE_MODEL_SIZE,
    SILENCE_THRESHOLD_SECONDS,
    TTS_DEFAULT_SPEED,
    VAD_SPEECH_THRESHOLD,
)


def _load_dotenv() -> None:
    """Load environment variables from .env file if it exists."""
    # Try multiple locations for .env file
    env_paths = [
        Path.cwd() / ".env",
        Path.home() / ".polyglott" / ".env",
        Path(__file__).parent.parent.parent / ".env",
    ]

    for env_path in env_paths:
        if env_path.exists():
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, _, value = line.partition("=")
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if key and key not in os.environ:
                                os.environ[key] = value
            except Exception:
                pass  # Ignore errors reading .env
            break  # Only load from first found .env


# Load .env on module import
_load_dotenv()


def _get_env_str(key: str, default: str) -> str:
    """Get string from environment."""
    return os.environ.get(key, default)


def _get_env_float(key: str, default: float) -> float:
    """Get float from environment."""
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _get_env_int(key: str, default: int) -> int:
    """Get int from environment."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    """Get bool from environment."""
    value = os.environ.get(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


@dataclass
class AudioConfig:
    """Audio processing configuration."""

    sample_rate: int = field(
        default_factory=lambda: _get_env_int("POLYGLOTT_AUDIO_SAMPLE_RATE", AUDIO_SAMPLE_RATE)
    )
    silence_timeout: float = field(
        default_factory=lambda: _get_env_float("POLYGLOTT_SILENCE_TIMEOUT", SILENCE_THRESHOLD_SECONDS)
    )


@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""

    backend: str = field(
        default_factory=lambda: _get_env_str("POLYGLOTT_VAD_BACKEND", DEFAULT_VAD_BACKEND)
    )
    threshold: float = field(
        default_factory=lambda: _get_env_float("POLYGLOTT_VAD_THRESHOLD", VAD_SPEECH_THRESHOLD)
    )


@dataclass
class STTConfig:
    """Speech-to-Text configuration."""

    backend: str = field(
        default_factory=lambda: _get_env_str("POLYGLOTT_STT_BACKEND", "moonshine")
    )
    model_size: str = field(
        default_factory=lambda: _get_env_str("POLYGLOTT_STT_MODEL_SIZE", MOONSHINE_MODEL_SIZE)
    )


@dataclass
class TTSConfig:
    """Text-to-Speech configuration."""

    backend: str = field(
        default_factory=lambda: _get_env_str("POLYGLOTT_TTS_BACKEND", "kokoro")
    )
    speed: float = field(
        default_factory=lambda: _get_env_float("POLYGLOTT_TTS_SPEED", TTS_DEFAULT_SPEED)
    )


@dataclass
class LLMConfig:
    """Language Model configuration."""

    model: str = field(
        default_factory=lambda: _get_env_str("POLYGLOTT_LLM_MODEL", DEFAULT_LLM_MODEL)
    )
    temperature: float = field(
        default_factory=lambda: _get_env_float("POLYGLOTT_LLM_TEMPERATURE", LLM_TEMPERATURE)
    )
    max_tokens: int = field(
        default_factory=lambda: _get_env_int("POLYGLOTT_LLM_MAX_TOKENS", MAX_LLM_RESPONSE_TOKENS)
    )
    timeout: float = field(
        default_factory=lambda: _get_env_float("POLYGLOTT_LLM_TIMEOUT", LLM_TIMEOUT_SECONDS)
    )
    max_history: int = field(
        default_factory=lambda: _get_env_int("POLYGLOTT_LLM_MAX_HISTORY", MAX_CONVERSATION_HISTORY)
    )


@dataclass
class ConversationConfig:
    """Conversation behavior configuration."""

    enable_barge_in: bool = field(
        default_factory=lambda: _get_env_bool("POLYGLOTT_ENABLE_BARGE_IN", True)
    )
    enable_followups: bool = field(
        default_factory=lambda: _get_env_bool("POLYGLOTT_ENABLE_FOLLOWUPS", True)
    )
    followup_tier1_timeout: float = field(
        default_factory=lambda: _get_env_float("POLYGLOTT_FOLLOWUP_TIER1_TIMEOUT", FOLLOWUP_TIER1_TIMEOUT)
    )
    followup_tier2_timeout: float = field(
        default_factory=lambda: _get_env_float("POLYGLOTT_FOLLOWUP_TIER2_TIMEOUT", FOLLOWUP_TIER2_TIMEOUT)
    )
    followup_tier3_timeout: float = field(
        default_factory=lambda: _get_env_float("POLYGLOTT_FOLLOWUP_TIER3_TIMEOUT", FOLLOWUP_TIER3_TIMEOUT)
    )


@dataclass
class PolyglottConfig:
    """Complete application configuration.

    All settings can be overridden via environment variables or .env file.

    Example .env file:
        POLYGLOTT_LLM_MODEL=qwen2.5:7b
        POLYGLOTT_TTS_SPEED=0.85
        POLYGLOTT_ENABLE_BARGE_IN=true
        POLYGLOTT_ENABLE_FOLLOWUPS=true
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)


# Global configuration instance
_config: Optional[PolyglottConfig] = None


def get_config() -> PolyglottConfig:
    """Get the global configuration instance.

    Returns:
        PolyglottConfig with all settings.
    """
    global _config
    if _config is None:
        _config = PolyglottConfig()
    return _config


def reload_config() -> PolyglottConfig:
    """Reload configuration from environment.

    Useful after changing environment variables.

    Returns:
        New PolyglottConfig instance.
    """
    global _config
    _load_dotenv()
    _config = PolyglottConfig()
    return _config


# Environment variable documentation
ENV_VARS = """
# Polyglott Configuration Environment Variables
# Copy this to .env and customize as needed

# Audio Configuration
POLYGLOTT_AUDIO_SAMPLE_RATE=16000
POLYGLOTT_SILENCE_TIMEOUT=1.5

# VAD Configuration
POLYGLOTT_VAD_BACKEND=silero  # Options: silero, ten_vad
POLYGLOTT_VAD_THRESHOLD=0.5

# STT Configuration
POLYGLOTT_STT_BACKEND=moonshine  # Options: moonshine, whisper_mlx
POLYGLOTT_STT_MODEL_SIZE=base  # Options: tiny, base

# TTS Configuration
POLYGLOTT_TTS_BACKEND=kokoro  # Options: kokoro, piper
POLYGLOTT_TTS_SPEED=0.9

# LLM Configuration
POLYGLOTT_LLM_MODEL=qwen2.5:7b
POLYGLOTT_LLM_TEMPERATURE=0.7
POLYGLOTT_LLM_MAX_TOKENS=256
POLYGLOTT_LLM_TIMEOUT=30.0
POLYGLOTT_LLM_MAX_HISTORY=20

# Conversation Features
POLYGLOTT_ENABLE_BARGE_IN=true
POLYGLOTT_ENABLE_FOLLOWUPS=true
POLYGLOTT_FOLLOWUP_TIER1_TIMEOUT=10.0
POLYGLOTT_FOLLOWUP_TIER2_TIMEOUT=20.0
POLYGLOTT_FOLLOWUP_TIER3_TIMEOUT=30.0
"""
