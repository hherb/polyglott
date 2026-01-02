"""Speech-to-Text module.

This module provides speech recognition using Moonshine ASR
with fallback to Whisper via mlx-whisper for Apple Silicon.
"""

from polyglott.stt.transcriber import SpeechTranscriber

__all__ = ["SpeechTranscriber"]
