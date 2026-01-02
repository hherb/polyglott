"""Polyglott - A LLM-based conversational multi-lingual language tutor.

This package provides a complete voice-based language tutoring system
that runs entirely offline on macOS with Apple Silicon.

Modules:
    vad: Voice Activity Detection using Silero VAD
    stt: Speech-to-Text using Moonshine/Whisper
    llm: Language Model integration via Ollama
    tts: Text-to-Speech using Kokoro/Piper
    audio: Audio I/O and pipeline management
    conversation: Conversation flow and tutoring logic
"""

import warnings

# Suppress noisy warnings from TTS/ML dependencies before they're imported
warnings.filterwarnings("ignore", message=".*dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm is deprecated.*")
warnings.filterwarnings("ignore", message=".*Defaulting repo_id.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

__version__ = "0.1.0"
__author__ = "Polyglott Team"
