# Polyglott

A LLM-based conversational multi-lingual language tutor designed for children.

## Overview

Polyglott is a voice-based language learning application that runs **completely offline** on your Mac. It provides patient, encouraging language tutoring for children aged 3-12, supporting conversation practice in multiple languages.

### Features

- **Voice-based interaction**: Speak naturally and get spoken responses
- **Fully offline**: No internet required after installation
- **Child-friendly**: Age-appropriate content and patient tutoring
- **Multilingual**: Learn German, Spanish, Japanese, or Mandarin
- **Privacy-focused**: All processing happens locally on your device

### Supported Languages

| Language | Status |
|----------|--------|
| English | Native/Student |
| German | Student/Target |
| Spanish | Target |
| Japanese | Target |
| Mandarin | Target |

## Quickstart

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB RAM minimum (32GB recommended)
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai) for the LLM

### Installation

```bash
# 1. Install Ollama (if not already installed)
brew install ollama

# 2. Start Ollama and pull the language model
ollama serve &
ollama pull qwen2.5:7b

# 3. Clone the repository
git clone https://github.com/yourusername/polyglott.git
cd polyglott

# 4. Install dependencies
uv sync

# 5. (Optional) Install macOS-specific optimizations
uv sync --extra macos
```

### Running Polyglott

```bash
# Start the interactive tutor
uv run polyglott
```

You'll be guided through:
1. Selecting a language to learn
2. Choosing an age group
3. Entering the student's name
4. Selecting text or voice mode

### Text Mode Example

```
╔══════════════════════════════════════════════════════════════╗
║  Polyglott v0.1.0                                            ║
║  Your friendly language learning companion                    ║
╚══════════════════════════════════════════════════════════════╝

What language would you like to learn?
  1. English
  2. German / Deutsch
  3. Spanish / Español
  4. Japanese / 日本語
  5. Mandarin / 中文

Enter number (1-5): 3

🎯 Great! Let's learn Spanish, Emma!

🎓 Tutor: Hi there! Are you ready to learn some Spanish today? Let's have fun!

👤 Emma: Yes! How do I say hello?

🎓 Tutor: Great question! In Spanish, we say "Hola" (OH-lah)! Can you try saying it?
```

## Project Status

### Implemented

- [x] Voice Activity Detection (Silero VAD)
- [x] Speech-to-Text (Moonshine ASR)
- [x] LLM Integration (Ollama/Qwen)
- [x] Text-to-Speech (Kokoro + Piper)
- [x] Audio Pipeline
- [x] CLI Interface
- [x] Test Suite

### In Progress

- [ ] Voice mode testing
- [ ] Performance optimization
- [ ] UI improvements

### Planned

- [ ] Progress tracking
- [ ] Vocabulary games
- [ ] Pronunciation feedback
- [ ] Parent dashboard

## Architecture

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Silero  │───▶│Moonshine │───▶│  Qwen    │───▶│ Kokoro/  │
│   VAD    │    │   ASR    │    │  (LLM)   │    │  Piper   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     ▲                                               │
     │              Audio Pipeline                   ▼
┌─────────────────────────────────────────────────────────────┐
│                       sounddevice                           │
└─────────────────────────────────────────────────────────────┘
```

## Development

### Setup

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run linting
uv run ruff check src tests

# Type checking
uv run mypy src
```

### Documentation

- **User Guide**: [doc/user/getting_started.md](doc/user/getting_started.md)
- **Architecture**: [doc/developers/architecture.md](doc/developers/architecture.md)
- **Contributing**: [doc/developers/contributing.md](doc/developers/contributing.md)
- **Golden Rules**: [doc/llm/golden_rules.md](doc/llm/golden_rules.md)

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| VAD | Silero VAD | Voice activity detection |
| STT | Moonshine ASR | Speech recognition |
| LLM | Qwen 2.5 (via Ollama) | Conversation generation |
| TTS | Kokoro + Piper | Speech synthesis |
| Audio | sounddevice | Audio I/O |

## Requirements

### Hardware
- Mac with Apple Silicon (M1/M2/M3/M4)
- 16GB+ unified memory
- Microphone and speakers

### Software
- macOS 12+
- Python 3.11+
- Ollama

## License

AGPL-3.0-or-later

## Acknowledgments

- [Silero VAD](https://github.com/snakers4/silero-vad) - Voice activity detection
- [Moonshine](https://github.com/moonshine-ai/moonshine) - Speech recognition
- [Kokoro TTS](https://github.com/hexgrad/kokoro) - Text-to-speech
- [Piper](https://github.com/rhasspy/piper) - German TTS
- [Ollama](https://ollama.ai) - Local LLM runtime
- [Qwen](https://github.com/QwenLM/Qwen) - Language model
