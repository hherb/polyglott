# Polyglott Architecture

This document describes the architecture of the Polyglott language tutor application.

## Overview

Polyglott is a voice-based language tutoring application that runs entirely offline. It uses a pipeline architecture to process audio through several stages:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Audio Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │   VAD    │───▶│   STT    │───▶│   LLM    │───▶│   TTS    │ │
│   │  Silero  │    │Moonshine │    │  Qwen    │    │ Kokoro/  │ │
│   │          │    │          │    │ (Ollama) │    │  Piper   │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│        ▲                                               │       │
│        │                                               ▼       │
│   ┌──────────┐                                    ┌──────────┐ │
│   │ Recorder │                                    │  Player  │ │
│   └──────────┘                                    └──────────┘ │
│        ▲                                               │       │
│        │              Audio I/O                        ▼       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                      sounddevice                         │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
└──────────────────────────────┼──────────────────────────────────┘
                               │
                        ┌──────┴──────┐
                        │  Hardware   │
                        │  (Mic/Spk)  │
                        └─────────────┘
```

## Module Structure

```
src/polyglott/
├── __init__.py          # Package root
├── constants.py         # Global constants and configuration
├── cli.py               # Command-line interface
├── vad/                 # Voice Activity Detection
│   ├── __init__.py
│   └── detector.py      # Silero VAD wrapper
├── stt/                 # Speech-to-Text
│   ├── __init__.py
│   └── transcriber.py   # Moonshine/Whisper transcription
├── llm/                 # Language Model
│   ├── __init__.py
│   ├── prompts.py       # System prompts and templates
│   └── tutor.py         # Ollama integration
├── tts/                 # Text-to-Speech
│   ├── __init__.py
│   └── synthesizer.py   # Kokoro/Piper synthesis
├── audio/               # Audio I/O
│   ├── __init__.py
│   ├── recorder.py      # Microphone input
│   ├── player.py        # Speaker output
│   └── pipeline.py      # Pipeline orchestration
└── conversation/        # Conversation management
    ├── __init__.py
    ├── session.py       # Session tracking
    └── manager.py       # High-level conversation flow
```

## Component Details

### VAD (Voice Activity Detection)

**Module:** `polyglott.vad`
**Technology:** Silero VAD

The VAD module detects when the user starts and stops speaking:

- Processes audio in 30ms chunks
- Returns speech probability (0.0-1.0)
- Tracks state transitions (silence → speaking → silence)
- Uses padding frames to avoid false triggers

Key classes:
- `VoiceActivityDetector`: Main detector class
- `SpeechState`: Enum for state machine
- `VADResult`: Result from processing a chunk

### STT (Speech-to-Text)

**Module:** `polyglott.stt`
**Technologies:** Moonshine ASR, mlx-whisper

The STT module transcribes spoken audio to text:

- Primary: Moonshine (5-15x faster than Whisper)
- Fallback: Whisper via mlx-whisper (macOS)
- Language-specific model selection
- Supports streaming audio input

Key classes:
- `SpeechTranscriber`: Unified transcriber interface
- `MoonshineTranscriber`: Moonshine backend
- `WhisperMLXTranscriber`: Whisper backend

### LLM (Language Model)

**Module:** `polyglott.llm`
**Technology:** Ollama (Qwen 2.5)

The LLM module generates tutor responses:

- Runs locally via Ollama
- Uses specialized prompts for language tutoring
- Age-appropriate vocabulary and pacing
- Maintains conversation history

Key classes:
- `LanguageTutor`: Main tutor class
- `TutorConfig`: Configuration dataclass
- `Message`: Conversation message

### TTS (Text-to-Speech)

**Module:** `polyglott.tts`
**Technologies:** Kokoro TTS, Piper TTS

The TTS module synthesizes spoken responses:

- Kokoro: English, Spanish, Japanese, Mandarin
- Piper: German (Kokoro doesn't support German)
- Slower speech speed for children
- 24kHz high-quality output

Key classes:
- `SpeechSynthesizer`: Unified synthesizer interface
- `KokoroSynthesizer`: Kokoro backend
- `PiperSynthesizer`: Piper backend

### Audio Pipeline

**Module:** `polyglott.audio`
**Technology:** sounddevice

The audio module orchestrates the full pipeline:

- `AudioRecorder`: Records with VAD-based endpoint detection
- `AudioPlayer`: Plays synthesized audio
- `AudioPipeline`: Coordinates all components

### Conversation Manager

**Module:** `polyglott.conversation`

The conversation module manages sessions:

- `ConversationSession`: Tracks session metadata
- `ConversationManager`: High-level conversation flow

## Data Flow

1. **Recording Phase**
   - `AudioRecorder` captures microphone input
   - `VoiceActivityDetector` processes chunks
   - Recording stops when speech ends

2. **Transcription Phase**
   - `SpeechTranscriber` converts audio to text
   - Language-specific model selected

3. **Generation Phase**
   - `LanguageTutor` generates response
   - Ollama processes with conversation context

4. **Synthesis Phase**
   - `SpeechSynthesizer` converts text to audio
   - Backend selected based on language

5. **Playback Phase**
   - `AudioPlayer` plays the response
   - Pipeline returns to recording phase

## Configuration

All configuration is centralized in `constants.py`:

- Audio parameters (sample rate, chunk size)
- VAD thresholds
- LLM settings (model, temperature)
- TTS settings (speed, sample rate)
- Language mappings

## Extending Polyglott

### Adding a New Language

1. Add to `TargetLanguage` enum in `constants.py`
2. Add language name to `LANGUAGE_NAMES`
3. Add TTS mapping in `synthesizer.py`
4. Add STT model mapping in `transcriber.py`
5. Add greetings to `prompts.py`

### Adding a New TTS Backend

1. Create a new synthesizer class implementing `SynthesizerProtocol`
2. Add backend to `TTSBackend` enum
3. Update `SpeechSynthesizer._get_backend()`

### Adding a New STT Backend

1. Create a new transcriber class implementing `TranscriberProtocol`
2. Add backend to `TranscriberBackend` enum
3. Update `SpeechTranscriber._detect_best_backend()`
