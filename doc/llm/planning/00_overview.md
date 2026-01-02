# Polyglott Implementation Plan - Overview

**Document Version:** 1.0
**Last Updated:** 2026-01-02

## Executive Summary

This document outlines the complete implementation plan for Polyglott, an offline voice-based language tutor for children. The implementation is divided into 6 phases, each building upon the previous.

## Project Goals

1. **Primary Goal**: Create a patient, voice-based language tutor for children aged 3-12
2. **Technical Goal**: Run completely offline on MacBook Pro with 32GB RAM
3. **Quality Goal**: Fluent, natural conversation with minimal latency

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Polyglott System                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────┐ │
│  │   Audio     │   │   Speech    │   │    LLM      │   │   UI     │ │
│  │  Pipeline   │   │ Processing  │   │ Integration │   │  Layer   │ │
│  │             │   │             │   │             │   │          │ │
│  │ • VAD       │   │ • STT       │   │ • Ollama    │   │ • CLI    │ │
│  │ • Recorder  │──▶│ • TTS       │──▶│ • Prompts   │──▶│ • Voice  │ │
│  │ • Player    │   │ • Languages │   │ • History   │   │ • Text   │ │
│  └─────────────┘   └─────────────┘   └─────────────┘   └──────────┘ │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Core Infrastructure                         │   │
│  │  • Constants  • Configuration  • Utilities  • Error Handling  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Implementation Phases

| Phase | Name | Description | Dependencies |
|-------|------|-------------|--------------|
| 1 | Core Infrastructure | Project setup, dependencies, base utilities | None |
| 2 | Audio Pipeline | VAD, recording, playback with sounddevice | Phase 1 |
| 3 | Speech Processing | STT (Moonshine) and TTS (Kokoro/Piper) | Phase 2 |
| 4 | LLM Integration | Ollama setup, prompts, conversation flow | Phase 1 |
| 5 | User Interface | CLI, session management, user experience | Phases 3, 4 |
| 6 | Testing & Polish | Integration tests, performance tuning | All phases |

## Technology Stack

### Core Technologies

| Component | Primary | Fallback | Notes |
|-----------|---------|----------|-------|
| VAD | Silero VAD 5.1+ | - | MIT license, <1ms/chunk |
| STT | Moonshine | mlx-whisper | 5-15x faster than Whisper |
| LLM | Qwen 2.5 7B | Qwen 3B, Llama 3.1 | Via Ollama |
| TTS | Kokoro 82M | - | Apache 2.0, 24kHz |
| TTS (German) | Piper | - | German voice support |
| Audio I/O | sounddevice | - | Cross-platform |
| Package Mgr | uv | - | **Never use pip!** |

### Language Support Matrix

| Language | Code | STT Model | TTS Engine | Status |
|----------|------|-----------|------------|--------|
| English | en | moonshine/base | Kokoro | Ready |
| German | de | moonshine/base | Piper | Ready |
| Spanish | es | moonshine/base | Kokoro | Ready |
| Japanese | ja | moonshine/tiny-ja | Kokoro | Ready |
| Mandarin | zh | moonshine/tiny-zh | Kokoro | Ready |

## Memory Budget (32GB Target)

| Component | Estimated RAM | Notes |
|-----------|---------------|-------|
| OS + Python | ~4GB | Base system |
| Silero VAD | ~50MB | Lightweight |
| Moonshine Base | ~400MB | ONNX runtime |
| Qwen 2.5 7B (4-bit) | ~8GB | Via Ollama |
| Kokoro 82M | ~300MB | All languages |
| Piper German | ~100MB | German only |
| Audio buffers | ~200MB | Recording + playback |
| **Total** | **~13GB** | Comfortable margin |

## Latency Targets

| Stage | Target | Technology |
|-------|--------|------------|
| VAD processing | <1ms | Silero |
| Speech recording | Real-time | sounddevice |
| STT transcription | <500ms per utterance | Moonshine |
| LLM response | <2s first token | Qwen via Ollama |
| TTS synthesis | <300ms | Kokoro |
| **Total round-trip** | **<4s** | End-to-end |

## File Structure

```
polyglott/
├── src/polyglott/
│   ├── __init__.py
│   ├── constants.py          # All configuration constants
│   ├── cli.py                # Main entry point
│   ├── vad/
│   │   ├── __init__.py
│   │   └── detector.py       # Voice activity detection
│   ├── stt/
│   │   ├── __init__.py
│   │   └── transcriber.py    # Speech-to-text
│   ├── tts/
│   │   ├── __init__.py
│   │   └── synthesizer.py    # Text-to-speech
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── prompts.py        # System prompts
│   │   └── tutor.py          # LLM integration
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── recorder.py       # Microphone input
│   │   ├── player.py         # Speaker output
│   │   └── pipeline.py       # Orchestration
│   └── conversation/
│       ├── __init__.py
│       ├── session.py        # Session tracking
│       └── manager.py        # Conversation flow
├── tests/                    # Mirror of src/ structure
├── doc/
│   ├── llm/                  # LLM assistant docs
│   │   └── planning/         # This directory
│   ├── user/                 # End-user docs
│   └── developers/           # Developer docs
├── pyproject.toml
└── README.md
```

## Success Criteria

1. **Functional**: Complete voice conversation loop works offline
2. **Performance**: Round-trip latency under 4 seconds
3. **Quality**: Natural-sounding TTS, accurate STT
4. **Usability**: Child can use with minimal adult help
5. **Reliability**: No crashes during normal use

## Next Steps

Proceed to Phase 1 documentation: [01_core_infrastructure.md](01_core_infrastructure.md)

## Related Documents

- [Phase 1: Core Infrastructure](01_core_infrastructure.md)
- [Phase 2: Audio Pipeline](02_audio_pipeline.md)
- [Phase 3: Speech Processing](03_speech_processing.md)
- [Phase 4: LLM Integration](04_llm_integration.md)
- [Phase 5: User Interface](05_user_interface.md)
- [Phase 6: Testing & Polish](06_testing_polish.md)
