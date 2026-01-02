# TEN VAD API Reference

**Version:** Latest
**License:** Apache 2.0
**Repository:** https://github.com/TEN-framework/ten-vad
**Hugging Face:** https://huggingface.co/TEN-framework/ten-vad

## Overview

TEN VAD is a high-performance, low-latency Voice Activity Detection system that offers:

- **Faster transitions**: Rapid detection of speech-to-non-speech transitions
- **Smaller footprint**: 277KB-731KB vs Silero's 2.2MB
- **No PyTorch dependency**: Uses ONNX Runtime
- **Lower latency**: 10ms or 16ms frame sizes vs 32ms

## Performance Comparison

| Metric | TEN VAD | Silero VAD |
|--------|---------|------------|
| Library Size | 277KB-731KB | ~2.2MB |
| Frame Size | 10ms / 16ms | 32ms |
| Dependencies | ONNX Runtime | PyTorch |
| RTF (M1 Mac) | 0.016 | ~0.04 |
| Python Support | Linux, macOS (ONNX) | All platforms |

## Installation

TEN VAD requires building from source:

```bash
# Clone repository
git clone https://github.com/TEN-framework/ten-vad.git
cd ten-vad/examples_onnx/python

# Build for macOS
./build-and-deploy-macos.sh

# Or for Linux
./build-and-deploy-linux.sh
```

### Requirements

- CMake (macOS: `brew install cmake`)
- ONNX Runtime >= 1.17.1
- Python 3.8+
- pybind11, NumPy

## Basic Usage

### Direct API

```python
import ten_vad_python

# Initialize with hop size and threshold
vad = ten_vad_python.VAD(hop_size=256, threshold=0.5)

# Process audio frame-by-frame
for frame in audio_frames:
    prob, is_voice = vad.process(frame)
    if is_voice:
        # Speech detected
        pass
```

### Using Polyglott Wrapper

```python
from polyglott.vad import create_vad_detector, TEN_VAD_AVAILABLE

if TEN_VAD_AVAILABLE:
    # Create TEN VAD detector
    detector = create_vad_detector(backend="ten_vad")

    # Or with custom hop size (10ms frames)
    detector = create_vad_detector(
        backend="ten_vad",
        hop_size=160,  # 160 samples = 10ms at 16kHz
    )

    # Process audio
    for chunk in audio_chunks:
        result = detector.process_chunk(chunk)
        if result.state == SpeechState.SPEECH_END:
            # End of utterance detected
            pass
```

## Hop Size Options

| Hop Size | Duration | Samples at 16kHz | Use Case |
|----------|----------|------------------|----------|
| 160 | 10ms | 160 | Lowest latency, more CPU |
| 256 | 16ms | 256 | Balanced (recommended) |

## Audio Format Requirements

- **Sample rate:** 16000 Hz only
- **Channels:** Mono (single channel)
- **Format:** Float32 array normalized to [-1.0, 1.0]

## Platform Support

| Platform | Architecture | Python ONNX | Native C |
|----------|--------------|-------------|----------|
| Linux | x64 | ✅ | ✅ |
| macOS | arm64, x86_64 | ✅ | ✅ |
| Windows | x86, x64 | ❌ | ✅ |
| Android | arm64-v8a, armeabi-v7a | ❌ | ✅ |
| iOS | arm64 | ❌ | ✅ |

## Polyglott Integration

TEN VAD is integrated as an optional VAD backend:

```python
from polyglott.vad import (
    get_available_backends,
    create_vad_detector,
    TEN_VAD_AVAILABLE,
)

# Check available backends
print(get_available_backends())
# ['silero']  - if TEN VAD not installed
# ['silero', 'ten_vad']  - if TEN VAD installed

# Use TEN VAD if available, fallback to Silero
backend = "ten_vad" if TEN_VAD_AVAILABLE else "silero"
detector = create_vad_detector(backend=backend)
```

## When to Use TEN VAD

**Choose TEN VAD when:**
- You need lower latency speech detection
- You want to reduce dependency footprint (no PyTorch)
- You're deploying on Linux or macOS

**Stick with Silero VAD when:**
- You need cross-platform Python support
- You want pip-installable dependencies
- You're already using PyTorch for other components

## Troubleshooting

### TenVadNotInstalledError

If you see this error, TEN VAD is not installed:

```
TenVadNotInstalledError: TEN VAD is not installed. To use TEN VAD:
1. Clone: git clone https://github.com/TEN-framework/ten-vad.git
2. Build: cd ten-vad/examples_onnx/python && ./build-and-deploy-macos.sh
3. Install: pip install -e .
```

### Sample Rate Error

TEN VAD only supports 16kHz audio:

```
ValueError: TEN VAD only supports 16kHz sample rate, got 8000
```

Resample your audio to 16kHz before processing.
