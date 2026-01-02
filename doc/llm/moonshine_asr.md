# Moonshine ASR API Reference

**Version:** Latest (2025)
**License:** MIT
**Repository:** https://github.com/moonshine-ai/moonshine

## Overview

Moonshine is a family of speech-to-text models optimized for fast, accurate ASR on resource-constrained devices. It's 5-15x faster than Whisper while maintaining similar or better accuracy.

## Installation

### ONNX Runtime (Recommended for Production)

```bash
uv add useful-moonshine-onnx
```

Or install from git:

```bash
uv pip install useful-moonshine-onnx@git+https://github.com/moonshine-ai/moonshine.git#subdirectory=moonshine-onnx
```

### Hugging Face Transformers (For Development)

```bash
uv add transformers torch
```

## Model Sizes

### English Models
| Model | Parameters | Size | Performance |
|-------|-----------|------|-------------|
| `moonshine/tiny` | 27M | ~190MB | Beats Whisper Small (9x larger) |
| `moonshine/base` | 62M | ~400MB | Matches Whisper Medium (28x larger) |

### Language-Specific Models
| Model | Language |
|-------|----------|
| `moonshine/tiny-ja` | Japanese |
| `moonshine/tiny-zh` | Chinese (Mandarin) |
| `moonshine/tiny-ko` | Korean |
| `moonshine/tiny-ar` | Arabic |
| `moonshine/base-es` | Spanish |

## Basic Usage (ONNX)

```python
import moonshine_onnx

# Transcribe from file
result = moonshine_onnx.transcribe(
    'audio.wav',
    'moonshine/base'
)
print(result)  # ['Transcribed text here']
```

## Usage with Transformers

```python
from transformers import AutoProcessor, MoonshineForConditionalGeneration
import torch

# Load model and processor
processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-base")
model = MoonshineForConditionalGeneration.from_pretrained("UsefulSensors/moonshine-base")

def transcribe(audio_array: torch.Tensor, sample_rate: int = 16000) -> str:
    """Transcribe audio array to text.

    Args:
        audio_array: Audio samples as tensor
        sample_rate: Audio sample rate (should be 16000)

    Returns:
        Transcribed text
    """
    inputs = processor(audio_array, return_tensors="pt", sampling_rate=sample_rate)
    generated_ids = model.generate(**inputs)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription
```

## Real-Time Transcription

For streaming/real-time use, process audio in chunks:

```python
import moonshine_onnx
import numpy as np
from collections import deque

class StreamingTranscriber:
    """Real-time audio transcriber using Moonshine."""

    def __init__(self, model_name: str = 'moonshine/tiny'):
        self.model_name = model_name
        self.audio_buffer = deque(maxlen=16000 * 30)  # 30 seconds max

    def add_audio(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk to buffer."""
        self.audio_buffer.extend(audio_chunk.tolist())

    def transcribe(self) -> str:
        """Transcribe current buffer contents."""
        if len(self.audio_buffer) < 16000 * 0.5:  # Minimum 0.5 seconds
            return ""

        audio_array = np.array(list(self.audio_buffer), dtype=np.float32)
        result = moonshine_onnx.transcribe(audio_array, self.model_name)
        return result[0] if result else ""

    def clear(self) -> None:
        """Clear the audio buffer."""
        self.audio_buffer.clear()
```

## Audio Format Requirements

- **Sample rate:** 16000 Hz (required)
- **Channels:** Mono
- **Format:** Float32 normalized [-1.0, 1.0] or Int16

## Language Selection for Polyglott

```python
# Mapping of target languages to Moonshine models
LANGUAGE_MODELS = {
    'en': 'moonshine/base',      # English - use base for quality
    'de': 'moonshine/base',      # German - use base (multilingual)
    'es': 'moonshine/base-es',   # Spanish - specialized model
    'ja': 'moonshine/tiny-ja',   # Japanese - specialized model
    'zh': 'moonshine/tiny-zh',   # Mandarin - specialized model
}

def get_model_for_language(language: str) -> str:
    """Get the best Moonshine model for a language."""
    return LANGUAGE_MODELS.get(language, 'moonshine/base')
```

## Performance Characteristics

- **Speed:** 5-15x faster than Whisper
- **Latency:** Processes variable-length audio (not fixed 30s chunks like Whisper)
- **Accuracy:** Matches or exceeds Whisper Medium on most benchmarks
- **Memory:** Low footprint suitable for edge devices

## Comparison with Whisper

| Feature | Moonshine | Whisper |
|---------|-----------|---------|
| Speed | 5-15x faster | Baseline |
| Model size | 27-62M | 39M-1.5B |
| Chunk processing | Variable length | Fixed 30s |
| Language models | Specialized per-language | Single multilingual |
| Best for | Real-time, edge | Batch processing |

## Fallback to Whisper

For languages without specialized Moonshine models, fall back to mlx-whisper:

```python
# If on macOS with Apple Silicon
import mlx_whisper

def transcribe_with_fallback(audio_path: str, language: str) -> str:
    """Transcribe with Moonshine, fallback to Whisper if needed."""
    moonshine_langs = {'en', 'es', 'ja', 'zh'}

    if language in moonshine_langs:
        import moonshine_onnx
        model = LANGUAGE_MODELS[language]
        return moonshine_onnx.transcribe(audio_path, model)[0]
    else:
        # Fallback to Whisper for German and others
        return mlx_whisper.transcribe(audio_path)['text']
```
