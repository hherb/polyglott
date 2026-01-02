# Silero VAD API Reference

**Version:** 6.0+
**License:** MIT
**Repository:** https://github.com/snakers4/silero-vad

> **Note:** Silero VAD v6+ requires minimum 32ms audio chunks (512 samples at 16kHz).

## Installation

```bash
uv add silero-vad
```

## Basic Usage

### Method 1: Using silero_vad Package (Recommended)

```python
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

# Initialize the model (loads ~2MB model)
model = load_silero_vad()

# Load audio file
wav = read_audio('path_to_audio_file')

# Detect speech timestamps
speech_timestamps = get_speech_timestamps(
    wav,
    model,
    return_seconds=True,  # Returns results in seconds (default is samples)
)
```

### Method 2: Using PyTorch Hub

```python
import torch
torch.set_num_threads(1)  # Recommended for consistent performance

# Load model and utilities
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad'
)
(get_speech_timestamps, _, read_audio, _, _) = utils

# Process audio
wav = read_audio('path_to_audio_file')
speech_timestamps = get_speech_timestamps(wav, model, return_seconds=True)
```

## Real-Time Processing

For streaming audio, process chunks directly:

```python
import torch
from silero_vad import load_silero_vad

model = load_silero_vad()

def process_chunk(audio_chunk: torch.Tensor) -> float:
    """Process a single audio chunk and return speech probability.

    Args:
        audio_chunk: Audio tensor, should be 32ms at 16kHz (512 samples)

    Returns:
        Speech probability between 0.0 and 1.0
    """
    # Reset states if needed for new utterance
    # model.reset_states()

    speech_prob = model(audio_chunk, 16000).item()
    return speech_prob
```

## Key Parameters for get_speech_timestamps()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wav` | Tensor | required | Audio waveform |
| `model` | Model | required | Silero VAD model |
| `return_seconds` | bool | False | Return timestamps in seconds |
| `threshold` | float | 0.5 | Speech probability threshold |
| `sampling_rate` | int | 16000 | Audio sample rate |
| `min_speech_duration_ms` | int | 250 | Minimum speech duration |
| `min_silence_duration_ms` | int | 100 | Minimum silence between speech |

## Performance Characteristics

- **Model size:** ~2MB
- **Processing time:** <1ms per 32ms audio chunk on CPU
- **Supported sample rates:** 8000 Hz, 16000 Hz
- **Minimum chunk size:** 32ms (512 samples at 16kHz)
- **Languages:** Trained on 6000+ languages

## State Management

For streaming applications, manage model states:

```python
# Reset states at start of new utterance
model.reset_states()

# Process multiple chunks
for chunk in audio_stream:
    speech_prob = model(chunk, 16000).item()
    if speech_prob > 0.5:
        # Speech detected
        pass
```

## Audio Format Requirements

- Sample rate: 8000 Hz or 16000 Hz
- Channels: Mono (single channel)
- Format: Float32 tensor normalized to [-1.0, 1.0]
