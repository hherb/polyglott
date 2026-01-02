# Kokoro TTS API Reference

**Version:** 0.9.4+
**License:** Apache 2.0
**Repository:** https://github.com/hexgrad/kokoro
**Model:** 82M parameters, 24kHz output

## Installation

```bash
uv add kokoro

# For Japanese support
uv add "misaki[ja]"

# For Mandarin Chinese support
uv add "misaki[zh]"

# System dependency (Linux)
apt-get install espeak-ng
```

## Basic Usage

```python
from kokoro import KPipeline
import soundfile as sf

# Initialize pipeline with language
pipeline = KPipeline(lang_code='a')  # American English

# Generate speech
text = "Hello! How are you today?"
generator = pipeline(text, voice='af_heart')

# Save audio
for i, (gs, ps, audio) in enumerate(generator):
    sf.write(f'output_{i}.wav', audio, 24000)
```

## Language Codes

| Code | Language | Notes |
|------|----------|-------|
| `'a'` | American English | Default voice |
| `'b'` | British English | |
| `'e'` | Spanish (Español) | |
| `'f'` | French | |
| `'h'` | Hindi | |
| `'i'` | Italian | |
| `'j'` | Japanese | Requires `misaki[ja]` |
| `'p'` | Portuguese (Brazilian) | |
| `'z'` | Mandarin Chinese | Requires `misaki[zh]` |

## Voice Selection

Each language has multiple voice options. Format: `{lang}_{type}_{name}`

### American English Voices
- `af_heart` - Female, warm
- `af_bella` - Female, clear
- `am_adam` - Male, neutral

### Accessing Available Voices

```python
from kokoro import KPipeline

pipeline = KPipeline(lang_code='a')
# Check pipeline documentation for available voices
```

## Advanced Configuration

```python
from kokoro import KPipeline

pipeline = KPipeline(lang_code='e')  # Spanish

generator = pipeline(
    text="¡Hola! ¿Cómo estás?",
    voice='ef_dora',           # Spanish female voice
    speed=0.9,                 # Slightly slower (good for learners)
    split_pattern=r'\n+'       # Split on newlines
)

for gs, ps, audio in generator:
    # gs = graphemes (original text)
    # ps = phonemes (pronunciation)
    # audio = numpy array at 24kHz
    pass
```

## Custom Voice Tensors

Load custom voice embeddings:

```python
import torch
from kokoro import KPipeline

pipeline = KPipeline(lang_code='a')
voice_tensor = torch.load('custom_voice.pt', weights_only=True)

generator = pipeline(text, voice=voice_tensor, speed=1.0)
```

## Streaming Audio Output

```python
import sounddevice as sd
from kokoro import KPipeline

pipeline = KPipeline(lang_code='a')
generator = pipeline("Hello world!", voice='af_heart')

for gs, ps, audio in generator:
    sd.play(audio, samplerate=24000)
    sd.wait()
```

## Multi-Language Support for Polyglott

```python
from kokoro import KPipeline
from typing import Optional
import numpy as np

# Language code mapping
LANG_TO_KOKORO = {
    'en': 'a',  # American English
    'es': 'e',  # Spanish
    'ja': 'j',  # Japanese (requires misaki[ja])
    'zh': 'z',  # Mandarin (requires misaki[zh])
}

# Default voices per language
DEFAULT_VOICES = {
    'en': 'af_heart',
    'es': 'ef_dora',
    'ja': 'jf_alpha',
    'zh': 'zf_xiaobei',
}

def synthesize(
    text: str,
    language: str,
    speed: float = 0.9
) -> Optional[np.ndarray]:
    """Synthesize speech for the given language."""
    lang_code = LANG_TO_KOKORO.get(language)
    if lang_code is None:
        return None

    pipeline = KPipeline(lang_code=lang_code)
    voice = DEFAULT_VOICES.get(language, 'af_heart')

    audio_chunks = []
    for gs, ps, audio in pipeline(text, voice=voice, speed=speed):
        audio_chunks.append(audio)

    return np.concatenate(audio_chunks) if audio_chunks else None
```

## Performance Notes

- **Model size:** 82M parameters (much smaller than XTTS-v2's 467M)
- **Output quality:** 24kHz high-fidelity audio
- **Speed:** Real-time or faster on modern CPUs
- **Memory:** Low footprint, suitable for edge devices

## German Language Note

Kokoro does not natively support German. For German TTS, use Piper TTS instead.
See `doc/llm/piper_tts.md` for German language support.
