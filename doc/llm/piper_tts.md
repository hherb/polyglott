# Piper TTS API Reference

**Version:** 1.2.0+
**License:** MIT
**Repository:** https://github.com/rhasspy/piper
**Purpose:** German language TTS (Kokoro doesn't support German)

## Overview

Piper is a fast, local neural text-to-speech system optimized for edge devices. In Polyglott, it's used specifically for German language support since Kokoro TTS doesn't support German.

## Installation

```bash
uv add piper-tts
```

Note: Piper may require voice model downloads on first use.

## Basic Usage

```python
from piper import PiperVoice

# Load a voice model
voice = PiperVoice.load("de_DE-thorsten-high")

# Synthesize speech
audio = voice.synthesize("Hallo, wie geht es dir?")

# audio is a numpy array at the voice's sample rate
```

## Available German Voices

| Voice ID | Quality | Description |
|----------|---------|-------------|
| `de_DE-thorsten-high` | High | Male, clear pronunciation |
| `de_DE-thorsten-medium` | Medium | Male, faster synthesis |
| `de_DE-thorsten-low` | Low | Male, smallest model |
| `de_DE-eva_k-x_low` | Low | Female voice |
| `de_DE-karlsson-low` | Low | Alternative male |

## Recommended Voice

For Polyglott, we use **`de_DE-thorsten-high`** because:
- High quality pronunciation important for language learning
- Clear articulation helps children understand
- Good balance of quality vs. speed

## API Details

### PiperVoice.load()

```python
@classmethod
def load(
    cls,
    model_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    use_cuda: bool = False,
) -> "PiperVoice":
    """Load a Piper voice model.

    Args:
        model_path: Path to .onnx model file or voice name.
        config_path: Optional path to config JSON.
        use_cuda: Whether to use GPU acceleration.

    Returns:
        Loaded PiperVoice instance.
    """
```

### PiperVoice.synthesize()

```python
def synthesize(
    self,
    text: str,
    speaker_id: Optional[int] = None,
    length_scale: float = 1.0,
    noise_scale: float = 0.667,
    noise_w: float = 0.8,
) -> np.ndarray:
    """Synthesize speech from text.

    Args:
        text: Text to synthesize.
        speaker_id: Speaker ID for multi-speaker models.
        length_scale: Speech speed (1.0 = normal, <1 = faster, >1 = slower).
        noise_scale: Variability in pronunciation.
        noise_w: Variability in duration.

    Returns:
        Audio samples as numpy array.
    """
```

## Integration with Polyglott

In `src/polyglott/tts/synthesizer.py`:

```python
class PiperSynthesizer:
    """Speech synthesizer using Piper TTS for German."""

    PIPER_VOICES: dict[str, str] = {
        "de": "de_DE-thorsten-high",
        "en": "en_US-lessac-high",
        "es": "es_ES-davefx-medium",
    }

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate
        self._piper = None
        self._voice_cache: dict[str, object] = {}

    def synthesize(
        self,
        text: str,
        language: str = "de",
        speed: float = 0.9,
        voice: Optional[str] = None,
    ) -> SynthesisResult:
        """Synthesize text to speech."""
        self._ensure_loaded()

        voice_name = voice or self.PIPER_VOICES.get(language)

        # Get or create voice
        if voice_name not in self._voice_cache:
            self._voice_cache[voice_name] = PiperVoice.load(voice_name)

        piper_voice = self._voice_cache[voice_name]

        # length_scale controls speed (inverse: smaller = faster)
        length_scale = 1.0 / speed
        audio_data = piper_voice.synthesize(text, length_scale=length_scale)

        return SynthesisResult(
            audio=audio_data.astype(np.float32),
            sample_rate=self.sample_rate,
            duration_seconds=len(audio_data) / self.sample_rate,
        )
```

## Speed Adjustment

Piper uses `length_scale` for speed control (inverse of Kokoro's `speed`):
- `length_scale = 1.0`: Normal speed
- `length_scale = 0.8`: 25% faster (1.0 / 0.8 = 1.25x)
- `length_scale = 1.2`: 20% slower (1.0 / 1.2 = 0.83x)

For children's speed of 0.85, use:
```python
length_scale = 1.0 / 0.85  # ≈ 1.18
```

## Sample Rate

Piper outputs at **22050 Hz** by default (may vary by voice model).

To match Kokoro's 24000 Hz, you may need resampling:

```python
import scipy.signal

def resample_audio(audio: np.ndarray,
                   source_rate: int,
                   target_rate: int) -> np.ndarray:
    """Resample audio to target sample rate."""
    if source_rate == target_rate:
        return audio

    num_samples = int(len(audio) * target_rate / source_rate)
    return scipy.signal.resample(audio, num_samples)
```

## Voice Model Download

On first use, Piper may need to download voice models. They are cached locally.

Default cache location:
- Linux: `~/.local/share/piper`
- macOS: `~/Library/Application Support/piper`

## Comparison with Kokoro

| Feature | Piper | Kokoro |
|---------|-------|--------|
| German support | ✅ Yes | ❌ No |
| Model size | ~40-100MB | ~300MB |
| Sample rate | 22050 Hz | 24000 Hz |
| Speed control | length_scale | speed |
| Quality | Good | Excellent |

## Troubleshooting

### "Voice model not found"
```bash
# Download voice models manually
pip install piper-tts[voices]
# Or specify full path to .onnx file
```

### Slow first synthesis
First call downloads and loads the model. Subsequent calls are faster.

### Audio quality issues
- Use "high" quality voices for language learning
- Avoid "low" quality voices for pronunciation teaching
