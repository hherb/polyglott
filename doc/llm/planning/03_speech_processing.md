# Phase 3: Speech Processing

**Phase:** 3 of 6
**Prerequisites:** Phase 2 (Audio Pipeline)
**Estimated Complexity:** High

## Objectives

1. Implement Speech-to-Text with Moonshine ASR (primary) and mlx-whisper (fallback)
2. Implement Text-to-Speech with Kokoro (multi-language) and Piper (German)
3. Support all target languages: English, German, Spanish, Japanese, Mandarin
4. Optimize for low latency on Apple Silicon

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Speech Processing                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Speech-to-Text (STT)                       │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │                                                               │    │
│  │  ┌────────────────────┐    ┌────────────────────┐           │    │
│  │  │  MoonshineTranscr. │    │  WhisperMLXTransc. │           │    │
│  │  │                    │    │                    │           │    │
│  │  │  • ONNX Runtime    │    │  • MLX Framework   │           │    │
│  │  │  • 27-62M params   │    │  • macOS only      │           │    │
│  │  │  • 5-15x faster    │    │  • Fallback        │           │    │
│  │  └────────────────────┘    └────────────────────┘           │    │
│  │              │                       │                       │    │
│  │              └───────────┬───────────┘                       │    │
│  │                          ▼                                   │    │
│  │              ┌────────────────────┐                          │    │
│  │              │  SpeechTranscriber │  ← Unified interface     │    │
│  │              └────────────────────┘                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Text-to-Speech (TTS)                       │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │                                                               │    │
│  │  ┌────────────────────┐    ┌────────────────────┐           │    │
│  │  │  KokoroSynthesizer │    │  PiperSynthesizer  │           │    │
│  │  │                    │    │                    │           │    │
│  │  │  • 82M params      │    │  • German voices   │           │    │
│  │  │  • EN,ES,JA,ZH     │    │  • Neural TTS      │           │    │
│  │  │  • 24kHz output    │    │  • 22kHz output    │           │    │
│  │  └────────────────────┘    └────────────────────┘           │    │
│  │              │                       │                       │    │
│  │              └───────────┬───────────┘                       │    │
│  │                          ▼                                   │    │
│  │              ┌────────────────────┐                          │    │
│  │              │ SpeechSynthesizer  │  ← Unified interface     │    │
│  │              └────────────────────┘                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Implementation

### Step 3.1: Speech-to-Text Base Classes

**File:** `src/polyglott/stt/transcriber.py`

**Data classes:**

```python
class TranscriberBackend(str, Enum):
    """Available transcription backends."""
    MOONSHINE = "moonshine"
    WHISPER_MLX = "whisper_mlx"

@dataclass
class TranscriptionResult:
    """Result from speech transcription."""
    text: str
    language: str
    confidence: Optional[float] = None
    duration_seconds: Optional[float] = None
```

**Protocol for backends:**

```python
class TranscriberProtocol(Protocol):
    """Protocol for transcription backends."""

    def transcribe(
        self,
        audio: Union[np.ndarray, Path, str],
        language: Optional[str] = None,
    ) -> TranscriptionResult: ...
```

### Step 3.2: Moonshine Transcriber

**Primary STT backend - optimized for speed and edge devices.**

```python
# Language to model mapping
MOONSHINE_LANGUAGE_MODELS: dict[str, str] = {
    "en": "moonshine/base",
    "de": "moonshine/base",      # Multilingual base
    "es": "moonshine/base",      # Spanish
    "ja": "moonshine/tiny-ja",   # Japanese specialized
    "zh": "moonshine/tiny-zh",   # Chinese specialized
}

class MoonshineTranscriber:
    """Moonshine ASR - 5-15x faster than Whisper."""

    def __init__(
        self,
        model_size: str = MOONSHINE_MODEL_SIZE,
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> None: ...

    def transcribe(
        self,
        audio: Union[np.ndarray, Path, str],
        language: Optional[str] = None,
    ) -> TranscriptionResult: ...

    def _get_model_for_language(self, language: str) -> str: ...
    def _transcribe_file(self, path: str, language: str) -> TranscriptionResult: ...
    def _transcribe_array(self, audio: np.ndarray, language: str) -> TranscriptionResult: ...
```

**Key Implementation Details:**

1. **ONNX Runtime Usage:**
   ```python
   import moonshine_onnx

   result = moonshine_onnx.transcribe(audio_path, model_name)
   # Returns: ['Transcribed text here']
   ```

2. **Audio Preparation:**
   ```python
   def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
       if audio.dtype != np.float32:
           audio = audio.astype(np.float32)
       max_val = np.abs(audio).max()
       if max_val > 1.0:
           audio = audio / max_val
       return audio
   ```

3. **Language-specific models:**
   - Japanese: `moonshine/tiny-ja` (specialized)
   - Chinese: `moonshine/tiny-zh` (specialized)
   - Others: `moonshine/base` (multilingual)

### Step 3.3: Whisper MLX Transcriber (Fallback)

**Fallback for macOS when Moonshine unavailable.**

```python
class WhisperMLXTranscriber:
    """Whisper via MLX - macOS Apple Silicon only."""

    def __init__(
        self,
        model_size: str = "base",
        sample_rate: int = AUDIO_SAMPLE_RATE,
    ) -> None: ...

    def transcribe(
        self,
        audio: Union[np.ndarray, Path, str],
        language: Optional[str] = None,
    ) -> TranscriptionResult: ...
```

**Key Implementation Details:**

```python
result = self._mlx_whisper.transcribe(
    audio_path,
    path_or_hf_repo=f"mlx-community/whisper-{self.model_size}-mlx",
    language=language,
)
```

### Step 3.4: Unified Speech Transcriber

```python
class SpeechTranscriber:
    """Unified transcriber with automatic backend selection."""

    def __init__(
        self,
        backend: Optional[TranscriberBackend] = None,
        model_size: str = MOONSHINE_MODEL_SIZE,
    ) -> None: ...

    def transcribe(
        self,
        audio: Union[np.ndarray, Path, str],
        language: Optional[str] = None,
    ) -> TranscriptionResult: ...

    def _detect_best_backend(self) -> TranscriberBackend:
        """Auto-detect best available backend."""
        try:
            import moonshine_onnx
            return TranscriberBackend.MOONSHINE
        except ImportError:
            pass

        try:
            import mlx_whisper
            return TranscriberBackend.WHISPER_MLX
        except ImportError:
            pass

        return TranscriberBackend.MOONSHINE  # Will fail with helpful message
```

### Step 3.5: Text-to-Speech Base Classes

**File:** `src/polyglott/tts/synthesizer.py`

```python
class TTSBackend(str, Enum):
    """Available TTS backends."""
    KOKORO = "kokoro"
    PIPER = "piper"

@dataclass
class SynthesisResult:
    """Result from speech synthesis."""
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
```

### Step 3.6: Kokoro Synthesizer

**Primary TTS - supports EN, ES, JA, ZH.**

```python
# Language code mapping for Kokoro
KOKORO_LANG_CODES: dict[str, str] = {
    "en": "a",  # American English
    "es": "e",  # Spanish
    "ja": "j",  # Japanese (requires misaki[ja])
    "zh": "z",  # Mandarin (requires misaki[zh])
}

# Default voices per language
KOKORO_VOICES: dict[str, str] = {
    "en": "af_heart",   # Warm female
    "es": "ef_dora",    # Spanish female
    "ja": "jf_alpha",   # Japanese female
    "zh": "zf_xiaobei", # Chinese female
}

class KokoroSynthesizer:
    """Kokoro TTS - 82M params, high quality."""

    def __init__(self, sample_rate: int = TTS_SAMPLE_RATE) -> None: ...

    def synthesize(
        self,
        text: str,
        language: str = "en",
        speed: float = TTS_DEFAULT_SPEED,
        voice: Optional[str] = None,
    ) -> SynthesisResult: ...
```

**Key Implementation Details:**

```python
from kokoro import KPipeline

pipeline = KPipeline(lang_code='a')  # American English

audio_chunks = []
for gs, ps, audio in pipeline(text, voice='af_heart', speed=0.9):
    # gs = graphemes (text)
    # ps = phonemes
    # audio = numpy array at 24kHz
    audio_chunks.append(audio)

full_audio = np.concatenate(audio_chunks)
```

### Step 3.7: Piper Synthesizer

**German TTS - Kokoro doesn't support German.**

```python
class PiperSynthesizer:
    """Piper TTS - German language support."""

    PIPER_VOICES: dict[str, str] = {
        "de": "de_DE-thorsten-high",
        "en": "en_US-lessac-high",
        "es": "es_ES-davefx-medium",
    }

    def __init__(self, sample_rate: int = 22050) -> None: ...

    def synthesize(
        self,
        text: str,
        language: str = "de",
        speed: float = TTS_DEFAULT_SPEED,
        voice: Optional[str] = None,
    ) -> SynthesisResult: ...
```

### Step 3.8: Unified Speech Synthesizer

```python
class SpeechSynthesizer:
    """Unified synthesizer with automatic backend selection."""

    def __init__(self, speed: float = TTS_CHILDREN_SPEED) -> None: ...

    def synthesize(
        self,
        text: str,
        language: str = "en",
        speed: Optional[float] = None,
    ) -> SynthesisResult: ...

    def _get_backend(self, language: str) -> tuple[SynthesizerProtocol, TTSBackend]:
        """Select backend based on language."""
        if language == "de":
            # German → Piper
            return self._piper, TTSBackend.PIPER
        else:
            # Everything else → Kokoro
            return self._kokoro, TTSBackend.KOKORO
```

## Language Support Matrix

| Language | STT Model | TTS Engine | TTS Voice | Notes |
|----------|-----------|------------|-----------|-------|
| English | moonshine/base | Kokoro | af_heart | Primary support |
| German | moonshine/base | Piper | thorsten-high | Piper required |
| Spanish | moonshine/base | Kokoro | ef_dora | Good quality |
| Japanese | moonshine/tiny-ja | Kokoro | jf_alpha | Requires misaki[ja] |
| Mandarin | moonshine/tiny-zh | Kokoro | zf_xiaobei | Requires misaki[zh] |

## Verification Checklist

- [ ] Moonshine transcribes English accurately
- [ ] Moonshine handles all target languages
- [ ] Fallback to Whisper works on macOS
- [ ] Kokoro synthesizes natural speech
- [ ] Piper synthesizes German correctly
- [ ] Speed adjustment works for children
- [ ] Audio quality is consistent

## Testing Strategy

### Unit Tests

```python
def test_moonshine_language_models():
    """Test all languages have model mappings."""
    for lang in ["en", "de", "es", "ja", "zh"]:
        assert lang in MOONSHINE_LANGUAGE_MODELS

def test_kokoro_supported_languages():
    """Test Kokoro language codes."""
    synth = KokoroSynthesizer()
    assert "en" in synth.supported_languages
    assert "de" not in synth.supported_languages  # Uses Piper

def test_synthesizer_backend_selection():
    """Test automatic backend selection."""
    synth = SpeechSynthesizer()
    backend_de = synth._get_backend("de")
    assert backend_de[1] == TTSBackend.PIPER

    backend_en = synth._get_backend("en")
    assert backend_en[1] == TTSBackend.KOKORO
```

### Integration Tests

```python
def test_stt_tts_roundtrip():
    """Test STT → text → TTS flow."""
    # Would require audio fixtures

def test_language_switching():
    """Test switching between languages."""
    synth = SpeechSynthesizer()
    result_en = synth.synthesize("Hello", language="en")
    result_de = synth.synthesize("Hallo", language="de")
    assert result_en.sample_rate == 24000  # Kokoro
    assert result_de.sample_rate == 22050  # Piper
```

## Files Modified/Created

| File | Status | Lines |
|------|--------|-------|
| `src/polyglott/stt/__init__.py` | ✅ Created | ~9 |
| `src/polyglott/stt/transcriber.py` | ✅ Created | ~430 |
| `src/polyglott/tts/__init__.py` | ✅ Created | ~9 |
| `src/polyglott/tts/synthesizer.py` | ✅ Created | ~421 |
| `tests/stt/test_transcriber.py` | ✅ Created | ~106 |
| `tests/tts/test_synthesizer.py` | ✅ Created | ~111 |

## Performance Targets

| Operation | Target | Actual |
|-----------|--------|--------|
| STT (1s audio) | <500ms | TBD |
| TTS (short sentence) | <300ms | TBD |
| Language switch | <100ms | TBD |

## Known Issues

1. **Kokoro German:** Not supported - requires Piper
2. **Japanese/Chinese deps:** Need misaki[ja] and misaki[zh]
3. **Piper voice download:** May need first-run download

## Next Phase

Proceed to [Phase 4: LLM Integration](04_llm_integration.md)
