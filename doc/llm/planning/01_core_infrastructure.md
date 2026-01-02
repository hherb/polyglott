# Phase 1: Core Infrastructure

**Phase:** 1 of 6
**Prerequisites:** None
**Estimated Complexity:** Low

## Objectives

1. Establish project structure following golden rules
2. Configure dependencies in pyproject.toml
3. Create constants module with all configuration values
4. Set up base utilities and error handling
5. Configure testing framework

## Step-by-Step Implementation

### Step 1.1: Project Structure Verification

Verify the following directory structure exists:

```
polyglott/
├── src/polyglott/
│   ├── __init__.py
│   ├── constants.py
│   └── [module directories]
├── tests/
│   └── [test files]
├── doc/
│   ├── llm/
│   ├── user/
│   └── developers/
├── pyproject.toml
└── README.md
```

**Actions:**
- [x] Create all directories
- [x] Create `__init__.py` files with module docstrings
- [x] Ensure proper Python package structure

### Step 1.2: Dependencies Configuration (pyproject.toml)

**Core Dependencies:**

```toml
[project]
dependencies = [
    # Audio I/O
    "sounddevice>=0.4.6",
    "numpy>=1.24.0",

    # VAD
    "silero-vad>=5.1",

    # STT
    "onnxruntime>=1.16.0",

    # LLM
    "ollama>=0.3.0",

    # TTS
    "kokoro>=0.3.0",

    # Utilities
    "pydantic>=2.0.0",
    "rich>=13.0.0",
]
```

**Optional Dependencies:**

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]
macos = [
    "mlx>=0.5.0",
    "mlx-whisper>=0.1.0",
]
piper = [
    "piper-tts>=1.2.0",
]
```

**Actions:**
- [x] Create pyproject.toml with all dependencies
- [x] Configure build system (hatchling)
- [x] Set up pytest configuration
- [x] Configure ruff and mypy

### Step 1.3: Constants Module

**File:** `src/polyglott/constants.py`

All magic numbers must be defined as named constants. Group by functionality:

```python
# Audio Configuration
AUDIO_SAMPLE_RATE: Final[int] = 16000
AUDIO_BIT_DEPTH: Final[int] = 16
AUDIO_CHANNELS: Final[int] = 1
VAD_CHUNK_DURATION_MS: Final[int] = 30
VAD_CHUNK_SAMPLES: Final[int] = 480  # 16000 * 30 / 1000

# VAD Configuration
VAD_SPEECH_THRESHOLD: Final[float] = 0.5
VAD_SPEECH_PAD_FRAMES: Final[int] = 3
VAD_SILENCE_PAD_FRAMES: Final[int] = 10

# STT Configuration
MOONSHINE_MODEL_SIZE: Final[str] = "base"
MAX_TRANSCRIPTION_AUDIO_SECONDS: Final[float] = 30.0

# LLM Configuration
DEFAULT_LLM_MODEL: Final[str] = "qwen2.5:7b"
MAX_LLM_RESPONSE_TOKENS: Final[int] = 256
LLM_TEMPERATURE: Final[float] = 0.7
MAX_CONVERSATION_HISTORY: Final[int] = 20

# TTS Configuration
TTS_SAMPLE_RATE: Final[int] = 24000
TTS_DEFAULT_SPEED: Final[float] = 0.9
TTS_CHILDREN_SPEED: Final[float] = 0.85

# Language Enums
class StudentLanguage(str, Enum): ...
class TargetLanguage(str, Enum): ...
class AgeGroup(str, Enum): ...
```

**Actions:**
- [x] Define all audio constants
- [x] Define VAD thresholds
- [x] Define LLM parameters
- [x] Define TTS settings
- [x] Create language enums
- [x] Create language name mappings

### Step 1.4: Base Utilities (If Needed)

Create utility functions that are used across modules:

```python
# src/polyglott/utils.py (create if needed)

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1.0, 1.0] range."""
    ...

def ensure_sample_rate(audio: np.ndarray,
                       source_rate: int,
                       target_rate: int) -> np.ndarray:
    """Resample audio if needed."""
    ...
```

**Note:** Only create if multiple modules need the same utility. Avoid premature abstraction.

### Step 1.5: Error Handling

Define custom exceptions for the project:

```python
# src/polyglott/exceptions.py (create if needed)

class PolyglottError(Exception):
    """Base exception for Polyglott."""
    pass

class AudioError(PolyglottError):
    """Audio processing errors."""
    pass

class ModelNotFoundError(PolyglottError):
    """Model not available."""
    pass
```

**Note:** Only create when specific error types are needed. Start simple.

### Step 1.6: Testing Framework

**File:** `tests/conftest.py`

Set up shared fixtures:

```python
@pytest.fixture
def sample_rate() -> int:
    return AUDIO_SAMPLE_RATE

@pytest.fixture
def silence_chunk() -> np.ndarray:
    return np.zeros(VAD_CHUNK_SAMPLES, dtype=np.float32)

@pytest.fixture
def speech_chunk() -> np.ndarray:
    # Generate sine wave
    ...
```

**Actions:**
- [x] Create conftest.py with fixtures
- [x] Set up test directory structure
- [x] Verify pytest configuration works

## Verification Checklist

- [ ] `uv sync` completes without errors
- [ ] `uv run python -c "import polyglott"` works
- [ ] `uv run pytest --collect-only` finds all tests
- [ ] All constants are properly typed
- [ ] Module docstrings are complete

## Files Modified/Created

| File | Status | Lines |
|------|--------|-------|
| `pyproject.toml` | ✅ Created | ~96 |
| `src/polyglott/__init__.py` | ✅ Created | ~16 |
| `src/polyglott/constants.py` | ✅ Created | ~178 |
| `tests/conftest.py` | ✅ Created | ~90 |

## Known Issues

None at this phase.

## Next Phase

Proceed to [Phase 2: Audio Pipeline](02_audio_pipeline.md)
