# Phase 6: Testing & Polish

**Phase:** 6 of 6
**Prerequisites:** All previous phases
**Estimated Complexity:** Medium

## Objectives

1. Complete unit test coverage for all modules
2. Add integration tests for full pipeline
3. Performance testing and optimization
4. Documentation finalization
5. Bug fixes and polish

## Testing Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Test Architecture                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                        Unit Tests                              │   │
│  │                                                                │   │
│  │  tests/vad/test_detector.py        - VAD functionality        │   │
│  │  tests/stt/test_transcriber.py     - STT functionality        │   │
│  │  tests/tts/test_synthesizer.py     - TTS functionality        │   │
│  │  tests/llm/test_tutor.py           - LLM functionality        │   │
│  │  tests/conversation/test_session.py - Session management      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     Integration Tests                          │   │
│  │                                                                │   │
│  │  tests/integration/test_audio_pipeline.py                     │   │
│  │  tests/integration/test_speech_roundtrip.py                   │   │
│  │  tests/integration/test_conversation_flow.py                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     Performance Tests                          │   │
│  │                                                                │   │
│  │  tests/performance/test_latency.py                            │   │
│  │  tests/performance/test_memory.py                             │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Implementation

### Step 6.1: Complete Unit Test Coverage

**Target: 80%+ coverage on critical paths**

#### VAD Tests (tests/vad/test_detector.py)

```python
class TestVoiceActivityDetector:
    def test_initialization_default(self): ...
    def test_initialization_custom(self): ...
    def test_invalid_sample_rate(self): ...
    def test_reset(self): ...
    def test_process_chunk_wrong_size(self): ...
    def test_process_silence_chunk(self): ...
    def test_process_speech_chunk(self): ...
    def test_state_transition_silence_to_speech(self): ...
    def test_state_transition_speech_to_silence(self): ...
    def test_get_chunk_samples(self): ...
```

#### STT Tests (tests/stt/test_transcriber.py)

```python
class TestSpeechTranscriber:
    def test_initialization_default(self): ...
    def test_initialization_with_backend(self): ...
    def test_backend_detection(self): ...
    def test_language_model_mapping(self): ...
    def test_all_languages_supported(self): ...

class TestTranscriptionResult:
    def test_creation_minimal(self): ...
    def test_creation_full(self): ...
```

#### TTS Tests (tests/tts/test_synthesizer.py)

```python
class TestSpeechSynthesizer:
    def test_initialization(self): ...
    def test_backend_selection_german(self): ...
    def test_backend_selection_english(self): ...
    def test_supported_languages(self): ...
    def test_speed_configuration(self): ...

class TestSynthesisResult:
    def test_creation(self): ...
```

#### LLM Tests (tests/llm/test_tutor.py)

```python
class TestLanguageTutor:
    def test_initialization(self): ...
    def test_initialization_with_config(self): ...
    def test_system_prompt_generation(self): ...
    def test_reset_conversation(self): ...
    def test_get_available_lessons(self): ...
    def test_set_lesson_focus(self): ...
    def test_history_trimming(self): ...

class TestTutorConfig:
    def test_defaults(self): ...
    def test_custom_values(self): ...
```

### Step 6.2: Integration Tests

**File:** `tests/integration/test_audio_pipeline.py`

```python
import pytest
from polyglott.audio.pipeline import AudioPipeline, PipelineState
from polyglott.constants import TargetLanguage, AgeGroup


class TestAudioPipelineIntegration:
    """Integration tests for the audio pipeline."""

    @pytest.fixture
    def pipeline(self) -> AudioPipeline:
        return AudioPipeline(
            target_language=TargetLanguage.SPANISH,
            native_language="English",
            age_group=AgeGroup.EARLY_PRIMARY,
        )

    def test_pipeline_initialization(self, pipeline: AudioPipeline) -> None:
        """Test pipeline initializes correctly."""
        assert pipeline.state == PipelineState.IDLE

    def test_start_session(self, pipeline: AudioPipeline) -> None:
        """Test session start returns greeting."""
        greeting = pipeline.start_session()
        assert isinstance(greeting, str)
        assert len(greeting) > 0

    def test_state_transitions(self, pipeline: AudioPipeline) -> None:
        """Test state transitions during pipeline operation."""
        states_seen = []

        def record_state(state: PipelineState) -> None:
            states_seen.append(state)

        pipeline.on_state_change = record_state
        # Would need mock audio for full test

    def test_stop(self, pipeline: AudioPipeline) -> None:
        """Test pipeline can be stopped."""
        pipeline.stop()
        assert pipeline.state == PipelineState.IDLE
```

**File:** `tests/integration/test_conversation_flow.py`

```python
import pytest
from polyglott.conversation.manager import ConversationManager
from polyglott.constants import TargetLanguage, AgeGroup


class TestConversationFlowIntegration:
    """Integration tests for conversation flow."""

    @pytest.fixture
    def manager(self) -> ConversationManager:
        return ConversationManager(
            student_name="Test Student",
            target_language=TargetLanguage.SPANISH,
        )

    def test_session_creation(self, manager: ConversationManager) -> None:
        """Test session is created on start."""
        manager.start()
        assert manager.session is not None
        assert manager.session.turn_count == 0

    def test_lesson_focus_change(self, manager: ConversationManager) -> None:
        """Test changing lesson focus."""
        manager.start()
        lessons = manager.get_available_lessons()
        assert len(lessons) > 0

        manager.set_lesson_focus(lessons[0])
        assert manager.session.config.lesson_focus == lessons[0]
```

### Step 6.3: Performance Tests

**File:** `tests/performance/test_latency.py`

```python
import time
import pytest
import numpy as np
from polyglott.constants import AUDIO_SAMPLE_RATE, VAD_CHUNK_SAMPLES


class TestPerformance:
    """Performance and latency tests."""

    @pytest.mark.slow
    def test_vad_latency(self) -> None:
        """Test VAD processes chunks under 1ms."""
        from polyglott.vad.detector import VoiceActivityDetector

        detector = VoiceActivityDetector()
        chunk = np.random.randn(VAD_CHUNK_SAMPLES).astype(np.float32) * 0.1

        # Warm up
        detector.process_chunk(chunk)

        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            detector.process_chunk(chunk)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        assert avg_time < 1.0, f"VAD latency {avg_time:.2f}ms exceeds 1ms target"

    @pytest.mark.slow
    def test_memory_usage(self) -> None:
        """Test memory usage stays within bounds."""
        import tracemalloc

        tracemalloc.start()

        # Create all components
        from polyglott.vad.detector import VoiceActivityDetector
        from polyglott.stt.transcriber import SpeechTranscriber
        from polyglott.tts.synthesizer import SpeechSynthesizer

        vad = VoiceActivityDetector()
        stt = SpeechTranscriber()
        tts = SpeechSynthesizer()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Should be under 500MB for just the Python objects
        assert peak < 500 * 1024 * 1024, f"Memory usage {peak / 1024 / 1024:.1f}MB exceeds limit"
```

### Step 6.4: Test Fixtures

**File:** `tests/conftest.py` (update)

```python
import numpy as np
import pytest
from polyglott.constants import AUDIO_SAMPLE_RATE, VAD_CHUNK_SAMPLES


@pytest.fixture
def sample_rate() -> int:
    """Standard sample rate for tests."""
    return AUDIO_SAMPLE_RATE


@pytest.fixture
def silence_chunk() -> np.ndarray:
    """Chunk of silence audio."""
    return np.zeros(VAD_CHUNK_SAMPLES, dtype=np.float32)


@pytest.fixture
def speech_chunk() -> np.ndarray:
    """Chunk of simulated speech audio (sine wave)."""
    t = np.linspace(0, VAD_CHUNK_SAMPLES / AUDIO_SAMPLE_RATE, VAD_CHUNK_SAMPLES)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


@pytest.fixture
def sample_audio_1s() -> np.ndarray:
    """1 second of sample audio."""
    t = np.linspace(0, 1.0, AUDIO_SAMPLE_RATE)
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.1 * np.sin(2 * np.pi * 800 * t)
    )
    return audio.astype(np.float32)


@pytest.fixture
def sample_text_en() -> str:
    """Sample English text."""
    return "Hello, how are you today?"


@pytest.fixture
def sample_text_es() -> str:
    """Sample Spanish text."""
    return "Hola, ¿cómo estás hoy?"


@pytest.fixture
def sample_text_de() -> str:
    """Sample German text."""
    return "Hallo, wie geht es dir heute?"


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "requires_ollama: requires Ollama running")
```

### Step 6.5: Documentation Finalization

**Update all documentation:**

1. **README.md** - Ensure quickstart is accurate
2. **doc/user/getting_started.md** - Complete user guide
3. **doc/user/faq.md** - Address common questions
4. **doc/developers/architecture.md** - Update with final design
5. **doc/developers/contributing.md** - Clear contribution guidelines
6. **doc/llm/*.md** - Ensure library docs are current

### Step 6.6: Bug Fixes and Polish

**Common issues to check:**

1. **Audio glitches**
   - Buffer underruns
   - Sample rate mismatches
   - Normalization issues

2. **Timeout handling**
   - LLM response timeouts
   - Audio recording timeouts
   - Network errors (for Ollama)

3. **Error messages**
   - User-friendly error messages
   - Helpful suggestions for fixes

4. **Edge cases**
   - Empty input handling
   - Very long input handling
   - Special characters in text

## Test Commands

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=polyglott --cov-report=html

# Run specific test file
uv run pytest tests/vad/test_detector.py

# Run only fast tests
uv run pytest -m "not slow"

# Run integration tests
uv run pytest -m integration

# Run with verbose output
uv run pytest -v

# Run tests matching pattern
uv run pytest -k "test_initialization"
```

## Coverage Targets

| Module | Target | Current |
|--------|--------|---------|
| vad | 90% | TBD |
| stt | 80% | TBD |
| tts | 80% | TBD |
| llm | 85% | TBD |
| audio | 75% | TBD |
| conversation | 85% | TBD |
| cli | 70% | TBD |

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| VAD latency | <1ms | Per chunk |
| STT latency | <500ms | Per utterance |
| LLM first token | <2s | Initial response |
| TTS latency | <300ms | Per sentence |
| Total round-trip | <4s | End-to-end |
| Memory usage | <15GB | All components loaded |

## Final Checklist

### Functionality
- [ ] VAD correctly detects speech
- [ ] STT transcribes all languages
- [ ] LLM generates appropriate responses
- [ ] TTS speaks all languages
- [ ] Pipeline flows correctly
- [ ] Session management works

### Quality
- [ ] Code follows golden rules
- [ ] All modules under 500 lines
- [ ] No magic numbers
- [ ] Complete docstrings and type hints
- [ ] Tests pass

### Documentation
- [ ] README is current
- [ ] User docs are complete
- [ ] Developer docs are complete
- [ ] LLM docs are current

### Performance
- [ ] Latency targets met
- [ ] Memory usage acceptable
- [ ] No memory leaks

## Files Modified/Created

| File | Status | Lines |
|------|--------|-------|
| `tests/conftest.py` | ✅ Updated | ~90 |
| `tests/integration/test_audio_pipeline.py` | 🔲 To create | ~100 |
| `tests/integration/test_conversation_flow.py` | 🔲 To create | ~80 |
| `tests/performance/test_latency.py` | 🔲 To create | ~60 |

## Completion Criteria

The project is considered complete when:

1. All unit tests pass
2. Integration tests pass
3. Performance targets are met
4. Documentation is complete
5. Code review checklist passes
6. Manual testing confirms usability

## Post-Launch Improvements (Future)

- Progress tracking and statistics
- Vocabulary games
- Pronunciation feedback
- Parent dashboard
- More languages
- Custom voice cloning
