# Phase 2: Audio Pipeline

**Phase:** 2 of 6
**Prerequisites:** Phase 1 (Core Infrastructure)
**Estimated Complexity:** Medium

## Objectives

1. Implement Voice Activity Detection with Silero VAD
2. Create audio recorder with VAD-based endpoint detection
3. Create audio player for speech playback
4. Build pipeline orchestration layer

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Audio Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐                              ┌────────────────┐ │
│  │  Microphone │                              │    Speaker     │ │
│  └──────┬─────┘                              └───────▲────────┘ │
│         │                                            │          │
│         ▼                                            │          │
│  ┌────────────┐      ┌─────────────┐      ┌─────────┴────────┐ │
│  │  Recorder  │─────▶│     VAD     │      │     Player       │ │
│  │            │      │             │      │                  │ │
│  │ • Stream   │      │ • Silero    │      │ • Blocking mode  │ │
│  │ • Buffer   │      │ • States    │      │ • Async mode     │ │
│  │ • Chunks   │      │ • Threshold │      │ • Stop/resume    │ │
│  └────────────┘      └──────┬──────┘      └──────────────────┘ │
│                             │                       ▲          │
│                             ▼                       │          │
│                      ┌─────────────┐                │          │
│                      │  Pipeline   │────────────────┘          │
│                      │             │                           │
│                      │ • States    │                           │
│                      │ • Callbacks │                           │
│                      │ • Timing    │                           │
│                      └─────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Implementation

### Step 2.1: Voice Activity Detection

**File:** `src/polyglott/vad/detector.py`

**Classes to implement:**

```python
class SpeechState(str, Enum):
    """State machine for speech detection."""
    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEAKING = "speaking"
    SPEECH_END = "speech_end"

@dataclass
class VADResult:
    """Result from processing an audio chunk."""
    speech_probability: float
    is_speech: bool
    state: SpeechState

class VoiceActivityDetector:
    """Real-time VAD using Silero."""

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        threshold: float = VAD_SPEECH_THRESHOLD,
        speech_pad_frames: int = VAD_SPEECH_PAD_FRAMES,
        silence_pad_frames: int = VAD_SILENCE_PAD_FRAMES,
    ) -> None: ...

    def process_chunk(self, audio_chunk: np.ndarray) -> VADResult: ...
    def reset(self) -> None: ...

    @property
    def is_speaking(self) -> bool: ...
```

**Key Implementation Details:**

1. **Lazy Model Loading:**
   ```python
   @property
   def model(self) -> torch.jit.ScriptModule:
       if self._model is None:
           self._model = self._load_model()
       return self._model
   ```

2. **State Machine Logic:**
   ```python
   def _update_state(self, is_speech: bool) -> SpeechState:
       if is_speech:
           self._speech_frame_count += 1
           self._silence_frame_count = 0
       else:
           self._silence_frame_count += 1
           self._speech_frame_count = 0

       # Transition logic
       if not self._is_speaking:
           if self._speech_frame_count >= self.speech_pad_frames:
               self._is_speaking = True
               return SpeechState.SPEECH_START
           return SpeechState.SILENCE
       else:
           if self._silence_frame_count >= self.silence_pad_frames:
               self._is_speaking = False
               return SpeechState.SPEECH_END
           return SpeechState.SPEAKING
   ```

3. **Audio Normalization:**
   ```python
   def _prepare_audio(self, audio_chunk: np.ndarray) -> torch.Tensor:
       if audio_chunk.dtype != np.float32:
           audio_chunk = audio_chunk.astype(np.float32)
       max_val = np.abs(audio_chunk).max()
       if max_val > 1.0:
           audio_chunk = audio_chunk / max_val
       return torch.from_numpy(audio_chunk)
   ```

**Tests to write:**
- `test_initialization_default`
- `test_initialization_custom`
- `test_invalid_sample_rate`
- `test_reset`
- `test_process_chunk_wrong_size`
- `test_process_silence_chunk`
- `test_state_transitions`

### Step 2.2: Audio Recorder

**File:** `src/polyglott/audio/recorder.py`

**Classes to implement:**

```python
@dataclass
class RecordingResult:
    """Result from audio recording."""
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    was_speech_detected: bool

class AudioRecorder:
    """Real-time audio recorder with VAD endpoint detection."""

    def __init__(
        self,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        channels: int = AUDIO_CHANNELS,
        vad_detector: Optional[VoiceActivityDetector] = None,
    ) -> None: ...

    def record_utterance(
        self,
        max_duration: float = MAX_RECORDING_DURATION_SECONDS,
        silence_timeout: float = SILENCE_THRESHOLD_SECONDS,
        min_duration: float = MIN_SPEECH_DURATION_SECONDS,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
    ) -> RecordingResult: ...

    def record_fixed_duration(self, duration: float) -> RecordingResult: ...
    def stop(self) -> None: ...

    @property
    def is_recording(self) -> bool: ...
```

**Key Implementation Details:**

1. **Lazy sounddevice import:**
   ```python
   def _ensure_sounddevice(self) -> None:
       if self._sounddevice is None:
           import sounddevice as sd
           self._sounddevice = sd
   ```

2. **Streaming with VAD:**
   ```python
   with self._sounddevice.InputStream(
       samplerate=self.sample_rate,
       channels=self.channels,
       dtype=np.float32,
       blocksize=chunk_samples,
   ) as stream:
       for _ in range(max_chunks):
           chunk, overflowed = stream.read(chunk_samples)
           vad_result = self.vad.process_chunk(chunk)
           # Handle state transitions...
   ```

3. **Pre-speech buffer:**
   - Keep ~300ms of audio before speech detection
   - Prevents clipping the start of utterances

### Step 2.3: Audio Player

**File:** `src/polyglott/audio/player.py`

**Classes to implement:**

```python
class AudioPlayer:
    """Audio player for speech playback."""

    def __init__(self) -> None: ...

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int = TTS_SAMPLE_RATE,
        blocking: bool = False,
    ) -> None: ...

    def stop(self) -> None: ...
    def wait(self) -> None: ...

    @property
    def is_playing(self) -> bool: ...
```

**Key Implementation Details:**

1. **Non-blocking playback:**
   ```python
   def play(self, audio, sample_rate, blocking=False):
       if blocking:
           self._play_blocking(audio, sample_rate)
       else:
           self._playback_thread = Thread(
               target=self._play_blocking,
               args=(audio, sample_rate),
               daemon=True,
           )
           self._playback_thread.start()
   ```

2. **Interruption support:**
   ```python
   def stop(self) -> None:
       self._stop_event.set()
       if self._sounddevice is not None:
           self._sounddevice.stop()
       self._is_playing = False
   ```

### Step 2.4: Pipeline Orchestration

**File:** `src/polyglott/audio/pipeline.py`

**Classes to implement:**

```python
class PipelineState(str, Enum):
    """Current state of the audio pipeline."""
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"

@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    user_audio: Optional[np.ndarray] = None
    user_text: str = ""
    tutor_text: str = ""
    tutor_audio: Optional[np.ndarray] = None
    state: PipelineState = PipelineState.IDLE

class AudioPipeline:
    """Complete audio processing pipeline."""

    def __init__(
        self,
        target_language: TargetLanguage,
        native_language: str,
        age_group: AgeGroup,
        on_state_change: Optional[StateCallback] = None,
    ) -> None: ...

    def start_session(self) -> str: ...
    def process_turn(self) -> ConversationTurn: ...
    def speak(self, text: str, language: Optional[str] = None) -> None: ...
    def stop(self) -> None: ...
```

**Key Implementation Details:**

1. **Lazy component initialization:**
   ```python
   def _ensure_components(self) -> None:
       if self._recorder is None:
           self._recorder = AudioRecorder()
       if self._transcriber is None:
           self._transcriber = SpeechTranscriber()
       # etc...
   ```

2. **State callbacks:**
   ```python
   def _set_state(self, state: PipelineState) -> None:
       self._state = state
       if self.on_state_change:
           self.on_state_change(state)
   ```

3. **Turn processing flow:**
   ```python
   def process_turn(self) -> ConversationTurn:
       # 1. Record (LISTENING)
       self._set_state(PipelineState.LISTENING)
       recording = self._recorder.record_utterance()

       # 2. Transcribe (TRANSCRIBING)
       self._set_state(PipelineState.TRANSCRIBING)
       transcription = self._transcriber.transcribe(recording.audio)

       # 3. Generate (THINKING)
       self._set_state(PipelineState.THINKING)
       response = self._tutor.respond(transcription.text)

       # 4. Synthesize & Play (SPEAKING)
       self._set_state(PipelineState.SPEAKING)
       synthesis = self._synthesizer.synthesize(response.text)
       self._player.play(synthesis.audio, blocking=True)

       self._set_state(PipelineState.IDLE)
       return turn
   ```

## Verification Checklist

- [ ] VAD correctly detects speech start/end
- [ ] Recorder captures clean audio with pre-roll
- [ ] Player can play and be interrupted
- [ ] Pipeline state transitions are correct
- [ ] Callbacks fire at appropriate times
- [ ] No audio artifacts or glitches

## Testing Strategy

### Unit Tests

| Test | Description |
|------|-------------|
| `test_vad_silence` | VAD returns low probability for silence |
| `test_vad_state_machine` | State transitions work correctly |
| `test_recorder_stop` | Recorder can be stopped mid-recording |
| `test_player_blocking` | Blocking playback works |
| `test_player_nonblocking` | Non-blocking playback works |

### Integration Tests

| Test | Description |
|------|-------------|
| `test_record_and_play` | Record audio, play it back |
| `test_vad_with_recorder` | VAD integration with recorder |
| `test_pipeline_states` | Full pipeline state flow |

## Files Modified/Created

| File | Status | Lines |
|------|--------|-------|
| `src/polyglott/vad/__init__.py` | ✅ Created | ~9 |
| `src/polyglott/vad/detector.py` | ✅ Created | ~250 |
| `src/polyglott/audio/recorder.py` | ✅ Created | ~253 |
| `src/polyglott/audio/player.py` | ✅ Created | ~135 |
| `src/polyglott/audio/pipeline.py` | ✅ Created | ~275 |
| `tests/vad/test_detector.py` | ✅ Created | ~117 |

## Performance Considerations

1. **VAD Processing:** <1ms per 30ms chunk - well within budget
2. **Audio Buffering:** Use numpy arrays for efficiency
3. **Thread Safety:** Player uses daemon threads
4. **Memory:** Audio buffers are limited to max recording duration

## Next Phase

Proceed to [Phase 3: Speech Processing](03_speech_processing.md)
