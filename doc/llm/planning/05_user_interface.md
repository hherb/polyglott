# Phase 5: User Interface

**Phase:** 5 of 6
**Prerequisites:** Phases 3 & 4 (Speech Processing, LLM Integration)
**Estimated Complexity:** Medium

## Objectives

1. Create intuitive CLI with interactive setup
2. Support both text and voice interaction modes
3. Implement session management and tracking
4. Provide clear status feedback during operation

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                           CLI                                  │   │
│  │                                                                │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐ │   │
│  │  │  Interactive    │  │   Text Mode     │  │  Voice Mode   │ │   │
│  │  │  Setup          │  │                 │  │               │ │   │
│  │  │                 │  │  • Type input   │  │  • Speak      │ │   │
│  │  │  • Language     │  │  • Read output  │  │  • Listen     │ │   │
│  │  │  • Age group    │  │  • Commands     │  │  • Status     │ │   │
│  │  │  • Name         │  │                 │  │               │ │   │
│  │  └─────────────────┘  └─────────────────┘  └───────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Conversation Manager                         │   │
│  │                                                                │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐ │   │
│  │  │ ConversationSess│  │  State Machine  │  │   Callbacks   │ │   │
│  │  │                 │  │                 │  │               │ │   │
│  │  │  • session_id   │  │  • IDLE         │  │  • on_turn    │ │   │
│  │  │  • turn_count   │  │  • LISTENING    │  │  • on_state   │ │   │
│  │  │  • words_learned│  │  • THINKING     │  │               │ │   │
│  │  │  • duration     │  │  • SPEAKING     │  │               │ │   │
│  │  └─────────────────┘  └─────────────────┘  └───────────────┘ │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Implementation

### Step 5.1: CLI Main Module

**File:** `src/polyglott/cli.py`

**Entry Point:**

```python
def main() -> int:
    """Main entry point for the CLI."""
    print_banner()

    if not check_dependencies():
        return 1

    # Interactive setup
    target_language = select_language()
    age_group = select_age_group()
    student_name = get_student_name()

    # Choose mode
    mode = select_mode()

    if mode == "voice":
        run_voice_mode(target_language, native_language, age_group, student_name)
    else:
        run_text_mode(target_language, native_language, age_group, student_name)

    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### Step 5.2: Interactive Setup Functions

```python
def print_banner() -> None:
    """Print the application banner."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  {APP_NAME} v{APP_VERSION}                                           ║
║  Your friendly language learning companion                    ║
╚══════════════════════════════════════════════════════════════╝
""")

def select_language() -> TargetLanguage:
    """Interactive language selection."""
    print("\nWhat language would you like to learn?")
    languages = list(TargetLanguage)

    for i, lang in enumerate(languages, 1):
        name = LANGUAGE_NAMES.get(lang.value, lang.value)
        print(f"  {i}. {name}")

    while True:
        try:
            choice = input("\nEnter number (1-5): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(languages):
                return languages[idx]
        except ValueError:
            pass
        print("Please enter a valid number.")

def select_age_group() -> AgeGroup:
    """Interactive age group selection."""
    print("\nWhat's the learner's age group?")
    groups = [
        (AgeGroup.PRESCHOOL, "Preschool (3-5 years)"),
        (AgeGroup.EARLY_PRIMARY, "Early Primary (6-8 years)"),
        (AgeGroup.LATE_PRIMARY, "Late Primary (9-12 years)"),
    ]

    for i, (_, desc) in enumerate(groups, 1):
        print(f"  {i}. {desc}")

    while True:
        try:
            choice = input("\nEnter number (1-3): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(groups):
                return groups[idx][0]
        except ValueError:
            pass
        print("Please enter a valid number.")

def get_student_name() -> str:
    """Get the student's name."""
    name = input("\nWhat's your name? ").strip()
    return name if name else "Friend"
```

### Step 5.3: Dependency Checking

```python
def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    missing = []

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")

    try:
        import torch
    except ImportError:
        missing.append("torch (for Silero VAD)")

    # Check Ollama
    try:
        import ollama
        ollama.list()
    except Exception:
        print_status("Ollama not running. Start with: ollama serve", "warning")
        print_status("Then pull a model: ollama pull qwen2.5:7b", "info")

    if missing:
        print_status(f"Missing dependencies: {', '.join(missing)}", "error")
        print_status("Install with: uv sync", "info")
        return False

    return True
```

### Step 5.4: Text Mode Implementation

```python
def run_text_mode(
    target_language: TargetLanguage,
    native_language: str,
    age_group: AgeGroup,
    student_name: str,
) -> None:
    """Run the tutor in text-only mode."""
    from polyglott.llm.tutor import LanguageTutor, TutorConfig

    config = TutorConfig(
        target_language=target_language,
        native_language=native_language,
        age_group=age_group,
    )
    tutor = LanguageTutor(config)

    greeting = tutor.reset_conversation()
    print(f"\n🎓 Tutor: {greeting}\n")

    print("(Type 'quit' to exit, 'lesson' to change topic)\n")

    while True:
        try:
            user_input = input(f"👤 {student_name}: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("\n🎓 Tutor: Great job today! See you next time!")
            break

        if user_input.lower() == "lesson":
            lessons = tutor.get_available_lessons()
            print("\nAvailable lessons:")
            for i, lesson in enumerate(lessons, 1):
                print(f"  {i}. {lesson}")
            try:
                choice = int(input("Choose: ")) - 1
                if 0 <= choice < len(lessons):
                    tutor.set_lesson_focus(lessons[choice])
                    print(f"\n🎓 Tutor: Great! Let's practice {lessons[choice]}!\n")
            except (ValueError, IndexError):
                pass
            continue

        response = tutor.respond(user_input)
        print(f"\n🎓 Tutor: {response.text}\n")
```

### Step 5.5: Voice Mode Implementation

```python
def run_voice_mode(
    target_language: TargetLanguage,
    native_language: str,
    age_group: AgeGroup,
    student_name: str,
) -> None:
    """Run the tutor in voice mode with audio."""
    from polyglott.audio.pipeline import PipelineState
    from polyglott.conversation.manager import ConversationManager

    def on_state_change(state: PipelineState) -> None:
        """Handle state changes."""
        state_messages = {
            PipelineState.LISTENING: "Listening...",
            PipelineState.TRANSCRIBING: "Processing...",
            PipelineState.THINKING: "Thinking...",
            PipelineState.SPEAKING: "Speaking...",
        }
        if state in state_messages:
            print_status(state_messages[state], state.value)

    manager = ConversationManager(
        student_name=student_name,
        target_language=target_language,
        native_language=native_language,
        age_group=age_group,
        on_state_change=on_state_change,
    )

    print_status("Starting voice session...", "info")
    print("(Press Ctrl+C to exit)\n")

    greeting = manager.start()
    print(f"\n🎓 Tutor: {greeting}\n")

    try:
        manager.run_conversation_loop()
    except KeyboardInterrupt:
        pass
    finally:
        summary = manager.end_session(say_goodbye=True)
        print(f"\n{summary}")
```

### Step 5.6: Session Management

**File:** `src/polyglott/conversation/session.py`

```python
@dataclass
class SessionConfig:
    """Configuration for a tutoring session."""
    student_name: str = "Student"
    native_language: str = "English"
    target_language: TargetLanguage = TargetLanguage.SPANISH
    age_group: AgeGroup = AgeGroup.EARLY_PRIMARY
    lesson_focus: str = "greetings and saying hello/goodbye"

@dataclass
class ConversationSession:
    """A tutoring conversation session."""
    session_id: str = field(default_factory=lambda: str(uuid4())[:8])
    config: SessionConfig = field(default_factory=SessionConfig)
    started_at: datetime = field(default_factory=datetime.now)
    turn_count: int = 0
    words_learned: list[str] = field(default_factory=list)

    def record_turn(self, user_text: str, tutor_text: str) -> None:
        """Record a conversation turn."""
        self.turn_count += 1

    def add_word_learned(self, word: str) -> None:
        """Add a word to the learned vocabulary."""
        if word not in self.words_learned:
            self.words_learned.append(word)

    @property
    def duration_minutes(self) -> float:
        """Get session duration in minutes."""
        delta = datetime.now() - self.started_at
        return delta.total_seconds() / 60

    def get_summary(self) -> str:
        """Get a summary of the session."""
        return (
            f"Session {self.session_id}\n"
            f"Student: {self.config.student_name}\n"
            f"Learning: {self.config.target_language.value}\n"
            f"Duration: {self.duration_minutes:.1f} minutes\n"
            f"Turns: {self.turn_count}\n"
            f"Words learned: {len(self.words_learned)}"
        )
```

### Step 5.7: Conversation Manager

**File:** `src/polyglott/conversation/manager.py`

```python
class ConversationManager:
    """High-level conversation manager."""

    def __init__(
        self,
        student_name: str = "Student",
        target_language: TargetLanguage = TargetLanguage.SPANISH,
        native_language: str = "English",
        age_group: AgeGroup = AgeGroup.EARLY_PRIMARY,
        on_turn_complete: Optional[TurnCallback] = None,
        on_state_change: Optional[StateCallback] = None,
    ) -> None: ...

    def start(self) -> str:
        """Start a new conversation session."""
        ...

    def process_turn(self) -> ConversationTurn:
        """Process a single conversation turn."""
        ...

    def run_conversation_loop(
        self,
        max_turns: int = 100,
        exit_phrases: Optional[list[str]] = None,
    ) -> None:
        """Run the main conversation loop."""
        exit_phrases = exit_phrases or ["goodbye", "bye", "quit", "exit", "stop"]

        for _ in range(max_turns):
            if not self._is_running:
                break

            turn = self.process_turn()

            if turn.user_text:
                user_lower = turn.user_text.lower()
                if any(phrase in user_lower for phrase in exit_phrases):
                    self.end_session()
                    break

    def end_session(self, say_goodbye: bool = True) -> str:
        """End the current session."""
        ...
```

## Status Feedback System

```python
def print_status(message: str, status: str = "info") -> None:
    """Print a status message with formatting."""
    icons = {
        "info": "ℹ️ ",
        "success": "✓",
        "warning": "⚠️ ",
        "error": "✗",
        "listening": "🎤",
        "thinking": "💭",
        "speaking": "🔊",
    }
    icon = icons.get(status, "")
    print(f"{icon} {message}")
```

## Commands Available

| Command | Description |
|---------|-------------|
| `quit` / `exit` / `bye` | End the session |
| `lesson` | Change lesson topic |
| `Ctrl+C` | Interrupt immediately |

## Verification Checklist

- [ ] Banner displays correctly
- [ ] Language selection works
- [ ] Age group selection works
- [ ] Name input works
- [ ] Text mode conversation works
- [ ] Voice mode state display works
- [ ] Session summary displays
- [ ] Graceful exit on Ctrl+C

## Testing Strategy

### Manual Tests

1. **Fresh start:** Run `uv run polyglott` from scratch
2. **Language selection:** Test each language option
3. **Text mode:** Complete a short conversation
4. **Voice mode:** Test speech recognition and synthesis
5. **Exit commands:** Test quit, exit, bye, Ctrl+C

### Unit Tests

```python
def test_session_creation():
    session = create_session(student_name="Emma")
    assert session.config.student_name == "Emma"
    assert session.turn_count == 0

def test_session_summary():
    session = create_session()
    session.record_turn("Hi", "Hello!")
    summary = session.get_summary()
    assert "1" in summary  # turn count
```

## Files Modified/Created

| File | Status | Lines |
|------|--------|-------|
| `src/polyglott/cli.py` | ✅ Created | ~305 |
| `src/polyglott/conversation/session.py` | ✅ Created | ~121 |
| `src/polyglott/conversation/manager.py` | ✅ Created | ~239 |
| `tests/conversation/test_session.py` | ✅ Created | ~102 |

## Next Phase

Proceed to [Phase 6: Testing & Polish](06_testing_polish.md)
