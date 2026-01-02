# Phase 4: LLM Integration

**Phase:** 4 of 6
**Prerequisites:** Phase 1 (Core Infrastructure)
**Estimated Complexity:** Medium

## Objectives

1. Integrate with Ollama for local LLM inference
2. Create child-friendly, age-appropriate system prompts
3. Implement conversation history management
4. Support lesson-based tutoring with vocabulary focus

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LLM Integration                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      LanguageTutor                             │   │
│  │                                                                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐   │   │
│  │  │ TutorConfig │  │   History   │  │   System Prompt     │   │   │
│  │  │             │  │             │  │                     │   │   │
│  │  │ • Model     │  │ • Messages  │  │ • Language focus    │   │   │
│  │  │ • Language  │  │ • Max size  │  │ • Age appropriate   │   │   │
│  │  │ • Age group │  │ • Trimming  │  │ • Lesson topic      │   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘   │   │
│  │                                                                │   │
│  │  ┌────────────────────────────────────────────────────────┐   │   │
│  │  │                    Ollama Client                        │   │   │
│  │  │  • chat()  • stream()  • options: temp, tokens         │   │   │
│  │  └────────────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                        Prompts                                 │   │
│  │                                                                │   │
│  │  • BASE_TUTOR_PROMPT      - Core tutoring behavior            │   │
│  │  • AGE_DESCRIPTIONS       - Age-appropriate language          │   │
│  │  • LESSON_FOCUSES         - Topics by age group               │   │
│  │  • LANGUAGE_GREETINGS     - Greetings in all languages        │   │
│  │  • CONVERSATION_STARTERS  - Session openers                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Implementation

### Step 4.1: Prompts Module

**File:** `src/polyglott/llm/prompts.py`

**Base System Prompt:**

```python
BASE_TUTOR_PROMPT: Final[str] = """You are a friendly, patient language tutor \
helping a young child learn {target_language}.

Key guidelines:
- Be warm, encouraging, and use simple language appropriate for a {age_description}
- Keep your responses SHORT (1-3 sentences maximum)
- Focus on basic conversational phrases and vocabulary
- When the child makes a mistake, gently correct them and encourage them to try again
- Use lots of positive reinforcement ("Great job!", "Well done!", "You're learning so fast!")
- If the child seems confused, simplify and try a different approach
- Introduce new words gradually, one or two at a time
- Always provide pronunciation hints in parentheses when teaching new words

The child's native language is {native_language}.

Current lesson focus: {lesson_focus}

Remember: You are speaking with a young child. Be patient, kind, and make learning fun!"""
```

**Age-Appropriate Descriptions:**

```python
AGE_DESCRIPTIONS: Final[dict[AgeGroup, str]] = {
    AgeGroup.PRESCHOOL: "preschooler (3-5 years old)",
    AgeGroup.EARLY_PRIMARY: "young child (6-8 years old)",
    AgeGroup.LATE_PRIMARY: "child (9-12 years old)",
}
```

**Lesson Topics by Age:**

```python
LESSON_FOCUSES: Final[dict[AgeGroup, list[str]]] = {
    AgeGroup.PRESCHOOL: [
        "greetings and saying hello/goodbye",
        "colors and shapes",
        "numbers 1-10",
        "animals",
        "family members (mom, dad, etc.)",
        "basic feelings (happy, sad, hungry)",
    ],
    AgeGroup.EARLY_PRIMARY: [
        "introducing yourself",
        "asking simple questions",
        "days of the week",
        "food and drinks",
        "weather and seasons",
        "hobbies and activities",
    ],
    AgeGroup.LATE_PRIMARY: [
        "having a short conversation",
        "describing people and places",
        "talking about your day",
        "asking for help or directions",
        "expressing opinions",
        "telling a simple story",
    ],
}
```

**Greetings in All Languages:**

```python
LANGUAGE_GREETINGS: Final[dict[str, dict[str, str]]] = {
    "en": {
        "hello": "Hello!",
        "goodbye": "Goodbye!",
        "how_are_you": "How are you?",
        "thank_you": "Thank you!",
    },
    "de": {
        "hello": "Hallo!",
        "goodbye": "Tschüss!",
        "how_are_you": "Wie geht es dir?",
        "thank_you": "Danke!",
    },
    "es": {
        "hello": "¡Hola!",
        "goodbye": "¡Adiós!",
        "how_are_you": "¿Cómo estás?",
        "thank_you": "¡Gracias!",
    },
    "ja": {
        "hello": "こんにちは! (Konnichiwa)",
        "goodbye": "さようなら! (Sayounara)",
        "how_are_you": "お元気ですか? (Ogenki desu ka?)",
        "thank_you": "ありがとう! (Arigatou)",
    },
    "zh": {
        "hello": "你好! (Nǐ hǎo)",
        "goodbye": "再见! (Zàijiàn)",
        "how_are_you": "你好吗? (Nǐ hǎo ma?)",
        "thank_you": "谢谢! (Xièxiè)",
    },
}
```

**Helper Functions:**

```python
def build_system_prompt(
    target_language: TargetLanguage,
    native_language: str,
    age_group: AgeGroup,
    lesson_focus: str,
) -> str:
    """Build a complete system prompt for the tutor."""
    ...

def get_lesson_focuses(age_group: AgeGroup) -> list[str]:
    """Get available lesson focuses for an age group."""
    ...

def get_greeting(language: str, greeting_type: str = "hello") -> str:
    """Get a greeting in the specified language."""
    ...

def get_conversation_starter(
    native_language: str,
    target_language: str,
) -> str:
    """Get a conversation starter in the child's native language."""
    ...
```

### Step 4.2: Tutor Configuration

**File:** `src/polyglott/llm/tutor.py`

**Data Classes:**

```python
@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

@dataclass
class TutorResponse:
    """Response from the language tutor."""
    text: str
    is_complete: bool = True
    tokens_generated: Optional[int] = None

@dataclass
class TutorConfig:
    """Configuration for the language tutor."""
    model: str = DEFAULT_LLM_MODEL
    target_language: TargetLanguage = TargetLanguage.SPANISH
    native_language: str = "English"
    age_group: AgeGroup = AgeGroup.EARLY_PRIMARY
    lesson_focus: str = "greetings and saying hello/goodbye"
    temperature: float = LLM_TEMPERATURE
    max_tokens: int = MAX_LLM_RESPONSE_TOKENS
```

### Step 4.3: Language Tutor Class

```python
class LanguageTutor:
    """Conversational language tutor using local LLM."""

    def __init__(self, config: Optional[TutorConfig] = None) -> None:
        self.config = config or TutorConfig()
        self._client = None
        self._history: list[Message] = []
        self._system_prompt: Optional[str] = None

    @property
    def system_prompt(self) -> str:
        """Get the current system prompt."""
        if self._system_prompt is None:
            self._system_prompt = build_system_prompt(
                target_language=self.config.target_language,
                native_language=self.config.native_language,
                age_group=self.config.age_group,
                lesson_focus=self.config.lesson_focus,
            )
        return self._system_prompt

    def reset_conversation(self) -> str:
        """Reset conversation and return greeting."""
        self._history.clear()
        self._system_prompt = None
        return get_conversation_starter(...)

    def respond(self, user_input: str) -> TutorResponse:
        """Generate a response to user input."""
        self._history.append(Message(role="user", content=user_input))
        messages = self._build_messages()

        response = self._client.chat(
            model=self.config.model,
            messages=messages,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        )

        assistant_text = response["message"]["content"]
        self._history.append(Message(role="assistant", content=assistant_text))
        self._trim_history()

        return TutorResponse(text=assistant_text)

    def respond_stream(self, user_input: str) -> Generator[str, None, TutorResponse]:
        """Generate streaming response."""
        # Similar to respond() but yields chunks
        ...

    def _build_messages(self) -> list[dict[str, str]]:
        """Build message list for API call."""
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in self._history:
            messages.append(msg.to_dict())
        return messages

    def _trim_history(self) -> None:
        """Keep history within limits."""
        while len(self._history) > MAX_CONVERSATION_HISTORY:
            self._history.pop(0)
            if self._history:
                self._history.pop(0)  # Remove pairs
```

### Step 4.4: Ollama Integration

**Key Ollama API Usage:**

```python
from ollama import chat, ResponseError

# Basic chat
response = chat(
    model='qwen2.5:7b',
    messages=[
        {'role': 'system', 'content': '...'},
        {'role': 'user', 'content': 'Hello!'},
    ],
    options={
        'temperature': 0.7,
        'num_predict': 256,
    },
)
print(response['message']['content'])

# Streaming
stream = chat(
    model='qwen2.5:7b',
    messages=[...],
    stream=True,
)
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

### Step 4.5: Ollama Availability Check

```python
def check_ollama_available() -> tuple[bool, str]:
    """Check if Ollama is available and running."""
    try:
        import ollama
        models = ollama.list()
        model_names = [m["name"] for m in models.get("models", [])]

        if not model_names:
            return False, "No models installed. Run: ollama pull qwen2.5:7b"

        return True, f"Available models: {', '.join(model_names[:5])}"

    except ImportError:
        return False, "Ollama not installed. Run: uv add ollama"
    except Exception as e:
        return False, f"Ollama not running. Start with: ollama serve ({e})"
```

## Recommended LLM Models

| Model | Size | RAM | Speed | Quality | Best For |
|-------|------|-----|-------|---------|----------|
| qwen2.5:3b | 3B | ~4GB | Fast | Good | Quick responses |
| qwen2.5:7b | 7B | ~8GB | Medium | Better | **Default choice** |
| qwen2.5:14b | 14B | ~16GB | Slower | Best | Maximum quality |
| llama3.1:8b | 8B | ~8GB | Medium | Good | Alternative |

## Prompt Engineering Guidelines

### For Children

1. **Short responses:** 1-3 sentences maximum
2. **Simple vocabulary:** Match age group
3. **Encouragement:** Lots of positive feedback
4. **Patience:** Never express frustration
5. **Pronunciation:** Always include hints for new words

### Example Interactions

**Preschool (3-5):**
```
User: How do I say dog?
Tutor: In Spanish, we say "perro" (PEHR-roh)! Can you say "perro"? 🐕
```

**Early Primary (6-8):**
```
User: How do I ask someone their name?
Tutor: Great question! You can say "¿Cómo te llamas?" (KOH-moh teh YAH-mahs).
       It means "What is your name?" Try saying it!
```

**Late Primary (9-12):**
```
User: How would I introduce myself?
Tutor: You could say "Me llamo [your name]" (meh YAH-moh) which means
       "My name is..." For example, "Me llamo Emma." Want to try?
```

## Verification Checklist

- [ ] Ollama connection works
- [ ] System prompt generates correctly
- [ ] Response length is appropriate
- [ ] History management works
- [ ] Streaming responses work
- [ ] Age-appropriate vocabulary used
- [ ] Lesson topics are relevant

## Testing Strategy

### Unit Tests

```python
def test_message_to_dict():
    msg = Message(role="user", content="Hello")
    assert msg.to_dict() == {"role": "user", "content": "Hello"}

def test_tutor_config_defaults():
    config = TutorConfig()
    assert config.model == "qwen2.5:7b"
    assert config.target_language == TargetLanguage.SPANISH

def test_system_prompt_generation():
    config = TutorConfig(
        target_language=TargetLanguage.SPANISH,
        native_language="English",
        lesson_focus="greetings",
    )
    tutor = LanguageTutor(config)
    prompt = tutor.system_prompt
    assert "Spanish" in prompt
    assert "English" in prompt
    assert "greetings" in prompt
```

### Integration Tests

```python
@pytest.mark.skipif(not ollama_available(), reason="Ollama not running")
def test_basic_conversation():
    tutor = LanguageTutor()
    tutor.reset_conversation()
    response = tutor.respond("How do I say hello?")
    assert response.text
    assert response.is_complete
```

## Files Modified/Created

| File | Status | Lines |
|------|--------|-------|
| `src/polyglott/llm/__init__.py` | ✅ Created | ~9 |
| `src/polyglott/llm/prompts.py` | ✅ Created | ~183 |
| `src/polyglott/llm/tutor.py` | ✅ Created | ~350 |
| `tests/llm/test_tutor.py` | ✅ Created | ~153 |

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| First token | <2s | Initial response |
| Full response | <5s | Complete generation |
| History trim | <1ms | Efficient |

## Next Phase

Proceed to [Phase 5: User Interface](05_user_interface.md)
