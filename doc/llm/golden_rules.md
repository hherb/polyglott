# Golden Rules for Polyglott Development

**IMPORTANT: Always read this file before implementing any new functionality!**

## 0. Package Management
- **Use `uv` as the package manager, NEVER `pip` directly**
- All dependencies must be managed through `pyproject.toml`
- Use `uv add <package>` to add dependencies
- Use `uv sync` to install dependencies

## 1. Code Architecture
- **Prefer reusable pure functions over monolithic code**
- Functions should be small, focused, and testable
- Avoid side effects where possible
- Use dependency injection for external resources

## 2. Module Size Limits
- **No individual module should exceed 500 lines of code**
- If a module grows beyond this limit, refactor into smaller modules
- Split by functionality, not arbitrarily

## 3. No Magic Numbers
- **Always use helpfully named constants**
- Constants should be defined at the module level or in a dedicated `constants.py`
- Names should be descriptive and follow SCREAMING_SNAKE_CASE

```python
# Bad
if audio_length > 30:
    pass

# Good
MAX_AUDIO_CHUNK_SECONDS = 30
if audio_length > MAX_AUDIO_CHUNK_SECONDS:
    pass
```

## 4. Documentation Requirements
- **Always use docstrings and type hints - NO EXCEPTIONS!**
- Every function, class, and module must have a docstring
- Use Google-style docstrings
- Type hints are mandatory for all function parameters and return values

```python
def process_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> str:
    """Process audio data and return transcribed text.

    Args:
        audio_data: Raw audio samples as numpy array.
        sample_rate: Audio sample rate in Hz.

    Returns:
        Transcribed text from the audio.

    Raises:
        ValueError: If audio_data is empty.
    """
    pass
```

## 5. Documentation Maintenance
- **Keep updating documentation in three locations:**
  - `doc/llm/` - For LLMs assisting in implementation (like yourself)
  - `doc/user/` - For end users of the application
  - `doc/developers/` - For human developers joining the project

## 6. Testing Requirements
- **Always write a test suite for all new functionality**
- Tests go in `tests/` mirroring the `src/` structure
- Use pytest as the testing framework
- Aim for high coverage of critical paths

## 7. README Maintenance
- **Keep updating README.md about the current state of the project**
- Must include a "Quickstart" section with installation and usage instructions
- Document any breaking changes

## 8. Library Documentation
- **If unsure about a library or API, ALWAYS look up the newest documentation online first**
- Update `doc/llm/<library>.md` accordingly
- This prevents repeated web lookups for the same information
- Include version numbers and example usage

---

## Project-Specific Standards

### Audio Constants
- Sample rate: 16000 Hz (standard for speech recognition)
- Chunk duration: 30ms for VAD processing
- Audio format: 16-bit PCM mono

### Supported Languages
- Student languages: English, German
- Target languages: German, English, Spanish, Japanese, Mandarin

### Target Platform
- macOS with Apple Silicon (M1/M2/M3)
- 32GB unified RAM minimum
- Fully offline operation required
