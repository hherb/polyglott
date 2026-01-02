# Contributing to Polyglott

Thank you for your interest in contributing to Polyglott! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.11 or later
- uv package manager
- Git
- macOS with Apple Silicon (for full functionality)
- Ollama (for LLM testing)

### Setting Up Your Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/polyglott.git
cd polyglott

# Install dependencies including dev extras
uv sync --extra dev

# Install pre-commit hooks (optional but recommended)
uv run pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=polyglott

# Run specific test file
uv run pytest tests/vad/test_detector.py

# Run tests matching a pattern
uv run pytest -k "test_initialization"
```

### Code Quality

```bash
# Format code
uv run ruff format src tests

# Lint code
uv run ruff check src tests

# Type checking
uv run mypy src
```

## Golden Rules

**Before writing any code, read `doc/llm/golden_rules.md`!**

Key rules:
1. Use `uv` as package manager, never `pip`
2. Prefer reusable pure functions
3. Keep modules under 500 lines
4. No magic numbers - use named constants
5. Always use docstrings and type hints
6. Write tests for all new functionality
7. Keep documentation updated

## Code Style

### Python Style

- Follow PEP 8 with 100 character line limit
- Use Google-style docstrings
- Type hints are mandatory

```python
def process_audio(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
) -> TranscriptionResult:
    """Process audio data and return transcribed text.

    Args:
        audio_data: Raw audio samples as numpy array.
        sample_rate: Audio sample rate in Hz.

    Returns:
        TranscriptionResult with transcribed text.

    Raises:
        ValueError: If audio_data is empty.
    """
    pass
```

### Naming Conventions

- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Private: prefix with `_`

### Module Structure

Each module should have:
- `__init__.py` with public exports
- Implementation files under 500 lines
- Corresponding tests in `tests/`

## Pull Request Process

### Before Submitting

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Write tests for your changes
3. Run the test suite: `uv run pytest`
4. Run linting: `uv run ruff check src tests`
5. Update documentation if needed

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Docstrings added for new functions
- [ ] Type hints added
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)

### Commit Messages

Use conventional commits:

```
feat: add Japanese voice support
fix: correct audio buffer overflow
docs: update installation instructions
test: add VAD threshold tests
refactor: simplify audio pipeline
```

## Project Structure

```
polyglott/
├── src/polyglott/       # Main source code
├── tests/               # Test suite
├── doc/
│   ├── llm/             # LLM assistant documentation
│   ├── user/            # User documentation
│   └── developers/      # Developer documentation
├── pyproject.toml       # Project configuration
└── README.md            # Project overview
```

## Adding New Features

### 1. Design First

For significant features:
1. Open an issue to discuss the design
2. Get feedback before implementing
3. Consider backwards compatibility

### 2. Implement

1. Start with tests (TDD encouraged)
2. Implement the feature
3. Update constants if needed
4. Add error handling

### 3. Document

1. Add docstrings to all functions
2. Update `doc/llm/` if using new libraries
3. Update `doc/user/` for user-facing features
4. Update `doc/developers/` for architecture changes

### 4. Test

1. Unit tests for all functions
2. Integration tests for pipelines
3. Test edge cases

## Getting Help

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues and discussions first

## License

By contributing, you agree that your contributions will be licensed under the project's AGPL-3.0 license.
