"""Language tutor powered by local LLM via Ollama.

This module provides the conversational AI backend for the
language tutor, using Qwen or other local models through Ollama.
"""

from dataclasses import dataclass, field
from typing import Generator, Optional

from polyglott.constants import (
    DEFAULT_LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    MAX_CONVERSATION_HISTORY,
    MAX_LLM_RESPONSE_TOKENS,
    AgeGroup,
    TargetLanguage,
)
from polyglott.llm.prompts import (
    build_system_prompt,
    get_conversation_starter,
    get_lesson_focuses,
)


@dataclass
class Message:
    """A single message in the conversation.

    Attributes:
        role: Message role (system, user, assistant).
        content: Message text content.
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary for Ollama API.

        Returns:
            Dictionary with role and content keys.
        """
        return {"role": self.role, "content": self.content}


@dataclass
class TutorResponse:
    """Response from the language tutor.

    Attributes:
        text: Generated response text.
        is_complete: Whether generation completed normally.
        tokens_generated: Number of tokens in response.
    """

    text: str
    is_complete: bool = True
    tokens_generated: Optional[int] = None


@dataclass
class TutorConfig:
    """Configuration for the language tutor.

    Attributes:
        model: Ollama model name to use.
        target_language: Language the child is learning.
        native_language: Child's native language.
        age_group: Child's age group for difficulty.
        lesson_focus: Current lesson topic.
        temperature: LLM temperature for response variety.
        max_tokens: Maximum tokens in response.
    """

    model: str = DEFAULT_LLM_MODEL
    target_language: TargetLanguage = TargetLanguage.SPANISH
    native_language: str = "English"
    age_group: AgeGroup = AgeGroup.EARLY_PRIMARY
    lesson_focus: str = "greetings and saying hello/goodbye"
    temperature: float = LLM_TEMPERATURE
    max_tokens: int = MAX_LLM_RESPONSE_TOKENS


class LanguageTutor:
    """Conversational language tutor using local LLM.

    This class manages conversation with a child learning a new
    language, providing patient and age-appropriate responses.

    Example:
        >>> config = TutorConfig(target_language=TargetLanguage.SPANISH)
        >>> tutor = LanguageTutor(config)
        >>> response = tutor.respond("How do I say hello?")
        >>> print(response.text)
    """

    def __init__(self, config: Optional[TutorConfig] = None) -> None:
        """Initialize the language tutor.

        Args:
            config: Tutor configuration, or None for defaults.
        """
        self.config = config or TutorConfig()
        self._client = None
        self._history: list[Message] = []
        self._system_prompt: Optional[str] = None

    def _ensure_client(self) -> None:
        """Ensure Ollama client is initialized."""
        if self._client is None:
            try:
                import ollama

                self._client = ollama
            except ImportError as e:
                raise ImportError(
                    "Ollama not installed. Install with: uv add ollama"
                ) from e

    @property
    def system_prompt(self) -> str:
        """Get the current system prompt.

        Returns:
            System prompt string.
        """
        if self._system_prompt is None:
            self._system_prompt = build_system_prompt(
                target_language=self.config.target_language,
                native_language=self.config.native_language,
                age_group=self.config.age_group,
                lesson_focus=self.config.lesson_focus,
            )
        return self._system_prompt

    def reset_conversation(self) -> str:
        """Reset the conversation and return a greeting.

        Returns:
            Initial greeting from the tutor.
        """
        self._history.clear()
        self._system_prompt = None  # Rebuild with current config

        # Get a conversation starter
        starter = get_conversation_starter(
            native_language=self.config.native_language.lower()[:2],
            target_language=self.config.target_language.value,
        )

        return starter

    def set_lesson_focus(self, focus: str) -> None:
        """Change the current lesson focus.

        Args:
            focus: New lesson topic to focus on.
        """
        self.config.lesson_focus = focus
        self._system_prompt = None  # Force rebuild

    def get_available_lessons(self) -> list[str]:
        """Get available lesson topics for current age group.

        Returns:
            List of lesson focus topics.
        """
        return get_lesson_focuses(self.config.age_group)

    def respond(self, user_input: str) -> TutorResponse:
        """Generate a response to user input.

        Args:
            user_input: Text from the child.

        Returns:
            TutorResponse with generated text.
        """
        self._ensure_client()

        # Add user message to history
        self._history.append(Message(role="user", content=user_input))

        # Build messages for API call
        messages = self._build_messages()

        try:
            response = self._client.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            )

            assistant_text = response["message"]["content"]

            # Add assistant response to history
            self._history.append(Message(role="assistant", content=assistant_text))

            # Trim history if needed
            self._trim_history()

            return TutorResponse(
                text=assistant_text,
                is_complete=True,
            )

        except self._client.ResponseError as e:
            return TutorResponse(
                text=f"I'm having trouble thinking right now. Let's try again! ({e.error})",
                is_complete=False,
            )

    def respond_stream(
        self,
        user_input: str,
    ) -> Generator[str, None, TutorResponse]:
        """Generate a streaming response to user input.

        Args:
            user_input: Text from the child.

        Yields:
            Text chunks as they are generated.

        Returns:
            Final TutorResponse with complete text.
        """
        self._ensure_client()

        # Add user message to history
        self._history.append(Message(role="user", content=user_input))

        # Build messages for API call
        messages = self._build_messages()

        full_response = []

        try:
            stream = self._client.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
                stream=True,
            )

            for chunk in stream:
                text_chunk = chunk["message"]["content"]
                full_response.append(text_chunk)
                yield text_chunk

            assistant_text = "".join(full_response)

            # Add assistant response to history
            self._history.append(Message(role="assistant", content=assistant_text))

            # Trim history if needed
            self._trim_history()

            return TutorResponse(text=assistant_text, is_complete=True)

        except Exception as e:
            error_msg = f"Oops! Let's try again. ({e})"
            return TutorResponse(text=error_msg, is_complete=False)

    def _build_messages(self) -> list[dict[str, str]]:
        """Build the message list for the API call.

        Returns:
            List of message dictionaries.
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        for msg in self._history:
            messages.append(msg.to_dict())

        return messages

    def _trim_history(self) -> None:
        """Trim conversation history to maximum length."""
        while len(self._history) > MAX_CONVERSATION_HISTORY:
            # Remove oldest user-assistant pair
            self._history.pop(0)
            if self._history:
                self._history.pop(0)

    @property
    def conversation_length(self) -> int:
        """Get current conversation length.

        Returns:
            Number of messages in history.
        """
        return len(self._history)


def create_tutor(
    target_language: TargetLanguage = TargetLanguage.SPANISH,
    native_language: str = "English",
    age_group: AgeGroup = AgeGroup.EARLY_PRIMARY,
    model: str = DEFAULT_LLM_MODEL,
) -> LanguageTutor:
    """Factory function to create a configured tutor.

    Args:
        target_language: Language to teach.
        native_language: Child's native language.
        age_group: Child's age group.
        model: Ollama model to use.

    Returns:
        Configured LanguageTutor instance.
    """
    config = TutorConfig(
        model=model,
        target_language=target_language,
        native_language=native_language,
        age_group=age_group,
    )
    return LanguageTutor(config)


def check_ollama_available() -> tuple[bool, str]:
    """Check if Ollama is available and running.

    Returns:
        Tuple of (is_available, message).
    """
    try:
        import ollama

        # Try to list models
        models = ollama.list()
        model_names = [m["name"] for m in models.get("models", [])]

        if not model_names:
            return False, "Ollama is running but no models installed. Run: ollama pull qwen2.5:7b"

        return True, f"Ollama available with models: {', '.join(model_names[:5])}"

    except ImportError:
        return False, "Ollama package not installed. Run: uv add ollama"
    except Exception as e:
        return False, f"Ollama not running. Start with: ollama serve ({e})"
