"""System prompts and prompt templates for the language tutor.

This module contains all the prompts used by the LLM to guide
language learning conversations with children.
"""

from typing import Final

from polyglott.constants import AgeGroup, TargetLanguage

# Base system prompt for the language tutor
BASE_TUTOR_PROMPT: Final[str] = """You are a friendly, patient language tutor helping a young child learn {target_language}.

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

# Age-appropriate descriptions
AGE_DESCRIPTIONS: Final[dict[AgeGroup, str]] = {
    AgeGroup.PRESCHOOL: "preschooler (3-5 years old)",
    AgeGroup.EARLY_PRIMARY: "young child (6-8 years old)",
    AgeGroup.LATE_PRIMARY: "child (9-12 years old)",
}

# Lesson focus options for each level
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

# Language names in both English and the target language
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


def build_system_prompt(
    target_language: TargetLanguage,
    native_language: str,
    age_group: AgeGroup,
    lesson_focus: str,
) -> str:
    """Build a complete system prompt for the tutor.

    Args:
        target_language: Language the child is learning.
        native_language: Child's native language name.
        age_group: Child's age group for difficulty adjustment.
        lesson_focus: Current lesson topic.

    Returns:
        Complete system prompt string.
    """
    from polyglott.constants import LANGUAGE_NAMES

    target_name = LANGUAGE_NAMES.get(target_language.value, target_language.value)
    age_description = AGE_DESCRIPTIONS.get(age_group, "young child")

    return BASE_TUTOR_PROMPT.format(
        target_language=target_name,
        native_language=native_language,
        age_description=age_description,
        lesson_focus=lesson_focus,
    )


def get_lesson_focuses(age_group: AgeGroup) -> list[str]:
    """Get available lesson focuses for an age group.

    Args:
        age_group: Child's age group.

    Returns:
        List of lesson focus topics.
    """
    return LESSON_FOCUSES.get(age_group, LESSON_FOCUSES[AgeGroup.EARLY_PRIMARY])


def get_greeting(language: str, greeting_type: str = "hello") -> str:
    """Get a greeting in the specified language.

    Args:
        language: ISO language code.
        greeting_type: Type of greeting (hello, goodbye, etc.).

    Returns:
        Greeting string in the target language.
    """
    greetings = LANGUAGE_GREETINGS.get(language, LANGUAGE_GREETINGS["en"])
    return greetings.get(greeting_type, greetings["hello"])


# Conversation starters for beginning a lesson
CONVERSATION_STARTERS: Final[dict[str, str]] = {
    "en": "Hi there! Are you ready to learn some {language} today? Let's have fun!",
    "de": "Hallo! Bist du bereit, heute etwas {language} zu lernen? Lass uns Spaß haben!",
}


def get_conversation_starter(
    native_language: str,
    target_language: str,
) -> str:
    """Get a conversation starter in the child's native language.

    Args:
        native_language: Child's native language code.
        target_language: Language being learned (for display name).

    Returns:
        Conversation starter string.
    """
    from polyglott.constants import LANGUAGE_NAMES

    target_name = LANGUAGE_NAMES.get(target_language, target_language)
    template = CONVERSATION_STARTERS.get(
        native_language,
        CONVERSATION_STARTERS["en"],
    )
    return template.format(language=target_name)
