"""System prompts and prompt templates for the language tutor.

This module contains all the prompts used by the LLM to guide
language learning conversations with children.
"""

from typing import Final, Optional

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

IMPORTANT - Language tagging for pronunciation:
When you include words or phrases in {target_language}, wrap them in language tags so they are pronounced correctly.
Use this format: <lang:{target_language_code}>word or phrase</lang>

Examples:
- "To say hello, you can say: <lang:{target_language_code}>Guten Tag</lang>"
- "The word for dog is <lang:{target_language_code}>Hund</lang>. Can you say <lang:{target_language_code}>Hund</lang>?"
- "Great job! <lang:{target_language_code}>Sehr gut!</lang>"

Always use language tags when speaking {target_language} words so the child hears correct pronunciation.
Do NOT include pronunciation hints in parentheses - the audio will pronounce it correctly.

The child's native language is {native_language}.

Understanding student speech:
The child will mostly speak in {native_language}, sometimes attempting {target_language} words or phrases.
Their attempts may be transcribed phonetically based on how they sound (e.g., "gooten tog" for "Guten Tag").
Recognize and praise these attempts, then model the correct pronunciation using language tags.

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
        target_language_code=target_language.value,
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

# Follow-up prompts when user is silent (tiered by urgency)
# Level 1: Gentle check-in (after ~10 seconds of silence)
FOLLOWUP_TIER1: Final[dict[str, list[str]]] = {
    "en": [
        "Are you still there?",
        "Take your time, I'm here when you're ready!",
        "No rush! Just let me know when you want to continue.",
    ],
    "de": [
        "Bist du noch da?",
        "Lass dir Zeit, ich bin hier wenn du bereit bist!",
        "Kein Stress! Sag mir einfach, wenn du weitermachen möchtest.",
    ],
}

# Level 2: Encouraging prompt (after ~20 seconds)
FOLLOWUP_TIER2: Final[dict[str, list[str]]] = {
    "en": [
        "Would you like to try something different?",
        "Do you want me to repeat that?",
        "Would you like a hint?",
        "Should we practice something else?",
    ],
    "de": [
        "Möchtest du etwas anderes versuchen?",
        "Soll ich das wiederholen?",
        "Möchtest du einen Hinweis?",
        "Sollen wir etwas anderes üben?",
    ],
}

# Level 3: Re-engagement (after ~30 seconds)
FOLLOWUP_TIER3: Final[dict[str, list[str]]] = {
    "en": [
        "Let's try a fun game! Can you say 'hello' in {language}?",
        "How about we count to five in {language}? One, two...",
        "Can you tell me your favorite color in {language}?",
    ],
    "de": [
        "Lass uns ein lustiges Spiel spielen! Kannst du 'Hallo' auf {language} sagen?",
        "Wie wäre es, wenn wir bis fünf auf {language} zählen? Eins, zwei...",
        "Kannst du mir deine Lieblingsfarbe auf {language} sagen?",
    ],
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


# Time-based greeting templates
TIME_GREETINGS: Final[dict[str, dict[str, str]]] = {
    "en": {
        "morning": "Good morning",
        "afternoon": "Good afternoon",
        "evening": "Good evening",
    },
    "de": {
        "morning": "Guten Morgen",
        "afternoon": "Guten Tag",
        "evening": "Guten Abend",
    },
}

# Returning user greeting templates
RETURNING_USER_GREETINGS: Final[dict[str, list[str]]] = {
    "en": [
        "Welcome back, {name}! Great to see you again!",
        "Hey {name}! Ready for more {language} practice?",
        "{name}! So happy you're back! Let's continue learning!",
    ],
    "de": [
        "Willkommen zurück, {name}! Schön dich wiederzusehen!",
        "Hey {name}! Bereit für mehr {language} Übung?",
        "{name}! So schön, dass du wieder da bist! Lass uns weiterlernen!",
    ],
}

# First time user greeting templates
NEW_USER_GREETINGS: Final[dict[str, list[str]]] = {
    "en": [
        "Hello {name}! I'm so excited to meet you! Let's learn {language} together!",
        "Hi {name}! Welcome to your first {language} adventure!",
        "Hey there, {name}! Ready to become a {language} superstar?",
    ],
    "de": [
        "Hallo {name}! Ich freue mich so, dich kennenzulernen! Lass uns zusammen {language} lernen!",
        "Hi {name}! Willkommen zu deinem ersten {language}-Abenteuer!",
        "Hey {name}! Bereit, ein {language}-Superstar zu werden?",
    ],
}

# Session continuation greetings (when resuming a session)
CONTINUE_SESSION_GREETINGS: Final[dict[str, list[str]]] = {
    "en": [
        "Welcome back, {name}! Last time we were learning about {topic}. Want to continue?",
        "Hey {name}! Ready to pick up where we left off with {topic}?",
        "{name}! Let's continue our {topic} adventure!",
    ],
    "de": [
        "Willkommen zurück, {name}! Letztes Mal haben wir {topic} gelernt. Möchtest du weitermachen?",
        "Hey {name}! Bereit, bei {topic} weiterzumachen?",
        "{name}! Lass uns unser {topic}-Abenteuer fortsetzen!",
    ],
}


def _get_time_of_day() -> str:
    """Get the current time of day.

    Returns:
        One of 'morning', 'afternoon', or 'evening'.
    """
    from datetime import datetime
    hour = datetime.now().hour

    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    else:
        return "evening"


def get_contextual_greeting(
    student_name: str,
    native_language: str,
    target_language: str,
    is_returning_user: bool = False,
    total_sessions: int = 0,
    last_topic: Optional[str] = None,
) -> str:
    """Get a contextual greeting based on time, user history, and context.

    Args:
        student_name: Name of the student.
        native_language: Child's native language code.
        target_language: Language being learned.
        is_returning_user: Whether this is a returning user.
        total_sessions: Number of previous sessions.
        last_topic: Last lesson topic (for session continuation).

    Returns:
        Personalized greeting string.
    """
    import random
    from polyglott.constants import LANGUAGE_NAMES

    target_name = LANGUAGE_NAMES.get(target_language, target_language)
    lang_code = native_language[:2].lower()

    # Get time-based greeting prefix
    time_of_day = _get_time_of_day()
    time_greetings = TIME_GREETINGS.get(lang_code, TIME_GREETINGS["en"])
    time_greeting = time_greetings.get(time_of_day, time_greetings["morning"])

    # Select appropriate greeting based on context
    if last_topic and is_returning_user:
        # Continuing a session
        templates = CONTINUE_SESSION_GREETINGS.get(lang_code, CONTINUE_SESSION_GREETINGS["en"])
        template = random.choice(templates)
        greeting = template.format(name=student_name, topic=last_topic, language=target_name)
    elif is_returning_user and total_sessions > 0:
        # Returning user
        templates = RETURNING_USER_GREETINGS.get(lang_code, RETURNING_USER_GREETINGS["en"])
        template = random.choice(templates)
        greeting = template.format(name=student_name, language=target_name)
    else:
        # New user
        templates = NEW_USER_GREETINGS.get(lang_code, NEW_USER_GREETINGS["en"])
        template = random.choice(templates)
        greeting = template.format(name=student_name, language=target_name)

    # Combine time greeting with main greeting
    return f"{time_greeting}, {greeting}"


def get_followup_prompt(
    tier: int,
    native_language: str,
    target_language: str,
) -> str:
    """Get a follow-up prompt for re-engaging a silent user.

    Args:
        tier: Urgency tier (1-3, higher = more engaging).
        native_language: Child's native language code.
        target_language: Language being learned (for formatting).

    Returns:
        Follow-up prompt string.
    """
    import random
    from polyglott.constants import LANGUAGE_NAMES

    target_name = LANGUAGE_NAMES.get(target_language, target_language)

    tier_prompts = {
        1: FOLLOWUP_TIER1,
        2: FOLLOWUP_TIER2,
        3: FOLLOWUP_TIER3,
    }

    prompts_dict = tier_prompts.get(tier, FOLLOWUP_TIER1)
    prompts = prompts_dict.get(native_language, prompts_dict.get("en", ["Are you there?"]))

    prompt = random.choice(prompts)
    return prompt.format(language=target_name)
