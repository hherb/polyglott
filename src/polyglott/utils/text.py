"""Text processing utilities for the language tutor.

This module provides text filtering and cleaning functions
for preparing text for TTS synthesis.
"""

import re
import unicodedata
from typing import Final

# Regex pattern for emoji characters (Unicode emoji ranges)
# This covers most emoji: emoticons, symbols, pictographs, transport, flags, etc.
EMOJI_PATTERN: Final[re.Pattern] = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows-c
    "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
    "\U00002702-\U000027B0"  # dingbats
    "\U00002300-\U000023FF"  # misc technical (includes some emoji)
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "]+",
    flags=re.UNICODE,
)

# Pattern for markdown-style formatting
MARKDOWN_PATTERN: Final[re.Pattern] = re.compile(
    r"\*\*([^*]+)\*\*"  # **bold**
    r"|\*([^*]+)\*"     # *italic*
    r"|__([^_]+)__"     # __bold__
    r"|_([^_]+)_"       # _italic_
    r"|`([^`]+)`"       # `code`
    r"|\[([^\]]+)\]\([^)]+\)"  # [text](url)
)

# Pattern for action indicators like *smiles*, (laughs), etc.
# These are non-verbal cues that shouldn't be spoken
ACTION_WORDS = r"(?:smil|laugh|grin|nod|wav|sigh|gasp|chuckl|wink|giggl|beam|frown|shrug|paus|think|ponder|hug|clap)"
ACTION_PATTERN: Final[re.Pattern] = re.compile(
    rf"\*[^*]*{ACTION_WORDS}[^*]*\*"  # *action with keywords*
    rf"|\([^)]*{ACTION_WORDS}[^)]*\)"  # (action)
    rf"|\[[^\]]*{ACTION_WORDS}[^\]]*\]",  # [action]
    flags=re.IGNORECASE,
)

# Characters that should be removed for cleaner TTS
REMOVE_CHARS: Final[str] = "~•●○◆◇■□▪▫★☆♠♣♥♦"


def strip_emojis(text: str) -> str:
    """Remove emoji characters from text.

    Args:
        text: Input text potentially containing emojis.

    Returns:
        Text with emojis removed.
    """
    return EMOJI_PATTERN.sub("", text)


def strip_markdown(text: str) -> str:
    """Remove markdown formatting from text.

    Preserves the text content while removing formatting markers.

    Args:
        text: Input text with markdown formatting.

    Returns:
        Plain text without markdown.
    """
    # Replace markdown with just the text content
    def replace_markdown(match: re.Match) -> str:
        # Return first non-None group (the captured text)
        for group in match.groups():
            if group is not None:
                return group
        return ""

    return MARKDOWN_PATTERN.sub(replace_markdown, text)


def strip_actions(text: str) -> str:
    """Remove action indicators from text.

    Removes things like *smiles*, (laughs), etc. that are
    not meant to be spoken aloud.

    Args:
        text: Input text with action indicators.

    Returns:
        Text without action indicators.
    """
    return ACTION_PATTERN.sub("", text)


def strip_special_chars(text: str) -> str:
    """Remove special characters that TTS shouldn't pronounce.

    Args:
        text: Input text.

    Returns:
        Text with special characters removed.
    """
    for char in REMOVE_CHARS:
        text = text.replace(char, "")
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    Collapses multiple spaces, removes leading/trailing whitespace,
    and ensures proper spacing around punctuation.

    Args:
        text: Input text.

    Returns:
        Text with normalized whitespace.
    """
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    # Ensure space after punctuation (but not at end)
    text = re.sub(r"([.,!?;:])([^\s\d])", r"\1 \2", text)
    return text.strip()


def prepare_for_tts(text: str) -> str:
    """Prepare text for TTS synthesis.

    Removes all non-speakable content: emojis, markdown,
    action indicators, and normalizes whitespace.

    Args:
        text: Raw input text.

    Returns:
        Clean text ready for TTS.
    """
    # Apply filters in order (actions first since they use * like markdown)
    text = strip_emojis(text)
    text = strip_actions(text)  # Before markdown to catch *smiles* patterns
    text = strip_markdown(text)
    text = strip_special_chars(text)
    text = normalize_whitespace(text)
    return text


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS.

    Args:
        text: Input text.

    Returns:
        List of sentences.
    """
    # Split on sentence-ending punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def is_speakable(text: str) -> bool:
    """Check if text contains speakable content.

    Args:
        text: Input text.

    Returns:
        True if text has speakable words.
    """
    # Remove non-speakable content
    clean = prepare_for_tts(text)
    # Check if any word characters remain
    return bool(re.search(r"\w", clean))
