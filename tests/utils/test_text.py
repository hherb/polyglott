"""Tests for text processing utilities.

This module tests the text filtering, cleaning, and language tag
parsing functions used for TTS synthesis.
"""

import pytest

from polyglott.utils.text import (
    LanguageSegment,
    has_language_tags,
    is_speakable,
    normalize_whitespace,
    parse_language_tags,
    prepare_for_tts,
    split_sentences,
    strip_actions,
    strip_emojis,
    strip_language_tags,
    strip_markdown,
    strip_special_chars,
)


class TestLanguageSegment:
    """Tests for LanguageSegment dataclass."""

    def test_creation(self) -> None:
        """Test segment can be created."""
        segment = LanguageSegment(text="Hello", language="en")
        assert segment.text == "Hello"
        assert segment.language == "en"

    def test_language_normalized_to_lowercase(self) -> None:
        """Test language code is normalized to lowercase."""
        segment = LanguageSegment(text="Hallo", language="DE")
        assert segment.language == "de"

    def test_mixed_case_language(self) -> None:
        """Test mixed case language code is normalized."""
        segment = LanguageSegment(text="Hola", language="Es")
        assert segment.language == "es"


class TestParseLanguageTags:
    """Tests for parse_language_tags function."""

    def test_no_tags_returns_single_segment(self) -> None:
        """Test text without tags returns single segment."""
        text = "Hello, how are you?"
        segments = parse_language_tags(text, default_language="en")

        assert len(segments) == 1
        assert segments[0].text == "Hello, how are you?"
        assert segments[0].language == "en"

    def test_single_tag(self) -> None:
        """Test text with single language tag."""
        text = "To say hello: <lang:de>Guten Tag</lang>"
        segments = parse_language_tags(text, default_language="en")

        assert len(segments) == 2
        assert segments[0].text == "To say hello:"
        assert segments[0].language == "en"
        assert segments[1].text == "Guten Tag"
        assert segments[1].language == "de"

    def test_multiple_tags(self) -> None:
        """Test text with multiple language tags."""
        text = "Say <lang:de>Hallo</lang> or <lang:de>Guten Tag</lang> to greet."
        segments = parse_language_tags(text, default_language="en")

        assert len(segments) == 5
        assert segments[0].text == "Say"
        assert segments[0].language == "en"
        assert segments[1].text == "Hallo"
        assert segments[1].language == "de"
        assert segments[2].text == "or"
        assert segments[2].language == "en"
        assert segments[3].text == "Guten Tag"
        assert segments[3].language == "de"
        assert segments[4].text == "to greet."
        assert segments[4].language == "en"

    def test_tag_at_start(self) -> None:
        """Test tag at beginning of text."""
        text = "<lang:de>Hallo</lang>! That means hello!"
        segments = parse_language_tags(text, default_language="en")

        assert len(segments) == 2
        assert segments[0].text == "Hallo"
        assert segments[0].language == "de"
        assert segments[1].text == "! That means hello!"
        assert segments[1].language == "en"

    def test_tag_at_end(self) -> None:
        """Test tag at end of text."""
        text = "The word for dog is <lang:de>Hund</lang>"
        segments = parse_language_tags(text, default_language="en")

        assert len(segments) == 2
        assert segments[0].text == "The word for dog is"
        assert segments[0].language == "en"
        assert segments[1].text == "Hund"
        assert segments[1].language == "de"

    def test_adjacent_tags(self) -> None:
        """Test adjacent language tags."""
        text = "<lang:de>Eins</lang><lang:de>zwei</lang><lang:de>drei</lang>"
        segments = parse_language_tags(text, default_language="en")

        assert len(segments) == 3
        assert all(s.language == "de" for s in segments)
        assert segments[0].text == "Eins"
        assert segments[1].text == "zwei"
        assert segments[2].text == "drei"

    def test_different_languages(self) -> None:
        """Test mixing multiple target languages."""
        text = "Hello is <lang:de>Hallo</lang> in German and <lang:es>Hola</lang> in Spanish."
        segments = parse_language_tags(text, default_language="en")

        assert len(segments) == 5
        assert segments[0].language == "en"  # "Hello is"
        assert segments[1].language == "de"  # "Hallo"
        assert segments[2].language == "en"  # "in German and"
        assert segments[3].language == "es"  # "Hola"
        assert segments[4].language == "en"  # "in Spanish."

    def test_empty_text(self) -> None:
        """Test empty text returns empty list."""
        segments = parse_language_tags("", default_language="en")
        assert len(segments) == 0

    def test_whitespace_only(self) -> None:
        """Test whitespace-only text returns empty list."""
        segments = parse_language_tags("   ", default_language="en")
        assert len(segments) == 0

    def test_empty_tag_is_skipped(self) -> None:
        """Test empty tags are skipped."""
        text = "Hello <lang:de></lang> world"
        segments = parse_language_tags(text, default_language="en")

        # Empty tag is skipped, but segments before and after are separate
        assert len(segments) == 2
        assert segments[0].text == "Hello"
        assert segments[0].language == "en"
        assert segments[1].text == "world"
        assert segments[1].language == "en"

    def test_case_insensitive_tags(self) -> None:
        """Test language tags are case-insensitive."""
        text = "Say <LANG:DE>Hallo</LANG> please"
        segments = parse_language_tags(text, default_language="en")

        assert len(segments) == 3
        assert segments[0].text == "Say"
        assert segments[0].language == "en"
        assert segments[1].text == "Hallo"
        assert segments[1].language == "de"
        assert segments[2].text == "please"
        assert segments[2].language == "en"

    def test_multiline_tag_content(self) -> None:
        """Test tags can contain newlines."""
        text = "Say: <lang:de>Guten\nTag</lang>"
        segments = parse_language_tags(text, default_language="en")

        assert len(segments) == 2
        assert "Guten" in segments[1].text
        assert "Tag" in segments[1].text


class TestStripLanguageTags:
    """Tests for strip_language_tags function."""

    def test_removes_tags_preserves_content(self) -> None:
        """Test tags are removed but content preserved."""
        text = "Say <lang:de>Hallo</lang> to greet."
        result = strip_language_tags(text)
        assert result == "Say Hallo to greet."

    def test_multiple_tags(self) -> None:
        """Test multiple tags are all removed."""
        text = "<lang:de>Eins</lang>, <lang:de>zwei</lang>, <lang:de>drei</lang>"
        result = strip_language_tags(text)
        assert result == "Eins, zwei, drei"

    def test_no_tags(self) -> None:
        """Test text without tags is unchanged."""
        text = "Hello, how are you?"
        result = strip_language_tags(text)
        assert result == text


class TestHasLanguageTags:
    """Tests for has_language_tags function."""

    def test_detects_tag(self) -> None:
        """Test tags are detected."""
        assert has_language_tags("Say <lang:de>Hallo</lang>") is True

    def test_no_tags(self) -> None:
        """Test absence of tags."""
        assert has_language_tags("Hello, how are you?") is False

    def test_similar_but_invalid_tags(self) -> None:
        """Test similar patterns that are not valid tags."""
        assert has_language_tags("<lang>text</lang>") is False
        assert has_language_tags("<lang:>text</lang>") is False
        assert has_language_tags("<lang:abc>text</lang>") is False  # 3 chars


class TestStripEmojis:
    """Tests for strip_emojis function."""

    def test_removes_emoji(self) -> None:
        """Test emojis are removed."""
        assert strip_emojis("Hello 😊 World") == "Hello  World"

    def test_no_emojis(self) -> None:
        """Test text without emojis is unchanged."""
        assert strip_emojis("Hello World") == "Hello World"


class TestStripMarkdown:
    """Tests for strip_markdown function."""

    def test_removes_bold(self) -> None:
        """Test bold markers are removed."""
        assert "great" in strip_markdown("**great**")
        assert "**" not in strip_markdown("**great**")

    def test_removes_italic(self) -> None:
        """Test italic markers are removed."""
        assert "nice" in strip_markdown("*nice*")
        assert "*" not in strip_markdown("*nice*")


class TestStripActions:
    """Tests for strip_actions function."""

    def test_removes_actions(self) -> None:
        """Test action indicators are removed."""
        assert strip_actions("Hello *smiles*") == "Hello "
        assert strip_actions("Hi (laughs)") == "Hi "


class TestStripSpecialChars:
    """Tests for strip_special_chars function."""

    def test_removes_special_chars(self) -> None:
        """Test special characters are removed."""
        assert "★" not in strip_special_chars("★ Great job!")
        assert "●" not in strip_special_chars("● Item one")


class TestNormalizeWhitespace:
    """Tests for normalize_whitespace function."""

    def test_collapses_spaces(self) -> None:
        """Test multiple spaces are collapsed."""
        assert normalize_whitespace("Hello   World") == "Hello World"

    def test_trims_ends(self) -> None:
        """Test leading/trailing whitespace is removed."""
        assert normalize_whitespace("  Hello  ") == "Hello"


class TestPrepareForTts:
    """Tests for prepare_for_tts function."""

    def test_combined_cleaning(self) -> None:
        """Test all cleaning is applied."""
        text = "**Great** job! 😊 *smiles*"
        result = prepare_for_tts(text)
        assert "**" not in result
        assert "😊" not in result
        assert "*smiles*" not in result
        assert "Great" in result


class TestSplitSentences:
    """Tests for split_sentences function."""

    def test_splits_on_periods(self) -> None:
        """Test sentences are split on periods."""
        sentences = split_sentences("Hello. World.")
        assert len(sentences) == 2

    def test_splits_on_question_marks(self) -> None:
        """Test sentences are split on question marks."""
        sentences = split_sentences("Hello? World!")
        assert len(sentences) == 2


class TestIsSpeakable:
    """Tests for is_speakable function."""

    def test_speakable_text(self) -> None:
        """Test text with words is speakable."""
        assert is_speakable("Hello") is True

    def test_emoji_only_not_speakable(self) -> None:
        """Test emoji-only text is not speakable."""
        assert is_speakable("😊") is False

    def test_empty_not_speakable(self) -> None:
        """Test empty text is not speakable."""
        assert is_speakable("") is False
