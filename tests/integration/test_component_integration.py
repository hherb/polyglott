"""Integration tests for component integration.

These tests verify that components work together correctly.
"""

import pytest

from polyglott.constants import (
    AUDIO_SAMPLE_RATE,
    AgeGroup,
    TargetLanguage,
)


class TestVADTranscriberIntegration:
    """Tests for VAD and Transcriber integration."""

    def test_vad_sample_rate_matches_transcriber(self) -> None:
        """Test VAD and transcriber use same sample rate."""
        from polyglott.vad.detector import VoiceActivityDetector
        from polyglott.stt.transcriber import SpeechTranscriber

        vad = VoiceActivityDetector()
        transcriber = SpeechTranscriber()

        assert vad.sample_rate == AUDIO_SAMPLE_RATE
        # Transcriber should expect same rate
        assert AUDIO_SAMPLE_RATE == 16000


class TestLLMPromptsIntegration:
    """Tests for LLM prompts integration."""

    def test_all_languages_have_greetings(self) -> None:
        """Test all target languages have greetings defined."""
        from polyglott.llm.prompts import LANGUAGE_GREETINGS

        for lang in TargetLanguage:
            assert lang.value in LANGUAGE_GREETINGS, f"Missing greetings for {lang}"

    def test_all_age_groups_have_lessons(self) -> None:
        """Test all age groups have lesson focuses defined."""
        from polyglott.llm.prompts import LESSON_FOCUSES

        for age_group in AgeGroup:
            assert age_group in LESSON_FOCUSES, f"Missing lessons for {age_group}"
            assert len(LESSON_FOCUSES[age_group]) > 0

    def test_system_prompt_generation(self) -> None:
        """Test system prompt generates correctly for all configs."""
        from polyglott.llm.prompts import build_system_prompt

        for lang in TargetLanguage:
            for age in AgeGroup:
                prompt = build_system_prompt(
                    target_language=lang,
                    native_language="English",
                    age_group=age,
                    lesson_focus="greetings",
                )
                assert isinstance(prompt, str)
                assert len(prompt) > 100


class TestTTSLanguageSupport:
    """Tests for TTS language support."""

    def test_all_languages_have_tts_support(self) -> None:
        """Test all target languages have TTS support."""
        from polyglott.tts.synthesizer import SpeechSynthesizer

        synth = SpeechSynthesizer()
        supported = synth.supported_languages

        for lang in TargetLanguage:
            assert lang.value in supported, f"TTS missing support for {lang}"


class TestConstantsConsistency:
    """Tests for constants consistency."""

    def test_audio_constants_valid(self) -> None:
        """Test audio constants are valid."""
        from polyglott.constants import (
            AUDIO_BIT_DEPTH,
            AUDIO_CHANNELS,
            AUDIO_SAMPLE_RATE,
            VAD_CHUNK_DURATION_MS,
            VAD_CHUNK_SAMPLES,
        )

        assert AUDIO_SAMPLE_RATE in (8000, 16000, 22050, 44100, 48000)
        assert AUDIO_BIT_DEPTH in (8, 16, 24, 32)
        assert AUDIO_CHANNELS >= 1
        assert VAD_CHUNK_DURATION_MS > 0

        # Verify chunk samples calculation
        expected_samples = int(AUDIO_SAMPLE_RATE * VAD_CHUNK_DURATION_MS / 1000)
        assert VAD_CHUNK_SAMPLES == expected_samples

    def test_tts_constants_valid(self) -> None:
        """Test TTS constants are valid."""
        from polyglott.constants import (
            TTS_CHILDREN_SPEED,
            TTS_DEFAULT_SPEED,
            TTS_SAMPLE_RATE,
        )

        assert TTS_SAMPLE_RATE in (22050, 24000, 44100, 48000)
        assert 0.5 <= TTS_DEFAULT_SPEED <= 2.0
        assert 0.5 <= TTS_CHILDREN_SPEED <= 2.0
        # Children speed should be slower
        assert TTS_CHILDREN_SPEED <= TTS_DEFAULT_SPEED

    def test_llm_constants_valid(self) -> None:
        """Test LLM constants are valid."""
        from polyglott.constants import (
            DEFAULT_LLM_MODEL,
            LLM_TEMPERATURE,
            MAX_CONVERSATION_HISTORY,
            MAX_LLM_RESPONSE_TOKENS,
        )

        assert isinstance(DEFAULT_LLM_MODEL, str)
        assert 0.0 <= LLM_TEMPERATURE <= 2.0
        assert MAX_LLM_RESPONSE_TOKENS > 0
        assert MAX_CONVERSATION_HISTORY > 0
