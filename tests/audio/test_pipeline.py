"""Tests for Audio Pipeline module.

This module tests the audio pipeline orchestration.
"""

import pytest

from polyglott.audio.pipeline import (
    AudioPipeline,
    ConversationTurn,
    PipelineState,
    create_pipeline,
)
from polyglott.constants import AgeGroup, TargetLanguage


class TestPipelineState:
    """Tests for PipelineState enum."""

    def test_all_states_exist(self) -> None:
        """Test all expected pipeline states are defined."""
        assert PipelineState.IDLE.value == "idle"
        assert PipelineState.LISTENING.value == "listening"
        assert PipelineState.TRANSCRIBING.value == "transcribing"
        assert PipelineState.THINKING.value == "thinking"
        assert PipelineState.SPEAKING.value == "speaking"


class TestConversationTurn:
    """Tests for ConversationTurn dataclass."""

    def test_creation_default(self) -> None:
        """Test ConversationTurn with defaults."""
        turn = ConversationTurn()
        assert turn.user_audio is None
        assert turn.user_text == ""
        assert turn.tutor_text == ""
        assert turn.tutor_audio is None
        assert turn.state == PipelineState.IDLE

    def test_creation_with_values(self) -> None:
        """Test ConversationTurn with values."""
        import numpy as np

        audio = np.zeros(16000, dtype=np.float32)
        turn = ConversationTurn(
            user_audio=audio,
            user_text="Hello",
            tutor_text="Hi there!",
            state=PipelineState.SPEAKING,
        )
        assert turn.user_text == "Hello"
        assert turn.tutor_text == "Hi there!"


class TestAudioPipeline:
    """Tests for AudioPipeline class."""

    def test_initialization(self) -> None:
        """Test pipeline initializes correctly."""
        pipeline = AudioPipeline()
        assert pipeline.state == PipelineState.IDLE

    def test_initialization_with_params(self) -> None:
        """Test pipeline with custom parameters."""
        pipeline = AudioPipeline(
            target_language=TargetLanguage.GERMAN,
            native_language="English",
            age_group=AgeGroup.PRESCHOOL,
        )
        assert pipeline.target_language == TargetLanguage.GERMAN
        assert pipeline.native_language == "English"
        assert pipeline.age_group == AgeGroup.PRESCHOOL

    def test_state_property(self) -> None:
        """Test state property."""
        pipeline = AudioPipeline()
        assert pipeline.state == PipelineState.IDLE

    def test_stop(self) -> None:
        """Test stop method."""
        pipeline = AudioPipeline()
        pipeline.stop()
        assert pipeline.state == PipelineState.IDLE

    def test_state_callback(self) -> None:
        """Test state change callback."""
        states_seen = []

        def record_state(state: PipelineState) -> None:
            states_seen.append(state)

        pipeline = AudioPipeline(on_state_change=record_state)
        pipeline._set_state(PipelineState.LISTENING)

        assert PipelineState.LISTENING in states_seen


class TestCreatePipeline:
    """Tests for create_pipeline factory function."""

    def test_creates_pipeline(self) -> None:
        """Test factory creates valid pipeline."""
        pipeline = create_pipeline()
        assert isinstance(pipeline, AudioPipeline)

    def test_with_parameters(self) -> None:
        """Test factory accepts parameters."""
        pipeline = create_pipeline(
            target_language=TargetLanguage.JAPANESE,
            native_language="German",
            age_group=AgeGroup.LATE_PRIMARY,
        )
        assert pipeline.target_language == TargetLanguage.JAPANESE
        assert pipeline.native_language == "German"
        assert pipeline.age_group == AgeGroup.LATE_PRIMARY
