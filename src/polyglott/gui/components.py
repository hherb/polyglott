"""Reusable GUI components for Polyglott.

Contains chat bubbles, buttons, and other UI elements.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import flet as ft

from polyglott.gui.theme import (
    BORDER_RADIUS_BUBBLE,
    BUBBLE_MAX_WIDTH,
    BUBBLE_PADDING,
    FONT_SIZE_NORMAL,
    FONT_SIZE_SMALL,
    LISTENING_COLOR,
    PRIMARY_COLOR,
    REPLAY_BUTTON_SIZE,
    SPEAKING_COLOR,
    SPACING_SM,
    SPACING_XS,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    THINKING_COLOR,
    TUTOR_BUBBLE_BORDER,
    TUTOR_BUBBLE_COLOR,
    USER_BUBBLE_BORDER,
    USER_BUBBLE_COLOR,
)


@dataclass
class ChatMessage:
    """Represents a chat message.

    Attributes:
        text: The message text.
        is_user: True if this is a user message.
        language: Language code for TTS replay.
        timestamp: Optional timestamp string.
    """

    text: str
    is_user: bool
    language: str = "en"
    timestamp: Optional[str] = None


class ChatBubble(ft.Container):
    """A chat bubble component with optional replay button.

    Displays a message in a speech bubble style with a small
    speaker icon to replay the audio.
    """

    def __init__(
        self,
        message: ChatMessage,
        on_replay: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """Initialize the chat bubble.

        Args:
            message: The chat message to display.
            on_replay: Callback when replay button is clicked.
                       Receives (text, language) arguments.
        """
        self.message = message
        self.on_replay = on_replay

        # Determine bubble styling based on sender
        if message.is_user:
            bg_color = USER_BUBBLE_COLOR
            border_color = USER_BUBBLE_BORDER
            alignment = ft.MainAxisAlignment.END
            bubble_alignment = ft.alignment.center_right
            border_radius = ft.border_radius.only(
                top_left=BORDER_RADIUS_BUBBLE,
                top_right=BORDER_RADIUS_BUBBLE,
                bottom_left=BORDER_RADIUS_BUBBLE,
                bottom_right=4,  # Small corner for user
            )
        else:
            bg_color = TUTOR_BUBBLE_COLOR
            border_color = TUTOR_BUBBLE_BORDER
            alignment = ft.MainAxisAlignment.START
            bubble_alignment = ft.alignment.center_left
            border_radius = ft.border_radius.only(
                top_left=BORDER_RADIUS_BUBBLE,
                top_right=BORDER_RADIUS_BUBBLE,
                bottom_left=4,  # Small corner for tutor
                bottom_right=BORDER_RADIUS_BUBBLE,
            )

        # Create replay button
        replay_button = ft.IconButton(
            icon=ft.Icons.VOLUME_UP_ROUNDED,
            icon_size=16,
            icon_color=TEXT_SECONDARY,
            tooltip="Replay",
            style=ft.ButtonStyle(
                shape=ft.CircleBorder(),
                padding=4,
            ),
            on_click=self._handle_replay,
        )

        # Create message content
        message_text = ft.Text(
            message.text,
            size=FONT_SIZE_NORMAL,
            color=TEXT_PRIMARY,
            selectable=True,
        )

        # Layout: message text with replay button
        bubble_content = ft.Row(
            controls=[
                ft.Container(
                    content=message_text,
                    expand=True,
                ),
                replay_button,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=SPACING_SM,
        )

        # Sender label
        sender_label = ft.Text(
            "You" if message.is_user else "Tutor",
            size=FONT_SIZE_SMALL,
            color=TEXT_SECONDARY,
            weight=ft.FontWeight.W_500,
        )

        # Bubble container
        bubble = ft.Container(
            content=ft.Column(
                controls=[
                    sender_label,
                    bubble_content,
                ],
                spacing=SPACING_XS,
            ),
            bgcolor=bg_color,
            border=ft.border.all(1, border_color),
            border_radius=border_radius,
            padding=BUBBLE_PADDING,
            width=BUBBLE_MAX_WIDTH,
        )

        # Outer row for alignment
        super().__init__(
            content=ft.Row(
                controls=[bubble],
                alignment=alignment,
            ),
            margin=ft.margin.only(
                left=SPACING_SM if message.is_user else 0,
                right=0 if message.is_user else SPACING_SM,
                bottom=SPACING_SM,
            ),
        )

    def _handle_replay(self, e: ft.ControlEvent) -> None:
        """Handle replay button click."""
        if self.on_replay:
            self.on_replay(self.message.text, self.message.language)


class StatusIndicator(ft.Container):
    """Visual indicator for the current conversation state."""

    def __init__(self) -> None:
        """Initialize the status indicator."""
        self._status_text = ft.Text(
            "Ready",
            size=FONT_SIZE_NORMAL,
            color=TEXT_SECONDARY,
            weight=ft.FontWeight.W_500,
        )

        self._status_icon = ft.Icon(
            ft.Icons.RADIO_BUTTON_CHECKED,
            size=16,
            color=TEXT_SECONDARY,
        )

        self._animation_ring = ft.Container(
            width=40,
            height=40,
            border_radius=20,
            animate=ft.animation.Animation(1000, ft.AnimationCurve.EASE_IN_OUT),
        )

        super().__init__(
            content=ft.Row(
                controls=[
                    self._animation_ring,
                    ft.Container(
                        content=ft.Row(
                            controls=[
                                self._status_icon,
                                self._status_text,
                            ],
                            spacing=SPACING_SM,
                        ),
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=SPACING_SM,
            ),
            padding=SPACING_SM,
        )

    def set_listening(self) -> None:
        """Set status to listening."""
        self._status_text.value = "Listening..."
        self._status_text.color = LISTENING_COLOR
        self._status_icon.name = ft.Icons.MIC
        self._status_icon.color = LISTENING_COLOR
        self._animation_ring.bgcolor = LISTENING_COLOR + "40"
        self._animation_ring.border = ft.border.all(3, LISTENING_COLOR)
        if self.page:
            self.update()

    def set_thinking(self) -> None:
        """Set status to thinking."""
        self._status_text.value = "Thinking..."
        self._status_text.color = THINKING_COLOR
        self._status_icon.name = ft.Icons.PSYCHOLOGY
        self._status_icon.color = THINKING_COLOR
        self._animation_ring.bgcolor = THINKING_COLOR + "40"
        self._animation_ring.border = ft.border.all(3, THINKING_COLOR)
        if self.page:
            self.update()

    def set_speaking(self) -> None:
        """Set status to speaking."""
        self._status_text.value = "Speaking..."
        self._status_text.color = SPEAKING_COLOR
        self._status_icon.name = ft.Icons.VOLUME_UP
        self._status_icon.color = SPEAKING_COLOR
        self._animation_ring.bgcolor = SPEAKING_COLOR + "40"
        self._animation_ring.border = ft.border.all(3, SPEAKING_COLOR)
        if self.page:
            self.update()

    def set_ready(self) -> None:
        """Set status to ready/idle."""
        self._status_text.value = "Ready"
        self._status_text.color = TEXT_SECONDARY
        self._status_icon.name = ft.Icons.RADIO_BUTTON_CHECKED
        self._status_icon.color = TEXT_SECONDARY
        self._animation_ring.bgcolor = None
        self._animation_ring.border = None
        if self.page:
            self.update()

    def set_processing(self) -> None:
        """Set status to processing audio."""
        self._status_text.value = "Processing..."
        self._status_text.color = PRIMARY_COLOR
        self._status_icon.name = ft.Icons.HOURGLASS_EMPTY
        self._status_icon.color = PRIMARY_COLOR
        self._animation_ring.bgcolor = PRIMARY_COLOR + "40"
        self._animation_ring.border = ft.border.all(3, PRIMARY_COLOR)
        if self.page:
            self.update()


class MicrophoneButton(ft.Container):
    """Large microphone button for voice input."""

    def __init__(
        self,
        on_click: Optional[Callable[[], None]] = None,
        size: int = 80,
    ) -> None:
        """Initialize the microphone button.

        Args:
            on_click: Callback when button is clicked.
            size: Button diameter in pixels.
        """
        self._is_active = False
        self._on_click = on_click
        self._size = size

        self._button = ft.IconButton(
            icon=ft.Icons.MIC_ROUNDED,
            icon_size=size // 2,
            icon_color=ft.Colors.WHITE,
            style=ft.ButtonStyle(
                shape=ft.CircleBorder(),
                bgcolor=PRIMARY_COLOR,
                overlay_color=ft.Colors.with_opacity(0.1, ft.Colors.WHITE),
                padding=size // 4,
            ),
            on_click=self._handle_click,
            tooltip="Press to speak",
        )

        super().__init__(
            content=self._button,
            width=size,
            height=size,
            border_radius=size // 2,
            shadow=ft.BoxShadow(
                spread_radius=1,
                blur_radius=15,
                color=ft.Colors.with_opacity(0.3, PRIMARY_COLOR),
                offset=ft.Offset(0, 4),
            ),
        )

    def _handle_click(self, e: ft.ControlEvent) -> None:
        """Handle button click."""
        if self._on_click:
            self._on_click()

    def set_active(self, active: bool) -> None:
        """Set the button active/inactive state.

        Args:
            active: True for active (recording) state.
        """
        self._is_active = active
        if active:
            self._button.icon = ft.Icons.MIC_ROUNDED
            self._button.style.bgcolor = LISTENING_COLOR
            self.shadow = ft.BoxShadow(
                spread_radius=2,
                blur_radius=20,
                color=ft.Colors.with_opacity(0.4, LISTENING_COLOR),
                offset=ft.Offset(0, 4),
            )
        else:
            self._button.icon = ft.Icons.MIC_ROUNDED
            self._button.style.bgcolor = PRIMARY_COLOR
            self.shadow = ft.BoxShadow(
                spread_radius=1,
                blur_radius=15,
                color=ft.Colors.with_opacity(0.3, PRIMARY_COLOR),
                offset=ft.Offset(0, 4),
            )
        if self.page:
            self.update()

    def set_disabled(self, disabled: bool) -> None:
        """Set the button disabled state.

        Args:
            disabled: True to disable the button.
        """
        self._button.disabled = disabled
        if disabled:
            self._button.style.bgcolor = TEXT_SECONDARY
        else:
            self._button.style.bgcolor = PRIMARY_COLOR
        if self.page:
            self.update()


class WelcomeCard(ft.Container):
    """Welcome card for new users."""

    def __init__(
        self,
        user_name: str,
        language: str,
        on_start: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize the welcome card.

        Args:
            user_name: Name of the user.
            language: Target language name.
            on_start: Callback when start button is clicked.
        """
        from polyglott.gui.theme import (
            ACCENT_COLOR,
            BORDER_RADIUS_LG,
            CARD_BG,
            FONT_SIZE_LARGE,
            FONT_SIZE_TITLE,
            SPACING_LG,
            SPACING_MD,
            TEXT_PRIMARY,
        )

        super().__init__(
            content=ft.Column(
                controls=[
                    ft.Text(
                        f"Hello, {user_name}!",
                        size=FONT_SIZE_TITLE,
                        weight=ft.FontWeight.BOLD,
                        color=PRIMARY_COLOR,
                    ),
                    ft.Text(
                        f"Ready to learn {language}?",
                        size=FONT_SIZE_LARGE,
                        color=TEXT_PRIMARY,
                    ),
                    ft.Container(height=SPACING_MD),
                    ft.ElevatedButton(
                        text="Start Learning",
                        icon=ft.Icons.PLAY_ARROW_ROUNDED,
                        style=ft.ButtonStyle(
                            bgcolor=ACCENT_COLOR,
                            color=ft.Colors.WHITE,
                            padding=ft.padding.symmetric(horizontal=32, vertical=16),
                            shape=ft.RoundedRectangleBorder(radius=BORDER_RADIUS_LG),
                        ),
                        on_click=lambda e: on_start() if on_start else None,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=SPACING_MD,
            ),
            bgcolor=CARD_BG,
            border_radius=BORDER_RADIUS_LG,
            padding=SPACING_LG,
            shadow=ft.BoxShadow(
                spread_radius=0,
                blur_radius=20,
                color=ft.Colors.with_opacity(0.1, ft.Colors.BLACK),
                offset=ft.Offset(0, 4),
            ),
        )
