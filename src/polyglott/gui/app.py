"""Main Flet application for Polyglott.

This module provides the graphical user interface for the
language tutoring application.
"""

import threading

import flet as ft

from polyglott.constants import (
    APP_NAME,
    LANGUAGE_NAMES,
    AgeGroup,
    TargetLanguage,
    language_name_to_code,
)
from polyglott.gui.components import (
    ChatBubble,
    ChatMessage,
    MicrophoneButton,
    StatusIndicator,
)
from polyglott.gui.theme import (
    ACCENT_COLOR,
    BG_COLOR,
    BORDER_RADIUS_LG,
    BORDER_RADIUS_MD,
    CARD_BG,
    FONT_SIZE_LARGE,
    FONT_SIZE_NORMAL,
    FONT_SIZE_SMALL,
    FONT_SIZE_TITLE,
    LANGUAGE_ICONS,
    PRIMARY_COLOR,
    SPACING_LG,
    SPACING_MD,
    SPACING_SM,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    create_theme,
)
from polyglott.persistence import (
    UserProfile,
    list_users,
    load_user_profile,
    save_user_profile,
)


class PolyglottApp:
    """Main Polyglott GUI application."""

    def __init__(self, page: ft.Page) -> None:
        """Initialize the application.

        Args:
            page: Flet page instance.
        """
        self.page = page
        self.profile: UserProfile | None = None
        self.target_language: TargetLanguage | None = None
        self._messages: list[ChatMessage] = []
        self._is_conversation_active = False
        self._conversation_thread: threading.Thread | None = None

        # Audio components (lazy-loaded)
        self._synthesizer = None
        self._player = None
        self._manager = None

        # UI components
        self._chat_list: ft.ListView | None = None
        self._status_indicator: StatusIndicator | None = None
        self._mic_button: MicrophoneButton | None = None

        self._setup_page()

    def _setup_page(self) -> None:
        """Configure the page settings."""
        self.page.title = f"{APP_NAME} - Language Tutor"
        self.page.theme = create_theme()
        self.page.bgcolor = BG_COLOR
        self.page.window.width = 500
        self.page.window.height = 800
        self.page.window.min_width = 400
        self.page.window.min_height = 600
        self.page.padding = 0

        # Show initial view
        self._show_user_selection()

    def _show_user_selection(self) -> None:
        """Show the user selection/welcome screen."""
        # Get existing users
        users = list_users()

        # Create user list
        user_tiles = []
        for name in users:
            profile = load_user_profile(name)
            if profile:
                lang = profile.target_language or "?"
                lang_name = LANGUAGE_NAMES.get(lang, lang)
                icon = LANGUAGE_ICONS.get(lang, "🌐")

                user_tiles.append(
                    ft.ListTile(
                        leading=ft.CircleAvatar(
                            content=ft.Text(
                                name[0].upper(),
                                color=ft.Colors.WHITE,
                                weight=ft.FontWeight.BOLD,
                            ),
                            bgcolor=PRIMARY_COLOR,
                        ),
                        title=ft.Text(name, weight=ft.FontWeight.W_500),
                        subtitle=ft.Text(f"{icon} Learning {lang_name}"),
                        trailing=ft.Icon(ft.Icons.ARROW_FORWARD_IOS, size=16),
                        on_click=lambda e, n=name: self._select_user(n),
                    )
                )

        # Build the view
        content = ft.Column(
            controls=[
                # Header
                ft.Container(
                    content=ft.Column(
                        controls=[
                            ft.Text(
                                APP_NAME,
                                size=FONT_SIZE_TITLE,
                                weight=ft.FontWeight.BOLD,
                                color=PRIMARY_COLOR,
                            ),
                            ft.Text(
                                "Your friendly language tutor",
                                size=FONT_SIZE_NORMAL,
                                color=TEXT_SECONDARY,
                            ),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    padding=SPACING_LG,
                    alignment=ft.Alignment(0, 0),
                ),
                # User list or new user prompt
                ft.Container(
                    content=ft.Column(
                        controls=[
                            ft.Text(
                                "Who's learning today?",
                                size=FONT_SIZE_LARGE,
                                weight=ft.FontWeight.W_500,
                                color=TEXT_PRIMARY,
                            ),
                            ft.Container(height=SPACING_SM),
                            *user_tiles,
                            ft.Divider(height=SPACING_MD),
                            ft.ListTile(
                                leading=ft.CircleAvatar(
                                    content=ft.Icon(
                                        ft.Icons.ADD,
                                        color=ft.Colors.WHITE,
                                    ),
                                    bgcolor=ACCENT_COLOR,
                                ),
                                title=ft.Text(
                                    "New Learner",
                                    weight=ft.FontWeight.W_500,
                                ),
                                subtitle=ft.Text("Set up a new profile"),
                                trailing=ft.Icon(ft.Icons.ARROW_FORWARD_IOS, size=16),
                                on_click=lambda e: self._show_new_user_setup(),
                            ),
                        ],
                    )
                    if users
                    else ft.Column(
                        controls=[
                            ft.Container(height=SPACING_LG),
                            ft.Icon(
                                ft.Icons.WAVING_HAND,
                                size=64,
                                color=ACCENT_COLOR,
                            ),
                            ft.Text(
                                "Welcome!",
                                size=FONT_SIZE_LARGE,
                                weight=ft.FontWeight.BOLD,
                                color=TEXT_PRIMARY,
                            ),
                            ft.Text(
                                "Let's set up your profile",
                                size=FONT_SIZE_NORMAL,
                                color=TEXT_SECONDARY,
                            ),
                            ft.Container(height=SPACING_MD),
                            ft.ElevatedButton(
                                text="Get Started",
                                icon=ft.Icons.ARROW_FORWARD,
                                style=ft.ButtonStyle(
                                    bgcolor=PRIMARY_COLOR,
                                    color=ft.Colors.WHITE,
                                    padding=ft.padding.symmetric(
                                        horizontal=32, vertical=16
                                    ),
                                ),
                                on_click=lambda e: self._show_new_user_setup(),
                            ),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    bgcolor=CARD_BG,
                    border_radius=BORDER_RADIUS_LG,
                    padding=SPACING_LG,
                    margin=SPACING_MD,
                ),
            ],
            expand=True,
            scroll=ft.ScrollMode.AUTO,
        )

        self.page.controls.clear()
        self.page.add(content)
        self.page.update()

    def _select_user(self, name: str) -> None:
        """Select an existing user.

        Args:
            name: User name to select.
        """
        self.profile = load_user_profile(name)
        if self.profile:
            self.target_language = self.profile.get_target_language()
            self._show_chat_view()

    def _show_new_user_setup(self) -> None:
        """Show the new user setup screen."""
        name_field = ft.TextField(
            label="What's your name?",
            hint_text="Enter your name",
            border_radius=BORDER_RADIUS_MD,
            filled=True,
            autofocus=True,
        )

        language_dropdown = ft.Dropdown(
            label="What language do you want to learn?",
            hint_text="Select a language",
            border_radius=BORDER_RADIUS_MD,
            filled=True,
            options=[
                ft.dropdown.Option(
                    key=lang.value,
                    text=f"{LANGUAGE_ICONS.get(lang.value, '')} "
                    f"{LANGUAGE_NAMES.get(lang.value, lang.value)}",
                )
                for lang in TargetLanguage
            ],
        )

        age_dropdown = ft.Dropdown(
            label="Age group",
            hint_text="Select age group",
            border_radius=BORDER_RADIUS_MD,
            filled=True,
            options=[
                ft.dropdown.Option(key=AgeGroup.PRESCHOOL.value, text="Preschool (3-5)"),
                ft.dropdown.Option(
                    key=AgeGroup.EARLY_PRIMARY.value, text="Early Primary (6-8)"
                ),
                ft.dropdown.Option(
                    key=AgeGroup.LATE_PRIMARY.value, text="Late Primary (9-12)"
                ),
            ],
        )

        def on_create(e: ft.ControlEvent) -> None:
            name = name_field.value.strip() if name_field.value else ""
            if not name:
                name_field.error_text = "Please enter a name"
                self.page.update()
                return

            if not language_dropdown.value:
                language_dropdown.error_text = "Please select a language"
                self.page.update()
                return

            if not age_dropdown.value:
                age_dropdown.error_text = "Please select an age group"
                self.page.update()
                return

            # Create profile
            self.profile = UserProfile(
                name=name,
                target_language=language_dropdown.value,
                age_group=age_dropdown.value,
            )
            save_user_profile(self.profile)

            self.target_language = self.profile.get_target_language()
            self._show_chat_view()

        content = ft.Column(
            controls=[
                # Back button
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.IconButton(
                                icon=ft.Icons.ARROW_BACK,
                                on_click=lambda e: self._show_user_selection(),
                            ),
                            ft.Text(
                                "New Learner",
                                size=FONT_SIZE_LARGE,
                                weight=ft.FontWeight.W_500,
                            ),
                        ],
                    ),
                    padding=SPACING_SM,
                ),
                # Form
                ft.Container(
                    content=ft.Column(
                        controls=[
                            name_field,
                            ft.Container(height=SPACING_MD),
                            language_dropdown,
                            ft.Container(height=SPACING_MD),
                            age_dropdown,
                            ft.Container(height=SPACING_LG),
                            ft.ElevatedButton(
                                text="Start Learning",
                                icon=ft.Icons.ROCKET_LAUNCH,
                                style=ft.ButtonStyle(
                                    bgcolor=PRIMARY_COLOR,
                                    color=ft.Colors.WHITE,
                                    padding=ft.padding.symmetric(
                                        horizontal=32, vertical=16
                                    ),
                                ),
                                on_click=on_create,
                            ),
                        ],
                        horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
                    ),
                    bgcolor=CARD_BG,
                    border_radius=BORDER_RADIUS_LG,
                    padding=SPACING_LG,
                    margin=SPACING_MD,
                ),
            ],
            expand=True,
        )

        self.page.controls.clear()
        self.page.add(content)
        self.page.update()

    def _show_chat_view(self) -> None:
        """Show the main chat interface."""
        if not self.profile or not self.target_language:
            return

        lang_name = LANGUAGE_NAMES.get(self.target_language.value, "")
        lang_icon = LANGUAGE_ICONS.get(self.target_language.value, "🌐")

        # Create components
        self._status_indicator = StatusIndicator()
        self._mic_button = MicrophoneButton(on_click=self._toggle_listening)

        # Chat message list
        self._chat_list = ft.ListView(
            expand=True,
            spacing=SPACING_SM,
            padding=SPACING_MD,
            auto_scroll=True,
        )

        # Header
        header = ft.Container(
            content=ft.Row(
                controls=[
                    ft.IconButton(
                        icon=ft.Icons.ARROW_BACK,
                        on_click=lambda e: self._back_to_users(),
                    ),
                    ft.Column(
                        controls=[
                            ft.Text(
                                f"{lang_icon} {lang_name}",
                                size=FONT_SIZE_NORMAL,
                                weight=ft.FontWeight.W_500,
                            ),
                            ft.Text(
                                f"Hi, {self.profile.name}!",
                                size=FONT_SIZE_SMALL,
                                color=TEXT_SECONDARY,
                            ),
                        ],
                        spacing=0,
                        expand=True,
                    ),
                    ft.IconButton(
                        icon=ft.Icons.SETTINGS_OUTLINED,
                        tooltip="Settings",
                    ),
                ],
            ),
            bgcolor=CARD_BG,
            padding=SPACING_SM,
            border=ft.border.only(bottom=ft.BorderSide(1, BG_COLOR)),
        )

        # Bottom control bar
        bottom_bar = ft.Container(
            content=ft.Column(
                controls=[
                    self._status_indicator,
                    ft.Container(height=SPACING_SM),
                    self._mic_button,
                    ft.Container(height=SPACING_SM),
                    ft.Text(
                        "Tap to speak",
                        size=FONT_SIZE_SMALL,
                        color=TEXT_SECONDARY,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            bgcolor=CARD_BG,
            padding=SPACING_MD,
            border=ft.border.only(top=ft.BorderSide(1, BG_COLOR)),
        )

        # Main layout
        content = ft.Column(
            controls=[
                header,
                self._chat_list,
                bottom_bar,
            ],
            expand=True,
            spacing=0,
        )

        self.page.controls.clear()
        self.page.add(content)
        self.page.update()

        # Start conversation
        self._start_conversation()

    def _back_to_users(self) -> None:
        """Go back to user selection."""
        self._stop_conversation()
        self._show_user_selection()

    def _add_message(self, text: str, is_user: bool) -> None:
        """Add a message to the chat.

        Args:
            text: Message text.
            is_user: True if from user.
        """
        if not self._chat_list:
            return

        message = ChatMessage(
            text=text,
            is_user=is_user,
            language=self.target_language.value if self.target_language else "en",
        )
        self._messages.append(message)

        bubble = ChatBubble(
            message=message,
            on_replay=self._replay_message,
        )
        self._chat_list.controls.append(bubble)
        self.page.update()

    def _replay_message(self, text: str, language: str) -> None:
        """Replay a message using TTS.

        Args:
            text: Text to speak.
            language: Language code.
        """
        # Run TTS in background thread
        def speak() -> None:
            try:
                self._ensure_audio_components()
                if self._synthesizer and self._player:
                    # Get native language code for default voice
                    native_lang = language_name_to_code(
                        self.profile.native_language if self.profile else "en"
                    )
                    # Use multilingual synthesis to handle language tags
                    result = self._synthesizer.synthesize_multilingual(
                        text, default_language=native_lang
                    )
                    self._player.play(result.audio, result.sample_rate, blocking=True)
            except Exception as e:
                print(f"TTS error: {e}")

        threading.Thread(target=speak, daemon=True).start()

    def _ensure_audio_components(self) -> None:
        """Ensure audio components are initialized."""
        if self._synthesizer is None:
            from polyglott.tts.synthesizer import SpeechSynthesizer

            self._synthesizer = SpeechSynthesizer()

        if self._player is None:
            from polyglott.audio.player import AudioPlayer

            self._player = AudioPlayer()

    def _start_conversation(self) -> None:
        """Start the conversation session."""
        if not self.profile or not self.target_language:
            return

        self._is_conversation_active = True

        # Show loading message
        lang_name = LANGUAGE_NAMES.get(self.target_language.value, "")
        self._add_message(
            f"Hello {self.profile.name}! Let's practice {lang_name} together. "
            "Tap the microphone and say something!",
            is_user=False,
        )

        # Update status
        if self._status_indicator:
            self._status_indicator.set_ready()

    def _stop_conversation(self) -> None:
        """Stop the current conversation."""
        self._is_conversation_active = False
        if self._manager:
            self._manager.stop()
            self._manager = None

    def _toggle_listening(self) -> None:
        """Toggle the listening state."""
        if not self._is_conversation_active:
            return

        # Start listening in background
        if self._mic_button:
            self._mic_button.set_active(True)
            self._mic_button.set_disabled(True)

        if self._status_indicator:
            self._status_indicator.set_listening()

        # Run conversation turn in background
        def run_turn() -> None:
            try:
                self._process_conversation_turn()
            finally:
                # Reset UI on main thread
                self.page.run_thread(self._reset_after_turn)

        threading.Thread(target=run_turn, daemon=True).start()

    def _process_conversation_turn(self) -> None:
        """Process a single conversation turn."""
        from polyglott.audio.recorder import AudioRecorder
        from polyglott.llm.tutor import LanguageTutor, TutorConfig
        from polyglott.stt.transcriber import SpeechTranscriber

        self._ensure_audio_components()

        # Create components if needed
        recorder = AudioRecorder()
        transcriber = SpeechTranscriber()

        config = TutorConfig(
            target_language=self.target_language,
            native_language=self.profile.native_language if self.profile else "en",
            age_group=self.profile.get_age_group() if self.profile else AgeGroup.EARLY_PRIMARY,
        )
        tutor = LanguageTutor(config)

        # Record
        def update_status_listening() -> None:
            if self._status_indicator:
                self._status_indicator.set_listening()
            self.page.update()

        self.page.run_thread(update_status_listening)

        recording = recorder.record_utterance(
            on_speech_start=lambda: None,
            on_speech_end=lambda: None,
        )

        if not recording.was_speech_detected:
            return

        # Transcribe
        def update_status_processing() -> None:
            if self._status_indicator:
                self._status_indicator.set_processing()
            self.page.update()

        self.page.run_thread(update_status_processing)

        transcription = transcriber.transcribe(
            recording.audio,
            language=self.target_language.value if self.target_language else "en",
        )

        if not transcription.text.strip():
            return

        # Add user message
        def add_user_msg() -> None:
            self._add_message(transcription.text, is_user=True)

        self.page.run_thread(add_user_msg)

        # Generate response
        def update_status_thinking() -> None:
            if self._status_indicator:
                self._status_indicator.set_thinking()
            self.page.update()

        self.page.run_thread(update_status_thinking)

        response = tutor.respond(transcription.text)

        # Add tutor message
        def add_tutor_msg() -> None:
            self._add_message(response.text, is_user=False)

        self.page.run_thread(add_tutor_msg)

        # Speak response
        def update_status_speaking() -> None:
            if self._status_indicator:
                self._status_indicator.set_speaking()
            self.page.update()

        self.page.run_thread(update_status_speaking)

        if self._synthesizer and self._player:
            # Convert native language name to ISO code (e.g., "English" -> "en")
            native_lang = language_name_to_code(
                self.profile.native_language if self.profile else "en"
            )
            synthesis = self._synthesizer.synthesize_multilingual(
                response.text,
                default_language=native_lang,
            )
            self._player.play(synthesis.audio, synthesis.sample_rate, blocking=True)

    def _reset_after_turn(self) -> None:
        """Reset UI after a conversation turn."""
        if self._mic_button:
            self._mic_button.set_active(False)
            self._mic_button.set_disabled(False)

        if self._status_indicator:
            self._status_indicator.set_ready()

        self.page.update()


def main() -> None:
    """Main entry point for the GUI application."""

    def app(page: ft.Page) -> None:
        PolyglottApp(page)

    ft.app(target=app)


if __name__ == "__main__":
    main()
