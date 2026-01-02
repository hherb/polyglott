"""Command-line interface for Polyglott language tutor.

This module provides the main entry point and CLI for the
language tutoring application.
"""

import argparse
import sys
from typing import Optional

from polyglott.constants import (
    APP_NAME,
    APP_VERSION,
    LANGUAGE_NAMES,
    AgeGroup,
    TargetLanguage,
)
from polyglott.persistence import (
    UserProfile,
    get_last_user,
    list_users,
    load_user_profile,
    save_user_profile,
    user_exists,
)


class Speaker:
    """Handles speaking text aloud for users who cannot read."""

    def __init__(self, language: str = "en") -> None:
        """Initialize the speaker.

        Args:
            language: Language code for TTS.
        """
        self._language = language
        self._synthesizer = None
        self._player = None

    def _ensure_loaded(self) -> bool:
        """Ensure TTS components are loaded.

        Returns:
            True if loaded successfully.
        """
        if self._synthesizer is not None:
            return True

        try:
            from polyglott.audio.player import AudioPlayer
            from polyglott.tts.synthesizer import SpeechSynthesizer

            self._synthesizer = SpeechSynthesizer()
            self._player = AudioPlayer()
            return True
        except Exception:
            return False

    def speak(self, text: str, language: Optional[str] = None) -> None:
        """Speak text aloud.

        Args:
            text: Text to speak.
            language: Optional language override.
        """
        if not self._ensure_loaded():
            return

        lang = language or self._language
        try:
            result = self._synthesizer.synthesize(text, language=lang)
            self._player.play(result.audio, result.sample_rate, blocking=True)
        except Exception:
            pass  # Fail silently if TTS unavailable

    def set_language(self, language: str) -> None:
        """Set the TTS language.

        Args:
            language: Language code.
        """
        self._language = language


def print_banner() -> None:
    """Print the application banner."""
    print(f"""
+==============================================================+
|  {APP_NAME} v{APP_VERSION}                                           |
|  Your friendly language learning companion                    |
+==============================================================+
""")


def print_status(message: str, status: str = "info") -> None:
    """Print a status message with formatting.

    Args:
        message: Message to print.
        status: Status type (info, success, warning, error).
    """
    icons = {
        "info": "[i]",
        "success": "[ok]",
        "warning": "[!]",
        "error": "[X]",
        "listening": "[mic]",
        "thinking": "[...]",
        "speaking": "[>>]",
    }
    icon = icons.get(status, "")
    print(f"{icon} {message}")


def select_language(speaker: Optional[Speaker] = None) -> TargetLanguage:
    """Interactive language selection.

    Args:
        speaker: Optional speaker for audio prompts.

    Returns:
        Selected target language.
    """
    prompt = "What language would you like to learn?"
    print(f"\n{prompt}")
    if speaker:
        speaker.speak(prompt)

    languages = list(TargetLanguage)

    for i, lang in enumerate(languages, 1):
        name = LANGUAGE_NAMES.get(lang.value, lang.value)
        print(f"  {i}. {name}")

    while True:
        try:
            choice = input("\nEnter number (1-5): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(languages):
                return languages[idx]
        except ValueError:
            pass
        msg = "Please enter a valid number."
        print(msg)
        if speaker:
            speaker.speak(msg)


def select_age_group(speaker: Optional[Speaker] = None) -> AgeGroup:
    """Interactive age group selection.

    Args:
        speaker: Optional speaker for audio prompts.

    Returns:
        Selected age group.
    """
    prompt = "What is the learner's age group?"
    print(f"\n{prompt}")
    if speaker:
        speaker.speak(prompt)

    groups = [
        (AgeGroup.PRESCHOOL, "Preschool (3-5 years)"),
        (AgeGroup.EARLY_PRIMARY, "Early Primary (6-8 years)"),
        (AgeGroup.LATE_PRIMARY, "Late Primary (9-12 years)"),
    ]

    for i, (_, desc) in enumerate(groups, 1):
        print(f"  {i}. {desc}")

    while True:
        try:
            choice = input("\nEnter number (1-3): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(groups):
                return groups[idx][0]
        except ValueError:
            pass
        msg = "Please enter a valid number."
        print(msg)
        if speaker:
            speaker.speak(msg)


def get_student_name(speaker: Optional[Speaker] = None) -> str:
    """Get the student's name.

    Args:
        speaker: Optional speaker for audio prompts.

    Returns:
        Student name.
    """
    prompt = "What is your name?"
    print(f"\n{prompt}")
    if speaker:
        speaker.speak(prompt)

    name = input("Your name: ").strip()
    return name if name else "Friend"


def check_dependencies() -> bool:
    """Check if all required dependencies are available.

    Returns:
        True if all dependencies are available.
    """
    missing = []

    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")

    try:
        import sounddevice  # noqa: F401
    except ImportError:
        missing.append("sounddevice")

    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch (for Silero VAD)")

    try:
        import ollama
        ollama.list()
    except Exception:
        print_status(
            "Ollama not running. Start with: ollama serve",
            "warning"
        )
        print_status(
            "Then pull a model: ollama pull qwen2.5:7b",
            "info"
        )

    if missing:
        print_status(f"Missing dependencies: {', '.join(missing)}", "error")
        print_status("Install with: uv sync", "info")
        return False

    return True


def run_text_mode(
    target_language: TargetLanguage,
    native_language: str,
    age_group: AgeGroup,
    student_name: str,
    speaker: Optional[Speaker] = None,
) -> None:
    """Run the tutor in text mode with spoken output.

    Args:
        target_language: Language to learn.
        native_language: Student's native language.
        age_group: Student's age group.
        student_name: Student's name.
        speaker: Speaker for TTS output.
    """
    from polyglott.llm.tutor import LanguageTutor, TutorConfig

    config = TutorConfig(
        target_language=target_language,
        native_language=native_language,
        age_group=age_group,
    )
    tutor = LanguageTutor(config)

    greeting = tutor.reset_conversation()
    print(f"\n[Tutor]: {greeting}\n")
    if speaker:
        speaker.speak(greeting, language=target_language.value)

    instructions = "Type 'quit' to exit, 'lesson' to change topic"
    print(f"({instructions})\n")

    while True:
        try:
            user_input = input(f"[{student_name}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            farewell = "Great job today! See you next time!"
            print(f"\n[Tutor]: {farewell}")
            if speaker:
                speaker.speak(farewell, language=target_language.value)
            break

        if user_input.lower() == "lesson":
            lessons = tutor.get_available_lessons()
            print("\nAvailable lessons:")
            for i, lesson in enumerate(lessons, 1):
                print(f"  {i}. {lesson}")
            try:
                choice = int(input("Choose: ")) - 1
                if 0 <= choice < len(lessons):
                    tutor.set_lesson_focus(lessons[choice])
                    msg = f"Great! Let's practice {lessons[choice]}!"
                    print(f"\n[Tutor]: {msg}\n")
                    if speaker:
                        speaker.speak(msg, language=target_language.value)
            except (ValueError, IndexError):
                pass
            continue

        response = tutor.respond(user_input)
        print(f"\n[Tutor]: {response.text}\n")
        if speaker:
            speaker.speak(response.text, language=target_language.value)


def run_voice_mode(
    target_language: TargetLanguage,
    native_language: str,
    age_group: AgeGroup,
    student_name: str,
) -> None:
    """Run the tutor in voice mode with audio.

    Args:
        target_language: Language to learn.
        native_language: Student's native language.
        age_group: Student's age group.
        student_name: Student's name.
    """
    from polyglott.audio.pipeline import PipelineState
    from polyglott.conversation.manager import ConversationManager

    def on_state_change(state: PipelineState) -> None:
        """Handle state changes."""
        state_messages = {
            PipelineState.LISTENING: "Listening...",
            PipelineState.TRANSCRIBING: "Processing...",
            PipelineState.THINKING: "Thinking...",
            PipelineState.SPEAKING: "Speaking...",
        }
        if state in state_messages:
            print_status(state_messages[state], state.value)

    manager = ConversationManager(
        student_name=student_name,
        target_language=target_language,
        native_language=native_language,
        age_group=age_group,
        on_state_change=on_state_change,
    )

    # Preload all ML models before user interaction starts
    print_status("Loading AI models (this may take a moment)...", "info")
    manager.preload_models(
        status_callback=lambda msg: print_status(msg, "info")
    )

    print_status("Starting voice session...", "info")
    print("(Press Ctrl+C to exit)\n")

    greeting = manager.start()
    print(f"\n[Tutor]: {greeting}\n")

    try:
        manager.run_conversation_loop()
    except KeyboardInterrupt:
        pass
    finally:
        summary = manager.end_session(say_goodbye=True)
        print(f"\n{summary}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="polyglott",
        description="Your friendly language learning companion",
    )

    parser.add_argument(
        "--new-user",
        action="store_true",
        help="Start fresh with a new user (ignore saved profile)",
    )

    parser.add_argument(
        "--new-language",
        action="store_true",
        help="Choose a new language (keep other settings)",
    )

    parser.add_argument(
        "--user",
        "-u",
        type=str,
        help="Specify user name directly",
    )

    parser.add_argument(
        "--language",
        "-l",
        type=str,
        choices=["en", "de", "es", "ja", "zh"],
        help="Specify target language directly",
    )

    parser.add_argument(
        "--voice",
        "-v",
        action="store_true",
        help="Start in voice mode",
    )

    parser.add_argument(
        "--text",
        "-t",
        action="store_true",
        help="Start in text mode",
    )

    parser.add_argument(
        "--gui",
        "-g",
        action="store_true",
        help="Start the graphical user interface",
    )

    parser.add_argument(
        "--list-users",
        action="store_true",
        help="List saved users and exit",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"{APP_NAME} v{APP_VERSION}",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success).
    """
    args = parse_args()

    # Handle --list-users
    if args.list_users:
        users = list_users()
        if users:
            print("Saved users:")
            for name in users:
                print(f"  - {name}")
        else:
            print("No saved users found.")
        return 0

    # Handle --gui
    if args.gui:
        from polyglott.gui.app import main as gui_main

        gui_main()
        return 0

    print_banner()

    # Check dependencies
    if not check_dependencies():
        return 1

    # Initialize speaker for spoken prompts (assume user can't read)
    speaker = Speaker(language="en")

    # Try to load existing user
    profile: Optional[UserProfile] = None

    if not args.new_user:
        if args.user:
            profile = load_user_profile(args.user)
            if profile:
                msg = f"Welcome back, {profile.name}!"
                print(msg)
                speaker.speak(msg)
        else:
            profile = get_last_user()
            if profile:
                # Ask if this is the returning user
                msg = f"Hello! Are you {profile.name}?"
                print(f"\n{msg}")
                speaker.speak(msg)

                answer = input("(yes/no): ").strip().lower()
                if answer not in ("y", "yes"):
                    profile = None

    # Get user info if no profile
    if profile is None:
        student_name = get_student_name(speaker)
        if user_exists(student_name) and not args.new_user:
            profile = load_user_profile(student_name)
            if profile:
                msg = f"Welcome back, {profile.name}!"
                print(msg)
                speaker.speak(msg)

        if profile is None:
            profile = UserProfile(name=student_name)

    student_name = profile.name

    # Handle language selection
    if args.new_language or profile.target_language is None:
        target_language = select_language(speaker)
        profile.target_language = target_language.value
    elif args.language:
        profile.target_language = args.language
        target_language = profile.get_target_language()
    else:
        target_language = profile.get_target_language()

    # Set speaker language
    speaker.set_language(target_language.value)

    # Handle age group (only ask for new users)
    if profile.total_sessions == 0 and not args.user:
        age_group = select_age_group(speaker)
        profile.age_group = age_group.value
    else:
        age_group = profile.get_age_group()

    native_language = profile.native_language

    # Save profile
    save_user_profile(profile)

    # Announce what we're doing
    lang_name = LANGUAGE_NAMES.get(target_language.value, target_language.value)
    msg = f"Let's learn {lang_name}, {student_name}!"
    print(f"\n{msg}")
    speaker.speak(msg)

    # Choose mode
    if args.voice:
        mode = "2"
    elif args.text:
        mode = "1"
    else:
        print("\nHow would you like to practice?")
        print("  1. Text mode (type responses)")
        print("  2. Voice mode (speak responses)")
        speaker.speak("Press 1 for text mode, or 2 for voice mode")
        mode = input("\nEnter choice (1 or 2): ").strip()

    # Update session tracking
    profile.update_session()
    save_user_profile(profile)

    if mode == "2":
        run_voice_mode(target_language, native_language, age_group, student_name)
    else:
        run_text_mode(target_language, native_language, age_group, student_name, speaker)

    return 0


if __name__ == "__main__":
    sys.exit(main())
