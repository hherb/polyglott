"""Command-line interface for Polyglott language tutor.

This module provides the main entry point and CLI for the
language tutoring application.
"""

import sys
from typing import Optional

from polyglott.constants import (
    APP_NAME,
    APP_VERSION,
    LANGUAGE_NAMES,
    AgeGroup,
    TargetLanguage,
)


def print_banner() -> None:
    """Print the application banner."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  {APP_NAME} v{APP_VERSION}                                           ║
║  Your friendly language learning companion                    ║
╚══════════════════════════════════════════════════════════════╝
""")


def print_status(message: str, status: str = "info") -> None:
    """Print a status message with formatting.

    Args:
        message: Message to print.
        status: Status type (info, success, warning, error).
    """
    icons = {
        "info": "ℹ️ ",
        "success": "✓",
        "warning": "⚠️ ",
        "error": "✗",
        "listening": "🎤",
        "thinking": "💭",
        "speaking": "🔊",
    }
    icon = icons.get(status, "")
    print(f"{icon} {message}")


def select_language() -> TargetLanguage:
    """Interactive language selection.

    Returns:
        Selected target language.
    """
    print("\nWhat language would you like to learn?")
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
        print("Please enter a valid number.")


def select_age_group() -> AgeGroup:
    """Interactive age group selection.

    Returns:
        Selected age group.
    """
    print("\nWhat's the learner's age group?")
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
        print("Please enter a valid number.")


def get_student_name() -> str:
    """Get the student's name.

    Returns:
        Student name.
    """
    name = input("\nWhat's your name? ").strip()
    return name if name else "Friend"


def check_dependencies() -> bool:
    """Check if all required dependencies are available.

    Returns:
        True if all dependencies are available.
    """
    missing = []

    # Check core dependencies
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

    # Check if Ollama is running
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
) -> None:
    """Run the tutor in text-only mode (no audio).

    Args:
        target_language: Language to learn.
        native_language: Student's native language.
        age_group: Student's age group.
        student_name: Student's name.
    """
    from polyglott.llm.tutor import LanguageTutor, TutorConfig

    config = TutorConfig(
        target_language=target_language,
        native_language=native_language,
        age_group=age_group,
    )
    tutor = LanguageTutor(config)

    greeting = tutor.reset_conversation()
    print(f"\n🎓 Tutor: {greeting}\n")

    print("(Type 'quit' to exit, 'lesson' to change topic)\n")

    while True:
        try:
            user_input = input(f"👤 {student_name}: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("\n🎓 Tutor: Great job today! See you next time!")
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
                    print(f"\n🎓 Tutor: Great! Let's practice {lessons[choice]}!\n")
            except (ValueError, IndexError):
                pass
            continue

        response = tutor.respond(user_input)
        print(f"\n🎓 Tutor: {response.text}\n")


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

    print_status("Starting voice session...", "info")
    print("(Press Ctrl+C to exit)\n")

    greeting = manager.start()
    print(f"\n🎓 Tutor: {greeting}\n")

    try:
        manager.run_conversation_loop()
    except KeyboardInterrupt:
        pass
    finally:
        summary = manager.end_session(say_goodbye=True)
        print(f"\n{summary}")


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code (0 for success).
    """
    print_banner()

    # Check dependencies
    if not check_dependencies():
        return 1

    # Interactive setup
    target_language = select_language()
    age_group = select_age_group()
    student_name = get_student_name()

    # Determine native language
    native_language = "English"  # Default, could be made configurable

    lang_name = LANGUAGE_NAMES.get(target_language.value, target_language.value)
    print(f"\n🎯 Great! Let's learn {lang_name}, {student_name}!")

    # Choose mode
    print("\nHow would you like to practice?")
    print("  1. Text mode (type responses)")
    print("  2. Voice mode (speak responses)")

    mode = input("\nEnter choice (1 or 2): ").strip()

    if mode == "2":
        run_voice_mode(target_language, native_language, age_group, student_name)
    else:
        run_text_mode(target_language, native_language, age_group, student_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
