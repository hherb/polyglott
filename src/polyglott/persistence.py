"""User profile persistence for Polyglott.

This module handles saving and loading user preferences
so the tutor remembers returning users.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from polyglott.constants import AgeGroup, TargetLanguage


def get_data_dir() -> Path:
    """Get the user data directory for Polyglott.

    Returns:
        Path to ~/.polyglott/ directory.
    """
    data_dir = Path.home() / ".polyglott"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_users_dir() -> Path:
    """Get the directory for user profiles.

    Returns:
        Path to ~/.polyglott/users/ directory.
    """
    users_dir = get_data_dir() / "users"
    users_dir.mkdir(parents=True, exist_ok=True)
    return users_dir


@dataclass
class UserProfile:
    """User profile data.

    Attributes:
        name: User's name.
        target_language: Language being learned.
        native_language: User's native language.
        age_group: User's age group.
        created_at: When profile was created.
        last_session: When user last practiced.
        total_sessions: Number of practice sessions.
    """

    name: str
    target_language: str = "en"
    native_language: str = "English"
    age_group: str = "early_primary"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_session: Optional[str] = None
    total_sessions: int = 0

    def get_target_language(self) -> TargetLanguage:
        """Get target language as enum.

        Returns:
            TargetLanguage enum value.
        """
        for lang in TargetLanguage:
            if lang.value == self.target_language:
                return lang
        return TargetLanguage.ENGLISH

    def get_age_group(self) -> AgeGroup:
        """Get age group as enum.

        Returns:
            AgeGroup enum value.
        """
        for group in AgeGroup:
            if group.value == self.age_group:
                return group
        return AgeGroup.EARLY_PRIMARY

    def update_session(self) -> None:
        """Update the session tracking after a practice session."""
        self.last_session = datetime.now().isoformat()
        self.total_sessions += 1


def _get_profile_path(name: str) -> Path:
    """Get path to a user's profile file.

    Args:
        name: User's name.

    Returns:
        Path to profile JSON file.
    """
    safe_name = "".join(c for c in name.lower() if c.isalnum() or c in " _-")
    safe_name = safe_name.strip().replace(" ", "_") or "user"
    return get_users_dir() / f"{safe_name}.json"


def save_user_profile(profile: UserProfile) -> Path:
    """Save user profile to disk.

    Args:
        profile: UserProfile to save.

    Returns:
        Path where profile was saved.
    """
    profile_path = _get_profile_path(profile.name)
    with open(profile_path, "w") as f:
        json.dump(asdict(profile), f, indent=2)
    return profile_path


def load_user_profile(name: str) -> Optional[UserProfile]:
    """Load user profile from disk.

    Args:
        name: User's name.

    Returns:
        UserProfile if found, None otherwise.
    """
    profile_path = _get_profile_path(name)
    if not profile_path.exists():
        return None

    try:
        with open(profile_path) as f:
            data = json.load(f)
        return UserProfile(**data)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def get_last_user() -> Optional[UserProfile]:
    """Get the most recently active user.

    Returns:
        UserProfile of most recent user, or None if no users.
    """
    users_dir = get_users_dir()
    profiles: list[tuple[datetime, UserProfile]] = []

    for profile_file in users_dir.glob("*.json"):
        try:
            with open(profile_file) as f:
                data = json.load(f)
            profile = UserProfile(**data)
            last = profile.last_session or profile.created_at
            dt = datetime.fromisoformat(last)
            profiles.append((dt, profile))
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            continue

    if not profiles:
        return None

    profiles.sort(key=lambda x: x[0], reverse=True)
    return profiles[0][1]


def list_users() -> list[str]:
    """List all saved user names.

    Returns:
        List of user names.
    """
    users_dir = get_users_dir()
    names = []

    for profile_file in users_dir.glob("*.json"):
        try:
            with open(profile_file) as f:
                data = json.load(f)
            names.append(data.get("name", profile_file.stem))
        except (json.JSONDecodeError, KeyError):
            continue

    return sorted(names)


def user_exists(name: str) -> bool:
    """Check if a user profile exists.

    Args:
        name: User's name.

    Returns:
        True if profile exists.
    """
    return _get_profile_path(name).exists()
