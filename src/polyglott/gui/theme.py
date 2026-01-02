"""Theme configuration for Polyglott GUI.

Child-friendly color palette and styling constants.
"""

from typing import Final

import flet as ft

# =============================================================================
# Color Palette - Bright, child-friendly colors
# =============================================================================

# Primary colors
PRIMARY_COLOR: Final[str] = "#6C63FF"  # Friendly purple
PRIMARY_LIGHT: Final[str] = "#A5A0FF"
PRIMARY_DARK: Final[str] = "#4A42CC"

# Accent colors
ACCENT_COLOR: Final[str] = "#FF6B6B"  # Coral red
ACCENT_LIGHT: Final[str] = "#FF9E9E"

# Background colors
BG_COLOR: Final[str] = "#F8F9FF"  # Soft off-white with purple tint
BG_DARK: Final[str] = "#E8EAFF"
CARD_BG: Final[str] = "#FFFFFF"

# Chat bubble colors
TUTOR_BUBBLE_COLOR: Final[str] = "#E3F2FD"  # Light blue
TUTOR_BUBBLE_BORDER: Final[str] = "#90CAF9"
USER_BUBBLE_COLOR: Final[str] = "#E8F5E9"  # Light green
USER_BUBBLE_BORDER: Final[str] = "#A5D6A7"

# Text colors
TEXT_PRIMARY: Final[str] = "#2D3748"
TEXT_SECONDARY: Final[str] = "#718096"
TEXT_LIGHT: Final[str] = "#FFFFFF"

# Status colors
SUCCESS_COLOR: Final[str] = "#48BB78"  # Green
WARNING_COLOR: Final[str] = "#ECC94B"  # Yellow
ERROR_COLOR: Final[str] = "#F56565"  # Red
INFO_COLOR: Final[str] = "#4299E1"  # Blue

# State indicator colors
LISTENING_COLOR: Final[str] = "#48BB78"  # Green pulse
THINKING_COLOR: Final[str] = "#ECC94B"  # Yellow
SPEAKING_COLOR: Final[str] = "#4299E1"  # Blue


# =============================================================================
# Typography
# =============================================================================

FONT_FAMILY: Final[str] = "Nunito, Poppins, sans-serif"
FONT_SIZE_SMALL: Final[int] = 12
FONT_SIZE_NORMAL: Final[int] = 16
FONT_SIZE_LARGE: Final[int] = 20
FONT_SIZE_XLARGE: Final[int] = 28
FONT_SIZE_TITLE: Final[int] = 36


# =============================================================================
# Spacing & Sizing
# =============================================================================

SPACING_XS: Final[int] = 4
SPACING_SM: Final[int] = 8
SPACING_MD: Final[int] = 16
SPACING_LG: Final[int] = 24
SPACING_XL: Final[int] = 32

BORDER_RADIUS_SM: Final[int] = 8
BORDER_RADIUS_MD: Final[int] = 16
BORDER_RADIUS_LG: Final[int] = 24
BORDER_RADIUS_BUBBLE: Final[int] = 20

# Chat bubble sizing
BUBBLE_MAX_WIDTH: Final[int] = 500
BUBBLE_PADDING: Final[int] = 16

# Button sizing
BUTTON_HEIGHT: Final[int] = 56
BUTTON_ICON_SIZE: Final[int] = 24
REPLAY_BUTTON_SIZE: Final[int] = 32


# =============================================================================
# Flet Theme Configuration
# =============================================================================

def create_theme() -> ft.Theme:
    """Create the Flet theme for the application.

    Returns:
        Configured Flet theme.
    """
    return ft.Theme(
        color_scheme_seed=PRIMARY_COLOR,
        visual_density=ft.VisualDensity.COMFORTABLE,
        color_scheme=ft.ColorScheme(
            primary=PRIMARY_COLOR,
            secondary=ACCENT_COLOR,
            background=BG_COLOR,
            surface=CARD_BG,
            on_primary=TEXT_LIGHT,
            on_secondary=TEXT_LIGHT,
            on_background=TEXT_PRIMARY,
            on_surface=TEXT_PRIMARY,
        ),
    )


# =============================================================================
# Language-specific decorations
# =============================================================================

LANGUAGE_COLORS: Final[dict[str, str]] = {
    "en": "#4A90D9",  # Blue for English
    "de": "#FFD700",  # Gold for German
    "es": "#E74C3C",  # Red for Spanish
    "ja": "#E91E63",  # Pink for Japanese
    "zh": "#FF5722",  # Orange for Mandarin
}

LANGUAGE_ICONS: Final[dict[str, str]] = {
    "en": "🇬🇧",
    "de": "🇩🇪",
    "es": "🇪🇸",
    "ja": "🇯🇵",
    "zh": "🇨🇳",
}
