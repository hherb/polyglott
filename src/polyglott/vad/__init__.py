"""Voice Activity Detection module.

This module provides voice activity detection with support for
multiple backends:
- Silero VAD: PyTorch-based, 32ms chunks, well-tested
- TEN VAD: ONNX-based, 10/16ms chunks, lower latency

Example:
    # Use default backend (Silero)
    >>> detector = create_vad_detector()

    # Use specific backend
    >>> detector = create_vad_detector(backend="ten_vad")

    # Check available backends
    >>> print(get_available_backends())
    ['silero', 'ten_vad']  # if TEN VAD is installed
"""

from polyglott.constants import (
    DEFAULT_VAD_BACKEND,
    TEN_VAD_HOP_SIZE,
    VAD_SPEECH_THRESHOLD,
    VADBackendType,
)
from polyglott.vad.base import BaseVADDetector, SpeechState, VADBackend, VADResult
from polyglott.vad.detector import SileroVadDetector, VoiceActivityDetector
from polyglott.vad.ten_vad import (
    TEN_VAD_AVAILABLE,
    TenVadDetector,
    TenVadNotInstalledError,
    is_ten_vad_available,
)

__all__ = [
    # Base classes and types
    "BaseVADDetector",
    "VADBackend",
    "VADResult",
    "SpeechState",
    # Silero VAD
    "VoiceActivityDetector",
    "SileroVadDetector",
    # TEN VAD
    "TenVadDetector",
    "TenVadNotInstalledError",
    "TEN_VAD_AVAILABLE",
    "is_ten_vad_available",
    # Factory functions
    "create_vad_detector",
    "get_available_backends",
]


def get_available_backends() -> list[str]:
    """Get list of available VAD backends.

    Returns:
        List of backend names that are installed and available.
    """
    backends = [VADBackendType.SILERO]
    if TEN_VAD_AVAILABLE:
        backends.append(VADBackendType.TEN_VAD)
    return [b.value for b in backends]


def create_vad_detector(
    backend: str = DEFAULT_VAD_BACKEND,
    threshold: float = VAD_SPEECH_THRESHOLD,
    **kwargs,
) -> BaseVADDetector:
    """Factory function to create a VAD detector with specified backend.

    Args:
        backend: VAD backend to use ("silero" or "ten_vad").
        threshold: Speech probability threshold (0.0-1.0).
        **kwargs: Additional backend-specific arguments.
            For TEN VAD: hop_size (160 or 256, default 256)

    Returns:
        Configured VAD detector instance.

    Raises:
        ValueError: If backend is not recognized.
        TenVadNotInstalledError: If TEN VAD is requested but not installed.

    Example:
        >>> # Use Silero VAD (default)
        >>> detector = create_vad_detector()

        >>> # Use TEN VAD with custom hop size
        >>> detector = create_vad_detector(
        ...     backend="ten_vad",
        ...     hop_size=160,  # 10ms frames
        ... )
    """
    try:
        backend_type = VADBackendType(backend) if isinstance(backend, str) else backend
    except ValueError:
        available = get_available_backends()
        raise ValueError(
            f"Unknown VAD backend: {backend}. Available: {available}"
        )

    if backend_type == VADBackendType.SILERO:
        return VoiceActivityDetector(threshold=threshold, **kwargs)

    elif backend_type == VADBackendType.TEN_VAD:
        hop_size = kwargs.pop("hop_size", TEN_VAD_HOP_SIZE)
        return TenVadDetector(threshold=threshold, hop_size=hop_size, **kwargs)

    else:
        available = get_available_backends()
        raise ValueError(
            f"Unknown VAD backend: {backend}. Available: {available}"
        )
