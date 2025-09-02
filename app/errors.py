"""Custom exceptions."""

class TranscriptionError(Exception):
    """Raised when transcription fails."""


class DiarizationError(Exception):
    """Raised when diarization fails."""


class AudioProcessingError(Exception):
    """Raised for audio preprocessing issues."""


class ConfigError(Exception):
    """Raised when configuration is invalid."""
