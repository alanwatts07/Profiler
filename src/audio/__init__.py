"""Audio processing module for capture, processing, and speaker identification."""

from .audio_capture import AudioCapture
from .audio_processor import AudioProcessor
from .speaker_identifier import SpeakerIdentifier

__all__ = [
    "AudioCapture",
    "AudioProcessor",
    "SpeakerIdentifier",
]
