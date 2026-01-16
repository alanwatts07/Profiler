"""Audio processor for speech-to-text using OpenAI Whisper.

Handles audio preprocessing, noise reduction, and transcription.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass, field

import numpy as np

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

try:
    from scipy.io import wavfile
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..config import config
from .audio_capture import AudioChunk

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from audio transcription."""
    text: str = ""
    segments: list = field(default_factory=list)
    words: list = field(default_factory=list)
    language: str = "en"
    duration: float = 0.0

    @property
    def has_content(self) -> bool:
        """Check if transcription has meaningful content."""
        return bool(self.text.strip())


class AudioProcessor:
    """Processes audio and performs speech-to-text."""

    def __init__(
        self,
        model_size: str = None,
        device: str = None,
        language: str = None
    ):
        """Initialize audio processor.

        Args:
            model_size: Whisper model size (tiny/base/small/medium/large)
            device: Device to use (cpu/cuda)
            language: Language code for transcription
        """
        self.model_size = model_size or config.WHISPER_MODEL_SIZE
        self.device = device or config.WHISPER_DEVICE
        self.language = language or config.WHISPER_LANGUAGE

        self._model = None
        self._vad = None

        # Initialize VAD if available
        if VAD_AVAILABLE:
            try:
                self._vad = webrtcvad.Vad(2)  # Aggressiveness 0-3
            except Exception as e:
                logger.warning(f"Failed to initialize VAD: {e}")

    def _load_model(self):
        """Lazy load Whisper model."""
        if self._model is None:
            if not WHISPER_AVAILABLE:
                raise ImportError("whisper is required for transcription")

            logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
            self._model = whisper.load_model(self.model_size, device=self.device)
            logger.info("Whisper model loaded")

    def preprocess_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """Preprocess audio: noise reduction, normalization.

        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Preprocessed audio array
        """
        # Ensure float32
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Normalize
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            audio_data = audio_data / max_val * 0.95

        # Simple noise gate
        threshold = 0.01
        audio_data = np.where(np.abs(audio_data) < threshold, 0, audio_data)

        # High-pass filter to remove low-frequency noise (optional)
        if SCIPY_AVAILABLE and sample_rate > 0:
            try:
                # Remove frequencies below 80 Hz
                nyquist = sample_rate / 2
                cutoff = 80 / nyquist
                if cutoff < 1:
                    b, a = signal.butter(4, cutoff, btype='high')
                    audio_data = signal.filtfilt(b, a, audio_data)
            except Exception as e:
                logger.debug(f"High-pass filter skipped: {e}")

        return audio_data

    def detect_voice_activity(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        frame_duration: int = 30
    ) -> list:
        """Detect voice activity in audio.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            frame_duration: Frame duration in ms (10, 20, or 30)

        Returns:
            List of (start, end) tuples in seconds for speech segments
        """
        if not self._vad:
            # No VAD available, return entire audio as speech
            return [(0, len(audio_data) / sample_rate)]

        # WebRTC VAD requires specific sample rates
        if sample_rate not in [8000, 16000, 32000, 48000]:
            logger.debug("Sample rate not supported by VAD, assuming all speech")
            return [(0, len(audio_data) / sample_rate)]

        # Convert to int16 for VAD
        audio_int16 = (audio_data * 32767).astype(np.int16)

        frame_samples = int(sample_rate * frame_duration / 1000)
        frames = []

        for i in range(0, len(audio_int16) - frame_samples, frame_samples):
            frame = audio_int16[i:i + frame_samples]
            try:
                is_speech = self._vad.is_speech(frame.tobytes(), sample_rate)
                frames.append((i / sample_rate, is_speech))
            except Exception:
                frames.append((i / sample_rate, True))

        # Convert frames to segments
        segments = []
        in_speech = False
        speech_start = 0

        for time_sec, is_speech in frames:
            if is_speech and not in_speech:
                speech_start = time_sec
                in_speech = True
            elif not is_speech and in_speech:
                if time_sec - speech_start >= 0.2:  # Min 200ms
                    segments.append((speech_start, time_sec))
                in_speech = False

        # Close final segment
        if in_speech:
            segments.append((speech_start, len(audio_data) / sample_rate))

        return segments if segments else [(0, len(audio_data) / sample_rate)]

    def transcribe(
        self,
        audio: Union[np.ndarray, str, AudioChunk],
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """Transcribe audio to text with timestamps.

        Args:
            audio: Audio data (numpy array, file path, or AudioChunk)
            sample_rate: Sample rate if audio is numpy array

        Returns:
            TranscriptionResult with text and segments
        """
        self._load_model()

        # Handle different input types
        if isinstance(audio, AudioChunk):
            audio_data = audio.data
            sample_rate = audio.sample_rate
        elif isinstance(audio, str) or isinstance(audio, Path):
            # Load from file
            result = self._transcribe_file(str(audio))
            return result
        else:
            audio_data = audio

        # Preprocess
        audio_data = self.preprocess_audio(audio_data, sample_rate)

        # Save to temp file (Whisper works better with files)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp:
            self._save_temp_wav(audio_data, sample_rate, tmp.name)
            result = self._transcribe_file(tmp.name)

        return result

    def _transcribe_file(self, filepath: str) -> TranscriptionResult:
        """Transcribe audio file.

        Args:
            filepath: Path to audio file

        Returns:
            TranscriptionResult
        """
        try:
            result = self._model.transcribe(
                filepath,
                language=self.language if self.language != 'auto' else None,
                word_timestamps=True,
                verbose=False
            )

            # Parse result
            segments = []
            for seg in result.get('segments', []):
                segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text'].strip()
                })

            words = []
            for seg in result.get('segments', []):
                for word_info in seg.get('words', []):
                    words.append({
                        'word': word_info.get('word', '').strip(),
                        'start': word_info.get('start', 0),
                        'end': word_info.get('end', 0)
                    })

            return TranscriptionResult(
                text=result.get('text', '').strip(),
                segments=segments,
                words=words,
                language=result.get('language', 'en'),
                duration=segments[-1]['end'] if segments else 0
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptionResult()

    def _save_temp_wav(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        filepath: str
    ):
        """Save audio to temporary WAV file."""
        if SCIPY_AVAILABLE:
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wavfile.write(filepath, sample_rate, audio_int16)
        else:
            raise ImportError("scipy required for saving audio")

    def transcribe_with_vad(
        self,
        audio: Union[np.ndarray, AudioChunk],
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """Transcribe audio, only processing speech segments.

        Args:
            audio: Audio data
            sample_rate: Sample rate

        Returns:
            TranscriptionResult
        """
        if isinstance(audio, AudioChunk):
            audio_data = audio.data
            sample_rate = audio.sample_rate
        else:
            audio_data = audio

        # Detect speech segments
        vad_segments = self.detect_voice_activity(audio_data, sample_rate)

        if not vad_segments:
            return TranscriptionResult()

        # Only transcribe speech portions
        speech_audio = []
        for start, end in vad_segments:
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            speech_audio.append(audio_data[start_sample:end_sample])

        if not speech_audio:
            return TranscriptionResult()

        combined_audio = np.concatenate(speech_audio)
        return self.transcribe(combined_audio, sample_rate)
