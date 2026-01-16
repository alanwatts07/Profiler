"""Speaker identification and diarization module.

Uses pyannote.audio for speaker diarization and voice embeddings.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional, Union, Dict, List
from dataclasses import dataclass, field

import numpy as np

try:
    from pyannote.audio import Pipeline
    from pyannote.audio import Inference
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

try:
    from scipy.io import wavfile
    from scipy.spatial.distance import cosine
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..config import config
from .audio_capture import AudioChunk

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """A segment of speech attributed to a speaker."""
    speaker: str
    start: float
    end: float
    text: str = ""
    confidence: float = 1.0


@dataclass
class DiarizationResult:
    """Result from speaker diarization."""
    segments: List[SpeakerSegment] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)
    duration: float = 0.0

    def get_speaker_segments(self, speaker: str) -> List[SpeakerSegment]:
        """Get all segments for a specific speaker."""
        return [s for s in self.segments if s.speaker == speaker]

    def get_timeline(self) -> List[dict]:
        """Get segments as timeline dictionaries."""
        return [
            {
                "speaker": s.speaker,
                "start": s.start,
                "end": s.end,
                "text": s.text
            }
            for s in sorted(self.segments, key=lambda x: x.start)
        ]


class SpeakerIdentifier:
    """Identifies and tracks speakers using voice embeddings."""

    def __init__(self, hf_token: str = None):
        """Initialize speaker identifier.

        Args:
            hf_token: HuggingFace token for pyannote models
        """
        self.hf_token = hf_token or config.HF_TOKEN
        self.similarity_threshold = config.SPEAKER_SIMILARITY_THRESHOLD

        self._pipeline = None
        self._embedding_model = None
        self._known_speakers: Dict[str, np.ndarray] = {}
        self._speaker_counter = 0

    def _load_pipeline(self):
        """Lazy load diarization pipeline."""
        if self._pipeline is None:
            if not PYANNOTE_AVAILABLE:
                raise ImportError(
                    "pyannote.audio is required for speaker diarization. "
                    "Install with: pip install pyannote.audio"
                )

            if not self.hf_token:
                logger.warning(
                    "HuggingFace token not set. Some features may not work. "
                    "Set HF_TOKEN in .env file."
                )

            try:
                logger.info("Loading speaker diarization pipeline...")
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=self.hf_token
                )
                logger.info("Diarization pipeline loaded")
            except Exception as e:
                logger.error(f"Failed to load diarization pipeline: {e}")
                raise

    def _load_embedding_model(self):
        """Lazy load speaker embedding model."""
        if self._embedding_model is None:
            if not PYANNOTE_AVAILABLE:
                raise ImportError("pyannote.audio required for embeddings")

            try:
                logger.info("Loading speaker embedding model...")
                # Use Model.from_pretrained for embeddings in newer pyannote versions
                try:
                    from pyannote.audio import Model
                    model = Model.from_pretrained(
                        "pyannote/wespeaker-voxceleb-resnet34-LM",
                        use_auth_token=self.hf_token
                    )
                    self._embedding_model = Inference(model, window="whole")
                except Exception as e1:
                    logger.warning(f"Model approach failed: {e1}, trying direct Inference")
                    try:
                        self._embedding_model = Inference(
                            "pyannote/wespeaker-voxceleb-resnet34-LM",
                            use_auth_token=self.hf_token
                        )
                    except TypeError:
                        self._embedding_model = Inference(
                            "pyannote/wespeaker-voxceleb-resnet34-LM"
                        )
                logger.info("Embedding model loaded")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    def diarize(
        self,
        audio: Union[str, Path, np.ndarray, AudioChunk],
        sample_rate: int = 16000
    ) -> DiarizationResult:
        """Perform speaker diarization on audio.

        Args:
            audio: Audio file path, numpy array, or AudioChunk
            sample_rate: Sample rate if audio is numpy array

        Returns:
            DiarizationResult with speaker segments
        """
        self._load_pipeline()

        # Handle different input types - convert to waveform dict to avoid torchcodec issues
        import torch

        if isinstance(audio, AudioChunk):
            waveform = torch.from_numpy(audio.data).float().unsqueeze(0)
            sr = audio.sample_rate
        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float().unsqueeze(0)
            sr = sample_rate
        else:
            # Load from file using scipy
            from scipy.io import wavfile
            file_sr, file_data = wavfile.read(str(audio))
            # Convert to float and normalize
            if file_data.dtype == np.int16:
                file_data = file_data.astype(np.float32) / 32768.0
            elif file_data.dtype == np.int32:
                file_data = file_data.astype(np.float32) / 2147483648.0
            waveform = torch.from_numpy(file_data).float().unsqueeze(0)
            sr = file_sr

        # Pass as dictionary to bypass AudioDecoder
        audio_input = {"waveform": waveform, "sample_rate": sr}

        try:
            # Run diarization
            diarization = self._pipeline(audio_input)

            # Parse results
            segments = []
            speakers = set()

            # Handle different pyannote output formats
            annotation = None

            # Check for new API first (pyannote 3.x)
            if hasattr(diarization, 'speaker_diarization'):
                annotation = diarization.speaker_diarization
            elif hasattr(diarization, 'itertracks'):
                # Old API - diarization is the annotation itself
                annotation = diarization
            elif hasattr(diarization, 'annotation'):
                annotation = diarization.annotation

            if annotation is not None and hasattr(annotation, 'itertracks'):
                for turn, _, speaker in annotation.itertracks(yield_label=True):
                    segment = SpeakerSegment(
                        speaker=speaker,
                        start=turn.start,
                        end=turn.end
                    )
                    segments.append(segment)
                    speakers.add(speaker)
            else:
                logger.warning(f"Unknown diarization output format: {type(diarization)}")
                logger.warning(f"Available attributes: {dir(diarization)}")

            # Get duration from last segment
            duration = max(s.end for s in segments) if segments else 0.0

            return DiarizationResult(
                segments=segments,
                speakers=list(speakers),
                duration=duration
            )

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise

    def extract_embedding(
        self,
        audio: Union[str, Path, np.ndarray, AudioChunk],
        sample_rate: int = 16000
    ) -> Optional[np.ndarray]:
        """Extract voice embedding from audio.

        Args:
            audio: Audio data
            sample_rate: Sample rate

        Returns:
            Embedding vector (512-dim) or None
        """
        self._load_embedding_model()

        # Handle different input types - convert to waveform dict to avoid torchcodec issues
        import torch

        if isinstance(audio, AudioChunk):
            waveform = torch.from_numpy(audio.data).float().unsqueeze(0)
            sr = audio.sample_rate
        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float().unsqueeze(0)
            sr = sample_rate
        else:
            # Load from file using scipy
            from scipy.io import wavfile
            file_sr, file_data = wavfile.read(str(audio))
            if file_data.dtype == np.int16:
                file_data = file_data.astype(np.float32) / 32768.0
            elif file_data.dtype == np.int32:
                file_data = file_data.astype(np.float32) / 2147483648.0
            # Handle stereo -> mono
            if len(file_data.shape) > 1:
                file_data = np.mean(file_data, axis=1)
            waveform = torch.from_numpy(file_data).float().unsqueeze(0)
            sr = file_sr

        # Pass as dictionary to bypass AudioDecoder
        audio_input = {"waveform": waveform, "sample_rate": sr}

        try:
            embedding = self._embedding_model(audio_input)
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return None

    def match_speaker(
        self,
        embedding: np.ndarray,
        threshold: float = None
    ) -> Optional[str]:
        """Match embedding against known speakers.

        Args:
            embedding: Voice embedding to match
            threshold: Similarity threshold (uses config if None)

        Returns:
            Speaker ID if match found, None otherwise
        """
        if not self._known_speakers:
            return None

        threshold = threshold or self.similarity_threshold

        best_match = None
        best_similarity = 0.0

        for speaker_id, known_embedding in self._known_speakers.items():
            # Cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(embedding, known_embedding)

            if similarity > threshold and similarity > best_similarity:
                best_match = speaker_id
                best_similarity = similarity

        if best_match:
            logger.debug(f"Matched speaker {best_match} with similarity {best_similarity:.3f}")

        return best_match

    def register_speaker(
        self,
        embedding: np.ndarray,
        speaker_id: str = None
    ) -> str:
        """Register a new speaker with their embedding.

        Args:
            embedding: Voice embedding
            speaker_id: Optional ID (generated if None)

        Returns:
            Speaker ID
        """
        if speaker_id is None:
            speaker_id = self._generate_speaker_id()

        self._known_speakers[speaker_id] = embedding
        logger.info(f"Registered speaker: {speaker_id}")
        return speaker_id

    def rename_speaker(self, old_id: str, new_id: str) -> bool:
        """Rename a registered speaker.

        Args:
            old_id: Current speaker ID
            new_id: New speaker ID

        Returns:
            True if successful
        """
        if old_id not in self._known_speakers:
            return False

        self._known_speakers[new_id] = self._known_speakers.pop(old_id)
        logger.info(f"Renamed speaker {old_id} to {new_id}")
        return True

    def identify_or_register(
        self,
        audio: Union[np.ndarray, AudioChunk],
        sample_rate: int = 16000
    ) -> str:
        """Identify speaker or register if new.

        Args:
            audio: Audio data
            sample_rate: Sample rate

        Returns:
            Speaker ID (existing or new)
        """
        embedding = self.extract_embedding(audio, sample_rate)
        if embedding is None:
            return self._generate_speaker_id()

        # Try to match existing
        matched = self.match_speaker(embedding)
        if matched:
            return matched

        # Register as new speaker
        return self.register_speaker(embedding)

    def get_known_speakers(self) -> List[str]:
        """Get list of all known speaker IDs."""
        return list(self._known_speakers.keys())

    def load_embeddings(self, embeddings_dict: Dict[str, list]):
        """Load speaker embeddings from dictionary.

        Args:
            embeddings_dict: Dict of speaker_id -> embedding list
        """
        for speaker_id, embedding in embeddings_dict.items():
            self._known_speakers[speaker_id] = np.array(embedding)
        logger.info(f"Loaded {len(embeddings_dict)} speaker embeddings")

    def export_embeddings(self) -> Dict[str, list]:
        """Export speaker embeddings as dictionary.

        Returns:
            Dict of speaker_id -> embedding list
        """
        return {
            speaker_id: embedding.tolist()
            for speaker_id, embedding in self._known_speakers.items()
        }

    def _generate_speaker_id(self) -> str:
        """Generate a new speaker ID."""
        self._speaker_counter += 1
        # Use letters: Speaker A, Speaker B, etc.
        letter = chr(ord('A') + (self._speaker_counter - 1) % 26)
        suffix = "" if self._speaker_counter <= 26 else str(self._speaker_counter // 26)
        return f"Speaker {letter}{suffix}"

    def _save_temp_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> str:
        """Save audio to temporary file.

        Returns:
            Path to temporary file
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy required for audio processing")

        # Create temp file
        fd, path = tempfile.mkstemp(suffix='.wav')

        # Ensure float32 and normalize
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        max_val = np.abs(audio_data).max()
        if max_val > 0:
            audio_data = audio_data / max_val * 0.95

        # Convert to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(path, sample_rate, audio_int16)

        return path


class SimpleSpeakerTracker:
    """Simple speaker tracking without pyannote (fallback).

    Uses basic voice characteristics for rough speaker separation.
    """

    def __init__(self):
        self._speaker_profiles: Dict[str, dict] = {}
        self._speaker_counter = 0

    def estimate_speaker(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> str:
        """Estimate speaker based on basic audio features.

        Args:
            audio_data: Audio samples
            sample_rate: Sample rate

        Returns:
            Estimated speaker ID
        """
        # Extract simple features
        features = self._extract_simple_features(audio_data, sample_rate)

        # Try to match existing speaker
        best_match = None
        best_score = 0.0

        for speaker_id, profile in self._speaker_profiles.items():
            score = self._compare_features(features, profile)
            if score > 0.7 and score > best_score:
                best_match = speaker_id
                best_score = score

        if best_match:
            # Update profile with new features
            self._update_profile(best_match, features)
            return best_match

        # Create new speaker
        speaker_id = self._generate_id()
        self._speaker_profiles[speaker_id] = features
        return speaker_id

    def _extract_simple_features(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> dict:
        """Extract simple audio features for speaker comparison."""
        # Basic features that might distinguish speakers
        features = {
            'energy_mean': float(np.mean(np.abs(audio_data))),
            'energy_std': float(np.std(np.abs(audio_data))),
            'zero_crossings': int(np.sum(np.diff(np.sign(audio_data)) != 0)),
        }

        # Estimate pitch (very rough)
        try:
            # Simple autocorrelation-based pitch
            corr = np.correlate(audio_data, audio_data, mode='full')
            corr = corr[len(corr)//2:]
            # Find first peak after first valley
            d = np.diff(corr)
            peaks = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
            if len(peaks) > 1:
                period = peaks[1] - peaks[0]
                features['pitch_estimate'] = sample_rate / period if period > 0 else 0
            else:
                features['pitch_estimate'] = 0
        except Exception:
            features['pitch_estimate'] = 0

        return features

    def _compare_features(self, f1: dict, f2: dict) -> float:
        """Compare two feature sets, return similarity 0-1."""
        if not f1 or not f2:
            return 0.0

        # Simple comparison based on relative differences
        score = 0.0
        count = 0

        for key in ['energy_mean', 'energy_std', 'pitch_estimate']:
            v1 = f1.get(key, 0)
            v2 = f2.get(key, 0)
            if v1 > 0 and v2 > 0:
                diff = abs(v1 - v2) / max(v1, v2)
                score += 1 - min(diff, 1)
                count += 1

        return score / count if count > 0 else 0.0

    def _update_profile(self, speaker_id: str, new_features: dict):
        """Update speaker profile with exponential smoothing."""
        profile = self._speaker_profiles.get(speaker_id, {})
        alpha = 0.3  # Smoothing factor

        for key, new_val in new_features.items():
            old_val = profile.get(key, new_val)
            profile[key] = alpha * new_val + (1 - alpha) * old_val

        self._speaker_profiles[speaker_id] = profile

    def _generate_id(self) -> str:
        """Generate new speaker ID."""
        self._speaker_counter += 1
        letter = chr(ord('A') + (self._speaker_counter - 1) % 26)
        return f"Speaker {letter}"
