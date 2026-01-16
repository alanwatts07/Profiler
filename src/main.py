#!/usr/bin/env python3
"""Main entry point for the Audio Profiling System.

This module provides the main application class that orchestrates:
- Audio capture from system output
- Speech-to-text transcription
- Speaker identification
- Linguistic analysis and profiling
- Database storage

Usage:
    python -m src.main          # Run interactive mode
    python -m src.main --help   # Show help
    profiler                    # If installed as CLI tool
"""

import logging
import signal
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import config
from .storage.database import init_db, Database
from .storage.models import Session, Utterance
from .audio.audio_capture import AudioCapture, AudioChunk
from .audio.audio_processor import AudioProcessor, TranscriptionResult
from .audio.speaker_identifier import SpeakerIdentifier, SimpleSpeakerTracker
from .profiling.behavioral_profiler import BehavioralProfiler
from .profiling.linguistic_analyzer import LinguisticAnalyzer

logger = logging.getLogger(__name__)


class AudioProfiler:
    """Main application class for audio profiling."""

    def __init__(
        self,
        database: Optional[Database] = None,
        use_gpu: bool = False
    ):
        """Initialize the audio profiler.

        Args:
            database: Database instance (created if None)
            use_gpu: Whether to use GPU for models
        """
        self.db = database or init_db()
        self.device = "cuda" if use_gpu else "cpu"

        # Components (lazy loaded)
        self._capture: Optional[AudioCapture] = None
        self._processor: Optional[AudioProcessor] = None
        self._speaker_id: Optional[SpeakerIdentifier] = None
        self._profiler: Optional[BehavioralProfiler] = None

        # Session state
        self._current_session: Optional[Session] = None
        self._db_session = None
        self._running = False
        self._lock = threading.Lock()

    @property
    def capture(self) -> AudioCapture:
        """Get audio capture instance."""
        if self._capture is None:
            self._capture = AudioCapture(callback=self._on_audio_chunk)
        return self._capture

    @property
    def processor(self) -> AudioProcessor:
        """Get audio processor instance."""
        if self._processor is None:
            self._processor = AudioProcessor(device=self.device)
        return self._processor

    @property
    def speaker_identifier(self) -> SpeakerIdentifier:
        """Get speaker identifier instance."""
        if self._speaker_id is None:
            try:
                self._speaker_id = SpeakerIdentifier()
            except ImportError:
                logger.warning("pyannote not available, using simple speaker tracking")
                self._speaker_id = SimpleSpeakerTracker()
        return self._speaker_id

    def start_recording(self, session_name: str = None) -> str:
        """Start a new recording session.

        Args:
            session_name: Optional name for the session

        Returns:
            Session ID
        """
        if self._running:
            raise RuntimeError("Already recording")

        # Create database session
        self._db_session = self.db.get_new_session()

        # Create profiler with session
        self._profiler = BehavioralProfiler(self._db_session)

        # Create recording session
        session_id = str(uuid.uuid4())
        self._current_session = Session(
            session_id=session_id,
            name=session_name or f"Session {datetime.now().strftime('%Y%m%d_%H%M')}",
            start_time=datetime.utcnow(),
            status="active"
        )
        self._db_session.add(self._current_session)
        self._db_session.commit()

        # Start capture
        self._running = True
        self.capture.start_recording()

        logger.info(f"Recording started: {session_id}")
        return session_id

    def stop_recording(self) -> Optional[Session]:
        """Stop the current recording session.

        Returns:
            The completed Session object
        """
        if not self._running:
            return None

        self._running = False
        self.capture.stop_recording()

        # Finalize session
        if self._current_session:
            self._current_session.end_time = datetime.utcnow()
            self._current_session.status = "completed"
            if self._current_session.start_time:
                delta = self._current_session.end_time - self._current_session.start_time
                self._current_session.duration_seconds = int(delta.total_seconds())
            self._db_session.commit()

        session = self._current_session
        self._current_session = None

        # Close database session
        if self._db_session:
            self._db_session.close()
            self._db_session = None

        logger.info("Recording stopped")
        return session

    def _on_audio_chunk(self, chunk: AudioChunk):
        """Process incoming audio chunk.

        Called by AudioCapture when a chunk is ready.
        """
        if not self._running:
            return

        try:
            with self._lock:
                self._process_chunk(chunk)
        except Exception as e:
            logger.error(f"Error processing chunk: {e}")

    def _process_chunk(self, chunk: AudioChunk):
        """Process a single audio chunk."""
        # Transcribe
        result = self.processor.transcribe(chunk)

        if not result.has_content:
            logger.debug("No speech in chunk")
            return

        # Identify speaker (simplified - uses whole chunk)
        try:
            if hasattr(self.speaker_identifier, 'identify_or_register'):
                speaker_id = self.speaker_identifier.identify_or_register(chunk)
            else:
                speaker_id = self.speaker_identifier.estimate_speaker(
                    chunk.data, chunk.sample_rate
                )
        except Exception as e:
            logger.warning(f"Speaker identification failed: {e}")
            speaker_id = "Speaker A"

        # Process utterance
        if self._profiler and self._current_session:
            self._profiler.process_utterance(
                speaker_id=speaker_id,
                text=result.text,
                session_id=self._current_session.id,
                start_time=chunk.timestamp,
                end_time=chunk.timestamp + chunk.duration
            )

        logger.info(f"[{speaker_id}] {result.text}")

    def process_file(
        self,
        audio_file: str,
        speaker_id: str = "Speaker A",
        save_to_db: bool = True
    ) -> dict:
        """Process a pre-recorded audio file.

        Args:
            audio_file: Path to audio file
            speaker_id: Speaker identifier
            save_to_db: Whether to save results to database

        Returns:
            Processing results dictionary
        """
        logger.info(f"Processing file: {audio_file}")

        # Transcribe
        result = self.processor.transcribe(audio_file)

        if not result.has_content:
            return {"error": "No speech detected", "file": audio_file}

        # Analyze
        analyzer = LinguisticAnalyzer()
        analysis = analyzer.analyze(result.text)

        # Save to database
        if save_to_db:
            with self.db.get_session() as session:
                profiler = BehavioralProfiler(session)
                profile = profiler.create_profile(speaker_id, result.text)

                return {
                    "file": audio_file,
                    "speaker": speaker_id,
                    "duration": result.duration,
                    "word_count": analysis.word_count,
                    "transcript": result.text,
                    "confidence": profile.confidence_level,
                    "vak": analysis.vak_scores,
                    "social_needs": analysis.social_needs,
                    "profile_id": profile.id
                }

        return {
            "file": audio_file,
            "duration": result.duration,
            "word_count": analysis.word_count,
            "transcript": result.text,
            "vak": analysis.vak_scores,
            "social_needs": analysis.social_needs
        }

    def analyze_text(self, text: str) -> dict:
        """Analyze text without audio processing.

        Args:
            text: Text to analyze

        Returns:
            Analysis results dictionary
        """
        analyzer = LinguisticAnalyzer()
        result = analyzer.analyze(text)

        return {
            "word_count": result.word_count,
            "unique_words": result.unique_words,
            "complexity": result.complexity,
            "sentiment": result.sentiment,
            "vak_scores": result.vak_scores,
            "dominant_vak": result.get_dominant_vak(),
            "social_needs": result.social_needs,
            "top_needs": result.get_top_needs(2),
            "decision_styles": result.decision_styles,
            "communication_patterns": result.communication_patterns
        }

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._running


def setup_signal_handlers(profiler: AudioProfiler):
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info("Signal received, stopping...")
        if profiler.is_recording:
            profiler.stop_recording()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point."""
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run CLI
    from .cli import main as cli_main
    cli_main()


if __name__ == '__main__':
    main()
