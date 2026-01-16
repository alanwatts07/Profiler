"""SQLAlchemy ORM models for the Audio Profiling System."""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import JSON

Base = declarative_base()


class Speaker(Base):
    """Represents a unique speaker identified by voice."""

    __tablename__ = "speakers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    speaker_id = Column(String(255), unique=True, index=True, nullable=False)
    display_name = Column(String(255), nullable=True)
    voice_embedding = Column(JSON, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    profiles = relationship("SpeakerProfile", back_populates="speaker", cascade="all, delete-orphan")
    utterances = relationship("Utterance", back_populates="speaker", cascade="all, delete-orphan")

    def __repr__(self):
        name = self.display_name or self.speaker_id
        return f"<Speaker(id={self.id}, name='{name}')>"

    @property
    def name(self) -> str:
        """Get display name or fall back to speaker_id."""
        return self.display_name or self.speaker_id


class SpeakerProfile(Base):
    """Behavioral profile for a speaker based on linguistic analysis."""

    __tablename__ = "speaker_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    speaker_db_id = Column(Integer, ForeignKey("speakers.id", ondelete="CASCADE"), nullable=False)

    # Social Needs Scores (0.0 - 1.0)
    significance_score = Column(Float, default=0.0)
    approval_score = Column(Float, default=0.0)
    acceptance_score = Column(Float, default=0.0)
    intelligence_score = Column(Float, default=0.0)
    pity_score = Column(Float, default=0.0)
    power_score = Column(Float, default=0.0)

    # Top needs (derived from scores)
    dominant_needs = Column(JSON, default=list)

    # VAK Distribution (should sum to 1.0)
    visual_score = Column(Float, default=0.0)
    auditory_score = Column(Float, default=0.0)
    kinesthetic_score = Column(Float, default=0.0)

    # Decision Styles (list of detected styles)
    decision_styles = Column(JSON, default=list)

    # Communication Patterns
    certainty_level = Column(Float, default=0.5)
    question_ratio = Column(Float, default=0.0)
    active_voice_ratio = Column(Float, default=0.5)
    time_orientation = Column(String(50), default="present")

    # Underlying Values (dictionary of value dimensions)
    values = Column(JSON, default=dict)

    # Average sentiment (-1.0 to 1.0)
    average_sentiment = Column(Float, default=0.0)

    # Text complexity (Flesch-Kincaid grade level)
    average_complexity = Column(Float, default=0.0)

    # Metadata
    confidence_level = Column(String(50), default="insufficient")
    sample_size = Column(Integer, default=0)
    unique_words_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    speaker = relationship("Speaker", back_populates="profiles")

    def __repr__(self):
        return f"<SpeakerProfile(speaker_id={self.speaker_db_id}, confidence='{self.confidence_level}')>"

    def get_social_needs_dict(self) -> dict:
        """Return social needs as a dictionary."""
        return {
            "significance": self.significance_score,
            "approval": self.approval_score,
            "acceptance": self.acceptance_score,
            "intelligence": self.intelligence_score,
            "pity": self.pity_score,
            "power": self.power_score,
        }

    def get_vak_dict(self) -> dict:
        """Return VAK scores as a dictionary."""
        return {
            "visual": self.visual_score,
            "auditory": self.auditory_score,
            "kinesthetic": self.kinesthetic_score,
        }

    def get_top_needs(self, n: int = 2) -> list:
        """Get the top N social needs."""
        needs = self.get_social_needs_dict()
        sorted_needs = sorted(needs.items(), key=lambda x: x[1], reverse=True)
        return [need for need, score in sorted_needs[:n] if score > 0.1]


class Session(Base):
    """Represents a recording session."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    start_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, default=0)
    audio_file_path = Column(String(512), nullable=True)
    status = Column(String(50), default="active")
    notes = Column(Text, nullable=True)

    # Relationships
    utterances = relationship("Utterance", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Session(id='{self.session_id}', status='{self.status}')>"

    @property
    def speaker_count(self) -> int:
        """Get unique speaker count in this session."""
        return len(set(u.speaker_db_id for u in self.utterances))


class Utterance(Base):
    """Individual speech segment from a speaker."""

    __tablename__ = "utterances"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    speaker_db_id = Column(Integer, ForeignKey("speakers.id", ondelete="CASCADE"), nullable=False)

    # Content
    text = Column(Text, nullable=False)
    word_count = Column(Integer, default=0)

    # Timing (seconds from session start)
    start_time = Column(Float, default=0.0)
    end_time = Column(Float, default=0.0)

    # Quick analysis results for this utterance
    vak_detected = Column(String(20), nullable=True)
    dominant_need = Column(String(50), nullable=True)
    sentiment = Column(Float, default=0.0)

    # Raw analysis data
    analysis_data = Column(JSON, nullable=True)

    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    session = relationship("Session", back_populates="utterances")
    speaker = relationship("Speaker", back_populates="utterances")

    # Indexes for common queries
    __table_args__ = (
        Index("ix_utterances_session_time", "session_id", "start_time"),
        Index("ix_utterances_speaker_time", "speaker_db_id", "timestamp"),
    )

    def __repr__(self):
        preview = self.text[:30] + "..." if len(self.text) > 30 else self.text
        return f"<Utterance(id={self.id}, text='{preview}')>"
