"""Behavioral profiler for generating and updating speaker profiles.

Based on Chase Hughes' "6 Minute X-Ray" and "The Ellipsis Manual" methodologies.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy.orm import Session as DBSession

from .linguistic_analyzer import LinguisticAnalyzer, AnalysisResult
from ..storage.models import Speaker, SpeakerProfile, Utterance
from ..config import config


class BehavioralProfiler:
    """Creates and updates behavioral profiles from linguistic analysis."""

    def __init__(self, db_session: DBSession):
        """Initialize the profiler.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session
        self.analyzer = LinguisticAnalyzer()

    def create_or_get_speaker(self, speaker_id: str) -> Speaker:
        """Get existing speaker or create new one.

        Args:
            speaker_id: Unique identifier for the speaker

        Returns:
            Speaker database object
        """
        speaker = self.db.query(Speaker).filter_by(speaker_id=speaker_id).first()
        if not speaker:
            speaker = Speaker(
                speaker_id=speaker_id,
                created_at=datetime.utcnow()
            )
            self.db.add(speaker)
            self.db.flush()
        return speaker

    def create_profile(
        self,
        speaker_id: str,
        transcript_text: str,
        save: bool = True
    ) -> SpeakerProfile:
        """Create initial profile from transcript.

        Args:
            speaker_id: Speaker identifier
            transcript_text: Text to analyze
            save: Whether to save to database

        Returns:
            New SpeakerProfile object
        """
        speaker = self.create_or_get_speaker(speaker_id)
        analysis = self.analyzer.analyze(transcript_text)

        profile = SpeakerProfile(
            speaker_db_id=speaker.id,
            # Social needs
            significance_score=analysis.social_needs.get("significance", 0.0),
            approval_score=analysis.social_needs.get("approval", 0.0),
            acceptance_score=analysis.social_needs.get("acceptance", 0.0),
            intelligence_score=analysis.social_needs.get("intelligence", 0.0),
            pity_score=analysis.social_needs.get("pity", 0.0),
            power_score=analysis.social_needs.get("power", 0.0),
            dominant_needs=analysis.get_top_needs(2),
            # VAK
            visual_score=analysis.vak_scores.get("visual", 0.0),
            auditory_score=analysis.vak_scores.get("auditory", 0.0),
            kinesthetic_score=analysis.vak_scores.get("kinesthetic", 0.0),
            # Decision styles
            decision_styles=analysis.decision_styles,
            # Communication patterns
            certainty_level=analysis.communication_patterns.get("certainty", 0.5),
            question_ratio=analysis.communication_patterns.get("question_ratio", 0.0),
            active_voice_ratio=analysis.communication_patterns.get("active_voice", 0.5),
            time_orientation=analysis.communication_patterns.get("time_orientation", "present"),
            # Metrics
            average_sentiment=analysis.sentiment,
            average_complexity=analysis.complexity,
            confidence_level=config.get_confidence_level(analysis.word_count),
            sample_size=analysis.word_count,
            unique_words_count=analysis.unique_words,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        if save:
            self.db.add(profile)
            self.db.commit()

        return profile

    def update_profile(
        self,
        speaker_id: str,
        new_transcript: str,
        weight_new: float = 0.3
    ) -> Optional[SpeakerProfile]:
        """Incrementally update existing profile with new data.

        Uses weighted averaging to blend new analysis with existing profile.

        Args:
            speaker_id: Speaker identifier
            new_transcript: New text to analyze
            weight_new: Weight for new data (0.0 to 1.0)

        Returns:
            Updated SpeakerProfile or None if speaker not found
        """
        speaker = self.db.query(Speaker).filter_by(speaker_id=speaker_id).first()
        if not speaker:
            return self.create_profile(speaker_id, new_transcript)

        # Get most recent profile
        profile = (
            self.db.query(SpeakerProfile)
            .filter_by(speaker_db_id=speaker.id)
            .order_by(SpeakerProfile.updated_at.desc())
            .first()
        )

        if not profile:
            return self.create_profile(speaker_id, new_transcript)

        # Analyze new text
        analysis = self.analyzer.analyze(new_transcript)
        weight_old = 1.0 - weight_new

        # Update social needs with weighted average
        profile.significance_score = self._weighted_avg(
            profile.significance_score,
            analysis.social_needs.get("significance", 0.0),
            weight_old, weight_new
        )
        profile.approval_score = self._weighted_avg(
            profile.approval_score,
            analysis.social_needs.get("approval", 0.0),
            weight_old, weight_new
        )
        profile.acceptance_score = self._weighted_avg(
            profile.acceptance_score,
            analysis.social_needs.get("acceptance", 0.0),
            weight_old, weight_new
        )
        profile.intelligence_score = self._weighted_avg(
            profile.intelligence_score,
            analysis.social_needs.get("intelligence", 0.0),
            weight_old, weight_new
        )
        profile.pity_score = self._weighted_avg(
            profile.pity_score,
            analysis.social_needs.get("pity", 0.0),
            weight_old, weight_new
        )
        profile.power_score = self._weighted_avg(
            profile.power_score,
            analysis.social_needs.get("power", 0.0),
            weight_old, weight_new
        )

        # Update VAK scores
        profile.visual_score = self._weighted_avg(
            profile.visual_score,
            analysis.vak_scores.get("visual", 0.0),
            weight_old, weight_new
        )
        profile.auditory_score = self._weighted_avg(
            profile.auditory_score,
            analysis.vak_scores.get("auditory", 0.0),
            weight_old, weight_new
        )
        profile.kinesthetic_score = self._weighted_avg(
            profile.kinesthetic_score,
            analysis.vak_scores.get("kinesthetic", 0.0),
            weight_old, weight_new
        )

        # Normalize VAK to sum to 1.0
        vak_total = profile.visual_score + profile.auditory_score + profile.kinesthetic_score
        if vak_total > 0:
            profile.visual_score = round(profile.visual_score / vak_total, 3)
            profile.auditory_score = round(profile.auditory_score / vak_total, 3)
            profile.kinesthetic_score = round(profile.kinesthetic_score / vak_total, 3)

        # Update communication patterns
        profile.certainty_level = self._weighted_avg(
            profile.certainty_level,
            analysis.communication_patterns.get("certainty", 0.5),
            weight_old, weight_new
        )
        profile.question_ratio = self._weighted_avg(
            profile.question_ratio,
            analysis.communication_patterns.get("question_ratio", 0.0),
            weight_old, weight_new
        )
        profile.active_voice_ratio = self._weighted_avg(
            profile.active_voice_ratio,
            analysis.communication_patterns.get("active_voice", 0.5),
            weight_old, weight_new
        )

        # Time orientation - use most recent if different
        new_time = analysis.communication_patterns.get("time_orientation", "present")
        if new_time != profile.time_orientation:
            profile.time_orientation = new_time

        # Update metrics
        profile.average_sentiment = self._weighted_avg(
            profile.average_sentiment,
            analysis.sentiment,
            weight_old, weight_new
        )
        profile.average_complexity = self._weighted_avg(
            profile.average_complexity,
            analysis.complexity,
            weight_old, weight_new
        )

        # Update sample size
        profile.sample_size += analysis.word_count
        profile.unique_words_count += analysis.unique_words

        # Update decision styles (merge lists)
        existing_styles = set(profile.decision_styles or [])
        new_styles = set(analysis.decision_styles or [])
        profile.decision_styles = list(existing_styles | new_styles)

        # Recalculate dominant needs
        profile.dominant_needs = profile.get_top_needs(2)

        # Update confidence
        profile.confidence_level = config.get_confidence_level(profile.sample_size)

        # Update timestamp
        profile.updated_at = datetime.utcnow()
        speaker.last_seen = datetime.utcnow()

        self.db.commit()
        return profile

    def get_profile(self, speaker_id: str) -> Optional[SpeakerProfile]:
        """Get the current profile for a speaker.

        Args:
            speaker_id: Speaker identifier

        Returns:
            Most recent SpeakerProfile or None
        """
        speaker = self.db.query(Speaker).filter_by(speaker_id=speaker_id).first()
        if not speaker:
            return None

        return (
            self.db.query(SpeakerProfile)
            .filter_by(speaker_db_id=speaker.id)
            .order_by(SpeakerProfile.updated_at.desc())
            .first()
        )

    def get_all_speakers(self) -> List[Speaker]:
        """Get all speakers in the database."""
        return self.db.query(Speaker).order_by(Speaker.last_seen.desc()).all()

    def rename_speaker(self, old_id: str, new_name: str) -> bool:
        """Rename a speaker.

        Args:
            old_id: Current speaker_id
            new_name: New display name

        Returns:
            True if successful, False if speaker not found
        """
        speaker = self.db.query(Speaker).filter_by(speaker_id=old_id).first()
        if not speaker:
            return False

        speaker.display_name = new_name
        self.db.commit()
        return True

    def process_utterance(
        self,
        speaker_id: str,
        text: str,
        session_id: int,
        start_time: float = 0.0,
        end_time: float = 0.0
    ) -> Utterance:
        """Process a single utterance and update profile.

        Args:
            speaker_id: Speaker identifier
            text: Utterance text
            session_id: Session database ID
            start_time: Start time in seconds
            end_time: End time in seconds

        Returns:
            Created Utterance object
        """
        speaker = self.create_or_get_speaker(speaker_id)
        analysis = self.analyzer.analyze(text)

        # Create utterance
        utterance = Utterance(
            session_id=session_id,
            speaker_db_id=speaker.id,
            text=text,
            word_count=analysis.word_count,
            start_time=start_time,
            end_time=end_time,
            vak_detected=analysis.get_dominant_vak(),
            dominant_need=analysis.get_top_needs(1)[0] if analysis.get_top_needs(1) else None,
            sentiment=analysis.sentiment,
            analysis_data={
                "vak_scores": analysis.vak_scores,
                "social_needs": analysis.social_needs,
                "decision_styles": analysis.decision_styles,
            },
            timestamp=datetime.utcnow()
        )

        self.db.add(utterance)

        # Update speaker's profile if we have enough words
        if analysis.word_count >= 10:
            self.update_profile(speaker_id, text, weight_new=0.2)

        self.db.commit()
        return utterance

    def generate_profile_summary(self, speaker_id: str) -> dict:
        """Generate a human-readable profile summary.

        Args:
            speaker_id: Speaker identifier

        Returns:
            Dictionary with formatted profile information
        """
        profile = self.get_profile(speaker_id)
        if not profile:
            return {"error": f"No profile found for {speaker_id}"}

        speaker = self.db.query(Speaker).filter_by(speaker_id=speaker_id).first()

        # Determine dominant VAK
        vak_scores = profile.get_vak_dict()
        dominant_vak = max(vak_scores, key=vak_scores.get)

        # Format social needs
        needs = profile.get_social_needs_dict()
        sorted_needs = sorted(needs.items(), key=lambda x: x[1], reverse=True)

        return {
            "speaker": {
                "id": speaker_id,
                "name": speaker.name if speaker else speaker_id,
                "last_seen": speaker.last_seen.isoformat() if speaker and speaker.last_seen else None,
            },
            "confidence": profile.confidence_level,
            "sample_size": profile.sample_size,
            "dominant_needs": {
                "primary": sorted_needs[0][0] if sorted_needs else None,
                "secondary": sorted_needs[1][0] if len(sorted_needs) > 1 else None,
                "scores": {k: round(v, 3) for k, v in sorted_needs[:3]},
            },
            "vak": {
                "dominant": dominant_vak,
                "distribution": {k: round(v, 3) for k, v in vak_scores.items()},
            },
            "decision_styles": profile.decision_styles,
            "communication": {
                "certainty": round(profile.certainty_level, 3),
                "question_ratio": round(profile.question_ratio, 3),
                "active_voice": round(profile.active_voice_ratio, 3),
                "time_orientation": profile.time_orientation,
            },
            "sentiment": {
                "average": round(profile.average_sentiment, 3),
                "label": self._sentiment_label(profile.average_sentiment),
            },
            "complexity": {
                "score": round(profile.average_complexity, 2),
                "label": self._complexity_label(profile.average_complexity),
            },
        }

    def _weighted_avg(
        self,
        old_val: float,
        new_val: float,
        weight_old: float,
        weight_new: float
    ) -> float:
        """Calculate weighted average of two values."""
        return round((old_val * weight_old) + (new_val * weight_new), 4)

    def _sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score >= 0.5:
            return "very positive"
        elif score >= 0.2:
            return "positive"
        elif score > -0.2:
            return "neutral"
        elif score > -0.5:
            return "negative"
        else:
            return "very negative"

    def _complexity_label(self, grade: float) -> str:
        """Convert Flesch-Kincaid grade to label."""
        if grade <= 6:
            return "simple"
        elif grade <= 10:
            return "moderate"
        elif grade <= 14:
            return "complex"
        else:
            return "very complex"
