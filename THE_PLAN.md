# THE PLAN: Audio Profiling System
## Complete Implementation Blueprint

**Version**: 1.0  
**Date**: 2026-01-15  
**Purpose**: Comprehensive guide for building an audio-based behavioral profiling system using Chase Hughes' methodologies

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Requirements](#system-requirements)
3. [Architecture](#architecture)
4. [Profiling Framework](#profiling-framework)
5. [Component Specifications](#component-specifications)
6. [Database Schema](#database-schema)
7. [Implementation Sequence](#implementation-sequence)
8. [CLI Commands](#cli-commands)
9. [Web UI Design](#web-ui-design)
10. [Testing Strategy](#testing-strategy)
11. [Future Enhancements](#future-enhancements)

---

## 1. Project Overview

### 1.1 Goal
Build a system that:
- Captures system audio output (Discord, etc.) via PulseAudio/PipeWire loopback
- Converts speech to text using OpenAI Whisper
- Identifies different speakers using voice diarization (pyannote.audio)
- Profiles speakers based on linguistic patterns from Chase Hughes' "6 Minute X-Ray" and "The Ellipsis Manual"
- Stores profiles in a database for tracking over time
- Provides CLI interface with expansion to web UI

### 1.2 What It Does NOT Do
- Body language analysis (audio-only)
- Real-time video processing
- Facial expression analysis
- Deception detection (requires visual cues)

### 1.3 Key Features
- ✅ Real-time audio capture from system output
- ✅ Speaker identification (Speaker A, Speaker B, etc.)
- ✅ Speaker renaming capability (user can assign names later)
- ✅ Linguistic profiling across 6 social needs
- ✅ Decision pattern detection
- ✅ VAK (Visual/Auditory/Kinesthetic) modality detection
- ✅ Profile history tracking
- ✅ CLI for control and viewing
- ✅ Web UI for visualization and tracking (future)

---

## 2. System Requirements

### 2.1 Operating System
- **Primary**: Linux (Ubuntu/Debian/Fedora)
- **Audio System**: PulseAudio or PipeWire

### 2.2 Python Version
- Python 3.9 or higher (for compatibility with PyTorch and modern libraries)

### 2.3 Hardware Requirements
- **RAM**: Minimum 8GB (16GB recommended for larger Whisper models)
- **CPU**: Multi-core processor (for real-time processing)
- **GPU**: Optional but recommended for faster Whisper transcription
- **Disk**: 5GB+ for models and database storage

### 2.4 Key Dependencies
```
# Audio Processing
sounddevice>=0.4.6
pyaudio>=0.2.13
pulsectl>=23.5.2
librosa>=0.10.1
pydub>=0.25.1
webrtcvad>=2.0.10

# Speech-to-Text
openai-whisper>=20231117
torch>=2.0.0
torchaudio>=2.0.0

# Speaker Diarization
pyannote.audio>=3.1.0
speechbrain>=0.5.16

# NLP and Analysis
spacy>=3.7.0
nltk>=3.8.1
transformers>=4.35.0
scikit-learn>=1.3.0

# Database
sqlalchemy>=2.0.0
alembic>=1.12.0

# CLI Interface
click>=8.1.7
rich>=13.7.0
tabulate>=0.9.0

# Web UI (Future)
flask>=3.0.0
flask-sqlalchemy>=3.1.0
flask-cors>=4.0.0

# Utilities
python-dotenv>=1.0.0
pyyaml>=6.0.1
python-dateutil>=2.8.2
```

---

## 3. Architecture

### 3.1 System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        AUDIO INPUT LAYER                         │
│  System Audio → PulseAudio Monitor → Audio Capture Module       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    AUDIO PROCESSING LAYER                        │
│  • Noise Reduction                                               │
│  • Voice Activity Detection (VAD)                                │
│  • Audio Buffering & Segmentation                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  SPEECH-TO-TEXT LAYER (Whisper)                  │
│  Audio Chunks → Transcription with Timestamps                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              SPEAKER DIARIZATION LAYER (pyannote)                │
│  • Extract voice embeddings                                      │
│  • Cluster/identify speakers                                     │
│  • Label as Speaker A, B, C, etc.                                │
│  • Match against known voice profiles                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LINGUISTIC ANALYSIS LAYER                     │
│  • Tokenization & POS Tagging                                    │
│  • Keyword/Phrase Extraction                                     │
│  • Pattern Matching (VAK, Social Needs, Decision Styles)         │
│  • Sentiment & Emotional Tone Analysis                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      PROFILING ENGINE                            │
│  • Social Needs Scoring (6 categories)                           │
│  • Decision Pattern Detection (6 styles)                         │
│  • VAK Modality Distribution                                     │
│  • Underlying Values Extraction                                  │
│  • Confidence Scoring                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        STORAGE LAYER                             │
│  • SQLite Database                                               │
│  • Speaker Profiles                                              │
│  • Transcripts & Sessions                                        │
│  • Voice Embeddings                                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      INTERFACE LAYER                             │
│  • CLI (Primary): Start/Stop, View Profiles, Rename Speakers    │
│  • Web UI (Future): Dashboard, Visualizations, History          │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Directory Structure

```
Profiler/
├── src/
│   ├── __init__.py
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── audio_capture.py      # System audio monitoring
│   │   ├── audio_processor.py    # Preprocessing, VAD
│   │   └── speaker_identifier.py # Speaker diarization
│   ├── profiling/
│   │   ├── __init__.py
│   │   ├── linguistic_analyzer.py  # NLP & pattern matching
│   │   ├── behavioral_profiler.py  # Profile generation
│   │   ├── profile_models.py       # Data models
│   │   └── scoring_engine.py       # Confidence scoring
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── database.py           # SQLAlchemy ORM
│   │   ├── models.py             # DB models
│   │   └── migrations/           # Alembic migrations
│   ├── web/
│   │   ├── __init__.py
│   │   ├── app.py                # Flask application
│   │   ├── routes.py             # API endpoints
│   │   └── templates/            # HTML templates
│   ├── cli.py                    # CLI interface
│   ├── config.py                 # Configuration management
│   └── main.py                   # Main entry point
├── docs/
│   ├── profiling_framework.md    # Chase Hughes reference
│   └── API.md                    # API documentation
├── data/
│   ├── models/                   # Downloaded AI models
│   └── profiles.db               # SQLite database
├── logs/                         # Application logs
├── tests/                        # Unit & integration tests
├── requirements.txt
├── setup.py
├── .env.example
├── THE_PLAN.md                   # This document
└── README.md
```

---

## 4. Profiling Framework

### 4.1 Social Needs (6 Categories)

Each person is driven by 1-2 dominant needs. The system scores each category based on linguistic markers.

#### **Significance**
- **Drive**: Feel important, make a difference, stand out
- **Fear**: Being dismissed or mocked
- **Keywords/Patterns**:
  - High "I/me/my" usage (>8% of total words)
  - Achievement words: "best", "first", "won", "achieved", "accomplished"
  - Competitive language: "better than", "nobody else could"
  - Status references: "my title", "my position", name-dropping
  - Quantifying accomplishments: numbers, metrics, rankings

#### **Approval**
- **Drive**: Validation, praise, being liked
- **Fear**: Rejection or disdain
- **Keywords/Patterns**:
  - Hedging: "maybe", "I think", "kind of", "sort of", "if that's okay"
  - Apologetic: "sorry", "I hope you don't mind", "excuse me"
  - Validation-seeking: "is that good?", "do you like it?", "right?"
  - Agreement-seeking tags: "don't you think?", "wouldn't you say?"
  - Minimizing self: "I'm not sure but", "I could be wrong"

#### **Acceptance**
- **Drive**: Belong, be included, be wanted
- **Fear**: Criticism or alienation
- **Keywords/Patterns**:
  - High "we/us/our" usage (>6% of total words)
  - Group references: "everyone", "they all", "the team"
  - Conforming: "everyone's doing it", "it's normal", "that's how we do it"
  - Belongingness: "part of", "member of", "one of us"
  - Fear of exclusion: "left out", "not invited", "alone"

#### **Intelligence**
- **Drive**: Be perceived as smart, competent
- **Fear**: Being seen as unintelligent
- **Keywords/Patterns**:
  - Correcting: "actually", "technically", "well, specifically"
  - Complex vocabulary (Flesch-Kincaid grade level >12)
  - Over-explaining simple concepts (word count per explanation)
  - Education references: "degree", "studied", "research shows"
  - Citing sources: "according to", "studies show", "data indicates"

#### **Pity**
- **Drive**: Be rescued, consoled, gain sympathy
- **Fear**: Being ignored or disbelieved
- **Keywords/Patterns**:
  - Victim language: "always happens to me", "nobody understands"
  - Struggle highlighting: "so hard", "difficult for me", "can't catch a break"
  - Self-deprecation for sympathy: "I'm terrible at", "I never succeed"
  - Sympathy-seeking: "you don't know how hard", "nobody cares"
  - Minimizing success: "just lucky", "not that good"

#### **Strength/Power**
- **Drive**: Feel in control, superior, influential
- **Fear**: Being disrespected or challenged
- **Keywords/Patterns**:
  - Commands: "you need to", "do this", "listen to me"
  - Absolute certainty: "definitely", "absolutely", "I'm telling you"
  - Dominance: "I don't need", "I can handle", "I'm in charge"
  - Toughness: "weak", "strong", "tough it out", "man up"
  - Dismissive: "that's nothing", "not a big deal", "get over it"

### 4.2 Decision Map (6 Styles)

Identified by analyzing how people discuss choices and decisions.

#### **Deviance**
- Goes against norms
- **Language**: "I don't care what they think", "I do my own thing", "break the rules"

#### **Novelty**
- Seeks new, exciting, different
- **Language**: "new", "never tried", "exciting", "different approach", "innovative"

#### **Social**
- Follows what others do
- **Language**: "everyone's doing it", "popular", "trending", "they said", "recommended"

#### **Conformity**
- Follows tradition, rules, expectations
- **Language**: "traditional", "proven", "right way", "by the book", "standard"

#### **Investment**
- Based on sunk costs
- **Language**: "already invested", "put so much into", "can't quit now", "too far in"

#### **Necessity**
- Based on survival/practical needs
- **Language**: "have to", "must", "need to", "no choice", "survival"

### 4.3 Sensory Modality (VAK)

Track frequency of sensory words to determine dominant modality.

#### **Visual (V)**
```
Keywords: see, look, view, show, picture, imagine, clear, bright, 
          colorful, focus, perspective, appears, visible, vision,
          envision, illustrate, reveal

Phrases: "see what I mean", "look at this", "picture this", 
         "appears that", "show me", "I see your point", "looks like",
         "clear as day", "in my view"
```

#### **Auditory (A)**
```
Keywords: hear, listen, sound, tell, say, ask, talk, speak, voice,
          tone, loud, quiet, resonate, harmony, click, ring, echo

Phrases: "hear me out", "listen to this", "sounds like", "rings a bell",
         "tell me", "loud and clear", "doesn't sound right", "tune in",
         "word for word"
```

#### **Kinesthetic (K)**
```
Keywords: feel, touch, grasp, hold, handle, solid, heavy, rough,
          smooth, warm, cold, move, push, pull, grab, carry

Phrases: "I feel you", "get a grip", "hold on", "touch base",
         "grasp the concept", "get in touch", "feels right",
         "hands-on", "concrete", "firm grasp"
```

### 4.4 Underlying Values

Track themes and priorities that emerge over time:
- **Security vs Risk**
- **Tradition vs Innovation**
- **Independence vs Community**
- **Achievement vs Relationships**
- **Logic vs Emotion**
- **Control vs Freedom**
- **Status vs Humility**

### 4.5 Communication Patterns

#### **Certainty vs Uncertainty**
- Certainty: "always", "never", "definitely", "absolutely", "certainly"
- Uncertainty: "maybe", "might", "possibly", "perhaps", "I think"

#### **Question vs Statement Ratio**
- High questions = seeking information/validation
- High statements = assertive/directive

#### **Active vs Passive Voice**
- Active: Taking action, agency
- Passive: Being acted upon, victim stance

#### **Time Orientation**
- Past-focused: "was", "had", "did", "used to", "back then"
- Present-focused: "am", "is", "are", "now", "currently"
- Future-focused: "will", "going to", "plan to", "soon", "next"

---

## 5. Component Specifications

### 5.1 Audio Capture Module (`src/audio/audio_capture.py`)

**Purpose**: Capture system audio output using PulseAudio/PipeWire loopback

**Key Functions**:

```python
class AudioCapture:
    def __init__(self, device_name=None, sample_rate=16000, chunk_duration=5.0):
        """
        Args:
            device_name: PulseAudio monitor device (auto-detect if None)
            sample_rate: 16kHz (optimal for Whisper)
            chunk_duration: Seconds per audio chunk (5s default)
        """
        
    def detect_monitor_device(self) -> str:
        """Find PulseAudio/PipeWire monitor source"""
        # Use pulsectl to list sources
        # Return .monitor device name
        
    def start_recording(self, callback=None):
        """Start continuous audio capture"""
        # Open audio stream
        # Buffer audio chunks
        # Call callback when chunk is ready
        
    def stop_recording(self):
        """Stop audio capture gracefully"""
        
    def save_chunk(self, audio_data, filename):
        """Save audio chunk as WAV file"""
```

**Implementation Notes**:
- Use `sounddevice` library for cross-platform audio
- Use `pulsectl` for PulseAudio device discovery
- Implement circular buffer for continuous recording
- Add Voice Activity Detection (VAD) to skip silence
- Handle audio device disconnections gracefully

---

### 5.2 Audio Processor (`src/audio/audio_processor.py`)

**Purpose**: Preprocess audio and perform speech-to-text

**Key Functions**:

```python
class AudioProcessor:
    def __init__(self, model_size="base", device="cpu"):
        """
        Args:
            model_size: Whisper model (tiny/base/small/medium/large)
            device: "cpu" or "cuda"
        """
        self.whisper_model = whisper.load_model(model_size, device=device)
        
    def preprocess_audio(self, audio_data):
        """
        Noise reduction, normalization, VAD
        Returns: Cleaned audio array
        """
        # Apply noise reduction (librosa)
        # Normalize volume
        # Apply VAD to detect speech segments
        
    def transcribe(self, audio_data, language="en"):
        """
        Transcribe audio to text with word-level timestamps
        Returns: {
            "text": "full transcription",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "Hello there"},
                ...
            ],
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5},
                ...
            ]
        }
        """
        cleaned_audio = self.preprocess_audio(audio_data)
        result = self.whisper_model.transcribe(
            cleaned_audio,
            language=language,
            word_timestamps=True
        )
        return result
```

**Implementation Notes**:
- Use Whisper "base" model as default (good accuracy/speed balance)
- Enable word-level timestamps for precise speaker attribution
- Implement audio quality checks
- Handle multiple languages (auto-detect or specify)

---

### 5.3 Speaker Identifier (`src/audio/speaker_identifier.py`)

**Purpose**: Identify and track different speakers

**Key Functions**:

```python
class SpeakerIdentifier:
    def __init__(self):
        """Initialize pyannote.audio pipeline"""
        from pyannote.audio import Pipeline
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1"
        )
        self.speaker_embeddings = {}  # speaker_id -> embedding vector
        
    def diarize(self, audio_file):
        """
        Perform speaker diarization
        Returns: [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5},
            {"speaker": "SPEAKER_01", "start": 2.5, "end": 5.0},
            ...
        ]
        """
        diarization = self.pipeline(audio_file)
        return self._format_diarization(diarization)
        
    def extract_embedding(self, audio_segment):
        """Extract voice embedding from audio segment"""
        # Use pyannote's embedding model
        # Return 512-dim vector
        
    def match_speaker(self, embedding, threshold=0.75):
        """
        Match embedding against known speakers
        Returns: speaker_id or None
        """
        # Compare cosine similarity with stored embeddings
        # Return match if similarity > threshold
        
    def register_speaker(self, embedding, speaker_id=None):
        """
        Register new speaker with optional ID
        If speaker_id is None, generate "Speaker A", "Speaker B", etc.
        """
        if speaker_id is None:
            speaker_id = self._generate_speaker_id()
        self.speaker_embeddings[speaker_id] = embedding
        return speaker_id
        
    def rename_speaker(self, old_id, new_id):
        """Allow user to rename a speaker"""
        if old_id in self.speaker_embeddings:
            self.speaker_embeddings[new_id] = self.speaker_embeddings.pop(old_id)
```

**Implementation Notes**:
- Use pyannote.audio 3.1+ for state-of-the-art diarization
- Store voice embeddings in database for persistence
- Implement similarity threshold tuning
- Handle embedding drift over time

---

### 5.4 Linguistic Analyzer (`src/profiling/linguistic_analyzer.py`)

**Purpose**: Analyze text for profiling markers

**Key Functions**:

```python
class LinguisticAnalyzer:
    def __init__(self):
        """Load NLP models and keyword dictionaries"""
        import spacy
        self.nlp = spacy.load("en_core_web_md")
        self.vak_keywords = self._load_vak_keywords()
        self.social_need_patterns = self._load_social_patterns()
        
    def analyze_text(self, text):
        """
        Comprehensive linguistic analysis
        Returns: {
            "vak_scores": {"visual": 0.3, "auditory": 0.2, "kinesthetic": 0.5},
            "social_needs": {
                "significance": 0.7,
                "approval": 0.3,
                ...
            },
            "decision_style": ["novelty", "social"],
            "communication_patterns": {
                "certainty": 0.8,
                "question_ratio": 0.15,
                "active_voice": 0.75,
                "time_orientation": "future"
            },
            "sentiment": 0.6,  # -1 to 1
            "word_count": 150,
            "unique_words": 87,
            "complexity": 9.2  # Flesch-Kincaid grade level
        }
        """
        doc = self.nlp(text)
        
        return {
            "vak_scores": self._analyze_vak(doc),
            "social_needs": self._analyze_social_needs(doc),
            "decision_style": self._detect_decision_style(doc),
            "communication_patterns": self._analyze_patterns(doc),
            "sentiment": self._calculate_sentiment(doc),
            "word_count": len([t for t in doc if not t.is_punct]),
            "unique_words": len(set([t.lemma_ for t in doc if not t.is_stop])),
            "complexity": self._calculate_complexity(text)
        }
        
    def _analyze_vak(self, doc):
        """Count VAK keywords and return normalized scores"""
        # Count visual/auditory/kinesthetic words
        # Normalize to percentages
        
    def _analyze_social_needs(self, doc):
        """Score each of 6 social needs based on markers"""
        # Pattern matching for each need
        # Return scores 0-1
        
    def _detect_decision_style(self, doc):
        """Identify decision-making style indicators"""
        # Look for decision-context keywords
        # Return list of detected styles
        
    def _analyze_patterns(self, doc):
        """Analyze communication patterns"""
        # Certainty: count absolute vs hedge words
        # Questions: count question marks and question words
        # Voice: detect active vs passive constructions
        # Time: count past/present/future verb tenses
```

**Implementation Notes**:
- Use spaCy for POS tagging and dependency parsing
- Maintain comprehensive keyword dictionaries (load from JSON/YAML)
- Implement weighted scoring (exact phrase matches > keyword matches)
- Handle negations properly ("I don't see" shouldn't count as visual)

---

### 5.5 Behavioral Profiler (`src/profiling/behavioral_profiler.py`)

**Purpose**: Generate and update speaker profiles

**Key Functions**:

```python
class BehavioralProfiler:
    def __init__(self, db_session):
        self.db = db_session
        self.analyzer = LinguisticAnalyzer()
        
    def create_profile(self, speaker_id, transcript_text):
        """Create initial profile from transcript"""
        analysis = self.analyzer.analyze_text(transcript_text)
        
        profile = SpeakerProfile(
            speaker_id=speaker_id,
            dominant_needs=self._get_top_needs(analysis["social_needs"], n=2),
            vak_distribution=analysis["vak_scores"],
            decision_styles=analysis["decision_style"],
            communication_style=analysis["communication_patterns"],
            confidence_level=self._calculate_confidence(analysis["word_count"]),
            sample_size=analysis["word_count"],
            created_at=datetime.utcnow()
        )
        
        self.db.add(profile)
        self.db.commit()
        return profile
        
    def update_profile(self, speaker_id, new_transcript):
        """Incrementally update existing profile with new data"""
        profile = self.db.query(SpeakerProfile).filter_by(speaker_id=speaker_id).first()
        analysis = self.analyzer.analyze_text(new_transcript)
        
        # Weighted average: 70% existing, 30% new
        profile.vak_distribution = self._weighted_merge(
            profile.vak_distribution,
            analysis["vak_scores"],
            weight_new=0.3
        )
        
        profile.sample_size += analysis["word_count"]
        profile.confidence_level = self._calculate_confidence(profile.sample_size)
        profile.updated_at = datetime.utcnow()
        
        self.db.commit()
        return profile
        
    def _calculate_confidence(self, word_count):
        """
        Confidence levels based on sample size:
        - <100 words: "insufficient"
        - 100-500: "low"
        - 500-1500: "medium"
        - 1500+: "high"
        """
        if word_count < 100:
            return "insufficient"
        elif word_count < 500:
            return "low"
        elif word_count < 1500:
            return "medium"
        else:
            return "high"
```

---

### 5.6 Database Models (`src/storage/models.py`)

**SQLAlchemy ORM Models**:

```python
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Speaker(Base):
    __tablename__ = 'speakers'
    
    id = Column(Integer, primary_key=True)
    speaker_id = Column(String, unique=True, index=True)  # "Speaker A" or user-defined
    voice_embedding = Column(JSON)  # 512-dim vector as JSON
    created_at = Column(DateTime)
    last_seen = Column(DateTime)
    
    profiles = relationship("SpeakerProfile", back_populates="speaker")
    utterances = relationship("Utterance", back_populates="speaker")

class SpeakerProfile(Base):
    __tablename__ = 'speaker_profiles'
    
    id = Column(Integer, primary_key=True)
    speaker_db_id = Column(Integer, ForeignKey('speakers.id'))
    
    # Social Needs (0-1 scores)
    significance_score = Column(Float, default=0.0)
    approval_score = Column(Float, default=0.0)
    acceptance_score = Column(Float, default=0.0)
    intelligence_score = Column(Float, default=0.0)
    pity_score = Column(Float, default=0.0)
    power_score = Column(Float, default=0.0)
    
    dominant_needs = Column(JSON)  # List of top 2-3 needs
    
    # VAK Distribution
    visual_score = Column(Float, default=0.0)
    auditory_score = Column(Float, default=0.0)
    kinesthetic_score = Column(Float, default=0.0)
    
    # Decision Styles
    decision_styles = Column(JSON)  # List of detected styles
    
    # Communication Patterns
    certainty_level = Column(Float)
    question_ratio = Column(Float)
    active_voice_ratio = Column(Float)
    time_orientation = Column(String)  # "past", "present", "future"
    
    # Metadata
    confidence_level = Column(String)  # "insufficient", "low", "medium", "high"
    sample_size = Column(Integer)  # Total words analyzed
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
    speaker = relationship("Speaker", back_populates="profiles")

class Session(Base):
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, index=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_seconds = Column(Integer)
    audio_file_path = Column(String)
    
    utterances = relationship("Utterance", back_populates="session")

class Utterance(Base):
    __tablename__ = 'utterances'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('sessions.id'))
    speaker_db_id = Column(Integer, ForeignKey('speakers.id'))
    
    text = Column(String)
    start_time = Column(Float)  # seconds from session start
    end_time = Column(Float)
    word_count = Column(Integer)
    
    # Quick analysis results
    vak_detected = Column(String)  # "visual", "auditory", or "kinesthetic"
    dominant_need = Column(String)
    
    timestamp = Column(DateTime)
    
    session = relationship("Session", back_populates="utterances")
    speaker = relationship("Speaker", back_populates="utterances")
```

---

## 6. Database Schema

### 6.1 Tables

**speakers**
- Primary key: id
- Unique: speaker_id (e.g., "Speaker A", "John Doe")
- Stores: voice_embedding (JSON), timestamps

**speaker_profiles**
- Primary key: id
- Foreign key: speaker_db_id → speakers.id
- Stores: All profiling scores, metadata

**sessions**
- Primary key: id
- Unique: session_id (UUID)
- Stores: Session metadata, timestamps, audio file reference

**utterances**
- Primary key: id
- Foreign keys: session_id, speaker_db_id
- Stores: Individual speech segments with timestamps and quick analysis

### 6.2 Indexes
- `speakers.speaker_id` (for fast lookup)
- `sessions.session_id` (for session retrieval)
- `utterances.session_id` (for transcript assembly)
- `utterances.speaker_db_id` (for speaker history)

---

## 7. Implementation Sequence

### Phase 1: Foundation (Week 1)
1. ✅ Set up project structure
2. ✅ Create profiling framework documentation
3. Create database schema and models
4. Set up configuration management (.env, config.py)
5. Install and verify dependencies
6. Create basic CLI skeleton

### Phase 2: Audio Pipeline (Week 1-2)
1. Implement PulseAudio monitor detection
2. Build audio capture module with buffering
3. Integrate Whisper for speech-to-text
4. Test audio → transcript pipeline
5. Add Voice Activity Detection (VAD)

### Phase 3: Speaker Identification (Week 2)
1. Integrate pyannote.audio for diarization
2. Implement voice embedding extraction
3. Build speaker matching/registration logic
4. Add speaker renaming functionality
5. Test multi-speaker scenarios

### Phase 4: Linguistic Analysis (Week 2-3)
1. Build VAK keyword analyzer
2. Implement social needs pattern matching
3. Add decision style detection
4. Create communication pattern analyzer
5. Test with sample transcripts

### Phase 5: Profiling Engine (Week 3)
1. Implement profile creation
2. Build incremental profile updating
3. Add confidence scoring
4. Create profile comparison tools
5. Test profile accuracy with known examples

### Phase 6: CLI Interface (Week 3-4)
1. Implement start/stop recording commands
2. Add profile viewing commands
3. Build speaker renaming interface
4. Create session history viewer
5. Add export functionality (JSON, CSV)

### Phase 7: Testing & Optimization (Week 4)
1. Unit tests for each module
2. Integration tests for full pipeline
3. Performance optimization
4. Error handling and edge cases
5. Documentation and examples

### Phase 8: Web UI (Future - Week 5+)
1. Flask application setup
2. API endpoints for profiles/sessions
3. Dashboard with profile visualizations
4. Speaker timeline view
5. Real-time monitoring page

---

## 8. CLI Commands

### 8.1 Core Commands

```bash
# Start recording from system audio
profiler record start

# Stop recording
profiler record stop

# List all speakers
profiler speakers list

# View detailed profile for a speaker
profiler speakers profile "Speaker A"

# Rename a speaker
profiler speakers rename "Speaker A" "John Doe"

# List all sessions
profiler sessions list

# View session transcript with speaker labels
profiler sessions view <session_id>

# Export profile to JSON
profiler export speaker "Speaker A" --format json --output john_profile.json

# Export session transcript
profiler export session <session_id> --format txt

# Show statistics
profiler stats

# Process a pre-recorded audio file
profiler process audio_file.wav
```

### 8.2 CLI Implementation (`src/cli.py`)

```python
import click
from rich.console import Console
from rich.table import Table

console = Console()

@click.group()
def cli():
    """Audio Profiling System CLI"""
    pass

@cli.group()
def record():
    """Recording controls"""
    pass

@record.command()
def start():
    """Start recording system audio"""
    console.print("[green]Starting audio capture...[/green]")
    # Implementation

@record.command()
def stop():
    """Stop recording"""
    console.print("[yellow]Stopping audio capture...[/yellow]")
    # Implementation

@cli.group()
def speakers():
    """Manage speakers"""
    pass

@speakers.command()
def list():
    """List all speakers"""
    # Query database
    # Display rich table

@speakers.command()
@click.argument('speaker_id')
def profile(speaker_id):
    """View speaker profile"""
    # Query profile
    # Display formatted profile

@speakers.command()
@click.argument('old_id')
@click.argument('new_id')
def rename(old_id, new_id):
    """Rename a speaker"""
    # Update database
    console.print(f"[green]Renamed {old_id} → {new_id}[/green]")

# Add more command groups...

if __name__ == '__main__':
    cli()
```

---

## 9. Web UI Design

### 9.1 Pages

**Dashboard** (`/`)
- Overview stats (total speakers, sessions, hours processed)
- Recent activity feed
- Quick actions (start/stop recording)

**Speakers** (`/speakers`)
- List all speakers with thumbnails
- Search and filter
- Click to view detailed profile

**Speaker Profile** (`/speakers/<id>`)
- Profile overview (VAK, social needs, decision style)
- Charts and visualizations
- Conversation history
- Edit speaker name/notes

**Sessions** (`/sessions`)
- List all recording sessions
- Timeline view
- Filter by date/speaker

**Session Viewer** (`/sessions/<id>`)
- Full transcript with speaker labels
- Timestamp navigation
- Audio playback (if saved)
- Export options

**Real-time Monitor** (`/monitor`)
- Live transcription
- Speaker identification in real-time
- Current session stats

### 9.2 Technology Stack
- **Backend**: Flask + Flask-SQLAlchemy
- **Frontend**: HTML5, CSS3, JavaScript (vanilla or Vue.js)
- **Charts**: Chart.js or D3.js
- **UI Framework**: Tailwind CSS or Bootstrap 5

---

## 10. Testing Strategy

### 10.1 Unit Tests
- Test each module independently
- Mock external dependencies (Whisper, pyannote)
- Test linguistic analyzer with known inputs
- Verify scoring algorithms

### 10.2 Integration Tests
- Test full pipeline with sample audio
- Verify database operations
- Test speaker identification accuracy
- Validate profile generation

### 10.3 Test Data
- Create sample audio files with multiple speakers
- Prepare transcripts with known profiling markers
- Test edge cases (overlapping speech, silence, background noise)

### 10.4 Performance Tests
- Measure processing time for 1-hour audio
- Monitor memory usage
- Check database query performance
- Test concurrent processing

---

## 11. Future Enhancements

### 11.1 Advanced Features
- **Multi-language support**: Extend beyond English
- **Emotion detection**: Integrate emotion/sentiment analysis
- **Topic modeling**: Automatically detect conversation topics
- **Relationship mapping**: Detect relationships between speakers
- **Vocal characteristics**: Analyze pitch, pace, volume patterns

### 11.2 Integrations
- **Discord bot**: Direct integration with Discord servers
- **Zoom/Teams**: Process meeting recordings
- **Calendar integration**: Auto-label sessions with calendar events
- **CRM integration**: Export profiles to CRM systems

### 11.3 Analysis Improvements
- **Deception indicators**: Linguistic markers of potential deception
- **Stress detection**: Analyze vocal stress markers
- **Personality types**: Map to MBTI, Big Five, etc.
- **Influence strategies**: Suggest communication approaches per person

---

## 12. Configuration

### 12.1 Environment Variables (`.env`)

```bash
# Database
DATABASE_URL=sqlite:///data/profiles.db

# Audio Settings
AUDIO_SAMPLE_RATE=16000
AUDIO_CHUNK_DURATION=5.0
PULSEAUDIO_DEVICE=auto  # or specific device name

# Whisper Settings
WHISPER_MODEL_SIZE=base  # tiny, base, small, medium, large
WHISPER_DEVICE=cpu  # or cuda
WHISPER_LANGUAGE=en

# Speaker Identification
SPEAKER_SIMILARITY_THRESHOLD=0.75
AUTO_REGISTER_SPEAKERS=true

# Profiling
MIN_WORDS_FOR_PROFILE=100
CONFIDENCE_THRESHOLDS=100,500,1500  # low, medium, high

# Web UI (if enabled)
FLASK_SECRET_KEY=your-secret-key
FLASK_PORT=5000
FLASK_DEBUG=false

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/profiler.log
```

---

## 13. Quick Start Guide

### Installation

```bash
# Clone/create project
cd /home/morpheus/Hackstuff/Profiler

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models
python -m spacy download en_core_web_md
python -c "import whisper; whisper.load_model('base')"

# Initialize database
python src/main.py init-db

# Configure
cp .env.example .env
# Edit .env with your settings
```

### First Use

```bash
# Start recording
profiler record start

# Let it run while you talk on Discord...

# Stop recording
profiler record stop

# View speakers
profiler speakers list

# View a profile
profiler speakers profile "Speaker A"

# Rename a speaker
profiler speakers rename "Speaker A" "Your Name"
```

---

## 14. Troubleshooting

### Common Issues

**PulseAudio device not found**
```bash
# List audio sources
pactl list sources short
# Set specific device in .env
PULSEAUDIO_DEVICE=alsa_output.pci-0000_00_1f.3.analog-stereo.monitor
```

**Whisper out of memory**
```bash
# Use smaller model
WHISPER_MODEL_SIZE=tiny
```

**Speaker identification not working**
```bash
# Lower similarity threshold
SPEAKER_SIMILARITY_THRESHOLD=0.65
```

---

## 15. API Reference (for Web UI)

### REST Endpoints

```
GET  /api/speakers              - List all speakers
GET  /api/speakers/<id>         - Get speaker profile
POST /api/speakers/<id>/rename  - Rename speaker
GET  /api/sessions              - List sessions
GET  /api/sessions/<id>         - Get session details
POST /api/record/start          - Start recording
POST /api/record/stop           - Stop recording
GET  /api/stats                 - Get system statistics
```

---

## 16. Handoff Checklist

If handing this project to another AI or developer, ensure they have:

- ✅ This complete plan document
- ✅ `/docs/profiling_framework.md` with Chase Hughes reference
- ✅ Access to the project directory structure
- ✅ Understanding of PulseAudio/PipeWire on Linux
- ✅ Familiarity with SQLAlchemy ORM
- ✅ Knowledge of async Python programming (if implementing real-time features)
- ✅ Understanding of NLP libraries (spaCy, NLTK)

**Key concepts to understand**:
1. Speaker diarization vs speaker identification
2. Voice embeddings and cosine similarity
3. Linguistic profiling vs sentiment analysis
4. Weighted incremental profile updating
5. Confidence scoring based on sample size

---

## END OF PLAN

This document should provide complete guidance for implementing the Audio Profiling System. For questions or clarifications, refer to the Chase Hughes books "6 Minute X-Ray" and "The Ellipsis Manual" for deeper understanding of the behavioral analysis framework.
