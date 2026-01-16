"""Configuration management for the Audio Profiling System."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Try UTF-8 first, fall back to UTF-16 for Windows compatibility
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    try:
        load_dotenv(_env_path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            load_dotenv(_env_path, encoding="utf-16")
        except UnicodeDecodeError:
            print(
                f"Error: Cannot read .env file. Please re-save it with UTF-8 encoding.",
                file=sys.stderr,
            )
            sys.exit(1)
else:
    load_dotenv()  # Will look for .env in default locations

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = DATA_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


class Config:
    """Application configuration."""

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/profiles.db")

    # Audio Settings
    AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    AUDIO_CHUNK_DURATION: float = float(os.getenv("AUDIO_CHUNK_DURATION", "5.0"))
    PULSEAUDIO_DEVICE: str = os.getenv("PULSEAUDIO_DEVICE", "auto")

    # Whisper Settings
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
    WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "en")

    # Speaker Identification
    SPEAKER_SIMILARITY_THRESHOLD: float = float(os.getenv("SPEAKER_SIMILARITY_THRESHOLD", "0.75"))
    AUTO_REGISTER_SPEAKERS: bool = os.getenv("AUTO_REGISTER_SPEAKERS", "true").lower() == "true"

    # Profiling
    MIN_WORDS_FOR_PROFILE: int = int(os.getenv("MIN_WORDS_FOR_PROFILE", "100"))

    # Parse confidence thresholds
    _thresholds = os.getenv("CONFIDENCE_THRESHOLDS", "100,500,1500").split(",")
    CONFIDENCE_LOW_THRESHOLD: int = int(_thresholds[0])
    CONFIDENCE_MEDIUM_THRESHOLD: int = int(_thresholds[1])
    CONFIDENCE_HIGH_THRESHOLD: int = int(_thresholds[2])

    # Web UI
    FLASK_SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "dev-key-change-in-production")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", str(LOGS_DIR / "profiler.log"))

    # HuggingFace Token
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # Anthropic API (for Claude analysis)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    @classmethod
    def get_confidence_level(cls, word_count: int) -> str:
        """Determine confidence level based on word count."""
        if word_count < cls.CONFIDENCE_LOW_THRESHOLD:
            return "insufficient"
        elif word_count < cls.CONFIDENCE_MEDIUM_THRESHOLD:
            return "low"
        elif word_count < cls.CONFIDENCE_HIGH_THRESHOLD:
            return "medium"
        else:
            return "high"


config = Config()
