"""Linguistic analyzer for behavioral profiling based on Chase Hughes' methodologies.

Analyzes text for:
- VAK (Visual/Auditory/Kinesthetic) modality
- Social needs (Significance, Approval, Acceptance, Intelligence, Pity, Power)
- Decision styles
- Communication patterns
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

from .pattern_data import (
    VAK_PATTERNS,
    SOCIAL_NEEDS_PATTERNS,
    DECISION_STYLE_PATTERNS,
    CERTAINTY_MARKERS,
    TIME_ORIENTATION_MARKERS,
    PRONOUN_CATEGORIES,
    EMOTIONAL_INDICATORS,
    SPEECH_PATTERNS,
    VALUE_INDICATORS,
    INFLUENCE_PATTERNS,
    LINGUISTIC_STRESS_INDICATORS,
)


@dataclass
class AnalysisResult:
    """Results from linguistic analysis."""

    # VAK scores (normalized to sum to 1.0)
    vak_scores: dict = field(default_factory=lambda: {
        "visual": 0.0, "auditory": 0.0, "kinesthetic": 0.0
    })

    # Social needs scores (0.0 to 1.0 each)
    social_needs: dict = field(default_factory=lambda: {
        "significance": 0.0, "approval": 0.0, "acceptance": 0.0,
        "intelligence": 0.0, "pity": 0.0, "power": 0.0
    })

    # Detected decision styles (list of styles)
    decision_styles: list = field(default_factory=list)

    # Communication patterns
    communication_patterns: dict = field(default_factory=lambda: {
        "certainty": 0.5,
        "question_ratio": 0.0,
        "active_voice": 0.5,
        "time_orientation": "present"
    })

    # Text metrics
    word_count: int = 0
    unique_words: int = 0
    complexity: float = 0.0
    sentiment: float = 0.0

    # Pronoun analysis
    pronoun_ratios: dict = field(default_factory=dict)

    # Emotional state indicators
    emotional_indicators: dict = field(default_factory=dict)

    # Value dimensions
    value_dimensions: dict = field(default_factory=dict)

    # Influence patterns detected
    influence_patterns: list = field(default_factory=list)

    # Stress/deception linguistic indicators
    stress_indicators: dict = field(default_factory=dict)

    # Filler word ratio
    filler_ratio: float = 0.0

    # Raw match counts for debugging
    raw_matches: dict = field(default_factory=dict)

    def get_dominant_vak(self) -> str:
        """Get the dominant VAK modality."""
        if not any(self.vak_scores.values()):
            return "unknown"
        return max(self.vak_scores, key=self.vak_scores.get)

    def get_top_needs(self, n: int = 2) -> list:
        """Get top N social needs."""
        sorted_needs = sorted(self.social_needs.items(), key=lambda x: x[1], reverse=True)
        return [need for need, score in sorted_needs[:n] if score > 0.1]

    def get_dominant_emotion(self) -> Optional[str]:
        """Get the dominant emotional state if detected."""
        if not self.emotional_indicators:
            return None
        sorted_emotions = sorted(
            self.emotional_indicators.items(),
            key=lambda x: x[1],
            reverse=True
        )
        if sorted_emotions and sorted_emotions[0][1] > 0:
            return sorted_emotions[0][0]
        return None


class LinguisticAnalyzer:
    """Analyzes text for behavioral profiling markers."""

    def __init__(self, use_spacy: bool = True):
        """Initialize the analyzer.

        Args:
            use_spacy: Whether to use spaCy for advanced NLP (if available)
        """
        self.nlp = None
        if use_spacy and SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_md")
                except OSError:
                    pass

        # Pre-compile regex patterns for phrases
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficient matching."""
        self._vak_phrase_patterns = {}
        for modality, data in VAK_PATTERNS.items():
            patterns = [re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
                       for phrase in data.get("phrases", [])]
            self._vak_phrase_patterns[modality] = patterns

        self._social_needs_phrase_patterns = {}
        for need, data in SOCIAL_NEEDS_PATTERNS.items():
            patterns = [re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
                       for phrase in data.get("phrases", [])]
            self._social_needs_phrase_patterns[need] = patterns

        self._decision_phrase_patterns = {}
        for style, data in DECISION_STYLE_PATTERNS.items():
            patterns = [re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
                       for phrase in data.get("phrases", [])]
            self._decision_phrase_patterns[style] = patterns

        # Negation patterns for context-aware matching
        self._negation_words = {
            "not", "no", "never", "don't", "doesn't", "didn't", "won't",
            "wouldn't", "couldn't", "shouldn't", "can't", "cannot", "neither",
            "nor", "nothing", "nobody", "nowhere", "hardly", "barely", "rarely",
        }

    def _is_negated(self, text: str, keyword: str, window: int = 3) -> bool:
        """Check if a keyword is negated within a window of words.

        Args:
            text: The full text (lowercase)
            keyword: The keyword to check
            window: Number of words before keyword to check for negation

        Returns:
            True if the keyword appears to be negated
        """
        # Find the keyword position
        pattern = r'\b' + re.escape(keyword) + r'\b'
        match = re.search(pattern, text)
        if not match:
            return False

        # Get words before the keyword
        text_before = text[:match.start()]
        words_before = text_before.split()[-window:]

        # Check for negation words
        for word in words_before:
            word_clean = re.sub(r'[^\w]', '', word.lower())
            if word_clean in self._negation_words:
                return True

        return False

    def analyze(self, text: str) -> AnalysisResult:
        """Perform comprehensive linguistic analysis.

        Args:
            text: The text to analyze

        Returns:
            AnalysisResult with all analysis metrics
        """
        if not text or not text.strip():
            return AnalysisResult()

        result = AnalysisResult()
        text_lower = text.lower()

        # Basic tokenization
        words = self._tokenize(text)
        result.word_count = len(words)
        result.unique_words = len(set(w.lower() for w in words))

        if result.word_count == 0:
            return result

        # Analyze different aspects
        result.vak_scores = self._analyze_vak(text_lower, words)
        result.social_needs = self._analyze_social_needs(text_lower, words)
        result.decision_styles = self._analyze_decision_styles(text_lower)
        result.communication_patterns = self._analyze_communication_patterns(text, words)
        result.pronoun_ratios = self._analyze_pronouns(words)
        result.complexity = self._calculate_complexity(text)
        result.sentiment = self._calculate_sentiment(text_lower, words)

        # Enhanced analysis from The Ellipsis Manual
        result.emotional_indicators = self._analyze_emotions(text_lower)
        result.value_dimensions = self._analyze_values(text_lower)
        result.influence_patterns = self._detect_influence_patterns(text_lower)
        result.stress_indicators = self._analyze_stress_indicators(text_lower)
        result.filler_ratio = self._calculate_filler_ratio(words)

        return result

    def _tokenize(self, text: str) -> list:
        """Tokenize text into words."""
        if self.nlp:
            doc = self.nlp(text)
            return [token.text for token in doc if not token.is_punct and not token.is_space]
        else:
            # Simple regex tokenization
            return re.findall(r'\b[a-zA-Z]+\b', text)

    def _analyze_vak(self, text_lower: str, words: list) -> dict:
        """Analyze VAK modality distribution.

        Returns normalized scores that sum to 1.0.
        """
        scores = {"visual": 0, "auditory": 0, "kinesthetic": 0}
        words_lower = [w.lower() for w in words]

        for modality, data in VAK_PATTERNS.items():
            # Count keyword matches
            keyword_matches = sum(1 for w in words_lower if w in data["keywords"])

            # Count phrase matches (weighted higher)
            phrase_matches = 0
            for pattern in self._vak_phrase_patterns.get(modality, []):
                phrase_matches += len(pattern.findall(text_lower))

            # Calculate weighted score
            scores[modality] = keyword_matches + (phrase_matches * 2)

        # Normalize to sum to 1.0
        total = sum(scores.values())
        if total > 0:
            scores = {k: round(v / total, 3) for k, v in scores.items()}
        else:
            scores = {k: round(1/3, 3) for k in scores}

        return scores

    def _analyze_social_needs(self, text_lower: str, words: list) -> dict:
        """Analyze social needs indicators.

        Returns scores from 0.0 to 1.0 for each need.
        """
        scores = {}
        words_lower = [w.lower() for w in words]
        word_count = len(words)

        for need, data in SOCIAL_NEEDS_PATTERNS.items():
            score = 0.0

            # Count keyword matches
            keywords = set(data.get("keywords", []))
            keyword_matches = sum(1 for w in words_lower if w in keywords)
            keyword_score = min(keyword_matches / max(word_count / 20, 1), 1.0)

            # Count phrase matches
            phrase_matches = 0
            for pattern in self._social_needs_phrase_patterns.get(need, []):
                phrase_matches += len(pattern.findall(text_lower))
            phrase_score = min(phrase_matches / max(word_count / 50, 1), 1.0)

            # Check pronoun patterns if defined
            pronoun_score = 0.0
            if "pronoun_pattern" in data:
                pronoun_data = data["pronoun_pattern"]
                pronoun_count = sum(1 for w in words_lower if w in pronoun_data["pronouns"])
                pronoun_ratio = pronoun_count / word_count if word_count > 0 else 0
                if pronoun_ratio >= pronoun_data["threshold"]:
                    pronoun_score = min(pronoun_ratio / pronoun_data["threshold"], 1.0) * 0.5

            # Combine scores with weights
            score = (keyword_score * 0.3) + (phrase_score * 0.5) + (pronoun_score * 0.2)
            scores[need] = round(min(score, 1.0), 3)

        return scores

    def _analyze_decision_styles(self, text_lower: str) -> list:
        """Detect decision-making styles.

        Returns list of detected styles (those with significant presence).
        """
        style_scores = {}

        for style, data in DECISION_STYLE_PATTERNS.items():
            score = 0

            # Count keyword matches
            keywords = set(data.get("keywords", []))
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    score += 1

            # Count phrase matches (weighted higher)
            for pattern in self._decision_phrase_patterns.get(style, []):
                score += len(pattern.findall(text_lower)) * 2

            style_scores[style] = score

        # Return styles with scores above threshold
        max_score = max(style_scores.values()) if style_scores else 0
        if max_score > 0:
            threshold = max_score * 0.5  # Include styles with at least 50% of max
            return [style for style, score in style_scores.items()
                   if score >= threshold and score > 0]

        return []

    def _analyze_communication_patterns(self, text: str, words: list) -> dict:
        """Analyze communication patterns."""
        patterns = {
            "certainty": 0.5,
            "question_ratio": 0.0,
            "active_voice": 0.5,
            "time_orientation": "present"
        }

        text_lower = text.lower()
        words_lower = [w.lower() for w in words]
        word_count = len(words)

        if word_count == 0:
            return patterns

        # Certainty analysis
        high_certainty_count = sum(
            1 for marker in CERTAINTY_MARKERS["high_certainty"]
            if re.search(r'\b' + re.escape(marker) + r'\b', text_lower)
        )
        low_certainty_count = sum(
            1 for marker in CERTAINTY_MARKERS["low_certainty"]
            if re.search(r'\b' + re.escape(marker) + r'\b', text_lower)
        )
        total_certainty = high_certainty_count + low_certainty_count
        if total_certainty > 0:
            patterns["certainty"] = round(high_certainty_count / total_certainty, 3)

        # Question ratio
        question_marks = text.count('?')
        sentences = max(len(re.findall(r'[.!?]+', text)), 1)
        patterns["question_ratio"] = round(question_marks / sentences, 3)

        # Time orientation
        time_counts = {}
        for orientation, markers in TIME_ORIENTATION_MARKERS.items():
            count = sum(1 for marker in markers
                       if re.search(r'\b' + re.escape(marker) + r'\b', text_lower))
            time_counts[orientation] = count

        if any(time_counts.values()):
            patterns["time_orientation"] = max(time_counts, key=time_counts.get)

        # Active voice estimation (simplified)
        if self.nlp:
            doc = self.nlp(text)
            passive_count = sum(1 for token in doc if token.dep_ in ["nsubjpass", "auxpass"])
            active_count = sum(1 for token in doc if token.dep_ == "nsubj")
            total = passive_count + active_count
            if total > 0:
                patterns["active_voice"] = round(active_count / total, 3)

        return patterns

    def _analyze_pronouns(self, words: list) -> dict:
        """Analyze pronoun usage patterns."""
        words_lower = [w.lower() for w in words]
        word_count = len(words)

        if word_count == 0:
            return {}

        ratios = {}
        for category, pronouns in PRONOUN_CATEGORIES.items():
            count = sum(1 for w in words_lower if w in pronouns)
            ratios[category] = round(count / word_count, 4)

        return ratios

    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity (Flesch-Kincaid grade level)."""
        if TEXTSTAT_AVAILABLE:
            try:
                return round(textstat.flesch_kincaid_grade(text), 2)
            except Exception:
                pass

        # Fallback: simple syllable-based estimation
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        if not words:
            return 0.0

        sentences = max(len(re.findall(r'[.!?]+', text)), 1)
        syllables = sum(self._count_syllables(word) for word in words)
        word_count = len(words)

        if word_count == 0 or sentences == 0:
            return 0.0

        # Flesch-Kincaid formula
        grade = (0.39 * (word_count / sentences)) + (11.8 * (syllables / word_count)) - 15.59
        return round(max(0, grade), 2)

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simple heuristic)."""
        word = word.lower()
        vowels = "aeiouy"
        count = 0
        prev_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Adjust for silent e
        if word.endswith('e') and count > 1:
            count -= 1

        return max(count, 1)

    def _calculate_sentiment(self, text_lower: str, words: list) -> float:
        """Calculate basic sentiment score (-1.0 to 1.0)."""
        positive_words = {
            "good", "great", "excellent", "amazing", "wonderful", "fantastic",
            "awesome", "love", "happy", "joy", "excited", "pleased", "glad",
            "delighted", "grateful", "thankful", "positive", "perfect", "best",
            "brilliant", "beautiful", "nice", "like", "enjoy", "fun",
        }

        negative_words = {
            "bad", "terrible", "awful", "horrible", "hate", "sad", "angry",
            "upset", "frustrated", "annoyed", "disappointed", "worried",
            "anxious", "scared", "afraid", "hurt", "pain", "wrong", "worst",
            "stupid", "dumb", "ugly", "boring", "fail", "problem", "issue",
        }

        words_lower = [w.lower() for w in words]
        pos_count = sum(1 for w in words_lower if w in positive_words)
        neg_count = sum(1 for w in words_lower if w in negative_words)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return round((pos_count - neg_count) / total, 3)

    def analyze_batch(self, texts: list) -> AnalysisResult:
        """Analyze multiple texts and aggregate results.

        Args:
            texts: List of texts to analyze

        Returns:
            Aggregated AnalysisResult
        """
        if not texts:
            return AnalysisResult()

        combined_text = " ".join(texts)
        return self.analyze(combined_text)

    def _analyze_emotions(self, text_lower: str) -> dict:
        """Analyze emotional state indicators from The Ellipsis Manual.

        Returns scores for anxiety, anger, sadness, joy.
        """
        emotions = {}

        for emotion, data in EMOTIONAL_INDICATORS.items():
            score = 0

            # Count keyword matches
            for keyword in data.get("keywords", []):
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    score += 1

            # Count phrase matches (weighted higher)
            for phrase in data.get("phrases", []):
                if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
                    score += 2

            emotions[emotion] = score

        # Normalize
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: round(v / total, 3) for k, v in emotions.items()}

        return emotions

    def _analyze_values(self, text_lower: str) -> dict:
        """Analyze value dimension indicators.

        Returns dict with value dimension scores.
        """
        values = {}

        for dimension, poles in VALUE_INDICATORS.items():
            pole_names = list(poles.keys())
            if len(pole_names) < 2:
                continue

            pole1_name, pole2_name = pole_names[0], pole_names[1]
            pole1_words = poles[pole1_name]
            pole2_words = poles[pole2_name]

            # Count matches for each pole
            pole1_count = sum(
                1 for word in pole1_words
                if re.search(r'\b' + re.escape(word) + r'\b', text_lower)
            )
            pole2_count = sum(
                1 for word in pole2_words
                if re.search(r'\b' + re.escape(word) + r'\b', text_lower)
            )

            total = pole1_count + pole2_count
            if total > 0:
                # Score from -1 (pole1) to +1 (pole2)
                score = (pole2_count - pole1_count) / total
                values[dimension] = {
                    "score": round(score, 3),
                    "dominant": pole2_name if score > 0 else pole1_name,
                    f"{pole1_name}_count": pole1_count,
                    f"{pole2_name}_count": pole2_count,
                }

        return values

    def _detect_influence_patterns(self, text_lower: str) -> list:
        """Detect influence and persuasion patterns from The Ellipsis Manual.

        Returns list of detected influence tactics.
        """
        detected = []

        for tactic, data in INFLUENCE_PATTERNS.items():
            phrases = data.get("phrases", [])
            for phrase in phrases:
                if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
                    if tactic not in detected:
                        detected.append(tactic)
                    break

        return detected

    def _analyze_stress_indicators(self, text_lower: str) -> dict:
        """Analyze linguistic stress indicators.

        These are NOT proof of deception, just indicators of potential stress.
        """
        indicators = {}

        # Qualifier overuse
        qualifiers = LINGUISTIC_STRESS_INDICATORS.get("qualifier_overuse", {})
        qualifier_count = 0
        for keyword in qualifiers.get("keywords", []):
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                qualifier_count += 1
        indicators["qualifier_count"] = qualifier_count
        indicators["high_qualifiers"] = qualifier_count > 3

        # Bolstering statements
        bolstering = LINGUISTIC_STRESS_INDICATORS.get("bolstering_statements", {})
        bolster_count = 0
        for phrase in bolstering.get("phrases", []):
            if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
                bolster_count += 1
        indicators["bolstering_count"] = bolster_count
        indicators["bolstering_detected"] = bolster_count > 0

        # Distancing language
        distancing = LINGUISTIC_STRESS_INDICATORS.get("distancing_language", {})
        distance_count = 0
        for phrase in distancing.get("phrases", []):
            if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
                distance_count += 1
        indicators["distancing_count"] = distance_count

        return indicators

    def _calculate_filler_ratio(self, words: list) -> float:
        """Calculate the ratio of filler words to total words."""
        if not words:
            return 0.0

        words_lower = [w.lower() for w in words]
        filler_words = set(SPEECH_PATTERNS.get("filler_words", []))

        filler_count = sum(1 for w in words_lower if w in filler_words)
        return round(filler_count / len(words), 4)
