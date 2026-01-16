#!/usr/bin/env python3
"""Standalone test script for the linguistic analyzer.

Run from the Profiler directory: python test_analyzer.py
"""

import re
from dataclasses import dataclass, field
from typing import Optional

# Import pattern data directly
exec(open("src/profiling/pattern_data.py").read())

# Simplified LinguisticAnalyzer for testing
@dataclass
class AnalysisResult:
    """Results from linguistic analysis."""
    vak_scores: dict = field(default_factory=lambda: {
        "visual": 0.0, "auditory": 0.0, "kinesthetic": 0.0
    })
    social_needs: dict = field(default_factory=lambda: {
        "significance": 0.0, "approval": 0.0, "acceptance": 0.0,
        "intelligence": 0.0, "pity": 0.0, "power": 0.0
    })
    decision_styles: list = field(default_factory=list)
    communication_patterns: dict = field(default_factory=lambda: {
        "certainty": 0.5, "question_ratio": 0.0, "time_orientation": "present"
    })
    word_count: int = 0
    emotional_indicators: dict = field(default_factory=dict)
    influence_patterns: list = field(default_factory=list)
    pronoun_ratios: dict = field(default_factory=dict)

    def get_dominant_vak(self) -> str:
        if not any(self.vak_scores.values()):
            return "unknown"
        return max(self.vak_scores, key=self.vak_scores.get)

    def get_top_needs(self, n: int = 2) -> list:
        sorted_needs = sorted(self.social_needs.items(), key=lambda x: x[1], reverse=True)
        return [need for need, score in sorted_needs[:n] if score > 0.1]


class TestAnalyzer:
    """Simplified analyzer for testing."""

    def __init__(self):
        self._compile_patterns()

    def _compile_patterns(self):
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

    def analyze(self, text: str) -> AnalysisResult:
        if not text or not text.strip():
            return AnalysisResult()

        result = AnalysisResult()
        text_lower = text.lower()
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        result.word_count = len(words)

        if result.word_count == 0:
            return result

        result.vak_scores = self._analyze_vak(text_lower, words)
        result.social_needs = self._analyze_social_needs(text_lower, words)
        result.decision_styles = self._analyze_decision_styles(text_lower)
        result.communication_patterns = self._analyze_communication(text_lower)
        result.emotional_indicators = self._analyze_emotions(text_lower)
        result.influence_patterns = self._detect_influence(text_lower)
        result.pronoun_ratios = self._analyze_pronouns(words)

        return result

    def _analyze_vak(self, text_lower, words):
        scores = {"visual": 0, "auditory": 0, "kinesthetic": 0}
        words_lower = [w.lower() for w in words]

        for modality, data in VAK_PATTERNS.items():
            keyword_matches = sum(1 for w in words_lower if w in data["keywords"])
            phrase_matches = 0
            for pattern in self._vak_phrase_patterns.get(modality, []):
                phrase_matches += len(pattern.findall(text_lower))
            scores[modality] = keyword_matches + (phrase_matches * 2)

        total = sum(scores.values())
        if total > 0:
            scores = {k: round(v / total, 3) for k, v in scores.items()}
        else:
            scores = {k: round(1/3, 3) for k in scores}
        return scores

    def _analyze_social_needs(self, text_lower, words):
        scores = {}
        words_lower = [w.lower() for w in words]
        word_count = len(words)

        for need, data in SOCIAL_NEEDS_PATTERNS.items():
            keywords = set(data.get("keywords", []))
            keyword_matches = sum(1 for w in words_lower if w in keywords)
            keyword_score = min(keyword_matches / max(word_count / 20, 1), 1.0)

            phrase_matches = 0
            for pattern in self._social_needs_phrase_patterns.get(need, []):
                phrase_matches += len(pattern.findall(text_lower))
            phrase_score = min(phrase_matches / max(word_count / 50, 1), 1.0)

            pronoun_score = 0.0
            if "pronoun_pattern" in data:
                pronoun_data = data["pronoun_pattern"]
                pronoun_count = sum(1 for w in words_lower if w in pronoun_data["pronouns"])
                pronoun_ratio = pronoun_count / word_count if word_count > 0 else 0
                if pronoun_ratio >= pronoun_data["threshold"]:
                    pronoun_score = min(pronoun_ratio / pronoun_data["threshold"], 1.0) * 0.5

            score = (keyword_score * 0.3) + (phrase_score * 0.5) + (pronoun_score * 0.2)
            scores[need] = round(min(score, 1.0), 3)

        return scores

    def _analyze_decision_styles(self, text_lower):
        style_scores = {}
        for style, data in DECISION_STYLE_PATTERNS.items():
            score = 0
            keywords = set(data.get("keywords", []))
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    score += 1
            for pattern in self._decision_phrase_patterns.get(style, []):
                score += len(pattern.findall(text_lower)) * 2
            style_scores[style] = score

        max_score = max(style_scores.values()) if style_scores else 0
        if max_score > 0:
            threshold = max_score * 0.5
            return [style for style, score in style_scores.items() if score >= threshold and score > 0]
        return []

    def _analyze_communication(self, text_lower):
        patterns = {"certainty": 0.5, "question_ratio": 0.0, "time_orientation": "present"}

        high_certainty = sum(1 for m in CERTAINTY_MARKERS["high_certainty"]
                            if re.search(r'\b' + re.escape(m) + r'\b', text_lower))
        low_certainty = sum(1 for m in CERTAINTY_MARKERS["low_certainty"]
                           if re.search(r'\b' + re.escape(m) + r'\b', text_lower))
        total = high_certainty + low_certainty
        if total > 0:
            patterns["certainty"] = round(high_certainty / total, 3)

        time_counts = {}
        for orientation, markers in TIME_ORIENTATION_MARKERS.items():
            count = sum(1 for m in markers if re.search(r'\b' + re.escape(m) + r'\b', text_lower))
            time_counts[orientation] = count
        if any(time_counts.values()):
            patterns["time_orientation"] = max(time_counts, key=time_counts.get)

        return patterns

    def _analyze_emotions(self, text_lower):
        emotions = {}
        for emotion, data in EMOTIONAL_INDICATORS.items():
            score = sum(1 for k in data.get("keywords", [])
                       if re.search(r'\b' + re.escape(k) + r'\b', text_lower))
            score += sum(2 for p in data.get("phrases", [])
                        if re.search(r'\b' + re.escape(p) + r'\b', text_lower))
            emotions[emotion] = score
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: round(v / total, 3) for k, v in emotions.items()}
        return emotions

    def _detect_influence(self, text_lower):
        detected = []
        for tactic, data in INFLUENCE_PATTERNS.items():
            for phrase in data.get("phrases", []):
                if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
                    if tactic not in detected:
                        detected.append(tactic)
                    break
        return detected

    def _analyze_pronouns(self, words):
        words_lower = [w.lower() for w in words]
        word_count = len(words)
        if word_count == 0:
            return {}
        ratios = {}
        for category, pronouns in PRONOUN_CATEGORIES.items():
            count = sum(1 for w in words_lower if w in pronouns)
            ratios[category] = round(count / word_count, 4)
        return ratios


def run_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Linguistic Analyzer Tests")
    print("Based on Chase Hughes' 6 Minute X-Ray & Ellipsis Manual")
    print("="*60 + "\n")

    analyzer = TestAnalyzer()
    passed = 0
    failed = 0

    # Test 1: VAK Visual
    try:
        text = "I see what you mean. Let me show you this picture. It looks clear to me."
        result = analyzer.analyze(text)
        assert result.vak_scores["visual"] > result.vak_scores["auditory"]
        assert result.get_dominant_vak() == "visual"
        print(f"✓ VAK Visual: {result.vak_scores}")
        passed += 1
    except Exception as e:
        print(f"✗ VAK Visual FAILED: {e}")
        failed += 1

    # Test 2: VAK Auditory
    try:
        text = "Listen to this. It sounds like a great idea. Tell me more."
        result = analyzer.analyze(text)
        assert result.vak_scores["auditory"] > result.vak_scores["visual"]
        assert result.get_dominant_vak() == "auditory"
        print(f"✓ VAK Auditory: {result.vak_scores}")
        passed += 1
    except Exception as e:
        print(f"✗ VAK Auditory FAILED: {e}")
        failed += 1

    # Test 3: VAK Kinesthetic
    try:
        text = "I feel you on this. Let me get a grip on it. It feels solid."
        result = analyzer.analyze(text)
        assert result.vak_scores["kinesthetic"] > result.vak_scores["visual"]
        assert result.get_dominant_vak() == "kinesthetic"
        print(f"✓ VAK Kinesthetic: {result.vak_scores}")
        passed += 1
    except Exception as e:
        print(f"✗ VAK Kinesthetic FAILED: {e}")
        failed += 1

    # Test 4: Significance Need
    try:
        text = "I was the first to do this. Nobody else could. I'm the best. They recognized my achievement."
        result = analyzer.analyze(text)
        assert result.social_needs["significance"] > 0.1
        print(f"✓ Significance Need: {result.social_needs['significance']:.3f}")
        passed += 1
    except Exception as e:
        print(f"✗ Significance Need FAILED: {e}")
        failed += 1

    # Test 5: Approval Need
    try:
        text = "I think maybe this is okay? Sorry if I'm wrong. Is that good? Do you like it?"
        result = analyzer.analyze(text)
        assert result.social_needs["approval"] > 0.1
        print(f"✓ Approval Need: {result.social_needs['approval']:.3f}")
        passed += 1
    except Exception as e:
        print(f"✗ Approval Need FAILED: {e}")
        failed += 1

    # Test 6: Acceptance Need
    try:
        text = "We all work together. Everyone's doing it this way. We belong to this team."
        result = analyzer.analyze(text)
        assert result.social_needs["acceptance"] > 0.1
        print(f"✓ Acceptance Need: {result.social_needs['acceptance']:.3f}")
        passed += 1
    except Exception as e:
        print(f"✗ Acceptance Need FAILED: {e}")
        failed += 1

    # Test 7: Intelligence Need
    try:
        text = "Well actually, technically speaking, the research shows this. Let me explain the nuance."
        result = analyzer.analyze(text)
        assert result.social_needs["intelligence"] > 0.1
        print(f"✓ Intelligence Need: {result.social_needs['intelligence']:.3f}")
        passed += 1
    except Exception as e:
        print(f"✗ Intelligence Need FAILED: {e}")
        failed += 1

    # Test 8: Pity Need
    try:
        text = "This always happens to me. Nobody understands. I can't catch a break. So hard for me."
        result = analyzer.analyze(text)
        assert result.social_needs["pity"] > 0.1
        print(f"✓ Pity Need: {result.social_needs['pity']:.3f}")
        passed += 1
    except Exception as e:
        print(f"✗ Pity Need FAILED: {e}")
        failed += 1

    # Test 9: Power Need
    try:
        text = "You need to listen to me. I'm in charge here. Trust me, I know best. Get over it."
        result = analyzer.analyze(text)
        assert result.social_needs["power"] > 0.1
        print(f"✓ Power Need: {result.social_needs['power']:.3f}")
        passed += 1
    except Exception as e:
        print(f"✗ Power Need FAILED: {e}")
        failed += 1

    # Test 10: Decision Style - Novelty
    try:
        text = "I've never tried this before! Something new sounds exciting. Innovative approach."
        result = analyzer.analyze(text)
        assert "novelty" in result.decision_styles
        print(f"✓ Novelty Decision Style: {result.decision_styles}")
        passed += 1
    except Exception as e:
        print(f"✗ Novelty Decision Style FAILED: {e}")
        failed += 1

    # Test 11: Decision Style - Social
    try:
        text = "Everyone's doing it. Really popular. They said it's good. My friends all use it."
        result = analyzer.analyze(text)
        assert "social" in result.decision_styles
        print(f"✓ Social Decision Style: {result.decision_styles}")
        passed += 1
    except Exception as e:
        print(f"✗ Social Decision Style FAILED: {e}")
        failed += 1

    # Test 12: High Certainty
    try:
        text = "This will definitely work. Absolutely certain. Guaranteed to succeed."
        result = analyzer.analyze(text)
        assert result.communication_patterns["certainty"] > 0.5
        print(f"✓ High Certainty: {result.communication_patterns['certainty']:.3f}")
        passed += 1
    except Exception as e:
        print(f"✗ High Certainty FAILED: {e}")
        failed += 1

    # Test 13: Low Certainty
    try:
        text = "Maybe this could work. I think it might be possible. Perhaps we should try."
        result = analyzer.analyze(text)
        assert result.communication_patterns["certainty"] < 0.5
        print(f"✓ Low Certainty: {result.communication_patterns['certainty']:.3f}")
        passed += 1
    except Exception as e:
        print(f"✗ Low Certainty FAILED: {e}")
        failed += 1

    # Test 14: Emotional Indicators
    try:
        text = "I'm so worried and scared. What if it goes wrong? This makes me anxious."
        result = analyzer.analyze(text)
        assert result.emotional_indicators.get("anxiety", 0) > 0
        print(f"✓ Anxiety Detection: {result.emotional_indicators}")
        passed += 1
    except Exception as e:
        print(f"✗ Anxiety Detection FAILED: {e}")
        failed += 1

    # Test 15: Influence Patterns
    try:
        text = "After all I've done for you. Everyone's doing it. This is a limited time offer."
        result = analyzer.analyze(text)
        assert len(result.influence_patterns) > 0
        print(f"✓ Influence Patterns: {result.influence_patterns}")
        passed += 1
    except Exception as e:
        print(f"✗ Influence Patterns FAILED: {e}")
        failed += 1

    # Test 16: Pronoun Analysis
    try:
        text = "I did this myself. My achievement is mine. I accomplished it by myself."
        result = analyzer.analyze(text)
        assert result.pronoun_ratios.get("self", 0) > 0.1
        print(f"✓ Pronoun Analysis: {result.pronoun_ratios}")
        passed += 1
    except Exception as e:
        print(f"✗ Pronoun Analysis FAILED: {e}")
        failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
