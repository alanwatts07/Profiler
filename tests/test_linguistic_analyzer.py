"""Tests for the linguistic analyzer module.

These tests verify the Chase Hughes methodology patterns are detected correctly.
"""

import sys
import importlib.util
from pathlib import Path

# Load the module directly to avoid SQLAlchemy dependency in tests
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Load pattern_data first
spec = importlib.util.spec_from_file_location(
    "pattern_data",
    src_path / "profiling" / "pattern_data.py"
)
pattern_data = importlib.util.module_from_spec(spec)
sys.modules["profiling.pattern_data"] = pattern_data
spec.loader.exec_module(pattern_data)

# Load linguistic_analyzer
spec = importlib.util.spec_from_file_location(
    "linguistic_analyzer",
    src_path / "profiling" / "linguistic_analyzer.py"
)
linguistic_analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(linguistic_analyzer)

LinguisticAnalyzer = linguistic_analyzer.LinguisticAnalyzer


def test_vak_visual():
    """Test detection of visual modality."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = "I see what you mean. Let me show you this picture. It looks clear to me."
    result = analyzer.analyze(text)

    assert result.vak_scores["visual"] > result.vak_scores["auditory"]
    assert result.vak_scores["visual"] > result.vak_scores["kinesthetic"]
    assert result.get_dominant_vak() == "visual"
    print(f"✓ Visual VAK test passed: {result.vak_scores}")


def test_vak_auditory():
    """Test detection of auditory modality."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = "Listen to this. It sounds like a great idea. Tell me more about it."
    result = analyzer.analyze(text)

    assert result.vak_scores["auditory"] > result.vak_scores["visual"]
    assert result.vak_scores["auditory"] > result.vak_scores["kinesthetic"]
    assert result.get_dominant_vak() == "auditory"
    print(f"✓ Auditory VAK test passed: {result.vak_scores}")


def test_vak_kinesthetic():
    """Test detection of kinesthetic modality."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = "I feel you on this. Let me get a grip on the situation. It feels solid."
    result = analyzer.analyze(text)

    assert result.vak_scores["kinesthetic"] > result.vak_scores["visual"]
    assert result.vak_scores["kinesthetic"] > result.vak_scores["auditory"]
    assert result.get_dominant_vak() == "kinesthetic"
    print(f"✓ Kinesthetic VAK test passed: {result.vak_scores}")


def test_social_need_significance():
    """Test detection of significance need."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = """I was the first to accomplish this. Nobody else could have done it.
    They recognized me for my achievement. I'm the best at what I do.
    My idea made it happen. I won the award."""
    result = analyzer.analyze(text)

    assert result.social_needs["significance"] > 0.2
    assert "significance" in result.get_top_needs(2)
    print(f"✓ Significance need test passed: {result.social_needs}")


def test_social_need_approval():
    """Test detection of approval need."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = """I think maybe this is okay? Sorry if I'm wrong. I hope you don't mind.
    Is that good? Do you like it? I guess I could be wrong about this.
    Does that make sense? I was just trying to help."""
    result = analyzer.analyze(text)

    assert result.social_needs["approval"] > 0.2
    assert "approval" in result.get_top_needs(2)
    print(f"✓ Approval need test passed: {result.social_needs}")


def test_social_need_acceptance():
    """Test detection of acceptance need."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = """We all need to work together on this. Everyone's doing it this way.
    It's normal for our team. We belong to this group. Just like everyone else,
    we go along with the rest of us."""
    result = analyzer.analyze(text)

    assert result.social_needs["acceptance"] > 0.2
    print(f"✓ Acceptance need test passed: {result.social_needs}")


def test_social_need_intelligence():
    """Test detection of intelligence need."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = """Well actually, technically speaking, the research indicates otherwise.
    Studies show that according to the data, what you don't understand is
    that it's more complex than that. Let me explain the nuance."""
    result = analyzer.analyze(text)

    assert result.social_needs["intelligence"] > 0.2
    print(f"✓ Intelligence need test passed: {result.social_needs}")


def test_social_need_pity():
    """Test detection of pity need."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = """This always happens to me. Nobody understands how hard it is.
    I can't catch a break. You don't know how difficult this has been.
    Story of my life. Nothing ever works out for me."""
    result = analyzer.analyze(text)

    assert result.social_needs["pity"] > 0.2
    print(f"✓ Pity need test passed: {result.social_needs}")


def test_social_need_power():
    """Test detection of power need."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = """You need to listen to me. I'm telling you how this works.
    I'm in charge here and I can handle it. Get over it, it's not a big deal.
    I demand that you do this. Trust me, I know best."""
    result = analyzer.analyze(text)

    assert result.social_needs["power"] > 0.2
    print(f"✓ Power need test passed: {result.social_needs}")


def test_decision_style_novelty():
    """Test detection of novelty decision style."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = """I've never tried this before! Something new sounds exciting.
    Let's explore a different approach. This innovative solution is cutting edge."""
    result = analyzer.analyze(text)

    assert "novelty" in result.decision_styles
    print(f"✓ Novelty decision style test passed: {result.decision_styles}")


def test_decision_style_social():
    """Test detection of social decision style."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = """Everyone's doing it. It's really popular and trending right now.
    They said it's good and highly recommended. My friends all use it."""
    result = analyzer.analyze(text)

    assert "social" in result.decision_styles
    print(f"✓ Social decision style test passed: {result.decision_styles}")


def test_decision_style_conformity():
    """Test detection of conformity decision style."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = """This is how it's done. The traditional way is proven and safe.
    By the book, standard practice. That's how it's always been."""
    result = analyzer.analyze(text)

    assert "conformity" in result.decision_styles
    print(f"✓ Conformity decision style test passed: {result.decision_styles}")


def test_certainty_high():
    """Test detection of high certainty communication."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = """This will definitely work. I'm absolutely certain about this.
    Without a doubt, it's guaranteed. I know for sure this is right."""
    result = analyzer.analyze(text)

    assert result.communication_patterns["certainty"] > 0.5
    print(f"✓ High certainty test passed: {result.communication_patterns}")


def test_certainty_low():
    """Test detection of low certainty communication."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = """Maybe this could work. I think it might be possible.
    Perhaps we should try it. I'm not sure, but I guess we could."""
    result = analyzer.analyze(text)

    assert result.communication_patterns["certainty"] < 0.5
    print(f"✓ Low certainty test passed: {result.communication_patterns}")


def test_emotional_indicators():
    """Test emotional state detection."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    # Test anxiety
    anxious_text = "I'm worried about this. What if it goes wrong? I'm scared and nervous."
    result = analyzer.analyze(anxious_text)
    assert result.emotional_indicators.get("anxiety", 0) > 0
    print(f"✓ Anxiety detection passed: {result.emotional_indicators}")

    # Test joy
    happy_text = "I'm so happy! This is amazing and wonderful. I love it!"
    result = analyzer.analyze(happy_text)
    assert result.emotional_indicators.get("joy", 0) > 0
    print(f"✓ Joy detection passed: {result.emotional_indicators}")


def test_influence_patterns():
    """Test detection of influence tactics."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = """After all I've done for you, you owe me this. Everyone's doing it
    and experts say it's the right choice. This is a limited time offer,
    your last chance."""
    result = analyzer.analyze(text)

    assert len(result.influence_patterns) > 0
    print(f"✓ Influence patterns detected: {result.influence_patterns}")


def test_stress_indicators():
    """Test detection of linguistic stress indicators."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    text = """Honestly, to tell you the truth, believe me I swear this is true.
    Why would I lie? You have to believe me. I promise I'm being honest."""
    result = analyzer.analyze(text)

    assert result.stress_indicators.get("qualifier_count", 0) > 0
    assert result.stress_indicators.get("bolstering_detected", False)
    print(f"✓ Stress indicators detected: {result.stress_indicators}")


def test_pronoun_analysis():
    """Test pronoun ratio analysis."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    # High self-reference
    self_text = "I did this myself. My achievement is mine. I accomplished it by myself."
    result = analyzer.analyze(self_text)
    assert result.pronoun_ratios.get("self", 0) > 0.1
    print(f"✓ Self-pronoun detection passed: {result.pronoun_ratios}")

    # High group reference
    group_text = "We all worked together. Our team helped us achieve our goals."
    result = analyzer.analyze(group_text)
    assert result.pronoun_ratios.get("group", 0) > 0.05
    print(f"✓ Group-pronoun detection passed: {result.pronoun_ratios}")


def test_complexity():
    """Test text complexity calculation."""
    analyzer = LinguisticAnalyzer(use_spacy=False)

    simple_text = "I like dogs. Dogs are fun. I play with my dog."
    result = analyzer.analyze(simple_text)
    simple_complexity = result.complexity

    complex_text = """The epistemological ramifications of phenomenological
    investigations notwithstanding the ontological presuppositions inherent
    in hermeneutical methodologies."""
    result = analyzer.analyze(complex_text)
    complex_complexity = result.complexity

    assert complex_complexity > simple_complexity
    print(f"✓ Complexity test passed: simple={simple_complexity}, complex={complex_complexity}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Linguistic Analyzer Tests")
    print("Based on Chase Hughes' 6 Minute X-Ray & Ellipsis Manual")
    print("="*60 + "\n")

    tests = [
        test_vak_visual,
        test_vak_auditory,
        test_vak_kinesthetic,
        test_social_need_significance,
        test_social_need_approval,
        test_social_need_acceptance,
        test_social_need_intelligence,
        test_social_need_pity,
        test_social_need_power,
        test_decision_style_novelty,
        test_decision_style_social,
        test_decision_style_conformity,
        test_certainty_high,
        test_certainty_low,
        test_emotional_indicators,
        test_influence_patterns,
        test_stress_indicators,
        test_pronoun_analysis,
        test_complexity,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
