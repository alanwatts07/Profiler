#!/usr/bin/env python3
"""Demo script showing linguistic analysis capabilities.

This demonstrates the Chase Hughes profiling methodology in action,
showing how different speech patterns reveal social needs, VAK modalities,
and decision styles.

Run from Profiler directory: python examples/demo_analysis.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import pattern data
exec(open("src/profiling/pattern_data.py").read())

import re
from dataclasses import dataclass, field


@dataclass
class Result:
    vak_scores: dict = field(default_factory=dict)
    social_needs: dict = field(default_factory=dict)
    decision_styles: list = field(default_factory=list)
    communication: dict = field(default_factory=dict)
    word_count: int = 0


def analyze(text):
    """Simple analysis for demo."""
    result = Result()
    text_lower = text.lower()
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    result.word_count = len(words)
    words_lower = [w.lower() for w in words]

    # VAK
    vak = {"visual": 0, "auditory": 0, "kinesthetic": 0}
    for modality, data in VAK_PATTERNS.items():
        for w in words_lower:
            if w in data["keywords"]:
                vak[modality] += 1
        for phrase in data.get("phrases", []):
            if phrase in text_lower:
                vak[modality] += 2
    total = sum(vak.values()) or 1
    result.vak_scores = {k: round(v/total, 2) for k, v in vak.items()}

    # Social needs
    needs = {}
    for need, data in SOCIAL_NEEDS_PATTERNS.items():
        score = 0
        for w in words_lower:
            if w in data.get("keywords", []):
                score += 1
        for phrase in data.get("phrases", []):
            if phrase in text_lower:
                score += 2
        needs[need] = score
    max_need = max(needs.values()) or 1
    result.social_needs = {k: round(v/max_need, 2) for k, v in needs.items()}

    # Decision styles
    for style, data in DECISION_STYLE_PATTERNS.items():
        for phrase in data.get("phrases", []):
            if phrase in text_lower:
                if style not in result.decision_styles:
                    result.decision_styles.append(style)

    # Communication
    high = sum(1 for m in CERTAINTY_MARKERS["high_certainty"] if m in text_lower)
    low = sum(1 for m in CERTAINTY_MARKERS["low_certainty"] if m in text_lower)
    result.communication["certainty"] = high / (high + low) if (high + low) else 0.5

    return result


def print_analysis(name, text, result):
    """Pretty print analysis results."""
    print(f"\n{'='*60}")
    print(f"PROFILE: {name}")
    print(f"{'='*60}")
    print(f"\nText ({result.word_count} words):")
    print(f'  "{text[:100]}..."' if len(text) > 100 else f'  "{text}"')

    # VAK
    print(f"\nVAK Modality:")
    dominant = max(result.vak_scores, key=result.vak_scores.get)
    for m, s in result.vak_scores.items():
        marker = "→" if m == dominant else " "
        bar = "█" * int(s * 10)
        print(f"  {marker} {m:12}: {bar} {s:.2f}")

    # Social Needs
    print(f"\nSocial Needs:")
    sorted_needs = sorted(result.social_needs.items(), key=lambda x: x[1], reverse=True)
    for need, score in sorted_needs[:3]:
        bar = "█" * int(score * 10)
        print(f"  → {need:12}: {bar} {score:.2f}")

    # Decision Style
    if result.decision_styles:
        print(f"\nDecision Style: {', '.join(result.decision_styles)}")

    # Communication
    cert = result.communication.get("certainty", 0.5)
    cert_label = "High" if cert > 0.6 else "Low" if cert < 0.4 else "Moderate"
    print(f"\nCertainty: {cert_label} ({cert:.2f})")


# =============================================================================
# SAMPLE PROFILES
# =============================================================================

SAMPLES = {
    "Significance Seeker": """
        I was the first person in my company to achieve this certification.
        Nobody else could have done what I did. They recognized me at the
        annual conference and I won the top performer award. My approach is
        unique - I'm known for being the best at solving these problems.
        I accomplished more in one year than most people do in five.
    """,

    "Approval Seeker": """
        I think maybe this approach could work? Sorry if I'm overstepping.
        I hope you don't mind me suggesting this. Is that okay with you?
        Do you like the direction I'm going? I'm not sure but I could be
        wrong about this. Does that make sense? I was just trying to help.
        I guess we could try something else if you prefer?
    """,

    "Acceptance Seeker": """
        We should all work together on this as a team. Everyone's doing it
        this way and it's completely normal. We belong to this community and
        that's how we do things here. Just like everyone else, we go along
        with the rest of the group. Our team always sticks together.
    """,

    "Intelligence Displayer": """
        Well actually, technically speaking, the research clearly indicates
        otherwise. Studies show that according to the empirical data, what
        most people don't understand is that it's far more complex than that.
        Let me explain the nuance here. From my academic background and years
        of studying this topic, I can tell you precisely what the evidence shows.
    """,

    "Pity Seeker": """
        This always happens to me, nobody understands how hard it is.
        I can't catch a break no matter what I do. You don't know how
        difficult this has been for me. It's not fair, nothing ever works
        out. Story of my life really. I try so hard but things just keep
        going wrong. Why does this always happen to me?
    """,

    "Power Displayer": """
        Listen to me, you need to do this my way. I'm telling you how this
        works and I'm in charge here. Trust me, I know best. Get over it,
        it's not a big deal. You must follow my instructions exactly.
        I don't need anyone's help, I can handle everything myself.
        That's weak thinking - just do what I say.
    """,

    "Visual Processor": """
        Let me show you what I mean. Can you see how this works? Picture
        this scenario in your mind. It looks like we have a clear path
        forward. From my perspective, this appears to be the right approach.
        I see your point completely. Let me illustrate the concept visually.
    """,

    "Auditory Processor": """
        Listen to what I'm saying here. Does that sound right to you?
        Tell me more about your thoughts. It rings a bell from our
        previous discussion. I hear what you're telling me loud and clear.
        That doesn't sound quite right though. Let's talk through this.
    """,

    "Kinesthetic Processor": """
        I feel like this is the right direction. Let me get a grip on
        the situation first. It feels solid to me. Can we touch base on
        this tomorrow? I need to get a handle on the details. My gut
        feeling is that we should move forward. Let's get in touch soon.
    """,

    "Novelty Decision Maker": """
        I've never tried this before and it sounds exciting! Let's explore
        something completely new and different. This innovative approach is
        cutting edge. I want to discover fresh possibilities and experiment
        with the latest methods. Something brand new is exactly what we need.
    """,

    "Social Proof Decision Maker": """
        Everyone's doing it this way, it's really popular right now.
        Trending across the industry. They said it's good and highly
        recommended. My friends all use this approach. The reviews are
        excellent and most people choose this option. It's the best seller.
    """,

    "Conformity Decision Maker": """
        This is how it's always been done. The traditional approach is
        proven and reliable. By the book, standard practice. That's the
        proper way to handle it. We should follow the established rules
        and stick with the conventional method. Play it safe.
    """,
}


def main():
    print("\n" + "="*60)
    print("CHASE HUGHES LINGUISTIC PROFILING DEMO")
    print("Based on '6 Minute X-Ray' & 'The Ellipsis Manual'")
    print("="*60)

    for name, text in SAMPLES.items():
        result = analyze(text.strip())
        print_analysis(name, text.strip(), result)

    print("\n" + "="*60)
    print("END OF DEMO")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
