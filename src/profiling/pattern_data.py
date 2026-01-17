"""Pattern dictionaries for linguistic analysis based on Chase Hughes' methodologies.

This file contains the keyword and phrase patterns used to identify:
- VAK (Visual/Auditory/Kinesthetic) modalities
- Social needs (Significance, Approval, Acceptance, Intelligence, Pity, Power)
- Decision styles (Deviance, Novelty, Social, Conformity, Investment, Necessity)
- Communication patterns (certainty, time orientation)

Based on "6 Minute X-Ray" and "The Ellipsis Manual" by Chase Hughes.
"""

# =============================================================================
# VAK (Visual/Auditory/Kinesthetic) Modality Patterns
# =============================================================================

VAK_PATTERNS = {
    "visual": {
        "keywords": [
            "see", "look", "view", "show", "picture", "imagine", "clear",
            "bright", "colorful", "focus", "perspective", "appears", "visible",
            "vision", "envision", "illustrate", "reveal", "watch", "observe",
            "glimpse", "gaze", "stare", "glance", "peek", "scan", "inspect",
            "examine", "display", "appear", "seem", "image", "scene", "scenery",
            "sight", "vivid", "dim", "dark", "light", "shadow", "reflect",
            "mirror", "transparent", "opaque", "hazy", "foggy", "crystal",
        ],
        "phrases": [
            "see what i mean",
            "look at this",
            "picture this",
            "appears that",
            "show me",
            "i see your point",
            "looks like",
            "clear as day",
            "in my view",
            "the way i see it",
            "from my perspective",
            "i can imagine",
            "take a look",
            "let me show you",
            "it looks to me",
            "i see what you're saying",
            "get the picture",
            "paint a picture",
            "beyond a shadow of doubt",
            "in light of",
            "shed light on",
            "see eye to eye",
            "turn a blind eye",
            "bird's eye view",
            "tunnel vision",
            "in hindsight",
            "short sighted",
            "keep an eye on",
        ],
        "weight": 1.0,
    },
    "auditory": {
        "keywords": [
            "hear", "listen", "sound", "tell", "say", "ask", "talk", "speak",
            "voice", "tone", "loud", "quiet", "resonate", "harmony", "click",
            "ring", "echo", "noise", "silent", "mute", "whisper", "shout",
            "yell", "scream", "call", "announce", "pronounce", "articulate",
            "express", "verbal", "oral", "acoustic", "audible", "inaudible",
            "pitch", "volume", "rhythm", "melody", "tune", "tempo", "beat",
            "chime", "buzz", "hum", "murmur",
        ],
        "phrases": [
            "hear me out",
            "listen to this",
            "sounds like",
            "rings a bell",
            "tell me",
            "loud and clear",
            "doesn't sound right",
            "tune in",
            "word for word",
            "manner of speaking",
            "in a manner of speaking",
            "so to speak",
            "clearly stated",
            "unheard of",
            "outspoken",
            "hold your tongue",
            "give me your ear",
            "all ears",
            "music to my ears",
            "strikes a chord",
            "on the same wavelength",
            "tongue tied",
            "power of speech",
            "voiced an opinion",
            "call to mind",
        ],
        "weight": 1.0,
    },
    "kinesthetic": {
        "keywords": [
            "feel", "touch", "grasp", "hold", "handle", "solid", "heavy",
            "rough", "smooth", "warm", "cold", "move", "push", "pull", "grab",
            "carry", "weight", "pressure", "tension", "relax", "tight", "loose",
            "firm", "soft", "hard", "gentle", "harsh", "sharp", "dull",
            "sticky", "slippery", "wet", "dry", "comfort", "uncomfortable",
            "pain", "pleasure", "sensation", "texture", "grip", "squeeze",
            "stretch", "bend", "twist", "shake", "hit", "strike", "tap",
            "concrete", "tangible", "physical",
        ],
        "phrases": [
            "i feel you",
            "get a grip",
            "hold on",
            "touch base",
            "grasp the concept",
            "get in touch",
            "feels right",
            "hands on",
            "firm grasp",
            "come to grips with",
            "get a handle on",
            "heated argument",
            "cool down",
            "warm up to",
            "cold shoulder",
            "throw around ideas",
            "kick around",
            "pain in the neck",
            "pull some strings",
            "sharp as a tack",
            "smooth operator",
            "under pressure",
            "hang in there",
            "solid foundation",
            "get a feel for",
            "gut feeling",
            "thick skinned",
            "rub the wrong way",
        ],
        "weight": 1.0,
    },
}

# =============================================================================
# Social Needs Patterns (6 Categories from Chase Hughes)
# =============================================================================

SOCIAL_NEEDS_PATTERNS = {
    "significance": {
        "description": "Need to feel important, make a difference, stand out",
        "fear": "Being dismissed or mocked",
        "keywords": [
            "best", "first", "won", "achieved", "accomplished", "success",
            "leading", "top", "elite", "exclusive", "unique", "special",
            "exceptional", "outstanding", "remarkable", "impressive",
            "recognized", "awarded", "honored", "distinguished", "superior",
            "champion", "winner", "leader", "expert", "authority", "master",
            "pioneer", "innovator", "genius",
        ],
        "phrases": [
            "better than",
            "nobody else could",
            "i was the first",
            "i'm the best at",
            "i accomplished",
            "my achievement",
            "they recognized me",
            "i made it happen",
            "i'm known for",
            "i stand out",
            "i'm not like others",
            "people look up to me",
            "i was chosen",
            "they picked me",
            "i'm the one who",
            "without me",
            "thanks to me",
            "i did that",
            "my idea",
            "i came up with",
        ],
        "pronoun_pattern": {
            "pronouns": ["i", "me", "my", "mine", "myself"],
            "threshold": 0.08,  # If >8% of words are self-referential
        },
        "behavioral_markers": [
            "name dropping",
            "status references",
            "quantifying accomplishments",
            "competitive comparisons",
        ],
        "weight": 1.0,
    },
    "approval": {
        "description": "Need for validation, praise, being liked",
        "fear": "Rejection or disdain",
        "keywords": [
            "maybe", "perhaps", "possibly", "sorry", "apologize", "excuse",
            "hope", "okay", "alright", "fine", "please", "thank", "grateful",
            "appreciate",
        ],
        "phrases": [
            "i think maybe",
            "kind of",
            "sort of",
            "if that's okay",
            "i hope you don't mind",
            "is that good",
            "do you like it",
            "don't you think",
            "wouldn't you say",
            "i'm not sure but",
            "i could be wrong",
            "sorry if",
            "excuse me for",
            "i didn't mean to",
            "i was just trying to",
            "please don't",
            "i hope i'm not",
            "am i doing this right",
            "is this what you wanted",
            "does that make sense",
            "right?",
            "you know?",
            "i guess",
        ],
        "behavioral_markers": [
            "hedging language",
            "seeking validation",
            "apologetic tone",
            "minimizing self",
            "agreement seeking",
        ],
        "weight": 1.0,
    },
    "acceptance": {
        "description": "Need to belong, be included, be wanted",
        "fear": "Criticism or alienation",
        "keywords": [
            "everyone", "group", "team", "together", "belong", "member",
            "included", "part", "community", "family", "friends", "people",
            "others", "normal", "common", "typical", "usual", "standard",
        ],
        "phrases": [
            "everyone's doing it",
            "it's normal",
            "that's how we do it",
            "part of the team",
            "one of us",
            "we all",
            "just like everyone",
            "they all say",
            "most people",
            "nobody would",
            "left out",
            "not invited",
            "alone",
            "don't want to be different",
            "fit in",
            "go along with",
            "the rest of us",
            "in the group",
            "together we",
            "our group",
        ],
        "pronoun_pattern": {
            "pronouns": ["we", "us", "our", "ours", "ourselves"],
            "threshold": 0.06,  # If >6% are group-referential
        },
        "behavioral_markers": [
            "group identification",
            "conforming language",
            "fear of exclusion",
            "avoiding standing out",
        ],
        "weight": 1.0,
    },
    "intelligence": {
        "description": "Need to be perceived as smart, competent",
        "fear": "Being seen as unintelligent",
        "keywords": [
            "actually", "technically", "specifically", "precisely", "research",
            "study", "data", "evidence", "according", "analysis", "fact",
            "logic", "reason", "rational", "scientific", "academic",
            "educated", "degree", "university", "expert", "knowledge",
            "understand", "comprehend", "intellectual", "cognitive",
        ],
        "phrases": [
            "well actually",
            "technically speaking",
            "to be precise",
            "studies show",
            "research indicates",
            "according to",
            "the data shows",
            "from my understanding",
            "as i've researched",
            "let me explain",
            "what you don't understand",
            "it's more complex than",
            "the nuance is",
            "to clarify",
            "i've studied this",
            "in my experience as",
            "based on my knowledge",
            "that's not quite right",
            "what people don't realize",
            "fundamentally",
        ],
        "behavioral_markers": [
            "correcting others",
            "complex vocabulary",
            "over-explaining",
            "citing sources",
            "education references",
        ],
        "weight": 1.0,
    },
    "pity": {
        "description": "Need to be rescued, consoled, gain sympathy",
        "fear": "Being ignored or disbelieved",
        "keywords": [
            "hard", "difficult", "struggle", "suffer", "pain", "hurt",
            "problem", "issue", "challenge", "obstacle", "unfortunate",
            "unlucky", "unfair", "terrible", "awful", "horrible", "disaster",
            "nightmare", "burden", "overwhelmed", "exhausted", "stressed",
            "anxious", "worried", "scared", "afraid", "helpless", "hopeless",
        ],
        "phrases": [
            "always happens to me",
            "nobody understands",
            "so hard for me",
            "can't catch a break",
            "i'm terrible at",
            "i never succeed",
            "you don't know how hard",
            "nobody cares",
            "just lucky",
            "not that good",
            "poor me",
            "why me",
            "it's not fair",
            "i've been through so much",
            "nobody has it as bad",
            "if only you knew",
            "i can't do anything right",
            "nothing ever works out",
            "story of my life",
            "i try so hard but",
        ],
        "behavioral_markers": [
            "victim language",
            "highlighting struggles",
            "self-deprecation",
            "sympathy seeking",
            "minimizing success",
        ],
        "weight": 1.0,
    },
    "power": {
        "description": "Need to feel in control, superior, influential",
        "fear": "Being disrespected or challenged",
        "keywords": [
            "control", "command", "lead", "direct", "decide", "authority",
            "power", "strong", "tough", "dominant", "superior", "charge",
            "boss", "manage", "handle", "definitely", "absolutely", "certainly",
            "obviously", "clearly", "must", "will", "demand", "require",
            "insist", "force", "weak", "pathetic", "loser",
        ],
        "phrases": [
            "you need to",
            "do this",
            "listen to me",
            "i'm telling you",
            "i don't need",
            "i can handle",
            "i'm in charge",
            "that's nothing",
            "not a big deal",
            "get over it",
            "man up",
            "stop complaining",
            "you should",
            "you must",
            "i demand",
            "i insist",
            "there's no question",
            "it's simple",
            "just do it",
            "that's weak",
            "don't be pathetic",
            "i'm right",
            "trust me",
            "believe me",
            "i know best",
        ],
        "behavioral_markers": [
            "commanding language",
            "absolute certainty",
            "dominance display",
            "dismissive of others",
            "refusing vulnerability",
        ],
        "weight": 1.0,
    },
}

# =============================================================================
# Decision Style Patterns (6 Styles from Chase Hughes)
# =============================================================================

DECISION_STYLE_PATTERNS = {
    "deviance": {
        "description": "Goes against norms, values independence and rebellion",
        "keywords": [
            "different", "unique", "rebel", "against", "unconventional",
            "alternative", "independent", "own", "break", "rules", "norm",
            "mainstream", "system", "conformist", "sheep",
        ],
        "phrases": [
            "i don't care what they think",
            "i do my own thing",
            "break the rules",
            "against the grain",
            "not like everyone else",
            "think outside the box",
            "who cares what others think",
            "i'll do it my way",
            "conventional thinking",
            "the system is",
            "i'm not a follower",
            "forget what they say",
            "screw the rules",
        ],
        "weight": 1.0,
    },
    "novelty": {
        "description": "Seeks new, exciting, different experiences",
        "keywords": [
            "new", "exciting", "different", "innovative", "fresh", "novel",
            "cutting-edge", "latest", "modern", "adventure", "discover",
            "explore", "experiment", "try", "experience", "first",
        ],
        "phrases": [
            "never tried before",
            "something new",
            "sounds exciting",
            "different approach",
            "innovative solution",
            "let's try something",
            "i want to explore",
            "this is fresh",
            "cutting edge",
            "the latest",
            "brand new",
            "haven't done this before",
            "always looking for",
            "love to discover",
        ],
        "weight": 1.0,
    },
    "social": {
        "description": "Follows what others do, values social proof",
        "keywords": [
            "everyone", "popular", "trending", "recommended", "reviews",
            "rating", "famous", "viral", "liked", "followed", "people",
            "friends", "others", "they", "said",
        ],
        "phrases": [
            "everyone's doing it",
            "really popular",
            "trending right now",
            "they said it's good",
            "highly recommended",
            "great reviews",
            "my friends all",
            "people say",
            "i heard that",
            "it's what everyone uses",
            "best seller",
            "top rated",
            "most people choose",
            "social proof",
        ],
        "weight": 1.0,
    },
    "conformity": {
        "description": "Follows tradition, rules, expectations",
        "keywords": [
            "traditional", "proven", "standard", "established", "classic",
            "conventional", "proper", "right", "correct", "should", "supposed",
            "expected", "normal", "usual", "safe", "reliable",
        ],
        "phrases": [
            "this is how it's done",
            "the right way",
            "proven method",
            "by the book",
            "standard practice",
            "tried and true",
            "the proper way",
            "that's how it's always been",
            "traditional approach",
            "play it safe",
            "follow the rules",
            "what's expected",
            "the established way",
            "conventional wisdom",
        ],
        "weight": 1.0,
    },
    "investment": {
        "description": "Based on sunk costs and prior investment",
        "keywords": [
            "invested", "spent", "committed", "dedicated", "effort", "time",
            "money", "years", "work", "built", "established", "already",
        ],
        "phrases": [
            "already invested",
            "put so much into",
            "can't quit now",
            "too far in",
            "after all this time",
            "sunk cost",
            "i've already",
            "we've come this far",
            "all the work we've done",
            "years of effort",
            "can't turn back now",
            "waste all that",
            "throw away",
            "start over",
        ],
        "weight": 1.0,
    },
    "necessity": {
        "description": "Based on survival and practical needs",
        "keywords": [
            "need", "must", "have to", "necessary", "essential", "required",
            "survival", "basic", "fundamental", "critical", "urgent",
            "important", "vital", "crucial",
        ],
        "phrases": [
            "have to do this",
            "no choice",
            "it's necessary",
            "we need to",
            "it's essential",
            "for survival",
            "can't live without",
            "it's critical",
            "must have",
            "absolutely need",
            "not optional",
            "required for",
            "have no other option",
            "matter of necessity",
        ],
        "weight": 1.0,
    },
}

# =============================================================================
# Communication Pattern Markers
# =============================================================================

CERTAINTY_MARKERS = {
    "high_certainty": [
        "always", "never", "definitely", "absolutely", "certainly", "surely",
        "obviously", "clearly", "undoubtedly", "without a doubt", "no question",
        "guaranteed", "positive", "certain", "confident", "know for sure",
        "100 percent", "completely", "totally", "entirely", "unquestionably",
        "indisputably", "undeniably", "without fail",
    ],
    "low_certainty": [
        "maybe", "perhaps", "possibly", "might", "could", "probably",
        "likely", "unlikely", "i think", "i guess", "i suppose", "i believe",
        "i feel like", "it seems", "appears to be", "sort of", "kind of",
        "somewhat", "fairly", "rather", "quite", "not sure", "uncertain",
        "i'm not certain", "hard to say", "difficult to know",
    ],
}

TIME_ORIENTATION_MARKERS = {
    "past": [
        "was", "were", "had", "did", "used to", "back then", "in the past",
        "previously", "before", "ago", "once", "formerly", "earlier",
        "remember when", "those days", "that time", "looking back",
        "historically", "tradition", "heritage", "legacy",
    ],
    "present": [
        "am", "is", "are", "now", "currently", "today", "right now",
        "at this moment", "presently", "these days", "lately", "recently",
        "this moment", "as we speak", "ongoing", "existing",
    ],
    "future": [
        "will", "going to", "plan to", "intend to", "soon", "next",
        "tomorrow", "upcoming", "future", "eventually", "someday", "one day",
        "looking forward", "ahead", "down the road", "in time", "later",
        "prospect", "potential", "possibility", "aspire", "hope to",
    ],
}

# =============================================================================
# Pronoun Categories for Analysis
# =============================================================================

PRONOUN_CATEGORIES = {
    "self": ["i", "me", "my", "mine", "myself"],
    "group": ["we", "us", "our", "ours", "ourselves"],
    "other": ["you", "your", "yours", "yourself", "yourselves"],
    "third": ["he", "she", "they", "him", "her", "them", "his", "hers", "their"],
}

# =============================================================================
# Complexity Indicators
# =============================================================================

COMPLEX_VOCABULARY_INDICATORS = [
    "notwithstanding", "nevertheless", "furthermore", "consequently",
    "subsequently", "heretofore", "wherein", "whereby", "inasmuch",
    "insofar", "henceforth", "aforementioned", "aforementioned",
    "paradigm", "ubiquitous", "ephemeral", "dichotomy", "juxtaposition",
    "quintessential", "synergy", "holistic", "proactive", "leverage",
]

# =============================================================================
# Emotional State Indicators (from The Ellipsis Manual)
# =============================================================================

EMOTIONAL_INDICATORS = {
    "anxiety": {
        "keywords": [
            "worried", "anxious", "nervous", "scared", "afraid", "fear",
            "panic", "stressed", "tense", "uneasy", "apprehensive",
            "concerned", "dread", "terrified", "frightened",
        ],
        "phrases": [
            "what if", "i'm worried that", "i'm afraid", "i can't stop thinking",
            "keeps me up at night", "i'm stressed about", "makes me nervous",
            "i'm concerned that", "i fear that", "scares me",
        ],
    },
    "anger": {
        "keywords": [
            "angry", "mad", "furious", "annoyed", "irritated", "frustrated",
            "pissed", "livid", "outraged", "enraged", "resentful", "bitter",
            "hostile", "aggravated", "infuriated",
        ],
        "phrases": [
            "makes me angry", "so annoying", "i can't stand", "drives me crazy",
            "sick of", "tired of", "had enough", "fed up", "pisses me off",
            "i hate when", "so frustrating", "really bothers me",
        ],
    },
    "sadness": {
        "keywords": [
            "sad", "depressed", "unhappy", "miserable", "heartbroken",
            "devastated", "disappointed", "hurt", "lonely", "empty",
            "hopeless", "despair", "grief", "sorrow", "melancholy",
        ],
        "phrases": [
            "i feel sad", "makes me sad", "breaks my heart", "so disappointed",
            "i miss", "i feel empty", "i feel alone", "nothing matters",
            "what's the point", "i feel lost", "can't get over",
        ],
    },
    "joy": {
        "keywords": [
            "happy", "excited", "thrilled", "delighted", "ecstatic",
            "overjoyed", "elated", "cheerful", "grateful", "blessed",
            "fortunate", "lucky", "wonderful", "amazing", "fantastic",
        ],
        "phrases": [
            "so happy", "i love", "makes me happy", "i'm excited",
            "can't wait", "looking forward", "so grateful", "so lucky",
            "best thing ever", "couldn't be happier", "dream come true",
        ],
    },
}

# =============================================================================
# Rapport & Trust Indicators
# =============================================================================

RAPPORT_INDICATORS = {
    "building_rapport": {
        "phrases": [
            "i understand", "that makes sense", "i hear you", "i get it",
            "i can see that", "you're right", "good point", "i agree",
            "absolutely", "exactly", "same here", "me too", "tell me more",
            "that's interesting", "how did that make you feel",
        ],
    },
    "distancing": {
        "phrases": [
            "i don't know", "not my problem", "whatever", "i don't care",
            "that's your issue", "not really", "i suppose", "if you say so",
            "i guess", "sure", "fine", "okay then", "anyway",
        ],
    },
    "mirroring_language": {
        "description": "When someone mirrors your words back, it indicates rapport",
        "detection_note": "Compare speaker utterances for repeated phrases",
    },
}

# =============================================================================
# Deception Indicators (Linguistic Only - from The Ellipsis Manual)
# Note: These are INDICATORS only, not proof of deception
# =============================================================================

LINGUISTIC_STRESS_INDICATORS = {
    "distancing_language": {
        "description": "Using language that creates psychological distance",
        "phrases": [
            "that woman", "that man", "that person", "the situation",
            "those people", "it happened", "things occurred",
        ],
        "note": "Avoiding names/pronouns when referring to people they know well",
    },
    "qualifier_overuse": {
        "description": "Excessive use of qualifiers may indicate uncertainty or evasion",
        "keywords": [
            "honestly", "truthfully", "frankly", "to be honest",
            "to tell you the truth", "believe me", "trust me",
            "i swear", "literally", "basically", "essentially",
        ],
    },
    "tense_inconsistency": {
        "description": "Switching between past and present tense unexpectedly",
        "detection_note": "Track verb tense changes within narratives",
    },
    "detail_imbalance": {
        "description": "Too much or too little detail in specific areas",
        "detection_note": "Compare detail density across narrative segments",
    },
    "bolstering_statements": {
        "description": "Statements that attempt to appear more credible",
        "phrases": [
            "why would i lie", "i have no reason to lie", "i'm being honest",
            "you can ask anyone", "everyone knows", "it's the truth",
            "i promise", "i wouldn't make this up", "you have to believe me",
        ],
    },
}

# =============================================================================
# Question Types (for analyzing communication style)
# =============================================================================

QUESTION_TYPES = {
    "open": {
        "starters": ["what", "how", "why", "tell me", "describe", "explain"],
        "description": "Encourages detailed responses",
    },
    "closed": {
        "starters": ["did", "do", "does", "is", "are", "was", "were", "can", "will", "have"],
        "description": "Yes/no or short factual answers",
    },
    "leading": {
        "phrases": [
            "don't you think", "wouldn't you say", "isn't it true",
            "you agree that", "surely you", "obviously you",
        ],
        "description": "Suggests the expected answer",
    },
    "rhetorical": {
        "phrases": [
            "who cares", "what's the point", "why bother", "how could anyone",
            "who would", "what kind of person",
        ],
        "description": "Not expecting an answer",
    },
}

# =============================================================================
# Value Indicators (expanded from underlying values)
# =============================================================================

VALUE_INDICATORS = {
    "security_vs_risk": {
        "security": [
            "safe", "secure", "stable", "reliable", "guaranteed", "protected",
            "insurance", "backup", "plan b", "safety net", "careful", "cautious",
        ],
        "risk": [
            "adventure", "chance", "gamble", "risk", "bet", "leap", "bold",
            "daring", "fearless", "spontaneous", "impulsive", "wing it",
        ],
    },
    "tradition_vs_innovation": {
        "tradition": [
            "traditional", "classic", "heritage", "history", "ancestors",
            "old school", "the way it's always been", "time-tested", "proven",
        ],
        "innovation": [
            "new", "innovative", "modern", "cutting-edge", "revolutionary",
            "disrupt", "change", "evolve", "progress", "future",
        ],
    },
    "independence_vs_community": {
        "independence": [
            "alone", "myself", "independent", "self-reliant", "my own",
            "don't need", "solo", "freedom", "autonomous", "individual",
        ],
        "community": [
            "together", "team", "group", "community", "family", "friends",
            "support", "help", "collaborate", "share", "collective",
        ],
    },
    "achievement_vs_relationships": {
        "achievement": [
            "success", "goal", "accomplish", "achieve", "win", "career",
            "ambition", "drive", "results", "performance", "compete",
        ],
        "relationships": [
            "love", "care", "connection", "bond", "relationship", "friend",
            "family", "trust", "loyalty", "support", "together",
        ],
    },
    "logic_vs_emotion": {
        "logic": [
            "think", "analyze", "reason", "logic", "rational", "facts",
            "data", "evidence", "objective", "practical", "realistic",
        ],
        "emotion": [
            "feel", "heart", "gut", "intuition", "passion", "love",
            "care", "sense", "vibe", "energy", "spirit",
        ],
    },
}

# =============================================================================
# Speech Patterns (filler words, pace indicators)
# =============================================================================

SPEECH_PATTERNS = {
    "filler_words": [
        "um", "uh", "like", "you know", "basically", "literally",
        "actually", "so", "well", "i mean", "right", "okay so",
        "anyway", "whatever", "kind of", "sort of",
    ],
    "hesitation_markers": [
        "let me think", "how do i say this", "what's the word",
        "i'm trying to", "bear with me", "one second", "wait",
    ],
    "confidence_markers": [
        "without a doubt", "absolutely", "definitely", "certainly",
        "no question", "for sure", "100 percent", "guaranteed",
    ],
    "uncertainty_markers": [
        "i think", "maybe", "possibly", "perhaps", "i guess",
        "i suppose", "not sure", "might be", "could be",
    ],
}

# =============================================================================
# Influence & Persuasion Patterns (from The Ellipsis Manual)
# =============================================================================

# =============================================================================
# Deception & Political Doublespeak Patterns
# Based on research on linguistic markers of deception and evasion
# =============================================================================

DECEPTION_PATTERNS = {
    "hedging": {
        "description": "Avoiding commitment to statements",
        "keywords": [
            "believe", "think", "suppose", "guess", "assume", "perhaps",
            "possibly", "probably", "might", "could", "may", "seems",
        ],
        "phrases": [
            "i believe", "i think", "to my knowledge", "as far as i know",
            "i don't recall", "i don't remember", "not to my knowledge",
            "i'm not aware", "i can't say for certain", "to the best of my recollection",
            "i have no memory of", "it's possible", "it's conceivable",
            "at this point in time", "at this juncture",
        ],
        "weight": 1.5,
    },
    "distancing": {
        "description": "Creating psychological distance from statements",
        "keywords": [
            "that", "those", "the", "one", "someone", "something",
        ],
        "phrases": [
            "that person", "that individual", "the situation", "those people",
            "one might say", "some would argue", "it has been said",
            "mistakes were made", "actions were taken", "decisions were made",
            "it was determined", "the decision was made", "it happened",
            "things occurred", "events transpired",
        ],
        "weight": 1.5,
        "note": "Passive voice and nominalization to obscure agency",
    },
    "non_answers": {
        "description": "Appearing to answer without actually answering",
        "phrases": [
            "that's a great question", "i'm glad you asked", "let me be clear",
            "what i can tell you is", "what i will say is", "look",
            "the fact of the matter is", "the reality is", "here's the thing",
            "at the end of the day", "when all is said and done",
            "the bottom line is", "in terms of", "with respect to",
            "as it relates to", "moving forward", "going forward",
        ],
        "weight": 1.2,
    },
    "weasel_words": {
        "description": "Vague attributions and unverifiable claims",
        "phrases": [
            "some people say", "many believe", "it's been said",
            "sources indicate", "reports suggest", "experts claim",
            "studies show", "research indicates", "they say",
            "people are saying", "i've heard", "there are those who",
            "some would argue", "critics say", "supporters believe",
            "many people", "a lot of people", "everybody knows",
            "nobody thinks", "most people agree",
        ],
        "weight": 1.3,
    },
    "false_dichotomy": {
        "description": "Presenting only two options when more exist",
        "phrases": [
            "either you're with us or against us", "you're either for or against",
            "there are only two choices", "it's simple", "it's black and white",
            "you can either", "the only options are", "there's no middle ground",
            "pick a side", "which side are you on",
        ],
        "weight": 1.4,
    },
    "emotional_manipulation": {
        "description": "Appeals to emotion over substance",
        "phrases": [
            "think of the children", "our brave", "hard-working families",
            "the american people", "real americans", "ordinary people",
            "common sense", "kitchen table issues", "main street",
            "taxpayer dollars", "your tax dollars", "our freedom",
            "our way of life", "under attack", "fighting for",
            "standing up for", "we cannot allow", "we must never",
        ],
        "weight": 1.2,
    },
    "blame_shifting": {
        "description": "Deflecting responsibility to others",
        "phrases": [
            "the previous administration", "my predecessor", "they didn't",
            "we inherited", "the other side", "our opponents",
            "the media", "fake news", "the establishment", "the elites",
            "special interests", "the system", "circumstances beyond",
            "factors outside our control", "we were given",
        ],
        "weight": 1.4,
    },
    "future_faking": {
        "description": "Vague promises without commitment",
        "phrases": [
            "we're looking into", "we're working on", "we're exploring",
            "in the coming weeks", "in the near future", "very soon",
            "stay tuned", "more to come", "we'll see", "we're committed to",
            "we're dedicated to", "it's a priority", "on the agenda",
            "under consideration", "being reviewed", "in due time",
        ],
        "weight": 1.2,
    },
    "tautology": {
        "description": "Circular statements that say nothing",
        "phrases": [
            "it is what it is", "at the end of the day it's",
            "the facts are the facts", "rules are rules", "fair is fair",
            "business is business", "boys will be boys", "war is war",
            "a deal is a deal", "enough is enough",
        ],
        "weight": 1.3,
    },
    "excessive_certainty": {
        "description": "Overclaiming certainty on uncertain matters",
        "phrases": [
            "100 percent", "absolutely certain", "no question whatsoever",
            "without any doubt", "guaranteed", "i can assure you",
            "make no mistake", "let me be perfectly clear", "mark my words",
            "you can count on", "i promise you", "believe me",
            "trust me", "honestly", "truthfully", "frankly",
            "to be honest", "to tell you the truth", "i swear",
        ],
        "weight": 1.3,
        "note": "Ironically, over-emphasis on honesty often indicates the opposite",
    },
    "gaslighting": {
        "description": "Making others question their reality",
        "phrases": [
            "that never happened", "you're misremembering", "that's not what i said",
            "you're taking it out of context", "that's not what i meant",
            "you're being too sensitive", "you're overreacting",
            "i never said that", "you're confused", "that's ridiculous",
            "that's absurd", "where did you hear that",
        ],
        "weight": 1.5,
    },
}

# Politician-specific speech patterns
POLITICIAN_PATTERNS = {
    "talking_points": {
        "description": "Rehearsed, scripted responses",
        "indicators": [
            "let me be clear", "make no mistake", "here's what i know",
            "what the american people want", "my record shows",
            "i have always believed", "throughout my career",
            "i've always said", "as i've said before", "my position has always been",
        ],
    },
    "pivot_phrases": {
        "description": "Phrases used to change the subject",
        "phrases": [
            "what's really important is", "the real issue here is",
            "what we should be talking about", "let me tell you what",
            "but let's focus on", "the bigger picture is",
            "what matters is", "the real question is",
        ],
    },
    "empty_platitudes": {
        "description": "Meaningless positive statements",
        "phrases": [
            "we're all in this together", "unity", "bringing people together",
            "bipartisan solutions", "common ground", "reach across the aisle",
            "work together", "find solutions", "move forward together",
            "bright future", "better tomorrow", "american dream",
        ],
    },
}

INFLUENCE_PATTERNS = {
    "reciprocity": {
        "description": "Creating obligation through giving",
        "phrases": [
            "i did this for you", "after all i've done", "i helped you",
            "remember when i", "you owe me", "i gave you",
        ],
    },
    "social_proof": {
        "description": "Using others' behavior as evidence",
        "phrases": [
            "everyone's doing it", "most people", "studies show",
            "experts say", "it's common", "normal people", "the majority",
        ],
    },
    "authority": {
        "description": "Appealing to expertise or status",
        "phrases": [
            "as an expert", "in my professional opinion", "i've been doing this",
            "trust me i know", "with my experience", "according to experts",
        ],
    },
    "scarcity": {
        "description": "Creating urgency through limited availability",
        "phrases": [
            "limited time", "only a few left", "last chance", "rare opportunity",
            "won't last", "exclusive", "once in a lifetime", "now or never",
        ],
    },
    "commitment": {
        "description": "Building on prior commitments",
        "phrases": [
            "you said you would", "you promised", "you agreed",
            "we talked about this", "you committed to", "your word",
        ],
    },
    "liking": {
        "description": "Building rapport to influence",
        "phrases": [
            "we're alike", "you and i", "we both", "same as you",
            "i like you", "we have so much in common", "just like me",
        ],
    },
}
