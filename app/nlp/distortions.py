"""
Cognitive distortion detection: spaCy for linguistic parsing (negation,
lemmatization, sentence splitting) + our own hand-written rule set on top.

This is OUR engineering, not a pretrained model. It is intentionally a
simple, explainable rule set rather than a classifier: it needs to be
defensible sentence-by-sentence in the viva, and the space of clearly
worded distortion cues is small enough that rules generalize reasonably
well for a first version.

Five distortions covered in Phase 1: all-or-nothing thinking, fortune
telling, self-labeling, catastrophizing, overgeneralization. Phase 2 adds
ten more (mind reading, mental filter, disqualifying the positive,
magnification/minimization, emotional reasoning, should statements,
personalization, blaming, comparison, control fallacy), drawn from the
standard CBT distortion taxonomy (Burns, D. D. (1980). Feeling Good: The
New Mood Therapy. William Morrow.), for 15 total.
"""

import re
import spacy

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# Each rule: (distortion_name, list of regex patterns, human explanation)
_RULES = [
    (
        "all_or_nothing",
        [
            r"\balways\b", r"\bnever\b", r"\bevery\s+(single\s+)?time\b",
            r"\bcompletely\s+(fail|failed|failing|ruined|useless)\b",
            r"\btotal(ly)?\s+(disaster|failure)\b",
            r"\beither\b.*\bor\b",
        ],
        "Seeing a situation in only two extreme categories, with no middle ground.",
    ),
    (
        "fortune_telling",
        [
            r"\bit'?s\s+going\s+to\s+(fail|go\s+wrong|be\s+a\s+disaster)\b",
            r"\bi\s+(know|just\s+know)\s+.*\s+(will|won'?t)\b",
            r"\bthere'?s\s+no\s+way\s+(this|it|i)\s+(will|can|could)\b",
            r"\bi'?m\s+going\s+to\s+(fail|lose|screw\s+up)\b",
            r"\bwill\s+never\s+work\b",
        ],
        "Predicting a negative future as if it were already a fact.",
    ),
    (
        "self_labeling",
        [
            r"\bi'?m\s+(such\s+a\s+|a\s+|an\s+)?(failure|loser|idiot|stupid|worthless|useless|fraud|impostor)\b",
            r"\bi'?m\s+not\s+(good|smart|talented)\s+enough\b",
            r"\bi'?m\s+broken\b",
        ],
        "Attaching a harsh, global label to yourself based on one event or trait.",
    ),
    (
        "catastrophizing",
        [
            r"\bthis\s+is\s+(a\s+)?(catastrophe|disaster|the\s+end)\b",
            r"\beverything\s+is\s+(ruined|falling\s+apart|over)\b",
            r"\bi\s+can'?t\s+(handle|cope\s+with|survive)\s+this\b",
            r"\bworst\s+(thing|case|possible)\b",
            r"\bit'?s\s+all\s+over\b",
        ],
        "Assuming the worst possible outcome and treating it as unbearable.",
    ),
    (
        "overgeneralization",
        [
            r"\bthis\s+always\s+happens\b",
            r"\bnothing\s+ever\s+(works|goes\s+right)\b",
            r"\beveryone\s+(always|thinks|hates)\b",
            r"\bi\s+can'?t\s+do\s+anything\s+right\b",
            r"\bno\s+one\s+(ever|will)\b",
        ],
        "Drawing a broad, permanent conclusion from a single event.",
    ),
    (
        "mind_reading",
        [
            r"\bthey\s+(must\s+)?(think|believe)\s+i'?m\b",
            r"\bhe\s+(must\s+)?(thinks?|believes?)\s+i'?m\b",
            r"\bshe\s+(must\s+)?(thinks?|believes?)\s+i'?m\b",
            r"\beveryone\s+(is\s+)?(judging|laughing\s+at)\s+me\b",
            r"\bi\s+can\s+tell\s+(they|he|she)\s+(hates?|doesn'?t\s+like)\s+me\b",
        ],
        "Assuming you know what others are thinking about you without evidence.",
    ),
    (
        "mental_filter",
        [
            r"\ball\s+i\s+(can\s+)?(see|think\s+about)\s+is\s+(the\s+)?(bad|negative|wrong)\b",
            r"\bthe\s+only\s+thing\s+that\s+matters\s+is\s+(the\s+)?(one\s+)?(mistake|failure)\b",
            r"\bnothing\s+good\s+(happened|counts)\b",
            r"\bignor\w*\s+(everything|all)\s+(good|positive)\b",
        ],
        "Dwelling on a single negative detail while filtering out everything positive.",
    ),
    (
        "disqualifying_positive",
        [
            r"\bthat\s+doesn'?t\s+count\b",
            r"\bit\s+was\s+just\s+luck\b",
            r"\banyone\s+(could|would)\s+have\s+done\s+(that|it)\b",
            r"\bi\s+got\s+lucky,?\s+that'?s\s+all\b",
        ],
        "Dismissing positive experiences by insisting they don't count.",
    ),
    (
        "magnification_minimization",
        [
            r"\bit'?s\s+a\s+huge\s+deal\b",
            r"\bthis\s+is\s+(way\s+)?blown\s+out\s+of\s+proportion\b",
            r"\bit'?s\s+nothing,?\s+i\s+guess\b",
            r"\bmy\s+(achievement|success)\s+(doesn'?t|is\s+not)\s+(matter|important)\b",
        ],
        "Either blowing a problem out of proportion or shrinking your own achievements.",
    ),
    (
        "emotional_reasoning",
        [
            r"\bi\s+feel\s+(like\s+)?(a\s+)?(failure|stupid|worthless|useless)\s*,?\s+so\s+i\s+must\s+be\b",
            r"\bi\s+feel\s+it,?\s+so\s+it\s+must\s+be\s+true\b",
            r"\bi\s+feel\s+(anxious|scared),?\s+so\s+something\s+bad\s+(must|will)\s+happen\b",
        ],
        "Assuming that because you feel a certain way, it must reflect objective reality.",
    ),
    (
        "should_statements",
        [
            r"\bi\s+should\s+(have\s+)?(be|been|do|done)\b",
            r"\bi\s+must\s+(be|always|never)\b",
            r"\bi\s+ought\s+to\s+(be|have)\b",
            r"\bthey\s+should\s+(have\s+)?(know|understand)\b",
        ],
        "Holding yourself or others to rigid 'should'/'must' rules that create unnecessary guilt or frustration.",
    ),
    (
        "personalization",
        [
            r"\bit'?s\s+(all\s+)?my\s+fault\b",
            r"\bi\s+caused\s+(this|it|everything)\b",
            r"\bif\s+(only\s+)?i\s+had\s+(done|been)\b.*\bwouldn'?t\s+have\s+happened\b",
            r"\bbecause\s+of\s+me,?\s+(everyone|everything)\b",
        ],
        "Taking responsibility for events that are largely outside your control.",
    ),
    (
        "blaming",
        [
            r"\bit'?s\s+(all\s+)?(his|her|their|your)\s+fault\b",
            r"\bthey\s+(ruined|wrecked)\s+(everything|my\s+life)\b",
            r"\bbecause\s+of\s+(him|her|them|you),?\s+i\b",
        ],
        "Holding someone else entirely responsible for your own feelings or situation.",
    ),
    (
        "comparison",
        [
            r"\beveryone\s+else\s+(is|has|seems)\b",
            r"\bcompared\s+to\s+(them|him|her|everyone)\b",
            r"\bwhy\s+can'?t\s+i\s+be\s+(more\s+)?like\b",
            r"\bthey'?re\s+(so\s+much\s+)?(better|more\s+successful)\s+than\s+me\b",
        ],
        "Measuring your own worth against others in a way that consistently favors them.",
    ),
    (
        "control_fallacy",
        [
            r"\bi\s+have\s+no\s+control\s+over\s+(anything|my\s+life)\b",
            r"\bthere'?s\s+nothing\s+i\s+can\s+do\s+about\s+it\b",
            r"\bi'?m\s+responsible\s+for\s+(everyone'?s|their)\s+happiness\b",
        ],
        "Seeing yourself as either helplessly controlled by external forces or as responsible for everyone around you.",
    ),
]

_COMPILED_RULES = [
    (name, [re.compile(p, re.IGNORECASE) for p in patterns], explanation)
    for name, patterns, explanation in _RULES
]

DISTORTION_EXPLANATIONS = {name: explanation for name, _, explanation in _RULES}


_FOCUS_STOPWORDS = {
    "i", "it", "this", "that", "something", "someone", "everything",
    "nothing", "anything", "everyone", "anyone", "me", "myself",
}


def extract_focus_phrase(text: str) -> str:
    """
    Pulls out the most concrete noun phrase from the user's own words (e.g.
    "my job interview", "my sister", "the exam") so the narrative and comic
    panels built from it can stay grounded in what they actually described,
    instead of falling back to a generic, unrelated scene. Picks the
    longest noun chunk headed by a real noun -- longer chunks tend to carry
    more of the concrete modifiers ("upcoming job interview" over just
    "interview"). Returns "" if nothing concrete is found (e.g. very short
    or abstract input), and callers should treat that as "no grounding
    available" rather than invent one.
    """
    if not text or not isinstance(text, str):
        return ""
    try:
        doc = _get_nlp()(text)
    except Exception:
        return ""

    candidates = []
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if len(phrase) < 3 or phrase.lower() in _FOCUS_STOPWORDS:
            continue
        if chunk.root.pos_ not in ("NOUN", "PROPN"):
            continue
        candidates.append(phrase)

    if not candidates:
        return ""
    candidates.sort(key=len, reverse=True)
    return candidates[0]


def detect_distortions(text: str) -> list:
    """
    Returns a list of distortion names detected in the text (deduplicated,
    order of first occurrence preserved). Uses spaCy to split into
    sentences first so multi-sentence inputs are checked sentence by
    sentence, then falls back to whole-text matching if spaCy fails.
    """
    if not text or not isinstance(text, str):
        return []

    found = []
    try:
        doc = _get_nlp()(text)
        sentences = [sent.text for sent in doc.sents] or [text]
    except Exception:
        sentences = [text]

    for sentence in sentences:
        for name, patterns, _ in _COMPILED_RULES:
            if name in found:
                continue
            for pattern in patterns:
                if pattern.search(sentence):
                    found.append(name)
                    break

    return found
