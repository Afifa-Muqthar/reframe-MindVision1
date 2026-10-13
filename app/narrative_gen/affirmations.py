"""
Short first-person affirmations for the narrated audio track.

Deliberately separate content from the four-part narrative shown in the
storyboard image: the audio is meant to be heard, not a read-aloud
duplicate of what's already on screen, so it uses entirely different
wording -- present-tense, first-person, affirmation-style, rather than the
narrative's second-person CURRENT REALITY / REFRAME / DESIRED FUTURE /
NEXT STEP structure.

Deterministic and template-based, not model-generated, for the same reason
NarrativeGenerator falls back to a deterministic template: predictable,
safety-reviewable output with no hallucination risk, and no extra model to
load just for a few short sentences. Never promises a guaranteed outcome,
consistent with generator.py's banned-phrase list.
"""

_EMOTION_LINES = {
    "sadness": "It's okay to feel sad right now. This feeling is real, and it will not last forever.",
    "anger": "It's okay to feel frustrated right now. I can feel this and still choose my next step.",
    "fear": "It's okay to feel afraid right now. I can take one small step even when I feel unsure.",
    "anxiety": "It's okay to feel anxious right now. My body is on alert, but I am safe in this moment.",
    "joy": "I notice this moment of ease, and I let myself have it.",
    "disgust": "It's okay to feel uneasy right now. I can look at this thought with a little more distance.",
    "surprise": "It's okay to feel caught off guard. I can take a breath and take this one step at a time.",
}
_DEFAULT_EMOTION_LINE = "Whatever I'm feeling right now is valid."


def build_affirmation(core_emotion: str, technique_name: str) -> str:
    emotion_line = _EMOTION_LINES.get((core_emotion or "").lower(), _DEFAULT_EMOTION_LINE)
    return (
        f"{emotion_line} "
        f"I am allowed to take this one moment at a time. "
        f"Using {technique_name.lower()}, I can look at my thoughts a little more gently. "
        f"I am doing my best, and that is enough for today."
    )
