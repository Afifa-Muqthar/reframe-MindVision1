"""
Principle Selector: matches a Case Frame against the knowledge base of
technique cards and returns the 1-3 most relevant ones.

OUR CODE. Deliberately simple scoring rather than a learned ranker: a
technique's relevance here needs to be explainable ("this card matched
because of X distortion and Y emotion"), which a black-box ranker would
undermine for the viva.
"""

import hashlib
import json
import os
import random

_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "knowledge_base", "techniques.json")

_kb_cache = None


def _load_kb():
    global _kb_cache
    if _kb_cache is None:
        with open(_KB_PATH, "r", encoding="utf-8") as f:
            _kb_cache = json.load(f)
    return _kb_cache


def select_techniques(case_frame, top_n: int = 3) -> list:
    """
    Scores every technique card: +2 per matching distortion, +1 per
    matching core emotion. Returns the top_n cards (list of dicts) with
    score > 0.

    Ties are broken by a shuffle seeded from the input text rather than
    fixed knowledge-base order: with only ~6 scoring points available,
    many sessions land on the same score, and always favoring whichever
    card happens to be listed first in techniques.json made the same 1-2
    cards dominate almost every result. The seed keeps this reproducible
    for the same input (so results are still explainable/testable) while
    varying across different inputs that happen to share a score profile.
    """
    kb = _load_kb()
    scored = []

    for card in kb:
        score = 0
        for d in case_frame.distortions:
            if d in card["applies_to_distortions"]:
                score += 2
        if case_frame.core_emotion in card["applies_to_emotions"]:
            score += 1

        if score > 0:
            scored.append((score, card))

    # hashlib, not Python's built-in hash(), since str hashing is randomized
    # per-process by default and would make this "reproducible seed" lie.
    seed = int(hashlib.sha256((case_frame.raw_text or "").encode("utf-8")).hexdigest(), 16) % (2 ** 32)
    random.Random(seed).shuffle(scored)
    scored.sort(key=lambda pair: pair[0], reverse=True)  # stable sort keeps the shuffle order within each score
    top_cards = [card for _, card in scored[:top_n]]

    # Fallback: if nothing matched (e.g. no distortion detected and a
    # neutral/joy emotion), default to the most broadly-applicable card
    # rather than returning nothing.
    if not top_cards:
        top_cards = [c for c in kb if c["id"] == "self_compassion"][:1]

    case_frame.selected_techniques = [c["id"] for c in top_cards]
    return top_cards
