"""
Narrative generation using google/flan-t5-base (fallback: flan-t5-small),
prompt-engineered only. No training/fine-tuning (hard constraint for
Phases 1-3).

flan-t5-large was tried and reverted: it fits comfortably in RAM on a
16GB machine, but CPU inference time is compute-bound, not RAM-bound, and
measured over 4 minutes for one stage-generation call here (base: ~20s)
-- large models hit CPU memory-bandwidth limits that don't show up in a
RAM budget. "Fits in RAM" does not imply "fast enough to be interactive,"
so model size stays fixed regardless of app.hardware.has_ample_ram();
that helper is only appropriate for RAM-bound tradeoffs (precision,
slicing, resolution), not CPU-bound ones like this.

Two APIs live here:
- generate() / parse_narrative_parts(): the original fixed 4-part
  narrative (current reality / reframe / desired future / next step).
  Kept for scripts/pregenerate_* and anything else still using it.
- generate_stages(): a dynamic 3-8 stage comic narrative (see
  compute_stage_count below) -- what the live app uses. Stage 1 is always
  the actual situation the user described, the middle stages gradually
  introduce the reframe (one per distortion detected, so a more tangled
  thought pattern gets more room to unpack), and the final stage is a
  realistic next moment, never a solved problem.

Both paths never promise a guaranteed outcome ("you will definitely
succeed") since that's both clinically irresponsible and not something the
model can know, and both reject any sign the model invented a resolution
the user never described (see _RESOLUTION_CLAIM_PATTERNS).
"""

import gc
import os
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.nlp.distortions import extract_focus_phrase

_MODEL_NAME_PRIMARY = "google/flan-t5-base"
_MODEL_NAME_FALLBACK = "google/flan-t5-small"

_BANNED_PHRASES = [
    "you will definitely", "you'll definitely", "guaranteed to", "i promise you",
    "you will certainly", "100% certain", "will always work out",
]

# FLAN-T5-base's free-form output was observed during testing to
# occasionally hallucinate content unrelated to the prompt, including
# insulting language directed at the reader (e.g. calling them a "cynical
# idiot"). A missing section header isn't the only failure mode worth
# catching, so generated text is also rejected outright (falling back to
# the deterministic template) if it contains any of these.
_HARMFUL_PATTERNS = [
    "idiot", "stupid", "worthless", "pathetic", "loser", "moron", "cynical",
    "hopeless case", "give up", "your fault", "deserve this",
]

# Also observed during testing: the model sometimes just echoes the
# instruction text back verbatim instead of writing a narrative. Since the
# prompt's own instructions literally contain the four required header
# words, an echo passes the header check while containing no real content.
# These phrases are unique to the instructions themselves (never something
# an actual narrative would say), so their presence means the model echoed
# rather than generated.
_ECHO_MARKERS = [
    "describe their situation and feeling",
    "gently introduce the technique above",
    "describe a realistic, modest, positive future",
    "give one small, concrete, doable action",
    "write a short, warm, second-person narrative",
]

# The comic's last stage must be a realistic next moment, not the model
# deciding the user's problem is already fixed -- nothing in the user's
# own text said it was. Any stage containing these phrases is rejected
# outright (falls back to the deterministic template) rather than risk
# showing the user a fabricated happy ending they never described.
_RESOLUTION_CLAIM_PATTERNS = [
    "problem is solved", "problem is now solved", "everything is fine now",
    "everything worked out", "completely fixed", "no longer a problem",
    "is now resolved", "has been resolved", "all better now", "now that it's over",
]

STAGE_SITUATION = "situation"
STAGE_REFRAMING = "reframing"
STAGE_NEXT_MOMENT = "next_moment"

_STAGE_LABELS = {STAGE_SITUATION: "SITUATION", STAGE_REFRAMING: "REFRAMING", STAGE_NEXT_MOMENT: "NEXT MOMENT"}


def compute_stage_count(case_frame) -> int:
    """
    3 stages minimum (situation, one reframing beat, next moment), 8
    maximum. Scales with how much progression the situation actually
    needs: each distinct cognitive distortion detected gets its own
    reframing stage, since each represents a separate thought pattern
    worth walking through on its own rather than collapsing into one
    generic "think positive" beat. A case with no detected distortion
    still gets exactly one reframing stage (the chosen technique itself).
    """
    distortions = getattr(case_frame, "distortions", []) or []
    reframing_stages = max(1, min(len(distortions), 6))
    return min(3 + reframing_stages - 1, 8)


def _stage_types_for_count(n: int) -> list:
    return [STAGE_SITUATION] + [STAGE_REFRAMING] * (n - 2) + [STAGE_NEXT_MOMENT]


def _short_quote(text: str, max_words: int = 10) -> str:
    first = re.split(r"[.!?]", (text or "").strip())[0].strip()
    words = first.split()
    if len(words) > max_words:
        first = " ".join(words[:max_words]) + "..."
    return first or "This is a lot right now."

_PROMPT_TEMPLATE = """You are a supportive, evidence-informed narrative writer. Someone shared this worry:
"{raw_text}"

Their main emotion is {core_emotion}. A relevant psychological technique is: {technique_name} - {technique_description}
{recurring_theme_line}
Write a short, warm, second-person narrative in exactly four labeled parts. Do not skip any part. Do not promise a guaranteed outcome or use words like "definitely", "guaranteed", or "always". Keep each part to 1-3 sentences.

CURRENT REALITY: describe their situation and feeling with empathy, using their own words where natural.
REFRAME: gently introduce the technique above as one way to look at the thought differently, without dismissing the feeling.
DESIRED FUTURE: describe a realistic, modest, positive future a few days or weeks out if they try this.
NEXT STEP: give one small, concrete, doable action they could take today.

Now write the four parts:
"""


class NarrativeGenerator:
    def __init__(self, use_small: bool = False):
        # Use every available CPU core for inference -- default detection
        # can under-count on some Windows setups, and this is otherwise a
        # single-threaded-feeling bottleneck during beam search.
        torch.set_num_threads(os.cpu_count() or 4)

        candidates = [_MODEL_NAME_FALLBACK] if use_small else [_MODEL_NAME_PRIMARY, _MODEL_NAME_FALLBACK]

        last_error = None
        for model_name in candidates:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                last_error = None
                break
            except Exception as e:
                last_error = e
        if last_error is not None:
            raise last_error

    def generate(self, case_frame, technique: dict) -> str:
        recurring_theme_line = ""
        if getattr(case_frame, "recurring_theme_note", ""):
            recurring_theme_line = (
                f"\nNote: {case_frame.recurring_theme_note}. Gently acknowledge this pattern "
                f"in the CURRENT REALITY part, without being repetitive or discouraging.\n"
            )

        prompt = _PROMPT_TEMPLATE.format(
            raw_text=case_frame.raw_text,
            core_emotion=case_frame.core_emotion or "unclear",
            technique_name=technique["name"],
            technique_description=technique["description"],
            recurring_theme_line=recurring_theme_line,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            num_beams=2,
            no_repeat_ngram_size=3,
        )
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = self._sanitize(text)
        text = self._validate_or_fallback(text, case_frame, technique)
        return text

    def generate_stages(self, case_frame, technique: dict) -> list:
        """
        Returns a list of stage dicts: {"index", "type", "caption",
        "dialogue"}. "type" is one of STAGE_SITUATION / STAGE_REFRAMING /
        STAGE_NEXT_MOMENT. "caption" is the full sentence(s) shown under
        the comic panel; "dialogue" is a short first-person line (<=10
        words) for the panel's speech/thought bubble.
        """
        n = compute_stage_count(case_frame)
        types = _stage_types_for_count(n)
        focus = extract_focus_phrase(case_frame.raw_text)

        prompt = self._build_stage_prompt(case_frame, technique, types)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=45 * n,
            do_sample=False,
            num_beams=2,
            no_repeat_ngram_size=3,
        )
        text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = self._sanitize(text)
        stages = self._parse_stages(text, types)
        if stages is None:
            stages = self._fallback_stages(case_frame, technique, types, focus)
        return stages

    def _build_stage_prompt(self, case_frame, technique: dict, types: list) -> str:
        reframing_total = types.count(STAGE_REFRAMING)
        headers = []
        for i, stype in enumerate(types):
            label = _STAGE_LABELS[stype]
            if stype == STAGE_REFRAMING and reframing_total > 1:
                label = f"{label} {types[:i + 1].count(STAGE_REFRAMING)} of {reframing_total}"
            headers.append(f"STAGE {i + 1} ({label}):")
        headers_block = "\n".join(headers)

        recurring_theme_line = ""
        if getattr(case_frame, "recurring_theme_note", ""):
            recurring_theme_line = f"\nNote: {case_frame.recurring_theme_note}. You may acknowledge this pattern briefly in stage 1 only.\n"

        return (
            "You are a supportive, evidence-informed narrative writer, writing a short comic-style "
            f"story about someone's real situation: \"{case_frame.raw_text}\"\n"
            f"Their main emotion is {case_frame.core_emotion or 'unclear'}. "
            f"A relevant psychological technique is: {technique['name']} - {technique['description']}\n"
            f"{recurring_theme_line}"
            f"Write exactly {len(types)} short stages (1-2 sentences each), one per labeled header below. "
            "Do not skip or merge any stage. Do not invent that the problem is already solved or fixed -- "
            "only describe what was actually said. Do not promise a guaranteed outcome or use words like "
            "\"definitely\", \"guaranteed\", or \"always\".\n\n"
            "The SITUATION stage describes what actually happened/was said, with empathy. "
            "Each REFRAMING stage gradually introduces the technique above, looking at one part of the "
            "thought a little differently, without dismissing the feeling. "
            "The NEXT MOMENT stage gives one small, realistic, doable action right now -- not a resolution.\n\n"
            f"{headers_block}\n\nNow write the stages:\n"
        )

    def _parse_stages(self, text: str, types: list):
        pattern = r"STAGE\s*\d+\s*\([^)]*\):\s*"
        pieces = re.split(pattern, text)
        # pieces[0] is any preamble before the first header; real content follows.
        bodies = [p.strip() for p in pieces[1:] if p.strip()]

        if len(bodies) != len(types):
            return None

        lowered_all = text.lower()
        if any(p in lowered_all for p in _HARMFUL_PATTERNS):
            return None
        if any(p in lowered_all for p in _ECHO_MARKERS):
            return None
        if any(p in lowered_all for p in _RESOLUTION_CLAIM_PATTERNS):
            return None

        stages = []
        for i, (stype, body) in enumerate(zip(types, bodies)):
            stages.append({
                "index": i + 1,
                "type": stype,
                "caption": body,
                "dialogue": _short_quote(body),
            })
        return stages

    def _fallback_stages(self, case_frame, technique: dict, types: list, focus: str) -> list:
        distortions = getattr(case_frame, "distortions", []) or []
        reframing_seen = 0
        stages = []
        about = f" about {focus}" if focus else ""

        for i, stype in enumerate(types):
            if stype == STAGE_SITUATION:
                caption = (
                    f"{case_frame.raw_text.strip()} It makes sense that this brings up "
                    f"{case_frame.core_emotion or 'difficult feelings'}."
                )
                dialogue = _short_quote(case_frame.raw_text)
            elif stype == STAGE_REFRAMING:
                if reframing_seen == 0:
                    caption = f"{technique['name']} suggests looking at this through a different lens: {technique['description']}"
                    dialogue = "Maybe there's another way to see this."
                else:
                    d = distortions[reframing_seen - 1] if reframing_seen - 1 < len(distortions) else None
                    if d:
                        caption = f"Noticing a pattern of {d.replace('_', ' ')} here, and gently questioning it rather than taking it as fact."
                    else:
                        caption = "Sitting with this thought a little longer, without judging it right away."
                    dialogue = "I don't have to believe every thought right away."
                reframing_seen += 1
            else:  # STAGE_NEXT_MOMENT
                caption = f"Right now, one small, realistic step: try a two-minute version of {technique['name'].lower()}{about}."
                dialogue = "One small step, right now."
            stages.append({"index": i + 1, "type": stype, "caption": caption, "dialogue": dialogue})
        return stages

    def _sanitize(self, text: str) -> str:
        lowered = text.lower()
        for phrase in _BANNED_PHRASES:
            if phrase in lowered:
                # Regenerate is expensive on CPU; instead soften in place.
                idx = lowered.find(phrase)
                text = text[:idx] + "this may help" + text[idx + len(phrase):]
                lowered = text.lower()
        return text

    def _validate_or_fallback(self, text: str, case_frame, technique: dict) -> str:
        """
        FLAN-T5-base's free-form output is not reliable enough to trust
        unconditionally: besides sometimes skipping a required section, it
        was observed during testing to occasionally hallucinate harmful or
        unrelated content, or simply echo the instructions back verbatim.
        Generated text is accepted only if it (a) contains all 4 required
        headers, (b) contains none of the harmful patterns, and (c) isn't
        just an echo of the prompt's own instructions; otherwise a
        deterministic template is used, which guarantees a safe, on-contract
        output regardless of generation quality.
        """
        required = ["CURRENT REALITY", "REFRAME", "DESIRED FUTURE", "NEXT STEP"]
        upper = text.upper()
        lowered = text.lower()
        has_all_headers = all(h in upper for h in required)
        has_harmful_content = any(p in lowered for p in _HARMFUL_PATTERNS)
        is_echo = any(p in lowered for p in _ECHO_MARKERS)

        if has_all_headers and not has_harmful_content and not is_echo:
            return text

        theme_sentence = ""
        if getattr(case_frame, "recurring_theme_note", ""):
            theme_sentence = f" You may notice that {case_frame.recurring_theme_note}."

        return (
            f"CURRENT REALITY: {case_frame.raw_text.strip()} It makes sense that this brings up "
            f"{case_frame.core_emotion or 'difficult feelings'}.{theme_sentence}\n\n"
            f"REFRAME: {technique['name']} suggests looking at this through a different lens: "
            f"{technique['description']}\n\n"
            f"DESIRED FUTURE: With some practice, it's realistic that this could feel a little more "
            f"manageable in the coming days or weeks.\n\n"
            f"NEXT STEP: Try one small action today related to this technique, even a two-minute version of it."
        )

    def unload(self):
        del self._model
        del self._tokenizer
        gc.collect()


_SECTION_HEADERS = ["CURRENT REALITY", "REFRAME", "DESIRED FUTURE", "NEXT STEP"]


def parse_narrative_parts(narrative: str) -> dict:
    """
    Splits a generated narrative into its four labeled parts, keyed
    "current_reality" / "reframe" / "desired_future" / "next_step". Used to
    build the storyboard image's per-panel captions (see
    app/image_gen/storyboard.py). Safe on both the FLAN-T5 output and the
    deterministic fallback template above, since both are guaranteed to
    contain all four headers by _validate_or_fallback.
    """
    pattern = r"(" + "|".join(_SECTION_HEADERS) + r"):\s*"
    pieces = re.split(pattern, narrative)
    parts = {}
    for i in range(1, len(pieces) - 1, 2):
        key = pieces[i].strip().lower().replace(" ", "_")
        parts[key] = pieces[i + 1].strip()
    for key in ("current_reality", "reframe", "desired_future", "next_step"):
        parts.setdefault(key, "")
    return parts
