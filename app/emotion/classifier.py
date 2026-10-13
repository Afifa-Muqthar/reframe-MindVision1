"""
Emotion detection: SamLowe/roberta-base-go_emotions-onnx, run via ONNX
Runtime for CPU speed, MIT licensed, pretrained as-is (no fine-tuning).

Precision auto-tuning: the int8-quantized export is faster and lighter but
less accurate than the full-precision one; on a machine with real RAM
headroom (see app.hardware.has_ample_ram) the accuracy is free to take, so
full precision is tried first, falling back to quantized if anything about
loading it fails (keeps this working on the original low-RAM target).

The 28 GoEmotions labels -> our 7 core emotions mapping (EMOTION_MAP below,
and collapse_to_core_emotion) is OUR CODE, written because the rest of the
pipeline (principle selector, narrative prompts) needs a small, stable
vocabulary rather than 28 fine-grained labels.
"""

import gc
import os

import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

from app.hardware import has_ample_ram

_MODEL_NAME = "SamLowe/roberta-base-go_emotions-onnx"

# GoEmotions' 28 labels collapsed to 7 core emotions the rest of the
# pipeline reasons about. "neutral" is kept as its own bucket rather than
# folded into another emotion, since forcing a neutral utterance into e.g.
# "sadness" would corrupt downstream technique matching.
EMOTION_MAP = {
    "sadness": "sadness", "grief": "sadness", "remorse": "sadness",
    "disappointment": "sadness", "embarrassment": "sadness",

    "fear": "fear",

    # GoEmotions has no literal "anxiety" label; "nervousness" is the
    # closest match and is specific enough to warrant its own bucket
    # (rather than folding into "fear"), since the knowledge base has
    # separate anxiety-oriented technique cards that would otherwise never
    # get their emotion-match bonus.
    "nervousness": "anxiety",

    "anger": "anger", "annoyance": "anger", "disgust": "anger", "disapproval": "anger",

    "joy": "joy", "amusement": "joy", "excitement": "joy", "gratitude": "joy",
    "love": "joy", "optimism": "joy", "pride": "joy", "relief": "joy",
    "admiration": "joy", "approval": "joy", "caring": "joy", "desire": "joy",

    "surprise": "surprise", "realization": "surprise", "curiosity": "surprise", "confusion": "surprise",

    "neutral": "neutral",
}

CORE_EMOTIONS = ["sadness", "fear", "anxiety", "anger", "joy", "surprise", "neutral"]


class EmotionClassifier:
    """
    Load on demand, release after use, per the RAM-discipline requirement.
    Usage:
        clf = EmotionClassifier()
        scores, core = clf.classify("I feel like nothing ever works out")
        clf.unload()
    """

    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        # ONNX Runtime's default intra-op thread count can under-use a
        # multi-core CPU; this classifier stays resident for the whole
        # process (see main.py), so setting it once here is worth it.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = os.cpu_count() or 4
        # The repo hosts both a full-precision and an int8-quantized ONNX
        # file. Full precision is more accurate; try it first on a machine
        # with headroom, falling back to quantized (lighter, slightly less
        # accurate) if that load fails for any reason.
        file_name = "onnx/model.onnx" if has_ample_ram() else "onnx/model_quantized.onnx"
        try:
            self._model = ORTModelForSequenceClassification.from_pretrained(
                _MODEL_NAME, file_name=file_name, session_options=session_options
            )
        except Exception:
            self._model = ORTModelForSequenceClassification.from_pretrained(
                _MODEL_NAME, file_name="onnx/model_quantized.onnx", session_options=session_options
            )
        self._pipe = pipeline(
            "text-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            top_k=None,
            function_to_apply="sigmoid",
        )

    def classify(self, text: str):
        """
        Returns (raw_scores: dict[label -> float], core_emotion: str).
        raw_scores contains all 28 GoEmotions labels with their scores.
        core_emotion is the collapsed label with the highest aggregated
        score across the 6 core buckets.
        """
        results = self._pipe(text)[0]  # list of {"label":.., "score":..}
        raw_scores = {r["label"]: float(r["score"]) for r in results}

        # Max, not sum, across each bucket's member labels. These are
        # independent sigmoid scores (not a softmax that sums to 1), and
        # "joy" absorbs 12 GoEmotions labels while "anxiety"/"fear" each
        # have only 1-2 -- summing let several weakly-activated joy-adjacent
        # labels (e.g. "love" from a phrase like "my love life", plus
        # "caring", "excitement") outvote one single strongly-activated
        # label like "nervousness", even when nothing joyful was expressed.
        core_totals = {c: 0.0 for c in CORE_EMOTIONS}
        for label, score in raw_scores.items():
            core = EMOTION_MAP.get(label, "neutral")
            core_totals[core] = max(core_totals[core], score)

        core_emotion = max(core_totals, key=core_totals.get)
        return raw_scores, core_emotion

    def unload(self):
        del self._pipe
        del self._model
        del self._tokenizer
        gc.collect()
