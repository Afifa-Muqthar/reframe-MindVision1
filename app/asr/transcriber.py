"""
Speech-to-text using faster-whisper (CTranslate2 build of Whisper, same
weights as openai-whisper, several times faster on CPU). Only invoked
when the user submitted audio instead of typed text.

Size auto-tuning: "small" is meaningfully more accurate than "base" (per
Whisper's published WER numbers) and still fast with int8 on a capable
CPU; a low-RAM machine stays on "base" to keep peak memory and model
download size down (see app.hardware.has_ample_ram).
"""

import gc
import os

from faster_whisper import WhisperModel

from app.hardware import has_ample_ram

_MODEL_SIZE = "small" if has_ample_ram() else "base"


class Transcriber:
    def __init__(self):
        # int8 compute type keeps this fast and light on a CPU-only
        # machine; cpu_threads defaults conservatively, so set it
        # explicitly to use the full core count.
        self._model = WhisperModel(
            _MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4
        )

    def transcribe(self, audio_path: str) -> str:
        segments, _info = self._model.transcribe(audio_path, beam_size=5)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def unload(self):
        del self._model
        gc.collect()
