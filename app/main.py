"""
Flask orchestrator. Wires the pipeline stages together in order:

  input -> (ASR if audio) -> safety gate -> distortion detection ->
  emotion detection -> Case Frame -> principle selector ->
  narrative generation -> [text returned] -> TTS (sync) ->
  image generation (background thread, subprocess-isolated) -> save to SQLite

RAM discipline: spaCy and the emotion classifier stay resident across
requests (small). Whisper, FLAN-T5, SD-Turbo, and Kokoro are loaded only
for the stage that needs them and released (`del model; gc.collect()`)
immediately after, since a typical student laptop can't hold all five in
memory at once.
"""

# Must be set before torch/CTranslate2/onnxruntime are imported anywhere
# in the process. faster-whisper (CTranslate2) and PyTorch's bundled MKL
# both link their own OpenMP runtime; loading both in one process aborts
# with "OMP: Error #15: Initializing libiomp5md.dll, but found
# libiomp5md.dll already initialized" the first time a request needs both
# ASR and any torch-based stage (which is every real request). This is the
# standard, documented workaround for that conflict.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import subprocess
import sys
import threading
import uuid

from flask import Flask, request, jsonify, render_template, send_from_directory

from app.safety.safety_gate import screen_for_crisis, get_helpline_message
from app.nlp.distortions import detect_distortions
from app.emotion.classifier import EmotionClassifier
from app.context_engine.case_frame import CaseFrame
from app.principle_selector.selector import select_techniques
from app.narrative_gen.generator import NarrativeGenerator
from app.narrative_gen.affirmations import build_affirmation
from app.db.schema import init_db
from app.db import repository

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(BASE_DIR, "data", "sessions")
os.makedirs(MEDIA_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
init_db()

# Emotion classifier + spaCy (via distortions module) stay resident.
_emotion_classifier = EmotionClassifier()

# Tracks in-flight background image generation per session id.
_image_status = {}  # session_id(str) -> {"status": "pending"|"done"|"error", "path": str|None}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/journey")
def journey():
    sessions = repository.list_sessions()
    recurring = repository.find_recurring_themes()
    return render_template("journey.html", sessions=sessions, recurring=recurring)


@app.route("/media/<path:filename>")
def media(filename):
    return send_from_directory(MEDIA_DIR, filename)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Speech-to-text only, no pipeline run. Called right after the browser
    finishes recording (or a file is chosen) so the recognized text shows
    up in the textarea immediately -- editable, like Google's voice typing
    -- instead of silently attaching raw audio to the /process submission.
    """
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio provided."}), 400
    text = _transcribe_audio(audio_file)
    if not text:
        return jsonify({"error": "Couldn't make out any speech in that recording."}), 422
    return jsonify({"text": text})


@app.route("/process", methods=["POST"])
def process():
    text = request.form.get("text", "").strip()
    audio_file = request.files.get("audio")

    if audio_file and not text:
        text = _transcribe_audio(audio_file)

    if not text:
        return jsonify({"error": "No text or audio provided."}), 400

    # 1. Safety gate — ALWAYS FIRST, fail-safe.
    if screen_for_crisis(text):
        return jsonify({
            "crisis": True,
            "message": get_helpline_message(),
        })

    # 2. Cache check (Phase 2): identical repeated input skips full pipeline.
    cached = _check_cache(text)
    if cached:
        return jsonify(cached)

    # 3. Distortion detection.
    distortions = detect_distortions(text)

    # 4. Emotion detection (resident classifier).
    emotion_scores, core_emotion = _emotion_classifier.classify(text)

    # 5. Case Frame.
    case_frame = CaseFrame(raw_text=text, distortions=distortions,
                            emotion_scores=emotion_scores, core_emotion=core_emotion)
    case_frame.build_summary()
    case_frame.recurring_theme_note = repository.build_recurring_theme_note()

    # 6. Principle selection.
    techniques = select_techniques(case_frame)
    primary_technique = techniques[0]

    # 7. Narrative generation (load -> use -> release). Dynamic 3-8 stage
    # comic narrative: stage 1 is the situation as described, middle
    # stages gradually introduce the reframe (one per distortion
    # detected), final stage is a realistic next moment -- never a
    # fabricated resolution (see generate_stages's validation).
    generator = NarrativeGenerator()
    try:
        stages = generator.generate_stages(case_frame, primary_technique)
    finally:
        generator.unload()
    narrative = "\n\n".join(
        f"{s['type'].upper().replace('_', ' ')}: {s['caption']}" for s in stages
    )

    # 8. Save to DB now (audio/image paths filled in below/async).
    session_id = repository.save_session(case_frame, narrative)

    # 9. TTS synchronously (fast enough not to block the response). Narrates
    # a short first-person affirmation, not the narrative text itself --
    # that's already shown, and now drawn, on screen, so reading it aloud
    # verbatim would just be redundant.
    affirmation = build_affirmation(case_frame.core_emotion, primary_technique["name"])
    audio_path = _run_tts(session_id, affirmation)

    # 10. Image generation in background thread; frontend polls /image_status.
    # Builds an N-panel comic storyboard (N = len(stages), 3-8) rather than
    # one plain illustration or a fixed panel count (see
    # app/image_gen/storyboard.py).
    _image_status[str(session_id)] = {"status": "pending", "path": None}
    panel_prompts = _build_stage_image_prompts(case_frame.raw_text, stages, case_frame.core_emotion)
    threading.Thread(
        target=_generate_image_background,
        args=(session_id, panel_prompts, stages),
        daemon=True,
    ).start()

    response = {
        "crisis": False,
        "session_id": session_id,
        "summary": case_frame.summary,
        "distortions": case_frame.distortions,
        "core_emotion": case_frame.core_emotion,
        "technique": {"name": primary_technique["name"], "citation": primary_technique["citation"]},
        "narrative": narrative,
        "stages": stages,
        "audio_url": f"/media/{os.path.basename(audio_path)}" if audio_path else None,
        "image_status_url": f"/image_status/{session_id}",
    }
    _simple_cache[text.strip().lower()] = response
    return jsonify(response)


@app.route("/image_status/<int:session_id>")
def image_status(session_id):
    status = _image_status.get(str(session_id), {"status": "unknown", "path": None})
    resp = {"status": status["status"]}
    if status["status"] == "done" and status["path"]:
        resp["image_url"] = f"/media/{os.path.basename(status['path'])}"
    return jsonify(resp)


# ---- helpers ----

_simple_cache = {}  # exact text -> response dict (Phase 2 requirement)


def _check_cache(text: str):
    return _simple_cache.get(text.strip().lower())


def _transcribe_audio(audio_file) -> str:
    from app.asr.transcriber import Transcriber

    # Keep the real extension (e.g. the browser mic recorder sends .webm,
    # not .wav) so ffmpeg's format probing has an accurate hint to work from.
    _, ext = os.path.splitext(audio_file.filename or "")
    ext = ext if ext else ".wav"
    tmp_path = os.path.join(MEDIA_DIR, f"upload_{uuid.uuid4().hex}{ext}")
    audio_file.save(tmp_path)
    transcriber = Transcriber()
    try:
        text = transcriber.transcribe(tmp_path)
    finally:
        transcriber.unload()
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return text


def _run_tts(session_id: int, narrative: str) -> str:
    from app.tts.narrator import Narrator

    output_path = os.path.join(MEDIA_DIR, f"session_{session_id}.wav")
    narrator = Narrator()
    try:
        narrator.narrate(narrative, output_path)
    except Exception:
        return None
    finally:
        narrator.unload()
    return output_path


def _build_stage_image_prompts(raw_text: str, stages: list, core_emotion: str):
    from app.image_gen.generator import build_stage_image_prompts
    return build_stage_image_prompts(raw_text, stages, core_emotion)


_IMAGE_WORKER_CODE = """
import os
import sys
sys.path.insert(0, {project_root!r})
from app.image_gen.generator import ImageGenerator
from app.image_gen.storyboard import compose_storyboard

prompts = {prompts!r}
panel_paths = {panel_paths!r}
seed = {seed!r}

gen = ImageGenerator()
for prompt, path in zip(prompts, panel_paths):
    gen.generate(prompt, path, seed=seed)  # resolution/steps auto-tuned to hardware, see image_gen/generator.py
gen.unload()

compose_storyboard(panel_paths, {stages!r}, {output_path!r})

for p in panel_paths:
    try:
        os.remove(p)
    except OSError:
        pass
"""


def _generate_image_background(session_id: int, prompts: list, stages: list):
    """
    Runs SD-Turbo (once per comic stage, 3-8 times) in a separate
    subprocess rather than in-thread. On a RAM-constrained machine,
    SD-Turbo's VAE-decode step can segfault (see README's RAM note) -- and a
    segfault kills the entire process it runs in, threads included, which
    would take down the whole Flask server and every other in-flight
    request along with it. Isolating it in a subprocess means a crash there
    only fails that one image. The panel compositing (storyboard.py) also
    runs in this same subprocess, after all panels are generated.
    """
    output_path = os.path.join(MEDIA_DIR, f"session_{session_id}.png")
    panel_paths = [os.path.join(MEDIA_DIR, f"session_{session_id}_panel{i + 1}.png") for i in range(len(prompts))]
    try:
        code = _IMAGE_WORKER_CODE.format(
            project_root=BASE_DIR,
            prompts=prompts,
            panel_paths=panel_paths,
            seed=session_id,  # fixed per-session seed keeps the panels' palette/style cohesive
            stages=stages,
            output_path=output_path,
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)

        if result.returncode == 0 and os.path.exists(output_path):
            repository.update_image_path(session_id, output_path)
            _image_status[str(session_id)] = {"status": "done", "path": output_path}
        else:
            _image_status[str(session_id)] = {"status": "error", "path": None, "error": result.stderr[-500:]}
    except Exception as e:
        _image_status[str(session_id)] = {"status": "error", "path": None, "error": str(e)}


if __name__ == "__main__":
    # debug=False by default: the debug reloader spawns a second process
    # that reloads every resident model (spaCy, emotion classifier) a
    # second time, which is a real problem on a RAM-constrained machine
    # (see README's RAM note). Set FLASK_DEBUG=1 if you want the reloader
    # on a machine with more headroom.
    app.run(debug=bool(os.environ.get("FLASK_DEBUG")), port=5000)
