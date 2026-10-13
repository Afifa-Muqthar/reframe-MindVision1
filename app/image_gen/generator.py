"""
Image generation using stabilityai/sd-turbo, 256x256/1 step on a low-RAM
machine, 384x384/3 steps on a machine with real headroom (see
app.hardware.has_ample_ram).

License note (say this in the viva if asked): SD-Turbo is under the
Stability AI Community/Research License - free to use for this academic
project, but non-commercial. It is also gated on Hugging Face: whoever
runs this needs a free HF account, to have clicked "accept" on the model
page, and to have run `huggingface-cli login` once before first download.

Runs in a background thread from the Flask route so it never blocks the
text response (see app/main.py /image_status/<session_id>).

Resolution note: the brief's spec is 512x512, but testing on the original
target laptop (Intel i3, 3.77GB RAM total) showed the VAE-decode step
segfaulting at 512x512 -- almost certainly a failed native memory
allocation under memory pressure rather than a code bug (it crashed at the
same point regardless of thread count). 384x384 is a middle ground that
still clears that machine's documented 8GB minimum (see SETUP_GUIDE.md)
comfortably; true 512x512 is left for a future bump once tested against
the segfault on a representative low-RAM box again.

Thread/slicing/resolution auto-tuning: attention/VAE slicing trades speed
for lower peak memory -- worth it on a low-RAM machine, pure overhead on
anything with real headroom, since it forces the UNet/VAE to process
smaller tiles sequentially instead of one batched op. Below
has_ample_ram()'s threshold, slicing stays on and resolution/steps stay at
the conservative defaults; above it, slicing is skipped, resolution and
step count both rise, and torch is told to use every available CPU core
(its default detection can under-count on some Windows setups).
"""

import gc
import os

import torch
from diffusers import AutoPipelineForText2Image

from app.hardware import has_ample_ram

_MODEL_NAME = "stabilityai/sd-turbo"
IMAGE_SIZE = 384 if has_ample_ram() else 256
DEFAULT_INFERENCE_STEPS = 3 if has_ample_ram() else 1


class ImageGenerator:
    def __init__(self):
        torch.set_num_threads(os.cpu_count() or 4)
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            _MODEL_NAME, torch_dtype=torch.float32
        )
        self._pipe.to("cpu")
        if not has_ample_ram():
            # Cuts peak memory during the UNet and VAE-decode steps by
            # processing attention/VAE tiles sequentially instead of all at
            # once -- the difference between segfaulting and completing on
            # a low-RAM machine. Skipped above the threshold since it's
            # pure overhead once there's enough RAM to not need it.
            self._pipe.enable_attention_slicing()
            self._pipe.enable_vae_slicing()

    def generate(self, prompt: str, output_path: str, num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
                 size: int = IMAGE_SIZE, seed: int = None) -> str:
        # A fixed seed across the three storyboard panels (see
        # build_panel_prompts) keeps their color palette and style visually
        # cohesive even though SD-Turbo has no built-in way to keep the same
        # illustrated figure identical across separate generations.
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        image = self._pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,  # SD-Turbo is trained for guidance_scale=0
            height=size,
            width=size,
            generator=generator,
        ).images[0]
        image.save(output_path)
        return output_path

    def unload(self):
        del self._pipe
        gc.collect()


def build_image_prompt(case_frame, technique: dict) -> str:
    """
    Turns the case frame + chosen technique into a short, calm, non-literal
    illustrative prompt (avoids depicting distressing scenes literally).
    Kept for scripts/pregenerate_samples.py's manifest field; the live app
    uses build_panel_prompts (three panels) instead -- see main.py.
    """
    return (
        f"A calm, gentle, hopeful illustration representing moving from {case_frame.core_emotion or 'worry'} "
        f"toward {technique['name'].lower()}, soft colors, minimalist, digital art, no text, no words"
    )


def build_panel_prompts(core_emotion: str, technique_name: str) -> list:
    """
    Three prompts for a CURRENT REALITY / REFRAME / DESIRED FUTURE
    storyboard: same illustrated-figure description and art style in each,
    varying only the concrete pose/scene, so that -- combined with a shared
    seed in ImageGenerator.generate -- the three panels read as one visual
    sequence rather than three unrelated pictures.

    Deliberately concrete (a specific pose/setting) rather than abstract
    psychological language: SD-Turbo has nothing to draw for a phrase like
    "using cognitive restructuring" (it's not a visual concept), and testing
    showed abstract prompts produced vague, figure-less color gradients
    instead of a recognizable scene. The technique name is shown as a text
    subtitle instead (see build_panel_subtitles), not asked of the image
    model. Also avoids depicting the distressing content literally.
    """
    emotion = core_emotion or "worried"
    figure = (
        "a simple minimalist illustration of one person, soft muted color palette, "
        "gentle lighting, clean flat digital art style, no text, no words"
    )
    return [
        f"{figure}, sitting at a desk with head resting on their hands, slouched and tired, feeling {emotion}",
        f"{figure}, sitting cross-legged, writing thoughtfully in an open notebook, calm focused expression",
        f"{figure}, standing outside looking toward a soft sunrise, relaxed steady posture, warm hopeful colors",
    ]


def build_stage_image_prompts(raw_text: str, stages: list, core_emotion: str) -> list:
    """
    One SD-Turbo prompt per comic stage (see narrative_gen.generator's
    dynamic STAGE_SITUATION/STAGE_REFRAMING/STAGE_NEXT_MOMENT structure).

    Every prompt shares the same consistent figure description and, where
    extractable, the same concrete focus phrase pulled from the user's own
    words (see app.nlp.distortions.extract_focus_phrase) -- e.g. "thinking
    about a job interview" rather than a generic, situation-less scene.
    Only the pose/expression varies per stage, tracking its type and (for
    reframing stages) how far through the reframe we are. SD-Turbo still
    can't reliably draw a *specific* recognizable scene (a particular
    interview room, a particular person) from a few CPU-budget inference
    steps -- the focus phrase grounds the *subject* of each panel, not a
    literal depiction of the scene. The final stage deliberately stays
    modest (a small step, same calm-but-real expression) rather than a
    triumphant/resolved pose, since the narrative never claims the
    situation is solved.
    """
    from app.nlp.distortions import extract_focus_phrase
    from app.narrative_gen.generator import STAGE_SITUATION, STAGE_REFRAMING, STAGE_NEXT_MOMENT

    focus = extract_focus_phrase(raw_text)
    about = f", thinking about {focus}" if focus else ""
    emotion = core_emotion or "worried"
    figure = (
        "a simple minimalist illustration of one person, soft muted color palette, "
        "gentle lighting, clean flat digital art style, no text, no words"
    )

    reframing_total = sum(1 for s in stages if s["type"] == STAGE_REFRAMING)
    reframing_seen = 0
    prompts = []
    for stage in stages:
        if stage["type"] == STAGE_SITUATION:
            prompts.append(f"{figure}, sitting, tense {emotion} expression, shoulders tight{about}")
        elif stage["type"] == STAGE_REFRAMING:
            reframing_seen += 1
            openness = reframing_seen / max(reframing_total, 1)
            pose = (
                "sitting cross-legged, pausing, starting to look thoughtful"
                if openness < 0.5 else
                "sitting cross-legged, writing in an open notebook, calmer focused expression"
            )
            prompts.append(f"{figure}, {pose}{about}")
        else:  # STAGE_NEXT_MOMENT
            prompts.append(
                f"{figure}, standing, taking one small steady step forward, "
                f"calm but realistic expression, not overly triumphant{about}"
            )
    return prompts


def build_panel_subtitles(core_emotion: str, technique_name: str) -> list:
    """
    Short captions shown under each panel's header. Panel 3 deliberately
    avoids promising an outcome (e.g. "you achieve your goal") -- the
    narrative generator's banned-phrase list exists for the same reason: no
    guaranteed outcome, ever.
    """
    return [
        f"Feeling {core_emotion or 'this'}",
        f"Try {technique_name}",
        "One step at a time",
    ]
