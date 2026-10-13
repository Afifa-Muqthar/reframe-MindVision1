"""
Composes N generated panel images (N = 3-8, one per dynamic comic stage --
see app.narrative_gen.generator.compute_stage_count) plus their captions
and speech-bubble dialogue into a single comic-strip storyboard, wrapped
into rows of up to 4 panels.

Built with Pillow only, no additional model -- this is layout/typesetting
("our own engineering"), not generation, so it stays within the project's
documented scope of five pretrained models (see DESIGN.md) plus this app's
own code around them. SD-Turbo cannot reliably render in-image text or keep
a character consistent across separate generations, so both the captions
and the panel arrangement are drawn here instead of asked of the model.
"""

import math
import textwrap

from PIL import Image, ImageDraw, ImageFont

_PANEL_SIZE = 200
_MARGIN = 18
_GAP = 18
_HEADER_H = 34
_CAPTION_H = 150
_PANEL_BORDER = 3  # black comic-panel outline
_MAX_COLUMNS = 4
_BUBBLE_FG = (40, 40, 40)

# (header background, header/text color) per stage type, soft pastel to
# match the calm, non-alarming tone required of everything shown to a user
# in distress (same intent as the narrative generator's tone constraints).
_TYPE_PALETTE = {
    "situation": ((247, 214, 212), (150, 45, 45)),     # soft red
    "reframing": ((210, 227, 248), (35, 65, 125)),     # soft blue
    "next_moment": ((214, 236, 217), (35, 100, 55)),   # soft green
}
_TYPE_LABELS = {"situation": "SITUATION", "reframing": "REFRAMING", "next_moment": "NEXT MOMENT"}
_CAPTION_BG = (250, 248, 244)
_CAPTION_FG = (55, 55, 55)
_BG = (255, 252, 247)


def _load_font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    candidates = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_and_draw(draw, text, font, box, fill, line_spacing=4):
    x0, y0, x1, y1 = box
    max_width = x1 - x0
    avg_char_w = font.getlength("n") or 7
    wrap_chars = max(10, int(max_width / avg_char_w))
    lines = []
    for paragraph in (text or "").split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=wrap_chars) or [""])
    y = y0
    line_h = getattr(font, "size", 13) + line_spacing
    for line in lines:
        if y + line_h > y1:
            break
        draw.text((x0, y), line, font=font, fill=fill)
        y += line_h


def _draw_speech_bubble(canvas, draw, text, font, panel_x, panel_y, panel_size):
    """
    Draws a comic-style chat bubble (rounded rect + tail) over the top of a
    panel's illustration, with a near-opaque white fill so the art stays
    faintly visible behind the text. Tail points down toward the figure.
    """
    pad = 7
    bubble_w = panel_size - 20
    bx0 = panel_x + (panel_size - bubble_w) // 2
    by0 = panel_y + 8

    max_width = bubble_w - pad * 2
    avg_char_w = font.getlength("n") or 7
    wrap_chars = max(10, int(max_width / avg_char_w))
    lines = textwrap.wrap(text or "", width=wrap_chars) or [""]
    line_h = getattr(font, "size", 13) + 3
    bubble_h = pad * 2 + line_h * len(lines)
    by1 = by0 + bubble_h
    bx1 = bx0 + bubble_w

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rounded_rectangle([bx0, by0, bx1, by1], radius=12, fill=(255, 255, 255, 235), outline=(30, 30, 30, 255), width=2)
    tail_cx = bx0 + bubble_w // 2
    odraw.polygon(
        [(tail_cx - 9, by1 - 1), (tail_cx + 9, by1 - 1), (tail_cx, by1 + 12)],
        fill=(255, 255, 255, 235), outline=(30, 30, 30, 255),
    )
    canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB"), (0, 0))

    y = by0 + pad
    for line in lines:
        draw.text((bx0 + pad, y), line, font=font, fill=_BUBBLE_FG)
        y += line_h


def compose_storyboard(panel_paths: list, stages: list, out_path: str) -> str:
    """
    panel_paths[i] is the generated illustration for stages[i], where each
    stage is {"index", "type", "caption", "dialogue"} from
    NarrativeGenerator.generate_stages(). Panels are laid out in reading
    order (left-to-right, top-to-bottom), wrapped at _MAX_COLUMNS per row,
    and keep their stage's own order -- the comic always has exactly as
    many meaningful panels as the narrative has stages, in the same order.
    """
    n = len(stages)
    columns = min(_MAX_COLUMNS, n)
    rows = math.ceil(n / columns)
    row_h = _HEADER_H + _PANEL_SIZE + _CAPTION_H

    total_w = _MARGIN * 2 + _PANEL_SIZE * columns + _GAP * (columns - 1)
    total_h = _MARGIN * 2 + row_h * rows + _GAP * (rows - 1)

    canvas = Image.new("RGB", (total_w, total_h), _BG)
    draw = ImageDraw.Draw(canvas)

    title_font = _load_font(bold=True, size=13)
    bubble_font = _load_font(bold=False, size=12)
    caption_font = _load_font(bold=False, size=12)

    reframing_total = sum(1 for s in stages if s["type"] == "reframing")
    reframing_seen = 0

    for i, stage in enumerate(stages):
        col = i % columns
        row = i // columns
        x = _MARGIN + col * (_PANEL_SIZE + _GAP)
        y0 = _MARGIN + row * (row_h + _GAP)

        header_bg, header_fg = _TYPE_PALETTE.get(stage["type"], _TYPE_PALETTE["reframing"])
        label = _TYPE_LABELS.get(stage["type"], stage["type"].upper())
        if stage["type"] == "reframing":
            reframing_seen += 1
            if reframing_total > 1:
                label = f"{label} {reframing_seen}/{reframing_total}"

        draw.rectangle([x, y0, x + _PANEL_SIZE, y0 + _HEADER_H], fill=header_bg)
        draw.text((x + 8, y0 + 8), f"{i + 1}. {label}", font=title_font, fill=header_fg)

        img_y = y0 + _HEADER_H
        panel_img = Image.open(panel_paths[i]).convert("RGB").resize((_PANEL_SIZE, _PANEL_SIZE))
        canvas.paste(panel_img, (x, img_y))

        # Comic-strip panel outline (gutter border), drawn after pasting the
        # art so it frames the illustration like a comic panel.
        draw.rectangle(
            [x, img_y, x + _PANEL_SIZE, img_y + _PANEL_SIZE],
            outline=(20, 20, 20), width=_PANEL_BORDER,
        )

        _draw_speech_bubble(canvas, draw, stage.get("dialogue", ""), bubble_font, x, img_y, _PANEL_SIZE)

        cap_y = img_y + _PANEL_SIZE
        draw.rectangle([x, cap_y, x + _PANEL_SIZE, cap_y + _CAPTION_H], fill=_CAPTION_BG)
        _wrap_and_draw(
            draw, stage.get("caption", ""), caption_font,
            (x + 8, cap_y + 6, x + _PANEL_SIZE - 8, cap_y + _CAPTION_H - 6),
            _CAPTION_FG,
        )

    canvas.save(out_path)
    return out_path
