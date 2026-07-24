"""Generate a transparent-background bufo emoji STICKER SET from the SD3.5 full-FT model.

v2 — RICH PROMPTS. Each emoji carries a list of rich expression phrases + an optional
action; each of the CAND candidates uses a *different* expression and a *different* framing
(genuine variety, not reseeds of one prompt). Identity + style anchors are kept fixed so the
character holds; "simple rounded hands" + hand negatives steer away from cursed fingers.

Output: per emoji, CAND candidates -> rembg cutout -> auto-trim + center -> 1024 master +
128px Slack PNG (RGBA). Incremental (AGENTS.md): each sticker written as made, run resumes by
skipping existing masters, progress streams with ETA.

Env: FT, BASE, OUT, RES (def 1024), CAND (def 4), SLACK (def 128),
EMOJI_LIMIT (first N emojis), EMOJI_NAMES (comma-separated subset; for tests).
"""

from __future__ import annotations

import json
import os
import time

import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from PIL import Image, ImageDraw

BASE = os.environ.get("BASE", "stabilityai/stable-diffusion-3.5-medium")
FT = os.environ.get("FT", "/mnt/ray/bufo-keep/sd35-medium-ft-1000/transformer")
OUT = os.environ.get("OUT", "/mnt/ray/bufo-runs/sd35-medium-ft/stickers-v2")
RES = int(os.environ.get("RES", "1024"))
CAND = int(os.environ.get("CAND", "4"))
SLACK = int(os.environ.get("SLACK", "128"))
EMOJI_LIMIT = int(os.environ.get("EMOJI_LIMIT", "0"))
EMOJI_NAMES = [s.strip() for s in os.environ.get("EMOJI_NAMES", "").split(",") if s.strip()]

IDENTITY = (
    "olive green adult bufo, a plump matte cartoon frog with a big rounded head, "
    "large round forward-set eyes, and simple rounded hands"
)
STYLE = "soft-shaded cartoon sticker, smooth matte shading, bold simple shapes, plain white background"
NEG = (
    "deformed, blurry, low quality, extra limbs, extra arms, malformed hands, extra fingers, "
    "fused fingers, claws, long fingers, teeth, fangs, text, watermark, photo, realistic, 3d render"
)
# Per-candidate framing — rotated across the CAND candidates for compositional variety.
FRAMINGS = [
    "front view, head and shoulders",
    "three-quarter view, upper body",
    "looking up, head and shoulders",
    "full body, sitting",
]

# (name, [rich expression phrases...], action)  action="" for pure-expression emoji.
EMOJIS: list[tuple[str, list[str], str]] = [
    # --- emotions / reactions ---
    (
        "happy",
        [
            "a wide warm smile, cheeks raised, bright eyes",
            "a big open-mouth grin, eyes curved with joy",
            "a gentle content smile, soft happy eyes",
            "beaming with closed happy eyes, rosy cheeks",
        ],
        "",
    ),
    (
        "grin",
        [
            "a huge toothless grin stretching wide, sparkling eyes",
            "a mischievous wide grin, raised brows",
            "a giddy ear-to-ear smile, shining eyes",
            "a goofy big grin, cheeks bunched up",
        ],
        "",
    ),
    (
        "sad",
        [
            "a downturned frown, droopy half-closed eyes",
            "a sorrowful pout, glistening eyes, lowered brows",
            "a quivering frown, big watery eyes",
            "a dejected look, eyes cast downward",
        ],
        "",
    ),
    (
        "cry",
        [
            "teary eyes with a single tear rolling down, wobbly frown",
            "watery eyes, one big tear, sad mouth",
            "glassy eyes brimming with tears, trembling lip",
            "a tear streaking down, downturned mouth",
        ],
        "",
    ),
    (
        "sob",
        [
            "streams of tears, mouth open wailing, eyes scrunched shut",
            "bawling with two waterfall tears, wide open mouth",
            "heavy sobbing, scrunched eyes, tears flying",
            "crying hard, open frowning mouth, gushing tears",
        ],
        "",
    ),
    (
        "angry",
        [
            "furrowed brows, a deep frown, narrowed eyes",
            "scowling with lowered angry brows",
            "a cross expression, puffed cheeks, sharp glare",
            "an irritated frown, eyes squinting in anger",
        ],
        "",
    ),
    (
        "rage",
        [
            "a furious red face, sharp V-shaped brows, shouting mouth, steam puffs",
            "an enraged glare, gritted open mouth, red cheeks",
            "seething fury, angry brows, steam rising from the head",
            "explosive anger, wide shouting mouth, red flush",
        ],
        "",
    ),
    (
        "love",
        [
            "large pink heart-shaped eyes, a dreamy smile, blush",
            "heart eyes and a loving grin, rosy cheeks",
            "a smitten expression, heart pupils, soft smile",
            "an adoring look, heart eyes, hands near the cheeks",
        ],
        "",
    ),
    (
        "laugh",
        [
            "eyes squeezed shut laughing, wide open smiling mouth",
            "head tipped back laughing hard, closed happy eyes",
            "joyful laughter, big open grin, bunched cheeks",
            "giggling with closed crescent eyes, open smile",
        ],
        "",
    ),
    (
        "smug",
        [
            "a sly smirk, half-lidded confident eyes, one brow raised",
            "a knowing smirk, lidded eyes",
            "a self-satisfied grin, raised chin",
            "a cheeky smirk, narrowed pleased eyes",
        ],
        "",
    ),
    (
        "surprised",
        [
            "wide round eyes, a small open mouth, raised brows",
            "startled wide eyes, a gasping o-shaped mouth",
            "shocked raised brows, big eyes, open mouth",
            "taken aback, wide eyes, a hand near the mouth",
        ],
        "",
    ),
    (
        "shocked",
        [
            "huge wide eyes, jaw dropped open, raised brows",
            "a stunned face, enormous eyes, gaping mouth",
            "an aghast expression, shrunken pupils, open mouth",
            "frozen in shock, wide staring eyes",
        ],
        "",
    ),
    (
        "confused",
        [
            "a puzzled look, one brow raised, slight frown, head tilted",
            "a baffled expression, squiggly mouth, tilted head",
            "a questioning look, raised brow, pursed mouth",
            "a perplexed look, eyes glancing aside, small frown",
        ],
        "",
    ),
    (
        "think",
        [
            "a thoughtful look, eyes glancing up",
            "a pondering expression, raised brow",
            "a contemplative look, gazing upward",
            "a musing look, eyes to the side",
        ],
        "one rounded hand resting on the chin",
    ),
    (
        "sleepy",
        [
            "heavy half-closed droopy eyes, a tiny yawn",
            "drowsy lidded eyes, a relaxed mouth",
            "a sleepy expression, one eye closed",
            "nodding off, half-lidded eyes, a small sleep bubble",
        ],
        "",
    ),
    (
        "cool",
        ["a calm confident smirk", "a relaxed cool smile", "a smug grin, chin up", "an effortless cool expression"],
        "wearing dark sunglasses",
    ),
    (
        "nervous",
        [
            "an anxious smile, a single sweat drop, wide eyes",
            "a worried grimace, a sweat bead, darting eyes",
            "an uneasy forced smile, a sweat drop, raised brows",
            "a tense expression, a sweat drop",
        ],
        "",
    ),
    (
        "pleading",
        [
            "huge teary puppy-dog eyes, a small trembling frown",
            "big glossy pleading eyes",
            "imploring teary eyes, a quivering lip",
            "wide hopeful watery eyes",
        ],
        "both rounded hands clasped together",
    ),
    (
        "dead",
        [
            "x-shaped eyes, tongue lolling out, a limp expression",
            "spiral dizzy eyes, a flat mouth",
            "x eyes, a slack open mouth",
            "a knocked-out face, x eyes, tongue out",
        ],
        "",
    ),
    (
        "wink",
        [
            "one eye closed in a playful wink, a cheeky smile, tongue out",
            "winking with a sly grin",
            "a friendly wink, a big smile",
            "a playful wink, tongue poking out, raised brow",
        ],
        "",
    ),
    (
        "blush",
        [
            "rosy blushing cheeks, a shy bashful smile, eyes averted",
            "a deep blush, a timid smile, looking away",
            "a flustered blush, a small smile",
            "bashful pink cheeks, a shy glance",
        ],
        "",
    ),
    (
        "scared",
        [
            "wide frightened eyes, a trembling open mouth, sweat",
            "a terrified face, shrunken pupils, open mouth",
            "fearful wide eyes, shaking, pale",
            "a petrified expression, huge eyes, sweat drops",
        ],
        "",
    ),
    (
        "bored",
        [
            "a flat unamused expression, half-lidded eyes, straight-line mouth",
            "a deadpan look, droopy eyes",
            "an uninterested face, eyes glazed",
            "a listless bored look, flat mouth",
        ],
        "",
    ),
    (
        "mindblown",
        [
            "amazed wide eyes, mouth agape",
            "an astonished expression, sparkles, wide eyes",
            "a blown-away look, huge eyes, open mouth",
            "an awestruck look, starry eyes",
        ],
        "both rounded hands on the head",
    ),
    # --- hand gestures ---
    (
        "thumbsup",
        ["a confident cheerful smile", "a friendly grin", "a pleased smile", "an encouraging smile"],
        "one arm raised giving a clear thumbs-up with a simple rounded hand",
    ),
    (
        "thumbsdown",
        ["a disapproving frown", "an unimpressed flat look", "a displeased frown", "a let-down expression"],
        "one arm lowered giving a thumbs-down with a simple rounded hand",
    ),
    (
        "wave",
        ["a warm friendly smile", "a cheerful welcoming grin", "a happy greeting smile", "a bright hello smile"],
        "one rounded hand raised waving hello",
    ),
    (
        "salute",
        ["a serious respectful expression", "a proud dutiful look", "a crisp confident look", "a solemn expression"],
        "one rounded hand raised to the brow in a salute",
    ),
    (
        "facepalm",
        ["an exasperated look, eyes closed", "a weary disappointed expression", "an embarrassed wince", "a tired sigh"],
        "one rounded hand covering the face",
    ),
    (
        "pray",
        [
            "a hopeful serene expression, eyes closed",
            "a grateful peaceful look",
            "a wishful look, eyes shut",
            "a thankful calm face",
        ],
        "both rounded hands pressed together in prayer",
    ),
    (
        "clap",
        ["a delighted smile", "a cheerful impressed grin", "an approving happy look", "an excited smile"],
        "both rounded hands together clapping, small motion lines",
    ),
    (
        "ok",
        ["a reassuring smile", "a cheerful confident look", "a friendly grin", "an approving smile"],
        "one rounded hand making an ok sign",
    ),
    (
        "shrug",
        [
            "a puzzled indifferent look, raised brows",
            "an unsure expression",
            "a clueless look, slight frown",
            "a casual whatever look",
        ],
        "both rounded hands raised palms-up in a shrug",
    ),
    # --- props / objects ---
    (
        "coffee",
        ["a cozy content smile", "a sleepy relaxed look", "a cheerful morning smile", "a satisfied sip expression"],
        "holding a steaming coffee mug in both rounded hands",
    ),
    (
        "heart",
        [
            "a loving warm smile, soft eyes",
            "an affectionate look, blush",
            "a tender happy smile",
            "a caring gentle look",
        ],
        "holding a big red heart in both rounded hands",
    ),
    (
        "fire",
        ["a determined look", "a mischievous grin", "a fierce expression", "a bold adventurous look"],
        "holding up a lit wooden torch with a bright flame",
    ),
    (
        "pizza",
        ["a delighted hungry smile", "a happy grin", "an eager excited look", "a satisfied yum expression"],
        "holding a slice of pepperoni pizza",
    ),
    (
        "cake",
        ["a joyful birthday smile", "a delighted grin", "an excited happy look", "a celebratory smile"],
        "holding a small birthday cake with lit candles",
    ),
    (
        "flowers",
        ["a gentle sweet smile, soft eyes", "a content happy look", "a shy warm smile", "a cheerful kind look"],
        "holding a colorful bouquet of flowers",
    ),
    (
        "book",
        ["a focused curious look", "a calm reading expression", "an absorbed interested look", "a studious expression"],
        "reading an open book held in both rounded hands",
    ),
    (
        "phone",
        ["an absorbed look, eyes on the screen", "a slight amused smile", "a focused scrolling look", "a relaxed look"],
        "looking down at a smartphone held in both rounded hands",
    ),
    (
        "popcorn",
        [
            "an entertained grin, wide interested eyes",
            "an amused look",
            "a gleeful watching expression",
            "an engrossed look",
        ],
        "holding a striped bucket of popcorn",
    ),
    (
        "money",
        ["a greedy delighted grin", "an excited thrilled look", "a smug rich grin", "a gleeful look"],
        "holding a fan of cash with both rounded hands",
    ),
    (
        "gaming",
        [
            "a focused excited look, intense eyes",
            "a competitive grin",
            "an absorbed determined look",
            "a thrilled gamer face",
        ],
        "holding a game controller with both rounded hands",
    ),
    (
        "balloon",
        ["a cheerful smile", "a happy delighted look", "a carefree grin", "a bright joyful smile"],
        "holding the string of a round balloon",
    ),
    (
        "gift",
        ["a delighted surprised smile", "a cheerful grin", "an excited happy look", "a warm giving smile"],
        "holding a wrapped present with a ribbon bow",
    ),
    (
        "music",
        [
            "a blissful relaxed look, eyes closed",
            "a happy groovy smile",
            "a content vibing look",
            "a cheerful bopping expression",
        ],
        "wearing headphones with floating music notes",
    ),
    # --- costumes / hats ---
    (
        "wizard",
        ["a wise mysterious look", "a curious expression", "a mischievous grin", "a serene knowing look"],
        "wearing a tall pointed wizard hat with stars",
    ),
    (
        "crown",
        ["a regal proud expression, chin up", "a haughty smile", "a noble dignified look", "a confident royal smile"],
        "wearing a shiny golden crown with jewels",
    ),
    (
        "party",
        [
            "a festive joyful grin",
            "an excited celebrating smile",
            "a delighted party face",
            "a cheerful whoop expression",
        ],
        "wearing a striped party hat with confetti falling",
    ),
    (
        "chef",
        ["a proud confident smile", "a focused cooking look", "a cheerful kitchen grin", "a satisfied chef expression"],
        "wearing a tall white chef's hat",
    ),
    (
        "graduate",
        ["a proud accomplished smile", "a hopeful look", "a cheerful celebratory grin", "a dignified happy look"],
        "wearing a graduation cap with a tassel",
    ),
    (
        "detective",
        [
            "a suspicious squint, one brow raised",
            "a focused investigating look",
            "a shrewd curious expression",
            "a sharp inquisitive look",
        ],
        "wearing a detective deerstalker hat and holding a magnifying glass",
    ),
    (
        "santa",
        ["a jolly warm smile", "a merry grin", "a cheerful festive look", "a kind ho-ho-ho smile"],
        "wearing a red santa hat with a fluffy white pom-pom",
    ),
    (
        "cowboy",
        ["an easygoing confident smirk", "a relaxed grin", "a laid-back cool look", "a friendly drawl smile"],
        "wearing a brown cowboy hat",
    ),
]


def build_prompt(expr: str, action: str, framing: str) -> str:
    parts = [IDENTITY, expr]
    if action:
        parts.append(action)
    parts.extend([framing, STYLE])
    return ", ".join(parts)


def cutout(img: Image.Image, size: int) -> Image.Image:
    """rembg cutout -> trim to alpha bbox -> center on a square transparent canvas."""
    from rembg import remove

    rgba = remove(img.convert("RGBA"))
    bbox = rgba.getbbox()
    if bbox:
        rgba = rgba.crop(bbox)
    side = max(rgba.size)
    margin = int(side * 0.08)
    canvas = Image.new("RGBA", (side + 2 * margin, side + 2 * margin), (0, 0, 0, 0))
    canvas.paste(rgba, ((canvas.width - rgba.width) // 2, (canvas.height - rgba.height) // 2), rgba)
    return canvas.resize((size, size), Image.LANCZOS)


def main() -> None:
    masters = os.path.join(OUT, "masters")
    slack = os.path.join(OUT, "slack")
    os.makedirs(masters, exist_ok=True)
    os.makedirs(slack, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} FT={FT} OUT={OUT} CAND={CAND}", flush=True)
    if dev.type != "cuda":
        raise SystemExit("no CUDA — refusing to generate on CPU (too slow)")

    emojis = EMOJIS
    if EMOJI_NAMES:
        emojis = [e for e in emojis if e[0] in EMOJI_NAMES]
    elif EMOJI_LIMIT > 0:
        emojis = emojis[:EMOJI_LIMIT]

    transformer = SD3Transformer2DModel.from_pretrained(FT, torch_dtype=torch.bfloat16)
    pipe = StableDiffusion3Pipeline.from_pretrained(BASE, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe = pipe.to(dev)
    pipe.set_progress_bar_config(disable=True)

    jobs = [(name, exprs, action, c) for (name, exprs, action) in emojis for c in range(CAND)]
    todo = [j for j in jobs if not os.path.exists(os.path.join(masters, f"{j[0]}_{j[3]}.png"))]
    print(f"total={len(jobs)} todo={len(todo)} (resuming, {len(jobs) - len(todo)} done)", flush=True)

    man = open(os.path.join(OUT, "manifest.jsonl"), "a")  # noqa: SIM115 (streaming append across loop)
    t0 = time.time()
    for i, (name, exprs, action, c) in enumerate(todo):
        expr = exprs[c % len(exprs)]
        framing = FRAMINGS[c % len(FRAMINGS)]
        prompt = build_prompt(expr, action, framing)
        g = torch.Generator(device=dev).manual_seed(100 + c)
        img = pipe(
            prompt=prompt,
            negative_prompt=NEG,
            num_inference_steps=28,
            guidance_scale=4.5,
            height=RES,
            width=RES,
            generator=g,
        ).images[0]
        master = cutout(img, RES)
        mp = os.path.join(masters, f"{name}_{c}.png")
        sp = os.path.join(slack, f"{name}_{c}.png")
        master.save(mp)
        cutout(img, SLACK).save(sp)
        man.write(json.dumps({"name": name, "cand": c, "prompt": prompt, "master": mp, "slack": sp}) + "\n")
        man.flush()
        done = i + 1
        rate = (time.time() - t0) / done
        eta = rate * (len(todo) - done)
        print(f"[{done}/{len(todo)}] {name}_{c}  {rate:.1f}s/img  eta {eta / 60:.1f}m", flush=True)
    man.close()

    # contact sheet: one row per emoji, CAND candidates across, composited on grey + label.
    cell = 240
    label_h = 22
    rows = len(emojis)
    sheet = Image.new("RGB", (CAND * cell, rows * (cell + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, (name, _exprs, _action) in enumerate(emojis):
        for c in range(CAND):
            mp = os.path.join(masters, f"{name}_{c}.png")
            if not os.path.exists(mp):
                continue
            im = Image.open(mp).convert("RGBA")
            im.thumbnail((cell - 12, cell - 12))
            tile = Image.new("RGB", (cell, cell), (235, 235, 235))
            tile.paste(im, ((cell - im.width) // 2, (cell - im.height) // 2), im)
            sheet.paste(tile, (c * cell, r * (cell + label_h)))
        draw.text((4, r * (cell + label_h) + cell + 4), name, fill=(0, 0, 0))
    sheet.save(os.path.join(OUT, "contact_sheet.png"))
    print("DONE wrote", os.path.join(OUT, "contact_sheet.png"), flush=True)


if __name__ == "__main__":
    main()
