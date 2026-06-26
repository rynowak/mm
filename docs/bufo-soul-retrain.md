# Bufo soul retrain — overnight findings

**Status:** identity + tiling **solved**; expression control still weak. The
structured-caption retrain plus an eval-time recipe gets clean, on-model bufos
with working props and no tiling. Expression (`happy`/`sad`/`angry`/…) is the one
axis that still mostly collapses to neutral — a data problem, not a recipe one.

**Winner:** `sdxl-soul-expr` (neutral-downsampled) **checkpoint-1000**, fused @ **0.80**,
**anti-tiling negative**, **DPM++ 2M Karras @ 1024**. Highest identity of any run
(CLIP 0.795), reliable props, zero tiling. See "v3 neutral-downsample" below.

## What we did

Retrained an SDXL LoRA on the **242 soul-consistent bufos** (Claude-vision curated)
with **structured captions** so training and inference share one schema:

```
bufo, {expression}, {prop}, {pose}, flat cartoon sticker
```

The trigger `bufo` binds the identity; the slots teach control. Recipe:

| Lever | Value | Why |
|---|---|---|
| Base | SDXL 1.0 | native 1024 |
| LoRA | rank 96, alpha 96, attn+FF, **+ text encoder** | enough capacity for identity + slots |
| Train | 1024, ~2000 steps | native res avoids the sample-res tiling we hit before |
| **Eval LoRA scale** | **~0.80** (fused) | rank-96 at full 1.0 over-applies → tiling |
| **Negative prompt** | base + **anti-tiling terms** | kills multi-subject generation |
| Sampler | **DPM++ 2M Karras @ 1024** | clean, deterministic |

The two eval-time levers (`--lora-scale`, anti-tiling negative) are the difference
between "lots of bugs" and clean output. Both are now first-class:
`bufo.eval --lora-scale 0.80` and `bufo/configs/eval-bufo-soul-clean.yaml`.

## Verdict (honest, flaws first)

Judged by eye against the real bufo reference, on the 16-prompt structured eval at
checkpoint-1000, scale 0.80, anti-tiling negative:

- **Tiling — FIXED.** Went from ~50% of cells showing a field of many small frogs
  (at scale 1.0 / default negative) to ~0% with scale 0.80 + anti-tiling negative.
  Tiling was *not* a LoRA-strength problem (0.70 and 0.85 tiled the same) and *not*
  a data-margin problem (training images fill the frame: median 0.96, mean 0.92). It
  was multi-subject generation that the negative prompt suppresses.
- **Soul — HOLDS.** Single-subject cells are on-model: rounded body, earnest face,
  muted olive. These read as bufo, not a generic cartoon frog.
- **Props — WORKING.** This is the payoff the inference levers (img2img / ControlNet
  / IP-Adapter) couldn't give cleanly. Reliable: wizard hat (3/3), flowers, pizza,
  balloon, sword, book, torch→fire, coffee→iced-coffee-cup. Occasional wrong prop
  (beer for flowers).
- **Expression — WEAK.** `happy`/`sad`/`angry`/`concerned`/`smug` mostly collapse to
  neutral or a generic open mouth. Root cause: the data is **46% neutral**
  (112/242), so the LoRA learns a strong "bufo = neutral face" prior that the
  expression word can't overcome.
- **Pose — partial.** `sitting` reads; `arms raised` does not (hallucinates a suit).
- **Minor:** expression-only prompts sometimes get a spurious prop.

## Why CLIP scores are ignored here

CLIP identity sat ~0.74–0.77 across every variant including the tiled ones — it
scores "frogginess," not "is-this-bufo," and is blind to tiling. All verdicts above
are by eye (and, separately, the Qwen VLM judge). CLIP is kept only as a regression
tripwire.

## Convergence + scale sweep (settled)

Resumed to **checkpoint-2000** and ran a clean-negative scale sweep. Both came back
negative for the open questions:

- **checkpoint-2000 is not better than 1000 — slightly worse.** Convergence *overfit*
  the prop-heavy training: expression-only prompts that were clean at 1000 produced
  **off-model creatures at 2000** ("sad, crying" → a pink flamingo; "angry" → a
  dragon), plus a little more tiling creep. CLIP agrees: ckpt-1000 @ 0.80-clean has
  the highest identity (0.768). **Winner stays ckpt-1000 @ scale 0.80.**
- **Scale can't fix expression — it's a hard trade-off.** At **0.60** the prompt
  comes through (adherence 0.796, the best of any run) **but the bufo identity breaks**
  (same flamingo/dragon). At **0.80** identity holds but expression flattens to
  neutral. There's no scale that gives both, because the base model doesn't know
  "bufo" — relaxing the LoRA to let "angry" through also lets the bufo go. Expression
  has to be taught *in the LoRA*, on expressive bufos. → the v3 experiment.

## v3 neutral-downsample (settled) — the winner

`bufo/data_canon_v3` (neutral 46%→28%, 180 imgs), fresh LoRA to step 1000, eval @
0.80-clean. Result, by eye and CLIP:

- **Best identity of any variant** (CLIP 0.795 vs base ckpt-1000's 0.768). Rock-solid:
  the off-model creatures that convergence introduced are **gone** — "sad, crying" and
  "angry" stay on-model bufos.
- **Props intact** despite the smaller set: wizard hat 3/3, flowers 3/3, pizza, sword,
  book, torch, balloon, coffee.
- **No tiling.**
- **Expression: marginally better, still the limiting axis.** A bit more variation
  (tongue, grimace, crossed arms, a tear) but `happy`/`sad`/`angry` are still subtle.
  Downsampling neutral *reduced the neutral prior* (helping identity robustness) but
  didn't *add* expressive examples — there are still only 7–26 per expression. So the
  direction is right; the dose isn't enough.
- Minor residue: `arms raised` → hallucinated suit; occasional prop confusion
  (book→flag, balloon→lollipop).

**Production pick: `sdxl-soul-expr` checkpoint-1000, fused @ 0.80, anti-tiling
negative, DPM++ 2M Karras @ 1024.** Contact sheets on the cluster:
`/mnt/ray/bufo-runs/sdxl-soul-expr/eval-1000-s080-clean/` (winner) and
`/mnt/ray/bufo-runs/sdxl-soul/eval-1000-s080-clean/` (base, for comparison).

## Recommendation

1. **Ship v3 as the soul LoRA** with the eval recipe above. Identity, props, and clean
   output are solved — the original "they don't look like bufos" + "lots of bugs"
   complaints are addressed.
2. **Expression is the next data task, not a recipe one.** Two concrete moves, in
   order of expected payoff:
   - *Enrich the expressive set*: pull more genuinely-expressive bufos from the ~1400
     raw corpus (the curation optimized for soul-consistency, which skews neutral).
     Target ≥30 strong examples each for happy/sad/angry/surprised.
   - *Make expression captions visually specific* while keeping the schema — e.g.
     `angry` → `angry, furrowed brow, frown`; `sad` → `sad, downturned mouth, teary`.
     Enrich the eval prompts to match (keep train/inference symmetry).
3. Don't pursue lower LoRA scale for expression — it trades away the bufo identity
   (proven by the 0.60 sweep).

## Recommendation

Ship the recipe (scale 0.80 + anti-tiling negative + DPM++ 2M Karras @1024) as the
default for the soul LoRA — identity, props, and clean output are there. Treat
**expression as the next data task**: rebalance away from neutral and/or enrich the
expression captions, then a short retrain. Don't chase expression with eval-time
tricks alone; it's in the training distribution.

---

## UPDATE — pivot to FLUX (supersedes the SDXL recommendations above)

The SDXL soul LoRA above was rejected on a close look: **cursed anatomy** (melted
eyes/mouths) — SDXL's coherence cap, which no recipe tweak fixes. We pivoted to FLUX,
which is coherent. The earlier FLUX failures were *style transfer*, not coherence: the
LoRA was attention-only rank-32 (a concept LoRA). Training **attention + all MLP
layers at rank 64** (`lora-flux-style.yaml`) on the structured data finally takes the
bufo style while staying coherent.

- **flux-soul-style** (v3 data): coherent + on-style + props + expression all working
  (identity 0.789 / adherence 0.800, best of any run). ~50% of cells near-perfect.
- **Remaining: consistency.** (1) warm-food prompts (coffee/pizza) recolor the frog
  **orange**; (2) ~30% of seeds drift to a **flat clipart** style.
- **v4** (`data_canon_v4`, running): re-caption `green bufo, …, soft-shaded cartoon
  sticker` to anchor color + shading. (`flat cartoon sticker` was actively wrong — bufo
  is soft-shaded.)
- Flux LoRA eval load: diffusers' `load_lora_weights` drops the attention adapter on
  the round-trip → load via the peft path (`--lora-config`, see `_apply_flux_lora`).

### Canonical bufo anatomy (off-model tells to encode — task #24)

The model gets the gestalt but drifts on specifics. Bake into the caption schema /
negative prompt in the next pass:
- **Short, stubby arms and legs** — drift-frogs render long thin limbs (negative:
  "long legs, lanky").
- **Long tongue** — a bufo signature; under-used. Add as a positive cue.
- **No teeth** — the "angry" cell rendered fangs (negative: "teeth, fangs").
- (v4) muted-**green** body, **soft-shaded** (not flat).
