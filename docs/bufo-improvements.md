# Bufo generator — improvement design

Status: proposed · Scope: the `bufo/` sample · Extends v1

## Context

The `bufo/` sample (v1) LoRA-fine-tunes Stable Diffusion 1.5 on the
[all-the-bufo](https://github.com/knobiknows/all-the-bufo) corpus (1,386 usable
PNGs) to generate bufo frog emoji. A 500-step run (~23 min, MPS) already produces
recognizable bufos that compose with unseen prompts (astronaut, pizza, king).

What's weak today, and what this design fixes:

| Gap (v1) | Symptom | Axis |
|----------|---------|------|
| Captions auto-derived from filenames, buggy | `"a bufo of offers cash money"`, dangling `a` on long names | Data |
| Captions ignore the actual image | No color/pose/prop/expression grounding | Data |
| No identity anchor / emoji interface | Prompts ad-hoc; cowboy rendered a literal *person* (concept bleed) | Prompting |
| No quantitative eval | We eyeball grids; can't tell what helped | Evals |
| Conservative training | rank 16, UNet-only, 1.4 epochs — undertrained | Training method |
| Old base model | SD 1.5 fidelity ceiling | Base model |
| No data review | 17% of captions are long "scene" narratives; never inspected | Data |

The end goal is **Slack-emoji generation**: a user types `:bufo-offers-cash-money:`
(or free-form "sad monday bufo") and gets a matching bufo. Quality is measured,
not vibed.

## Goals / non-goals

**North star:** clean **flat-cartoon** bufos that read as a **small square Slack
emoji (~48px)** — see *Definition of a good bufo* below.

**Goals.** (1) A CLIP eval harness so every change is measured. (2) A unified
`understand → design → rewrite → generate` pipeline that links training captions
and inference prompts through one shared schema. (3) Data quality on three fronts:
*fix*, *filter*, *recaption*. (4) A stronger training recipe (rank, text-encoder
LoRA, longer watched runs). (5) An SDXL path behind a flag. (6) Inference-side
wins (negative prompts, CLIP-reranked sampling, an emoji interface). (7) A
human-preference loop — rate → calibrate the eval / best-of-N / Diffusion-DPO.

**Non-goals.** Animated/GIF bufos. A hosted service. Training a base model from
scratch. **Prose prompt-upsampling** — our rewriter normalizes *into* the terse
sticker schema (≤77 CLIP tokens), it does not expand into DALL·E-style paragraphs;
that would push prompts off the LoRA's trained distribution. **Cursed/crude bufos
and photorealism** are excluded by the aesthetic target below.

---

## The organizing idea: one pipeline, applied at two times

The generator is a single pipeline:

```
understand  →  design  →  rewrite  →  generate
```

It runs at **training time** (to build captions) and at **inference time** (to
build prompts), and the two are deliberately symmetric:

| Stage | Training time (recaption) | Inference time |
|-------|---------------------------|----------------|
| **understand** | read the bufo's name + look at the image (VLM) | parse the user's intent (`:bufo-pls:`, "sad monday bufo") |
| **design** | pick the canonical caption fields | decide what the bufo should depict |
| **rewrite** | emit a caption in the schema | emit a prompt in the *same* schema |
| **generate** | — (train the LoRA on it) | run the diffusion model |

**The shared contract is the caption schema** — one grammar both ends rewrite
into:

```
bufo <action>, <expression>, <notable props/colors>, flat cartoon frog emoji sticker, bold simple shapes, white background
```

Because training captions and inference prompts occupy **one distribution**, the
model never sees a prompt shaped unlike what it trained on — the maximum-leverage
condition for prompt adherence. This is the DALL·E 3 insight: *better captions +
a matching rewriter*. The schema is terse on purpose (CLIP truncates at 77
tokens); for a narrow sticker LoRA you **normalize into** this dialect rather than
expand out of it.

Everything below is an instance of this pipeline or the harness that measures it.

## Definition of a good bufo (the `design` target)

A "good" bufo is optimized for its destination — a **small square Slack emoji
(~32–64px)**:

- **Flat cartoon** style, consistent green-frog identity (recognizably the same
  character).
- **Frog-dominant** — the bufo fills most of the frame; objects/backgrounds are
  allowed but also clean cartoon, never photoreal.
- **Legible at 48px** — bold shapes, few elements, high contrast; survives
  downscaling.

Explicitly **excluded**: "cursed/crude" bufos (deliberately ugly/distorted meme
drawings), photorealistic renders, text-heavy panels, and busy scenes where the
frog is tiny.

This is the shared target for the `design` stage — it drives what we *filter
toward* (A2/A5), what we *prompt for* (B), and what we *measure* (C, at emoji
size). Tension: the corpus is multi-artist/multi-style, so "consistent" means
filtering toward a coherent cartoon band + leaning on the trigger token to average
to a canonical frog.

## Data sources

| Source | Content | Useful for | Status |
|--------|---------|-----------|--------|
| [all-the-bufo](https://github.com/knobiknows/all-the-bufo) | ~1,400 PNGs, names = shortcodes | Primary corpus (already used) | ✅ in use |
| [bufo.zone](https://bufo.zone/) (`all-the.bufo.zone`) | Superset incl. GIFs, **1–5 🐸 community ratings** | Quality-weighted curation; more PNGs | ⚠️ scrape only, no API |
| [bufo.fun](https://bufo.fun/) | "Bufo Emoji Repository" | Possible extra images | ❓ JS-rendered, investigate |
| [bufopedia.com](https://bufopedia.com/) | Bufo reference site | Possible descriptions/labels | ❓ JS-rendered, investigate |

bufo.zone's per-bufo ratings are the most interesting external signal — a cheap
quality filter (train on ≥N-star bufos). Optional data-expansion, gated on review;
the GitHub corpus stays the baseline.

---

## A. Data — fix, filter, recaption

Three distinct capabilities, all feeding the curated `metadata.jsonl`.

### A1. Fix — the caption schema + deterministic filename mapping

`bufo/data.py::caption_for()` strips the substring `bufo` from each token, which
on `...-a-bufo-should-be-able-to-fly` leaves a dangling `a` and turns
`bufo-offers-cash-money` into `"a bufo of offers cash money"`. Replace with the
shared schema builder, used by captions **and** the inference interface:

```python
TRIGGER = "bufo"
SUFFIX  = ", flat cartoon frog emoji sticker, bold simple shapes, white background"

def filename_phrase(name: str) -> str:
    # whole-token removal of "bufo"/"smol", keep the action verbatim
    # "bufo-offers-cash-money" -> "offers cash money"
def caption_for(name: str) -> str:                 # the schema, action-only fields empty
    return f"{TRIGGER} {filename_phrase(name)}{SUFFIX}".strip()
```

This is the deterministic floor; recaption (A5) enriches `<expression>` and
`<props/colors>` on top of it. Verb forms stay raw (train and inference share the
path); leave an empty `VERB_FIXUPS` seam.

### A2. Filter — review tooling (you decide what stays)

- **`bufo/curate.py`** — heuristics + report: token count, `scene_like`
  (≥6 tokens ≈ the 231 long names), aspect ratio, `alpha_coverage`,
  **frog-dominance** (foreground fraction — a small frog in a busy frame fails at
  48px), an 8×8 average-hash (hand-rolled, no `imagehash` dep) for near-dup
  grouping, and a cheap text-in-image proxy. Emits `curate-report.{json,csv}`.
  CLI: `python -m bufo.curate report`.
- **Model-based quality signals** (toward the *good bufo* target, not popularity):
  CLIP **style score** (`cos(img,"flat cartoon sticker") − cos(img,"photograph")`),
  a **48px-legibility** check (downscale → concept-score), and VLM flags
  (cartoon? frog prominent? busy/text?) emitted during A5. Community **ratings are
  deliberately not used** — they track meme-worthiness (cursed bufos rate *high*),
  not training quality.
- **`bufo/gallery.py`** — single-file FastAPI app mirroring
  `wordle/dashboard/app.py` (inline CSS + htmx, no new deps). Browse all ~1,400
  with caption + heuristic chips, filter by category, toggle **keep/drop**.
  Chosen over static contact sheets because you want to *decide*, not just look.

### A3. Curation feedback loop

Decisions land in **`bufo/data/curation.jsonl`** (git-tracked), keyed by
`file_name` with `keep: bool` + optional `caption` override. `prepare()` applies
it once after preprocess, so `metadata.jsonl` becomes the curated manifest and
`BufoDataset` is unchanged. (Recaptioned text in A5 also flows through this file.)

### A4. Preprocessing

Square white-padding teaches white borders. Add a center-`crop` option; default to
crop for cleaner stickers. Optional duplicate removal from A2's hash groups.

### A5. Recaption — model-rewritten training captions ★

The filename encodes the *action* well (humans named each bufo by what it does),
but ignores the *image*. Recaptioning adds visual grounding (color, pose, props,
expression) and consistency — the training-time half of the unified pipeline.

- **`bufo/recaption.py`** — for each image: a small local **VLM**
  (moondream / Qwen2-VL-2B / LLaVA-class, via `transformers`, runs on MPS) reads
  the picture; a small instruct **LLM** merges *(filename action + VLM detail)* →
  one caption in the A1 schema. Output is written as `caption` overrides in
  `curation.jsonl` (A3), so it's inspectable and revertible.
- **Honest nuance:** the win here is grounding + consistency, *not* the core
  action (the name already nails that). So merge shortcode-intent with VLM-detail;
  don't drop the name.
- **Eval-gated:** A/B *filename captions* vs *recaptioned* on the eval harness's
  prompt-adherence + concept-fidelity metrics. Keep whichever wins — decide with a
  number. No new pip deps (VLM/LLM weights load via transformers; flag possible
  `sentencepiece` for some tokenizers).

---

## B. Emoji interface — the inference half of the pipeline

The inference `understand → design → rewrite` path. Same schema target as A5, so
the two ends stay aligned.

```python
def shortcode_to_prompt(code: str) -> str:
    # ":bufo-offers-cash-money:" -> schema prompt, via the SAME filename_phrase
    return caption_for(code.strip(":"))
```

- **`bufo/rewrite.py`** — a `Rewriter` Protocol with `rewrite(query) -> prompt`:
  - **`RulesRewriter`** (default): `shortcode_to_prompt`. Deterministic; sufficient
    when input already looks like a shortcode.
  - **`LLMRewriter`** (in-scope, measured): a small local instruct model maps
    *arbitrary* user intent → the A1 schema. This is what makes free-form Slack
    input ("sad monday bufo") work, and it's the same technique DALL·E 3 uses. Its
    target is the schema, not prose. Judged against rules on prompt-adherence; if
    it wins on free-form inputs, it becomes default.
- **Trigger token:** plain `bufo`, leading. Single-subject LoRA, so SD's frog
  prior helps; concept bleed was a *captioning* problem, fixed by the schema. A
  rare token / textual inversion stays an escape hatch via the `TRIGGER` constant.
- **`sample.py`:** add `--emoji :bufo-x-y:` (repeatable), `--rewriter
  {rules,llm}`, a default **negative prompt** (`photo, realistic, 3d render,
  cluttered, tiny, text, watermark`), and `--rerank` (generate N, keep top-K by
  CLIP prompt-adherence — reuses the eval embedder).

---

## C. Evaluation harness

Pure math split from orchestration so metrics are unit-testable with no model
download (mirrors `wordle3` `metrics.py` vs `steplog.py`).

- **`bufo/clip_metrics.py`** — `ClipEmbedder` (transformers `CLIPModel` +
  `CLIPProcessor`, `openai/clip-vit-base-patch32`; **no new dependency**) + pure
  functions over L2-normalized embeddings:

  | Metric | Definition | Catches |
  |--------|-----------|---------|
  | Concept fidelity | `2.5·max(cos(img, "a bufo, a green cartoon frog sticker"), 0)` | "is it a bufo?" |
  | Prompt adherence (CLIPScore) | `2.5·max(cos(img, its prompt), 0)` | "does it do the thing?" |
  | Diversity | mean pairwise `1−cos` (per-prompt + overall) | mode collapse |
  | Memorization | per image, `max cos` to ~1.4k **train** embeddings | copying training data |
  | Legibility @48px | concept fidelity after downscale→48px→upscale | survives emoji size |
  | Cartoon style | `cos(img,"flat cartoon sticker") − cos(img,"photograph")` | photoreal / off-style |

- **`bufo/configs/eval-bufo.yaml`** — fixed **held-out** prompt set (~24
  emoji-style actions absent from training) + a 4-prompt `step_prompts` subset for
  cheap in-training eval. Subjects only; harness appends `SUFFIX`.
- **`bufo/eval.py`** — `evaluate(checkpoint, ...) -> EvalScorecard`: reuse
  `load_inference_pipeline`, seeded generation, embed, score, write
  `scorecard.json` + labeled `contact_sheet.png`. CLI runs with `--lora` or with
  none (the base-model baseline). Train embeddings cached in
  `runs/.cache/clip_train_emb/` keyed by `(clip_model, file stats)`.
- **In-training `EvalReporter`** — beside `_snapshot()` in `train_lora.py`: loads
  CLIP once, reuses cached train embeddings, logs `eval/*` scalars every
  `eval_interval` to the same TensorBoard as `train/loss`. Watch fidelity rise and
  memorization stay flat; catch the overfit knee.
- **This is what makes A5 and B honest** — the A/B experiments (filename vs
  recaptioned; rules vs LLM rewriter) are decided here, not by hunch.
- **Tests** — Tier 1 (offline): metric math on fake embeddings + config
  invariants. Tier 2 (gated `BUFO_CLIP_SMOKE=1`): real embedder + cache round-trip.

---

## D. Training method

- **Rank 16 → 32**; `alpha = rank`.
- **Text-encoder LoRA** (`train_text_encoder`, default off) — biggest
  prompt-adherence lever for a new concept; CLIP attn targets
  `q_proj/k_proj/v_proj/out_proj`.
- **`cast_training_params`** (diffusers) replaces the manual fp32 loop, handles a
  model list.
- **Longer, watched runs** — 1,500–3,000 steps, stop at the
  memorization/diversity knee via the scorecard, not a fixed count.
- **Optional min-SNR weighting** (γ=5) behind a flag; measure before keeping.
- Extend `create_optimizer`/`clip_grad_norm` to a param iterable only when
  enabling text-encoder LoRA; ship UNet-only first.

---

## E. SDXL path (behind a flag)

SD 1.5 and SDXL differ in three places — component loading, conditioning prep,
save/load — everything else is identical. Branch on `base_kind: "sd15" | "sdxl"`
and isolate the divergence behind one function so the loop stays single-path:

```python
def encode_conditioning(comp, batch, *, resolution, device, dtype) -> dict:
    if comp.base_kind == "sd15":
        return {"encoder_hidden_states": comp.text_encoder(batch["input_ids"])[0]}
    # SDXL: concat PENULTIMATE hidden states of both encoders (768+1280=2048),
    # pooled embeds from encoder 2, + a constant 6-vector add_time_ids
    # (original+crops(0,0)+target; uniform since images are pre-squared).
unet(noisy, timesteps, return_dict=False, **cond)   # uniform call site
```

Verified against diffusers 0.38 / transformers 5.11:
- Base `stabilityai/stable-diffusion-xl-base-1.0` (ungated). Adds `tokenizer_2` +
  `CLIPTextModelWithProjection`. `BufoDataset` gains optional `tokenizer_2`.
- UNet LoRA targets unchanged. Save/load via `StableDiffusionXLPipeline`
  `save_lora_weights`/`load_lora_weights`. fp16-fix VAE **not** needed (fp32).
- **MPS:** memory is fine; *speed* is the limit. Default **768px**; iterate fast
  with `stabilityai/sdxl-turbo` for previews, final LoRA on base. LR `5e-5`.
- Ship UNet-only SDXL first, then text-encoder LoRA.

---

## F. Preference loop — rate → refine (capstone)

Closes the loop with **aligned** human feedback on **our** outputs (friends
briefed with the good-bufo rubric) — distinct from bufo.zone meme-ratings.
`understand → design → rewrite → generate → **rate → refine**`. Three uses,
escalating:

1. **Calibrate the eval** — friends rate a batch; verify CLIP metrics correlate
   with human taste, reweight if not. Makes the whole harness trustworthy.
2. **Best-of-N curation** — generate 8–16 per emoji, humans pick keepers, ship. No
   retraining; immediate product value (and reuses `sample.py --rerank`).
3. **Preference optimization (Diffusion-DPO)** — collect *pairwise* preferences,
   DPO-finetune the LoRA toward them. The image-generation analog of the wordle
   GRPO work — same reward-from-preference idea, different modality.

Optional **data flywheel**: fold high-rated generations back into training,
**guarded by the diversity + memorization metrics** (self-training can collapse).

- **Tooling:** extend `gallery.py` into a pairwise voter ("which is the better
  `:prompt:`?") → `bufo/data/preferences.jsonl`. Pairwise beats 1–5 stars (less
  scale noise, native DPO format).
- **Cold start:** needs a solid supervised model + the eval harness first — a late
  phase. A few friends = noisy; fine for calibration/best-of-N, stay humble on DPO.

---

## File manifest

**New**

| File | Purpose |
|------|---------|
| `bufo/clip_metrics.py` | `ClipEmbedder` + pure metric math + train-emb cache |
| `bufo/eval.py` | `evaluate()`, scorecard, contact sheet, CLI |
| `bufo/curate.py` | dataset heuristics + report |
| `bufo/gallery.py` | FastAPI keep/drop review gallery |
| `bufo/recaption.py` | VLM + LLM recaptioning → schema captions ★ |
| `bufo/rewrite.py` | `Rewriter` protocol + rules default + LLM rewriter |
| `bufo/configs/eval-bufo.yaml` | held-out eval prompt set |
| `bufo/configs/lora-sdxl.yaml` | SDXL training config |
| `bufo/data/curation.jsonl` | per-image keep/drop + caption overrides |
| `bufo/dpo.py` | Diffusion-DPO LoRA finetune from pairwise preferences (stretch) |
| `bufo/data/preferences.jsonl` | pairwise human preferences (gallery-written) |
| `bufo/tests/test_clip_metrics.py`, `test_curate.py`, `test_rewrite.py`, `test_recaption.py` | tests |

**Modified**

| File | Change |
|------|--------|
| `bufo/data.py` | schema builder (`caption_for`/`filename_phrase`/`shortcode_to_prompt`); `tokenizer_2`; curation apply; crop option |
| `bufo/config.py` | `base_kind`, `train_text_encoder`, `eval_interval`, `recaption`/`rewriter` settings; `EvalConfig` |
| `bufo/pipeline.py` | `base_kind` branches; `encode_conditioning`; `attach_lora` (model list + `cast_training_params`); SDXL save/load |
| `bufo/train_lora.py` | wire `base_kind`/`tokenizer_2`; `encode_conditioning`; `EvalReporter`; SDXL `_snapshot` |
| `bufo/sample.py` | `--emoji`, `--rewriter`, `--base-kind`, negative prompt, `--rerank` |
| `bufo/README.md` | document the pipeline + new commands |

No new runtime pip deps (CLIP/VLM/LLM via transformers; gallery via existing
FastAPI; average-hash hand-rolled). Possible `sentencepiece` for some VLM
tokenizers — add only if a chosen model needs it.

---

## Sequencing

All axes in scope; phases give order so each builds on a measurable base.

1. **Evals first** — `clip_metrics.py`, `eval.py`, `eval-bufo.yaml`, cache, tests.
   Baseline the base model + v1 checkpoint. *Now everything has a number.*
2. **Data: fix + filter** — schema `caption_for`, `curate.py` + `gallery.py` +
   `curation.jsonl`; review; re-`prepare`. Retrain v1 recipe, compare scorecard.
3. **Data: recaption (A5)** — VLM+LLM captions; A/B vs filename baseline; keep the
   winner.
4. **Training recipe** — rank 32, text-encoder LoRA, `min-SNR`, `EvalReporter`
   wired in; longer watched run.
5. **Emoji interface** — `rewrite.py` (rules + LLM), `sample.py`
   `--emoji`/`--rerank`/negatives; A/B rules vs LLM on free-form inputs.
6. **SDXL** — `base_kind` path, `lora-sdxl.yaml`, 768px, turbo-iterate then base.
7. **(Optional) data expansion** — bufo.zone scrape with rating filter, if review
   shows the corpus is the bottleneck.
8. **Preference loop (capstone)** — pairwise rating gallery → calibrate the eval +
   best-of-N curation; Diffusion-DPO as the stretch.

## Verification

- `make check` stays green; offline metric/curate/rewrite/recaption tests run in
  it. Gated smokes: `BUFO_SMOKE=1` (SD train), `BUFO_CLIP_SMOKE=1` (CLIP).
- Each phase: `python -m bufo.eval` scorecard vs the phase-1 baseline; the
  contact sheet for eyeballing; TensorBoard `eval/*` curves during training.
- End-to-end: free-form input → rewriter → `sample.py` → recognizable bufo;
  memorization_max stays well below 1.0.

## Risks / open questions

- **VLM recaptioning quality/cost** — captioning ~1,400 images on MPS is
  minutes–hours; VLMs can hallucinate. Mitigated by merging with the (reliable)
  filename action and by the A/B eval gate. Pick a small, fast VLM.
- **SDXL on MPS is slow** at fp32/768px — turbo-iterate; a full base run is hours.
- **CLIP-score is a proxy** — good for relative comparison; keep human eyeballing.
- **Style heterogeneity** — the multi-artist corpus fights "consistent"; filter
  toward a cartoon band + trigger-token averaging, accept moderate variety. The
  48px-legibility metric is itself a proxy (downscale method matters).
- **LLM rewriter drift** — must target the terse schema, not prose; constrain via
  few-shot / a fixed output format, and measure.
- **Text-encoder LoRA** needs the optimizer/grad-clip to accept multiple modules
  (small `mm_training` extension) — deferred to that phase.

## References

- all-the-bufo: https://github.com/knobiknows/all-the-bufo/tree/main/all-the-bufo
- bufo.zone (ratings + GIFs): https://bufo.zone/ · bufo.fun: https://bufo.fun/ · bufopedia: https://bufopedia.com/
- DALL·E 3, *Improving Image Generation with Better Captions* (recaption + matching rewriter)
- CLIPScore (arXiv:2104.08718); diffusers SDXL LoRA; min-SNR weighting (arXiv:2303.09556)
- Diffusion-DPO, *Diffusion Model Alignment Using Direct Preference Optimization* (arXiv:2311.12908)
