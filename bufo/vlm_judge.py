"""Qwen2.5-VL VLM-as-judge — scores generated bufo images for COHERENCE/QUALITY.

CLIP scores *semantic presence* ("a green frog and a bike are here") but is blind
to whether the image is well-formed: melted anatomy, extra/missing limbs, glitchy
noise backgrounds all sail past a good CLIP concept score. This module fills that
gap with a vision-language model that *looks* at each image and rates structural
quality, returning strict JSON we parse into a dataclass + aggregate.

Mirrors the ``clip_metrics`` split: the JSON parsing + aggregation math is pure and
unit-testable (no model download); the model wrapper loads lazily and is gated.

Python 3.9-compatible (runs on the Ray cluster's py3.9 nodes): no 3.10+ syntax,
``from __future__ import annotations`` for the type hints.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from PIL import Image

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# "broken" threshold: an image is broken if its structure scored at-or-below this
# coherence, OR the judge could not recognize a single clean frog at all.
BROKEN_COHERENCE = 2

# The rubric the model is asked to follow. Kept as a module constant so the
# validation report and tests can quote the *exact* text that produced a score.
SYSTEM_PROMPT = (
    "You are a STRICT visual quality judge for cartoon frog emoji stickers "
    "(a character called 'bufo': a simple green cartoon frog). You are shown one "
    "image at a time and must rate how clean and well-formed it is as a small "
    "Slack emoji. You judge STRUCTURE and COHERENCE, not whether the prompt's "
    "subject is present. Be harsh: melted/blobby anatomy, extra or missing or "
    "fused limbs, two heads, broken or nonsensical objects, smeared faces, and "
    "noisy/glitchy/distorted textures are all serious defects. When in doubt, "
    "score LOWER. Reply with ONLY a single JSON object, no prose, no markdown."
)

USER_PROMPT = (
    "Rate this image. Return STRICT JSON with exactly these keys:\n"
    '  "recognizable": 0 or 1 — 1 only if it is clearly a SINGLE, clean, '
    "well-formed cartoon frog. Broken anatomy, blobs, or multiple/merged frogs => 0.\n"
    '  "coherence": integer 1-5 — overall structural correctness. '
    "5 = crisp, correct anatomy, clean shapes; 3 = readable but with minor flaws; "
    "1 = melted/deformed/incoherent. Extra/missing/fused limbs or a broken object "
    "force coherence <= 2.\n"
    '  "artifacts": 0 or 1 — 1 if there is noisy, glitchy, grainy, or distorted '
    "texture (especially in the background) or JPEG-like smearing.\n"
    '  "emoji_ok": 0 or 1 — 1 only if it would still read cleanly as a recognizable '
    "frog when shrunk to a ~48px Slack emoji (bold simple shapes, clear silhouette).\n"
    '  "reason": a one-line string (<= 15 words) naming the main defect or "clean".\n'
    "Rules: if anatomy is broken or noise is present, set recognizable=0 AND "
    "coherence<=2. Output JSON only."
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class JudgeScore:
    """One image's verdict. ``parse_failed`` flags a worst-case fallback fill."""

    recognizable: int  # 0/1 — clearly a single clean cartoon frog
    coherence: int  # 1-5 — structural correctness
    artifacts: int  # 0/1 — noisy/glitchy textures present
    emoji_ok: int  # 0/1 — reads cleanly at ~48px
    reason: str
    raw: str  # the model's raw text output (for debugging)
    parse_failed: bool = False

    @property
    def broken(self) -> bool:
        """Our headline defect flag: low structure OR not a clean single frog."""
        return self.coherence <= BROKEN_COHERENCE or self.recognizable == 0


@dataclass
class JudgeAggregate:
    n: int
    mean_coherence: float
    recognizable_rate: float
    artifact_rate: float
    emoji_ok_rate: float
    broken_rate: float  # fraction with coherence<=2 OR recognizable==0
    parse_failure_rate: float


# ---------------------------------------------------------------------------
# Pure parsing + aggregation math (the unit-test surface — no model needed)
# ---------------------------------------------------------------------------

# Worst-case fallback when the model output can't be parsed: treat as fully broken
# so a silent parse failure never inflates the quality score.
_WORST_CASE = {
    "recognizable": 0,
    "coherence": 1,
    "artifacts": 1,
    "emoji_ok": 0,
}

# First balanced-ish {...} block; greedy to the last brace so a trailing keyed
# object survives even if the model wrapped it in prose.
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_int(value: Any, lo: int, hi: int, default: int) -> int:
    """Clamp ``value`` into [lo, hi]; fall back to ``default`` on junk.

    The model sometimes emits ``"1"``, ``1.0``, ``true``, or out-of-range ints —
    normalize all of them rather than trusting the raw field.
    """
    if isinstance(value, bool):
        value = int(value)
    try:
        out = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, out))


def parse_judge_json(text: str) -> JudgeScore:
    """Extract a ``JudgeScore`` from raw model text.

    Robust to markdown fences, leading/trailing prose, and missing/garbage fields.
    On a total parse failure returns the worst-case score with ``parse_failed=True``
    so the caller can both penalize the image and count the failure.
    """
    cleaned = text.strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```).
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, count=1).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    match = _JSON_RE.search(cleaned)
    if match is None:
        return JudgeScore(raw=text, parse_failed=True, reason="parse-failure: no JSON found", **_WORST_CASE)

    obj: dict[str, Any] | None = None
    candidate = match.group(0)
    # Greedy match can over-capture; retry by trimming to the last closing brace.
    for attempt in (candidate, candidate[: candidate.rfind("}") + 1]):
        try:
            parsed = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            obj = parsed
            break
    if obj is None:
        return JudgeScore(raw=text, parse_failed=True, reason="parse-failure: bad JSON", **_WORST_CASE)

    reason = obj.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    return JudgeScore(
        recognizable=_coerce_int(obj.get("recognizable"), 0, 1, default=0),
        coherence=_coerce_int(obj.get("coherence"), 1, 5, default=1),
        artifacts=_coerce_int(obj.get("artifacts"), 0, 1, default=1),
        emoji_ok=_coerce_int(obj.get("emoji_ok"), 0, 1, default=0),
        reason=reason[:200],
        raw=text,
        parse_failed=False,
    )


def aggregate_scores(scores: Sequence[JudgeScore]) -> JudgeAggregate:
    """Reduce per-image scores to the headline aggregate."""
    n = len(scores)
    if n == 0:
        return JudgeAggregate(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return JudgeAggregate(
        n=n,
        mean_coherence=sum(s.coherence for s in scores) / n,
        recognizable_rate=sum(s.recognizable for s in scores) / n,
        artifact_rate=sum(s.artifacts for s in scores) / n,
        emoji_ok_rate=sum(s.emoji_ok for s in scores) / n,
        broken_rate=sum(1 for s in scores if s.broken) / n,
        parse_failure_rate=sum(1 for s in scores if s.parse_failed) / n,
    )


# ---------------------------------------------------------------------------
# The model wrapper (loads Qwen2.5-VL lazily; not needed by the math tests)
# ---------------------------------------------------------------------------


@dataclass
class VLMJudge:
    """Qwen2.5-VL wrapped as a strict coherence/quality judge for bufo images."""

    model: Any
    processor: Any
    device: torch.device
    model_id: str
    max_new_tokens: int = 128

    @classmethod
    def load(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        device: torch.device | None = None,
        max_new_tokens: int = 128,
        torch_dtype: str = "auto",
    ) -> VLMJudge:
        """Load the model + processor.

        ``device_map="auto"`` is used when no explicit device is given (the cluster
        path: single GPU). ``use_safetensors`` is implied by Qwen's repo, which ships
        safetensors — so it loads on the cluster's torch<2.6 (no torch.load .bin).
        """
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        if device is None:
            kwargs["device_map"] = "auto"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        if device is not None:
            model = model.to(device)
        model.eval()
        resolved = device if device is not None else next(model.parameters()).device
        # min/max pixels cap the visual-token budget: emoji are small, so we don't
        # need the default high resolution and this keeps inference fast/cheap.
        processor = AutoProcessor.from_pretrained(model_id, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
        return cls(
            model=model,
            processor=processor,
            device=resolved,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
        )

    def _generate_one(self, image: Image.Image) -> str:
        """Run a single image through the model and return raw decoded text."""
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image.convert("RGB")},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = _vision_inputs(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        # Trim the prompt tokens, decode only the completion.
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]  # noqa: B905 — py3.9 has no strict=
        decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded[0]

    def score_images(self, images: Sequence[Image.Image], verbose: bool = False) -> list[JudgeScore]:
        """Score a batch of PIL images, one model call per image.

        One-image-per-call keeps the prompt unambiguous (the model never has to
        decide *which* image a JSON object refers to) and lets a single bad
        generation degrade gracefully instead of corrupting a batched response.
        """
        scores: list[JudgeScore] = []
        for i, image in enumerate(images):
            raw = self._generate_one(image)
            score = parse_judge_json(raw)
            scores.append(score)
            if verbose:
                tag = " PARSE-FAIL" if score.parse_failed else ""
                print(
                    f"  judge {i + 1}/{len(images)}: "
                    f"coh={score.coherence} rec={score.recognizable} "
                    f"art={score.artifacts} emoji={score.emoji_ok}{tag} | {score.reason}",
                    flush=True,
                )
        return scores

    def score_and_aggregate(
        self, images: Sequence[Image.Image], verbose: bool = False
    ) -> tuple[list[JudgeScore], JudgeAggregate]:
        scores = self.score_images(images, verbose=verbose)
        return scores, aggregate_scores(scores)


def _vision_inputs(messages: list[dict[str, Any]]) -> tuple[Any, Any]:
    """Extract vision inputs from chat messages.

    Prefers ``qwen_vl_utils.process_vision_info`` (the upstream-recommended path,
    which handles PIL images, paths, URLs, and base64). Falls back to pulling the
    PIL images straight out of the message content if that package is absent, so
    the judge still runs without the optional dependency.
    """
    try:
        from qwen_vl_utils import process_vision_info

        return process_vision_info(messages)
    except ImportError:
        images: list[Any] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image":
                    images.append(part["image"])
        return (images or None), None
