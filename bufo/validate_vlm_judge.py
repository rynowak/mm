"""Validate the Qwen2.5-VL coherence judge on a directory of generated bufo images.

Runs ``VLMJudge`` over every ``PP_NN.png`` in an image dir, writes ``scores.json``
(per-image verdicts + aggregate + per-prompt breakdown) and a labeled contact-sheet
montage (each cell tagged with its coherence score + recognizable flag; broken cells
boxed/labeled in red). The montage is the human-readable proof the judge separates
melted/glitched generations from clean ones.

Incremental + resumable (repo rule): each image's verdict is appended to a JSONL
sidecar as it is produced and flushed, so a crash/preemption loses at most the
image in flight; ``--resume`` skips already-scored files. ``scores.json`` + the
montage are (re)written from the JSONL at the end.

Usage (on the cluster):
    python -m bufo.validate_vlm_judge \
        --images /mnt/ray/bufo-eval/sdxl600/images \
        --out /mnt/ray/bufo-eval/vlm-judge/sdxl600 \
        --eval-config bufo/configs/eval-bufo-v2.yaml \
        --resume
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

from bufo.vlm_judge import (
    DEFAULT_MODEL_ID,
    JudgeAggregate,
    JudgeScore,
    VLMJudge,
    aggregate_scores,
)

# Files are PP_NN.png — PP = prompt index, NN = image index within the prompt.
_NAME_RE = re.compile(r"^(\d+)_(\d+)\.png$")


def list_images(images_dir: Path) -> list[Path]:
    """Sorted PP_NN.png paths under ``images_dir`` (prompt then image order)."""

    def key(p: Path) -> tuple[int, int]:
        m = _NAME_RE.match(p.name)
        return (int(m.group(1)), int(m.group(2))) if m else (1 << 30, 1 << 30)

    return sorted((p for p in images_dir.glob("*.png") if _NAME_RE.match(p.name)), key=key)


def prompt_index(path: Path) -> int | None:
    m = _NAME_RE.match(path.name)
    return int(m.group(1)) if m else None


def load_prompts(eval_config: Path | None) -> list[str]:
    """Subjects from an eval YAML's ``prompts:`` list (for montage labels)."""
    if eval_config is None or not eval_config.exists():
        return []
    data = yaml.safe_load(eval_config.read_text())
    return list(data.get("prompts", []))


def _load_done(jsonl_path: Path) -> dict[str, JudgeScore]:
    """Re-read already-scored verdicts from the JSONL sidecar (for --resume)."""
    done: dict[str, JudgeScore] = {}
    if not jsonl_path.exists():
        return done
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        done[rec["file"]] = JudgeScore(
            recognizable=rec["recognizable"],
            coherence=rec["coherence"],
            artifacts=rec["artifacts"],
            emoji_ok=rec["emoji_ok"],
            reason=rec["reason"],
            raw=rec.get("raw", ""),
            parse_failed=rec.get("parse_failed", False),
        )
    return done


def _append_jsonl(jsonl_path: Path, file_name: str, score: JudgeScore) -> None:
    rec = dataclasses.asdict(score)
    rec["file"] = file_name
    with jsonl_path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()


def score_directory(
    judge: VLMJudge,
    images_dir: Path,
    out_dir: Path,
    *,
    resume: bool = False,
    limit: int | None = None,
) -> tuple[list[Path], list[JudgeScore]]:
    """Score every image, streaming verdicts to a resumable JSONL sidecar."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "scores.jsonl"
    paths = list_images(images_dir)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"No PP_NN.png images found under {images_dir}")

    done = _load_done(jsonl_path) if resume else {}
    if not resume and jsonl_path.exists():
        jsonl_path.unlink()  # fresh run — drop any stale sidecar
    print(f"Scoring {len(paths)} images ({len(done)} already done) from {images_dir}", flush=True)

    scores: list[JudgeScore] = []
    for i, path in enumerate(paths):
        if path.name in done:
            scores.append(done[path.name])
            continue
        with Image.open(path) as im:
            im = im.convert("RGB")
            score = judge.score_images([im])[0]
        _append_jsonl(jsonl_path, path.name, score)
        scores.append(score)
        tag = " PARSE-FAIL" if score.parse_failed else ""
        print(
            f"  [{i + 1}/{len(paths)}] {path.name}: coh={score.coherence} "
            f"rec={score.recognizable} art={score.artifacts} emoji={score.emoji_ok}"
            f"{tag} | {score.reason}",
            flush=True,
        )
    return paths, scores


def per_prompt_aggregates(
    paths: list[Path], scores: list[JudgeScore], subjects: list[str]
) -> dict[str, dict[str, object]]:
    """Aggregate by prompt index → {label, n, mean_coherence, broken_rate, ...}."""
    buckets: dict[int, list[JudgeScore]] = {}
    for path, score in zip(paths, scores):  # noqa: B905 — py3.9 has no strict=
        pi = prompt_index(path)
        if pi is None:
            continue
        buckets.setdefault(pi, []).append(score)
    out: dict[str, dict[str, object]] = {}
    for pi in sorted(buckets):
        agg = aggregate_scores(buckets[pi])
        label = subjects[pi] if 0 <= pi < len(subjects) else f"prompt {pi}"
        out[f"{pi:02d}"] = {
            "subject": label,
            "n": agg.n,
            "mean_coherence": round(agg.mean_coherence, 3),
            "recognizable_rate": round(agg.recognizable_rate, 3),
            "artifact_rate": round(agg.artifact_rate, 3),
            "emoji_ok_rate": round(agg.emoji_ok_rate, 3),
            "broken_rate": round(agg.broken_rate, 3),
            "coherences": [s.coherence for s in buckets[pi]],
        }
    return out


def write_scores_json(
    out_dir: Path,
    images_dir: Path,
    model_id: str,
    paths: list[Path],
    scores: list[JudgeScore],
    subjects: list[str],
) -> JudgeAggregate:
    agg = aggregate_scores(scores)
    payload = {
        "model_id": model_id,
        "images_dir": str(images_dir),
        "n_images": len(scores),
        "aggregate": dataclasses.asdict(agg),
        "per_prompt": per_prompt_aggregates(paths, scores, subjects),
        "per_image": [
            {
                "file": path.name,
                "prompt_index": prompt_index(path),
                "coherence": s.coherence,
                "recognizable": s.recognizable,
                "artifacts": s.artifacts,
                "emoji_ok": s.emoji_ok,
                "broken": s.broken,
                "parse_failed": s.parse_failed,
                "reason": s.reason,
            }
            for path, s in zip(paths, scores)  # noqa: B905 — py3.9 has no strict=
        ],
    }
    (out_dir / "scores.json").write_text(json.dumps(payload, indent=2))
    return agg


def render_montage(
    paths: list[Path],
    scores: list[JudgeScore],
    subjects: list[str],
    out_path: Path,
    *,
    cols: int = 8,
    cell: int = 200,
) -> None:
    """Contact sheet: each cell shows the image with a coherence/recognizable
    banner; broken cells get a red border + red text so defects pop visually."""
    if not paths:
        return
    label_h = 26
    n = len(paths)
    rows = (n + cols - 1) // cols
    width = cols * cell
    height = rows * (cell + label_h)
    sheet = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)

    for i, (path, score) in enumerate(zip(paths, scores)):  # noqa: B905 — py3.9 has no strict=
        r, c = divmod(i, cols)
        x = c * cell
        y = r * (cell + label_h)
        with Image.open(path) as im:
            thumb = im.convert("RGB").resize((cell, cell))
        sheet.paste(thumb, (x, y))
        broken = score.broken
        color = (200, 0, 0) if broken else (0, 120, 0)
        if broken:  # red box around broken cells
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(200, 0, 0), width=4)
        pi = prompt_index(path)
        subj = subjects[pi] if (pi is not None and 0 <= pi < len(subjects)) else ""
        subj = (subj[:16] + "…") if len(subj) > 17 else subj
        flag = "PF" if score.parse_failed else ("BROKEN" if broken else "ok")
        banner_y = y + cell
        draw.rectangle([x, banner_y, x + cell - 1, banner_y + label_h - 1], fill=(255, 255, 255))
        draw.text(
            (x + 4, banner_y + 4),
            f"{path.name} c{score.coherence} r{score.recognizable} {flag}",
            fill=color,
        )
        if subj:
            draw.text((x + 4, banner_y + 14), subj, fill=(90, 90, 90))
    sheet.save(out_path)
    print(f"Wrote montage -> {out_path} ({width}x{height})", flush=True)


def run(args: argparse.Namespace) -> JudgeAggregate:
    import torch  # noqa: F401 — ensures torch present; device chosen by VLMJudge

    images_dir = Path(args.images)
    out_dir = Path(args.out)
    subjects = load_prompts(Path(args.eval_config) if args.eval_config else None)

    judge = VLMJudge.load(model_id=args.model_id, max_new_tokens=args.max_new_tokens)
    paths, scores = score_directory(judge, images_dir, out_dir, resume=args.resume, limit=args.limit)
    agg = write_scores_json(out_dir, images_dir, args.model_id, paths, scores, subjects)
    render_montage(paths, scores, subjects, out_dir / "montage.png", cols=args.cols)

    print(
        f"\nAGGREGATE [{out_dir.name}] n={agg.n} "
        f"mean_coherence={agg.mean_coherence:.3f} "
        f"recognizable={agg.recognizable_rate:.3f} "
        f"artifacts={agg.artifact_rate:.3f} "
        f"emoji_ok={agg.emoji_ok_rate:.3f} "
        f"broken={agg.broken_rate:.3f} "
        f"parse_fail={agg.parse_failure_rate:.3f}",
        flush=True,
    )
    # Spotlight the known-broken exemplars so the pass/fail check is visible in logs.
    pp = per_prompt_aggregates(paths, scores, subjects)
    for pi in ("08", "03"):
        if pi in pp:
            row = pp[pi]
            print(
                f"KNOWN-BROKEN prompt {pi} ({row['subject']}): "
                f"mean_coherence={row['mean_coherence']} "
                f"coherences={row['coherences']} broken_rate={row['broken_rate']}",
                flush=True,
            )
    return agg


def main() -> None:
    p = argparse.ArgumentParser(description="Validate the Qwen2.5-VL bufo coherence judge")
    p.add_argument("--images", required=True, help="Directory of PP_NN.png generations")
    p.add_argument("--out", required=True, help="Output dir for scores.json + montage.png")
    p.add_argument("--eval-config", default="bufo/configs/eval-bufo-v2.yaml", help="YAML with prompts: for labels")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--cols", type=int, default=8, help="Montage columns")
    p.add_argument("--limit", type=int, default=None, help="Score only the first N images (debug)")
    p.add_argument("--resume", action="store_true", help="Skip already-scored images in scores.jsonl")
    run(p.parse_args())


if __name__ == "__main__":
    main()
