"""VLM-judge JSON parsing + aggregation math (offline, always runs).

The model wrapper (VLMJudge.load) downloads a 7B Qwen checkpoint, so it is not
exercised here; this file pins the *robustness* contract that protects the scores:
strict JSON extraction, worst-case fallback on garbage, and the aggregate math.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

from bufo.validate_vlm_judge import (
    list_images,
    per_prompt_aggregates,
    prompt_index,
    render_montage,
)
from bufo.vlm_judge import (
    BROKEN_COHERENCE,
    JudgeScore,
    aggregate_scores,
    parse_judge_json,
)

# --------------------------------------------------------------------------
# parse_judge_json — strict extraction + robustness
# --------------------------------------------------------------------------


def test_parse_clean_json():
    raw = '{"recognizable": 1, "coherence": 5, "artifacts": 0, "emoji_ok": 1, "reason": "clean"}'
    s = parse_judge_json(raw)
    assert (s.recognizable, s.coherence, s.artifacts, s.emoji_ok) == (1, 5, 0, 1)
    assert s.reason == "clean"
    assert not s.parse_failed
    assert not s.broken


def test_parse_strips_markdown_fence():
    raw = '```json\n{"recognizable": 0, "coherence": 1, "artifacts": 1, "emoji_ok": 0, "reason": "melted"}\n```'
    s = parse_judge_json(raw)
    assert s.coherence == 1
    assert s.recognizable == 0
    assert not s.parse_failed
    assert s.broken


def test_parse_extracts_from_surrounding_prose():
    raw = 'Sure! Here is my verdict:\n{"recognizable":1,"coherence":4,"artifacts":0,"emoji_ok":1,"reason":"ok"} Thanks!'
    s = parse_judge_json(raw)
    assert s.coherence == 4
    assert not s.parse_failed


def test_parse_failure_defaults_to_worst_case_and_flags():
    s = parse_judge_json("I cannot answer that.")
    assert s.parse_failed
    assert s.coherence == 1
    assert s.recognizable == 0
    assert s.artifacts == 1
    assert s.emoji_ok == 0
    assert s.broken  # worst-case must count as broken


def test_parse_coerces_stringy_and_float_and_bool_values():
    raw = '{"recognizable": true, "coherence": "3", "artifacts": 0.0, "emoji_ok": 1.0, "reason": 42}'
    s = parse_judge_json(raw)
    assert s.recognizable == 1
    assert s.coherence == 3
    assert s.artifacts == 0
    assert s.emoji_ok == 1
    assert s.reason == "42"  # non-str reason stringified
    assert not s.parse_failed


def test_parse_clamps_out_of_range_coherence():
    assert parse_judge_json('{"coherence": 9}').coherence == 5
    assert parse_judge_json('{"coherence": -4}').coherence == 1
    assert parse_judge_json('{"recognizable": 7}').recognizable == 1


def test_parse_missing_fields_use_safe_defaults():
    s = parse_judge_json('{"reason": "partial"}')
    # missing -> safe defaults: not recognizable, worst coherence, artifacts present
    assert (s.recognizable, s.coherence, s.artifacts, s.emoji_ok) == (0, 1, 1, 0)
    assert not s.parse_failed  # it WAS valid JSON, just sparse


# --------------------------------------------------------------------------
# broken flag + aggregate
# --------------------------------------------------------------------------


def _score(coh: int, rec: int, art: int = 0, emoji: int = 1) -> JudgeScore:
    return JudgeScore(recognizable=rec, coherence=coh, artifacts=art, emoji_ok=emoji, reason="", raw="")


def test_broken_flag_threshold():
    assert _score(coh=2, rec=1).broken  # low coherence
    assert _score(coh=5, rec=0).broken  # not recognizable
    assert not _score(coh=3, rec=1).broken  # clean
    assert BROKEN_COHERENCE == 2


def test_aggregate_math():
    scores = [_score(5, 1, art=0, emoji=1), _score(1, 0, art=1, emoji=0), _score(4, 1, art=0, emoji=1)]
    agg = aggregate_scores(scores)
    assert agg.n == 3
    assert agg.mean_coherence == (5 + 1 + 4) / 3
    assert agg.recognizable_rate == 2 / 3
    assert agg.artifact_rate == 1 / 3
    assert agg.emoji_ok_rate == 2 / 3
    assert agg.broken_rate == 1 / 3  # only the coh=1,rec=0 one
    assert agg.parse_failure_rate == 0.0


def test_aggregate_empty():
    agg = aggregate_scores([])
    assert agg.n == 0
    assert agg.mean_coherence == 0.0
    assert agg.broken_rate == 0.0


def test_aggregate_counts_parse_failures():
    failed = parse_judge_json("garbage")
    agg = aggregate_scores([failed, _score(5, 1)])
    assert agg.parse_failure_rate == 0.5
    assert agg.broken_rate == 0.5  # the parse failure is worst-case => broken


# --------------------------------------------------------------------------
# validation-script helpers (filename parsing, per-prompt buckets, montage I/O)
# --------------------------------------------------------------------------


def test_prompt_index_and_list_images_ordering(tmp_path: Path):
    names = ["10_00.png", "02_01.png", "02_00.png", "skip.png", "notes.txt"]
    for nm in names:
        if nm.endswith(".png"):
            Image.new("RGB", (8, 8), (0, 200, 0)).save(tmp_path / nm)
        else:
            (tmp_path / nm).write_text("x")
    ordered = [p.name for p in list_images(tmp_path)]
    assert ordered == ["02_00.png", "02_01.png", "10_00.png"]  # sorted, skip.png excluded
    assert prompt_index(tmp_path / "08_03.png") == 8
    assert prompt_index(tmp_path / "weird.png") is None


def test_per_prompt_aggregates_buckets_by_prompt(tmp_path: Path):
    paths = [tmp_path / "03_00.png", tmp_path / "03_01.png", tmp_path / "08_00.png"]
    scores = [_score(1, 0), _score(2, 0), _score(5, 1)]
    subjects = ["a"] * 9
    subjects[3] = "drinking bubble tea"
    subjects[8] = "riding a bicycle"
    pp = per_prompt_aggregates(paths, scores, subjects)
    assert set(pp.keys()) == {"03", "08"}
    assert pp["03"]["subject"] == "drinking bubble tea"
    assert pp["03"]["n"] == 2
    assert pp["03"]["broken_rate"] == 1.0
    assert pp["03"]["coherences"] == [1, 2]
    assert pp["08"]["broken_rate"] == 0.0


def test_render_montage_writes_image(tmp_path: Path):
    paths = []
    for i in range(3):
        p = tmp_path / f"0{i}_00.png"
        Image.new("RGB", (32, 32), (0, 150 + i, 0)).save(p)
        paths.append(p)
    scores = [_score(5, 1), _score(1, 0), _score(3, 1)]
    out = tmp_path / "montage.png"
    render_montage(paths, scores, ["a", "b", "c"], out, cols=2, cell=40)
    assert out.exists()
    with Image.open(out) as im:
        assert im.size[0] == 2 * 40  # 2 columns
        assert im.size[1] > 0
