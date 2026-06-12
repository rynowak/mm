"""Offline tests for the curate heuristics (synthetic images, no model)."""

from __future__ import annotations

from PIL import Image, ImageDraw

from bufo.curate import analyze_image, assign_dup_groups, average_hash, report


def test_average_hash_identical_and_different():
    a = Image.new("L", (32, 32), 0)
    b = Image.new("L", (32, 32), 0)
    assert average_hash(a) == average_hash(b)


def test_analyze_image_flags_scene(tmp_path):
    name = "bufo-is-doing-doordash-deliveries-in-the-evening.png"
    Image.new("RGBA", (64, 64), (0, 200, 0, 255)).save(tmp_path / name)
    stats = analyze_image(tmp_path / name)
    assert stats.scene_like  # 6 action words -> scene-like
    assert stats.aspect == 1.0
    assert stats.alpha_coverage == 1.0  # fully opaque


def test_alpha_coverage_transparent(tmp_path):
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(tmp_path / "bufo-ghost.png")
    stats = analyze_image(tmp_path / "bufo-ghost.png")
    assert stats.alpha_coverage == 0.0


def _patterned(fill_left: bool) -> Image.Image:
    # Average-hash is brightness-invariant, so dups must differ in *layout*, not
    # just color: left-half-white vs top-half-white produce distinct hashes.
    img = Image.new("L", (64, 64), 0)
    box = (0, 0, 32, 64) if fill_left else (0, 0, 64, 32)
    ImageDraw.Draw(img).rectangle(box, fill=255)
    return img


def test_dup_grouping(tmp_path):
    # Two identical layouts + one different -> one dup group of 2, one unique.
    _patterned(fill_left=True).save(tmp_path / "bufo-a.png")
    _patterned(fill_left=True).save(tmp_path / "bufo-b.png")
    _patterned(fill_left=False).save(tmp_path / "bufo-c.png")
    stats = [analyze_image(p) for p in sorted(tmp_path.glob("*.png"))]
    assign_dup_groups(stats, threshold=2)
    groups = {s.file_name: s.dup_group for s in stats}
    assert groups["bufo-a.png"] == groups["bufo-b.png"] >= 0
    assert groups["bufo-c.png"] == -1  # unique


def test_report_writes_files(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(3):
        Image.new("RGBA", (64, 64), (i * 80, 100, 0, 255)).save(raw / f"bufo-x{i}.png")
    summary = report(tmp_path)
    assert summary["total"] == 3
    assert (tmp_path / "curate-report.json").exists()
    assert (tmp_path / "curate-report.csv").exists()
