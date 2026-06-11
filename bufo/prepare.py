"""CLI: download + preprocess the bufo corpus into a training-ready dataset.

Usage:
    uv run python -m bufo.prepare                       # full corpus, 512px
    uv run python -m bufo.prepare --limit 32            # quick subset
    uv run python -m bufo.prepare --config bufo/configs/lora-sd15.yaml
"""

from __future__ import annotations

import argparse

from bufo.config import BufoLoRAConfig, DataConfig
from bufo.data import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the bufo dataset")
    parser.add_argument("--config", type=str, default=None, help="YAML config (uses its data section)")
    parser.add_argument("--resolution", type=int, default=None, help="Override target resolution")
    parser.add_argument("--limit", type=int, default=None, help="Only fetch the first N images (quick runs)")
    args = parser.parse_args()

    cfg = BufoLoRAConfig.from_yaml(args.config).data if args.config else DataConfig()
    if args.resolution is not None:
        cfg.resolution = args.resolution
    prepare(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
