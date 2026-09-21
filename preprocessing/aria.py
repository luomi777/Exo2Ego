"""Aria Pilot aligned-frame intervals, transforms, and action split."""

import json
from collections import defaultdict
from pathlib import Path

from .common import clips, emit_clip, paired_files


INTERVALS_PATH = Path(__file__).resolve().parents[1] / "configs/aria_intervals.json"


def prepare_aria(args, writer):
    intervals = json.loads(INTERVALS_PATH.read_text())
    total = defaultdict(int)
    for key, (start, end) in intervals.items():
        exo_dir = args.input / key / "sync/exo"
        ego_dir = args.input / key / "sync/ego"
        if not exo_dir.is_dir() or not ego_dir.is_dir():
            raise FileNotFoundError(f"Missing aligned Aria frames under {args.input / key / 'sync'}")
        selected = [row for row in paired_files(exo_dir, ego_dir, prefix=True) if start <= int(row[0]) <= end]
        grouped = clips(selected)
        cut = int(len(grouped) * 0.8)
        for clip_id, group in enumerate(grouped):
            split = "training" if clip_id < cut else "testing"
            emit_clip(args, "aria", key.replace("/", "_"), "main", clip_id, split, group,
                      [1500, 1080, 0, 0], [1408, 1408, 0, 0], writer, rotate_ego=True)
            total[split] += 1
    return total
