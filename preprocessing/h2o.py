"""H2O cam 2/cam 4 pairing, scene crops, and benchmark splits."""

import json
from collections import defaultdict
from pathlib import Path

from .common import clips, emit_clip, paired_files


CROPS_PATH = Path(__file__).resolve().parents[1] / "configs/h2o_crops.json"


def prepare_h2o(args, writer):
    crops = json.loads(CROPS_PATH.read_text())
    total = defaultdict(int)
    for exo_dir in sorted(args.input.glob("subject[12]/*/*/cam2/rgb")):
        subject, scene, object_id = exo_dir.relative_to(args.input).parts[:3]
        if subject not in {"subject1", "subject2"}:
            continue
        if args.setting != "subject" and subject != "subject1":
            continue
        ego_dir = exo_dir.parents[1] / "cam4" / "rgb"
        if not ego_dir.is_dir():
            raise FileNotFoundError(ego_dir)
        frames = paired_files(exo_dir, ego_dir)
        sequence = f"{subject}_{scene}_{object_id}"
        grouped = clips(frames)
        cut = int(len(grouped) * 0.8)
        subject_test_count = len(grouped) - cut
        for clip_id, group in enumerate(grouped):
            if args.setting == "action":
                split = "training" if clip_id < cut else "testing"
            elif args.setting == "object":
                if object_id not in {"0", "1", "2", "3", "4", "5", "7"}:
                    continue
                split = "testing" if object_id == "7" else "training"
            elif args.setting == "scene":
                split = "testing" if scene in {"o1", "o2"} else "training"
            else:  # subject: subject1 training portion, subject2 first test clips
                if subject == "subject1":
                    if clip_id >= cut:
                        continue
                    split = "training"
                else:
                    if clip_id >= subject_test_count:
                        continue
                    split = "testing"
            camera_crops = crops[subject][scene]
            emit_clip(args, "h2o", sequence, "main", clip_id, split, group,
                      camera_crops["cam2"], camera_crops["cam4"], writer)
            total[split] += 1
    return total
