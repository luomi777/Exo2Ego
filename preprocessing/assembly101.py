"""Assembly101 v4/e3 decoding from explicit manually aligned frame pairs."""

import csv
import json
from collections import defaultdict

from PIL import Image

from .common import emit_clip


ASSEMBLY_RECORDINGS = {
    "nusar-2021_action_both_9051-c13a_9051_user_id_2021-02-22_121941",
    "nusar-2021_action_both_9056-c13a_9056_user_id_2021-02-22_145733",
    "nusar-2021_action_both_9071-c13a_9071_user_id_2021-02-11_090900",
    "nusar-2021_action_both_9081-c13a_9081_user_id_2021-02-12_162453",
    "nusar-2021_action_both_9086-c13a_9086_user_id_2021-02-16_151024",
    "nusar-2021_action_both_9086-c13a_9086_user_id_2021-02-16_152408",
}


class VideoReader:
    def __init__(self, exo_path, ego_path):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Assembly101 video decoding requires opencv-python-headless") from exc
        self.cv2 = cv2
        self.captures = [cv2.VideoCapture(str(exo_path)), cv2.VideoCapture(str(ego_path))]
        if not all(cap.isOpened() for cap in self.captures):
            self.close()
            raise ValueError(f"Cannot open videos: {exo_path}, {ego_path}")
        self.last = [-1, -1]

    def read_one(self, stream, index):
        index = int(index)
        cap = self.captures[stream]
        if index != self.last[stream] + 1:
            cap.set(self.cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot decode frame {index} from stream {stream}")
        self.last[stream] = index
        return Image.fromarray(self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB))

    def __call__(self, exo_index, ego_index):
        return self.read_one(0, exo_index), self.read_one(1, ego_index)

    def close(self):
        for cap in self.captures:
            cap.release()


def prepare_assembly(args, writer):
    if not args.alignment or not args.crops:
        raise ValueError("Assembly101 requires --alignment and --crops")
    crop_config = json.loads(args.crops.read_text())
    rows_by_clip = defaultdict(list)
    with args.alignment.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"recording", "segment", "clip", "split", "exo_frame", "ego_frame"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Alignment CSV needs columns: {', '.join(sorted(required))}")
        for row in reader:
            recording = row["recording"]
            if recording not in ASSEMBLY_RECORDINGS:
                raise ValueError(f"Unexpected Assembly101 recording: {recording}")
            if row["split"] not in {"training", "testing"}:
                raise ValueError(f"Invalid split: {row['split']}")
            key = (recording, row["segment"], row["clip"], row["split"])
            rows_by_clip[key].append((f"{row['exo_frame']}:{row['ego_frame']}", row["exo_frame"], row["ego_frame"]))
    total = defaultdict(int)
    for recording in sorted({key[0] for key in rows_by_clip}):
        directory = args.input / recording
        exo_path = directory / "C10119_rgb.mp4"
        ego_candidates = list(directory.glob("HMC_84355350_mono10bit.mp4")) + list(directory.glob("HMC_21110305_mono10bit.mp4"))
        if not exo_path.is_file() or len(ego_candidates) != 1:
            raise FileNotFoundError(f"Expected one v4 and one e3 video in {directory}")
        if recording not in crop_config:
            raise ValueError(f"Missing crop rectangles for {recording}")
        reader = None if args.dry_run else VideoReader(exo_path, ego_candidates[0])
        try:
            for key in sorted(k for k in rows_by_clip if k[0] == recording):
                _, segment, clip_id, split = key
                group = rows_by_clip[key]
                crop = crop_config[recording]
                emit_clip(args, "assembly101", recording, segment, clip_id, split, group,
                          crop["exo"], crop["ego"], writer, rotate_ego=True,
                          video_source=reader if reader else None)
                total[split] += 1
        finally:
            if reader:
                reader.close()
    return total
