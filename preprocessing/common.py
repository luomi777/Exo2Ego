"""Frame pairing, clipping, image transforms, and output naming."""

import re

from PIL import Image


EXTENSIONS = {".png", ".jpg", ".jpeg"}


def image_files(directory):
    return sorted((p for p in directory.iterdir() if p.suffix.lower() in EXTENSIONS), key=lambda p: p.name)


def paired_files(exo_dir, ego_dir, prefix=False):
    def key(path):
        return path.name.split("_", 1)[0] if prefix else path.stem

    exo = {key(path): path for path in image_files(exo_dir)}
    ego = {key(path): path for path in image_files(ego_dir)}
    keys = sorted(exo.keys() & ego.keys(), key=lambda value: int(value) if value.isdigit() else value)
    if not keys:
        raise ValueError(f"No paired frames: {exo_dir} and {ego_dir}")
    return [(key, exo[key], ego[key]) for key in keys]


def clips(frames):
    return [frames[start:start + 30] for start in range(0, len(frames) - 29, 30)]


def crop_and_save(image, crop, output, rotate=False):
    # Crop format follows the historical scripts: [width, height, left, top].
    width, height, left, top = (int(value) for value in crop)
    if min(width, height) <= 0 or min(left, top) < 0:
        raise ValueError(f"Invalid crop rectangle: {crop}")
    if image.size[0] < left + width or image.size[1] < top + height:
        raise ValueError(f"Crop {crop} exceeds image size {image.size}")
    frame = image.convert("RGB").crop((left, top, left + width, top + height))
    frame = frame.resize((256, 256), Image.Resampling.BICUBIC)
    if rotate:
        frame = frame.transpose(Image.Transpose.ROTATE_270)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.save(output)


def save_image_pair(exo, ego, exo_crop, ego_crop, exo_target, ego_target, rotate_ego=False):
    with Image.open(exo) as image:
        crop_and_save(image, exo_crop, exo_target)
    with Image.open(ego) as image:
        crop_and_save(image, ego_crop, ego_target, rotate=rotate_ego)


def safe_id(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def emit_clip(args, dataset, recording, segment, clip_id, split, frames, exo_crop, ego_crop,
              index_writer, rotate_ego=False, video_source=None):
    if len(frames) != 30:
        raise ValueError(f"Expected 30 pairs in {recording}/{segment}/{clip_id}")
    for position, (source_id, exo, ego) in enumerate(frames):
        name = safe_id(f"{dataset}_{recording}_{segment}_{clip_id}_{position:02d}") + ".png"
        exo_target = args.output / split / "exo" / name
        ego_target = args.output / split / "ego" / name
        if exo_target.exists() or ego_target.exists():
            raise FileExistsError(f"Refusing to replace existing pair: {name}")
        if not args.dry_run:
            if video_source is None:
                save_image_pair(exo, ego, exo_crop, ego_crop, exo_target, ego_target, rotate_ego)
            else:
                exo_image, ego_image = video_source(exo, ego)
                crop_and_save(exo_image, exo_crop, exo_target)
                crop_and_save(ego_image, ego_crop, ego_target, rotate_ego)
        index_writer.writerow([dataset, recording, segment, clip_id, position, split, source_id, str(exo), str(ego)])
