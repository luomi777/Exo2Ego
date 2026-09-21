"""Evaluate paired 256x256 ego images with SSIM, PSNR, FID, and LPIPS."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def image_map(directory):
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    images = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            if path.stem in images:
                raise ValueError(f"Duplicate image stem in {directory}: {path.stem}")
            images[path.stem] = path
    if not images:
        raise ValueError(f"No images in {directory}")
    return images


def matched_images(generated_dir, ground_truth_dir):
    generated = image_map(generated_dir)
    ground_truth = image_map(ground_truth_dir)
    if generated.keys() != ground_truth.keys():
        missing = sorted(ground_truth.keys() - generated.keys())
        extra = sorted(generated.keys() - ground_truth.keys())
        raise ValueError(
            f"Image stems differ: missing predictions={missing[:5]} ({len(missing)} total), "
            f"extra predictions={extra[:5]} ({len(extra)} total)"
        )
    if len(generated) < 2:
        raise ValueError("FID requires at least two image pairs")
    return [(generated[key], ground_truth[key]) for key in sorted(generated)]


def load_rgb(path):
    import numpy as np
    import torch

    with Image.open(path) as image:
        if image.size != (256, 256):
            raise ValueError(f"Expected 256x256 image: {path} has size {image.size}")
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB image: {path} has mode {image.mode}")
        array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1)


def evaluate(pairs, batch_size, device):
    import torch
    import torchmetrics
    from torchmetrics.functional.image import structural_similarity_index_measure
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    fid = FrechetInceptionDistance(feature=2048, normalize=False).set_dtype(torch.float64).to(device)
    lpips = {
        name: LearnedPerceptualImagePatchSimilarity(net_type=name, normalize=True).to(device)
        for name in ("squeeze", "alex", "vgg")
    }
    ssim_sum = 0.0
    psnr_sum = 0.0
    perfect_pairs = 0

    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            generated = torch.stack([load_rgb(pred) for pred, _ in chunk]).to(device)
            reference = torch.stack([load_rgb(real) for _, real in chunk]).to(device)
            fid.update(reference, real=True)
            fid.update(generated, real=False)

            generated_float = generated.float() / 255.0
            reference_float = reference.float() / 255.0
            ssim = structural_similarity_index_measure(
                generated_float, reference_float, data_range=1.0, reduction="none"
            )
            ssim_sum += float(ssim.sum().item())

            mse = (generated_float - reference_float).square().mean(dim=(1, 2, 3))
            perfect_pairs += int((mse == 0).sum().item())
            psnr_sum += float((-10.0 * torch.log10(mse[mse > 0])).sum().item())

            for metric in lpips.values():
                metric.update(generated_float, reference_float)

        result = {
            "num_pairs": len(pairs),
            "ssim": ssim_sum / len(pairs),
            "psnr_db": "inf" if perfect_pairs else psnr_sum / len(pairs),
            "fid": float(fid.compute().item()),
            "lpips_squeeze": float(lpips["squeeze"].compute().item()),
            "lpips_alex": float(lpips["alex"].compute().item()),
            "lpips_vgg": float(lpips["vgg"].compute().item()),
            "torch_version": torch.__version__,
            "torchmetrics_version": torchmetrics.__version__,
        }
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite metric: {key}={value}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True, help="Generated ego image directory")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Matching ground-truth ego image directory")
    parser.add_argument("--output", type=Path, help="Write results as JSON; also print to stdout")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        pairs = matched_images(args.generated, args.ground_truth)
        import torch

        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device == "auto":
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is not available; use --device cpu")
        result = evaluate(pairs, args.batch_size, device)
    except (ImportError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)


if __name__ == "__main__":
    main()
