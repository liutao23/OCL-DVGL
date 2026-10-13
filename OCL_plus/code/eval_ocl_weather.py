#!/usr/bin/env python3
"""Evaluate trained OCL image checkpoints under online UAV weather changes.

Weather is applied in memory after opening each UAV image. No augmented image is
written to disk. Satellite images always remain clear.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import dump, guard
from eval_ocl_image_retrieval import (fixed_protocol, image_list, load_encoder,
                                      retrieval_metrics)
from eval_ocl_support import sha256, write_csv
from iaa_trans import WEATHER_NAMES, apply_weather, build_weather_augmenters


# Clear weather already exists in the original University-1652 test set.  This
# evaluator therefore measures only the nine synthetic adverse conditions.
ADVERSE_WEATHER_NAMES = tuple(name for name in WEATHER_NAMES if name != "clear")


class OnlineWeatherDataset(Dataset):
    """Load and transform images in workers without materializing weather files."""

    def __init__(self, entries, transform, weather="clear", augmenters=None):
        self.entries = entries
        self.transform = transform
        self.weather = weather
        self.augmenters = augmenters

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, _ = self.entries[index]
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            if self.weather != "clear":
                # Each DataLoader process owns an independent augmenter state.
                if self.augmenters is None:
                    self.augmenters = build_weather_augmenters()
                image = Image.fromarray(apply_weather(image, self.weather, self.augmenters))
            return self.transform(image)


def _descriptor_forward(model, images, mode):
    return model.orbit(model.image_tokens(images)[:, None])


@torch.inference_mode()
def encode_online(model, entries, transform, batch_size, device, mode,
                  weather="clear", augmenters=None, description="Encoding",
                  workers=0, prefetch_factor=2):
    """Encode paths while applying an in-memory weather preset before preprocessing."""

    outputs = []
    dataset = OnlineWeatherDataset(entries, transform, weather, augmenters)
    loader_options = dict(
        batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=str(device).startswith("cuda"),
    )
    if workers:
        loader_options.update(
            persistent_workers=True, prefetch_factor=prefetch_factor,
        )
    loader = DataLoader(dataset, **loader_options)
    with tqdm(total=len(entries), desc=description, unit="image", dynamic_ncols=True) as progress:
        for images in loader:
            images = images.to(device, non_blocking=True)
            descriptors = _descriptor_forward(model, images, mode)
            if not torch.isfinite(descriptors).all():
                raise FloatingPointError("Nonfinite image descriptor")
            outputs.append(descriptors.float().cpu())
            progress.update(len(images))
    return torch.cat(outputs)


def evaluate(args, model, transform, metadata, mode, dataset, output):
    direction = "drone_to_satellite"
    uav_side = "query"
    queries = image_list(dataset / "test/query_drone")
    gallery = image_list(dataset / "test/gallery_satellite")
    fixed_protocol(queries, gallery, direction)
    output = guard(output)
    output.mkdir(parents=True, exist_ok=True)

    # Satellite images are the original dataset images and are encoded once.
    satellite_gallery = encode_online(
        model, gallery, transform, args.batch_size, args.device, mode,
        description="drone_to_satellite: original satellite gallery",
        workers=args.workers, prefetch_factor=args.prefetch_factor)

    summary = []
    for weather in args.weather:
        weather_output = output / weather
        metrics_path = weather_output / "metrics.json"
        if args.resume and metrics_path.is_file():
            saved = json.loads(metrics_path.read_text())
            expected = dict(
                direction=direction, weather=weather,
                variant=metadata["stored_variant"], checkpoint_sha=metadata["sha256"],
                query_count=len(queries), gallery_count=len(gallery),
                online_augmentation=True, augmented_images_written=False,
            )
            if any(saved.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Cannot resume incompatible weather result: {metrics_path}")
            summary.append({key: saved[key] for key in
                            ("direction", "weather", "r1", "r5", "r10", "mrr", "map")})
            print(f"Completed, skipped: {direction} / {weather}", flush=True)
            continue
        print(f"\n=== {direction} / {weather} ===", flush=True)
        q = encode_online(model, queries, transform, args.batch_size, args.device, mode,
                          weather, None, f"UAV query ({weather})",
                          args.workers, args.prefetch_factor)
        g = satellite_gallery

        metrics, rows, missing, per_query = retrieval_metrics(q, g, queries, gallery)
        weather_output.mkdir(parents=True, exist_ok=True)
        write_csv(weather_output / "per_query.csv", per_query)
        dump(weather_output / "retrieval.json", dict(
            metrics=metrics, evaluated_queries=len(rows), gallery_images=len(gallery),
            queries_without_positive=missing, results=rows,
            weather=weather, weather_applied_to=uav_side,
            augmented_images_written=False,
        ))
        record = dict(direction=direction, weather=weather, **metrics)
        dump(weather_output / "metrics.json", dict(
            **record, variant=metadata["stored_variant"], checkpoint=metadata["path"],
            checkpoint_sha=metadata["sha256"], query_count=len(queries),
            gallery_count=len(gallery), weather_applied_to=uav_side,
            online_augmentation=True, augmented_images_written=False,
        ))
        summary.append(record)
        print(json.dumps(metrics, indent=2), flush=True)

    write_csv(output / "weather_summary.csv", summary)
    dump(output / "weather_summary.json", dict(
        direction=direction, variant=metadata["stored_variant"], checkpoint=metadata,
        weather_presets=list(args.weather), weather_applied_to=uav_side,
        online_augmentation=True, augmented_images_written=False, results=summary,
        implementation=dict(
            evaluator_sha256=sha256(Path(__file__)),
            weather_transform_sha256=sha256(Path(__file__).with_name("iaa_trans.py")),
        ),
    ))
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Trained checkpoints/best_image.pt export")
    p.add_argument("--expected-variant", choices=(
        "mean_teacher",))
    p.add_argument("--dataset-root", type=Path, required=True,
                   help="University-1652 root containing test/query_drone, etc.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--weather", nargs="+", choices=ADVERSE_WEATHER_NAMES,
                   default=list(ADVERSE_WEATHER_NAMES),
                   help="Default: all nine non-clear weather presets")
    p.add_argument("--resume", action="store_true",
                   help="Verify and skip weather conditions with completed metrics")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel image decode/weather workers; default: 8")
    p.add_argument("--prefetch-factor", type=int, default=2,
                   help="Batches queued per worker; default: 2")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    if args.workers < 0 or args.prefetch_factor < 1:
        p.error("--workers must be nonnegative and --prefetch-factor positive")
    if len(set(args.weather)) != len(args.weather):
        p.error("--weather contains duplicates")

    args.encoder_mode = "ocl_image_model"
    args.dino_checkpoint = None
    dataset = guard(args.dataset_root)
    output = guard(args.output_dir)
    if output == dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError("Output must be separate from the University-1652 image root")

    model, transform, metadata, mode, _ = load_encoder(args)
    model.to(args.device)
    direction = "drone_to_satellite"
    results = evaluate(
        args, model, transform, metadata, mode, dataset, output / direction)
    write_csv(output / "weather_summary.csv", results)
    dump(output / "weather_summary.json", dict(
        variant=metadata["stored_variant"], checkpoint=metadata,
        directions=[direction], weather_presets=list(args.weather),
        online_augmentation=True, augmented_images_written=False, results=results,
    ))


if __name__ == "__main__":
    main()
