#!/usr/bin/env python3
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


DEFAULT_MODELS = ("a2", "a3", "indextts")


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Split generated wav files into a2/a3/indextts folders using output manifests."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=None,
        help="Manifest JSON path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=base_dir / "wav",
        help="Directory that currently contains all wav files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=base_dir / "wav_by_model",
        help="Directory where model folders will be created.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them. Default keeps originals in --wav-dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without copying or moving files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each copied or moved file.",
    )
    return parser.parse_args()


def load_manifest(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} should contain a JSON list")
    return data


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    manifests = args.manifest or [
        base_dir / "manifest_ta.json",
        base_dir / "manifest_indextts.json",
    ]

    entries = []
    for manifest in manifests:
        for item in load_manifest(manifest):
            item = dict(item)
            item["_manifest"] = str(manifest)
            entries.append(item)

    copied = Counter()
    missing = []
    skipped = Counter()
    action = shutil.move if args.move else shutil.copy2
    action_name = "move" if args.move else "copy"
    per_model_manifest = {model: [] for model in DEFAULT_MODELS}

    if not args.dry_run:
        for model in DEFAULT_MODELS:
            (args.out_dir / model).mkdir(parents=True, exist_ok=True)

    for item in entries:
        model = item.get("model")
        filename = item.get("file")
        if model not in DEFAULT_MODELS:
            skipped[str(model)] += 1
            continue
        if not filename:
            skipped[model] += 1
            continue

        src = args.wav_dir / filename
        dst = args.out_dir / model / filename
        if not src.exists():
            missing.append(str(src))
            continue

        per_model_manifest[model].append(item)
        copied[model] += 1
        if args.dry_run:
            if args.verbose:
                print(f"{action_name}: {src} -> {dst}")
            continue

        if args.verbose:
            print(f"{action_name}: {src} -> {dst}")

        action(src, dst)

    if not args.dry_run:
        for model, items in per_model_manifest.items():
            manifest_path = args.out_dir / model / "manifest.json"
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)
        if missing:
            missing_path = args.out_dir / "missing_wavs.txt"
            with missing_path.open("w", encoding="utf-8") as f:
                f.write("\n".join(missing))
                f.write("\n")

    print(f"action: {action_name}")
    print(f"wav_dir: {args.wav_dir}")
    print(f"out_dir: {args.out_dir}")
    print("done:", dict(copied))
    if skipped:
        print("skipped:", dict(skipped))
    if missing:
        print(f"missing: {len(missing)}")
        if not args.dry_run:
            print(f"missing_report: {args.out_dir / 'missing_wavs.txt'}")
        for path in missing[:20]:
            print("  ", path)
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


if __name__ == "__main__":
    main()
