#!/usr/bin/env python3
"""
split_domainnet.py — Build an ImageFolder-style train/test split for DomainNet.

DomainNet raw layout (after downloading from http://ai.bu.edu/M3SDA/):
  <src_dir>/clipart/<class>/image.jpg
  <src_dir>/infograph/<class>/image.jpg
  <src_dir>/painting/<class>/image.jpg
  <src_dir>/quickdraw/<class>/image.jpg
  <src_dir>/real/<class>/image.jpg
  <src_dir>/sketch/<class>/image.jpg

The official train/test split text files are expected alongside the images:
  <src_dir>/clipart_train.txt   (lines: "clipart/class/img.jpg <label>")
  <src_dir>/clipart_test.txt

This script merges all six domains, resolves class names to a unified 0-344
label space, and creates hard-links (or copies) under:
  <out_dir>/train/<class_name>/
  <out_dir>/test/<class_name>/

Usage:
  python scripts/split_domainnet.py \\
      --src_dir /path/to/raw/domainnet \\
      --out_dir ./data/domainnet \\
      [--domains clipart infograph painting quickdraw real sketch] \\
      [--copy]   # copy files instead of hard-linking

If the official split .txt files are absent, a random 80/20 per-class split
is used instead.
"""

import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def find_classes(src_dir: Path, domains: list[str]) -> list[str]:
    """Return a sorted list of class names that appear in ALL requested domains."""
    class_sets = []
    for domain in domains:
        domain_dir = src_dir / domain
        if not domain_dir.is_dir():
            raise FileNotFoundError(f"Domain directory not found: {domain_dir}")
        classes = {d.name for d in domain_dir.iterdir() if d.is_dir()}
        class_sets.append(classes)
    # Union across domains (some classes may be missing in some domains)
    all_classes = sorted(set().union(*class_sets))
    return all_classes


def load_official_splits(src_dir: Path, domains: list[str]):
    """
    Load official train/test splits from <domain>_train.txt / <domain>_test.txt.
    Returns (train_files, test_files) where each is a list of absolute Path objects.
    Returns (None, None) if split files are absent.
    """
    train_files: list[Path] = []
    test_files: list[Path] = []
    found_any = False

    for domain in domains:
        train_txt = src_dir / f"{domain}_train.txt"
        test_txt = src_dir / f"{domain}_test.txt"
        if not train_txt.exists() or not test_txt.exists():
            continue
        found_any = True
        for split_txt, bucket in [(train_txt, train_files), (test_txt, test_files)]:
            with open(split_txt) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    rel_path = parts[0]  # e.g. "clipart/apple/img_0001.jpg"
                    abs_path = src_dir / rel_path
                    if abs_path.exists():
                        bucket.append(abs_path)

    return (train_files, test_files) if found_any else (None, None)


def random_split(src_dir: Path, domains: list[str], classes: list[str]):
    """Fallback: random 80/20 per-class split across all domains."""
    rng = random.Random(RANDOM_SEED)
    train_files: list[Path] = []
    test_files: list[Path] = []

    for cls in classes:
        imgs: list[Path] = []
        for domain in domains:
            cls_dir = src_dir / domain / cls
            if cls_dir.is_dir():
                imgs.extend(p for p in cls_dir.iterdir() if p.is_file())
        rng.shuffle(imgs)
        split_idx = max(1, int(len(imgs) * TRAIN_RATIO))
        train_files.extend(imgs[:split_idx])
        test_files.extend(imgs[split_idx:])

    return train_files, test_files


def class_name_from_path(path: Path, src_dir: Path) -> str:
    """Extract class name from a path like src_dir/domain/class/img.jpg."""
    parts = path.relative_to(src_dir).parts
    # parts[0] = domain, parts[1] = class_name, parts[2] = image
    return parts[1]


def link_or_copy(src: Path, dst: Path, do_copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if do_copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            # Cross-device link not permitted — fall back to copy
            shutil.copy2(src, dst)


def build_output(
    files: list[Path], split_name: str, out_dir: Path, src_dir: Path, do_copy: bool
):
    counters: dict[str, int] = defaultdict(int)
    split_dir = out_dir / split_name
    for src_path in files:
        cls = class_name_from_path(src_path, src_dir)
        idx = counters[cls]
        counters[cls] += 1
        dst_path = split_dir / cls / f"{src_path.stem}_{idx}{src_path.suffix}"
        link_or_copy(src_path, dst_path, do_copy)
    return dict(counters)


def main():
    parser = argparse.ArgumentParser(
        description="Build ImageFolder train/test split for DomainNet."
    )
    parser.add_argument(
        "--src_dir",
        required=True,
        type=Path,
        help="Root directory containing DomainNet domain sub-folders.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        type=Path,
        help="Output directory for ImageFolder-style train/ and test/ splits.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=DOMAINS,
        choices=DOMAINS,
        help="Which domains to include (default: all six).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of hard-linking (slower but cross-device safe).",
    )
    args = parser.parse_args()

    src_dir: Path = args.src_dir.resolve()
    out_dir: Path = args.out_dir.resolve()
    domains: list[str] = args.domains
    do_copy: bool = args.copy

    print(f"Source : {src_dir}")
    print(f"Output : {out_dir}")
    print(f"Domains: {', '.join(domains)}")

    # Discover classes
    print("\nScanning class directories...")
    classes = find_classes(src_dir, domains)
    print(f"Found {len(classes)} classes across {len(domains)} domain(s).")

    # Load or build splits
    print("\nLoading train/test splits...")
    train_files, test_files = load_official_splits(src_dir, domains)
    if train_files is not None:
        print(
            f"Using official split files: {len(train_files)} train, "
            f"{len(test_files)} test images."
        )
    else:
        print(
            "Official split .txt files not found — using random 80/20 split "
            f"(seed={RANDOM_SEED})."
        )
        train_files, test_files = random_split(src_dir, domains, classes)
        print(f"Random split: {len(train_files)} train, {len(test_files)} test images.")

    # Build output
    print("\nBuilding train split...")
    train_counts = build_output(train_files, "train", out_dir, src_dir, do_copy)
    print(f"  {len(train_counts)} classes, {sum(train_counts.values())} images.")

    print("Building test split...")
    test_counts = build_output(test_files, "test", out_dir, src_dir, do_copy)
    print(f"  {len(test_counts)} classes, {sum(test_counts.values())} images.")

    print(f"\nDone. ImageFolder data written to: {out_dir}")
    print("  train/  →", out_dir / "train")
    print("  test/   →", out_dir / "test")


if __name__ == "__main__":
    main()
