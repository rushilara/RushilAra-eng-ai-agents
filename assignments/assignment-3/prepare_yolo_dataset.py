#!/usr/bin/env python3
"""
Create Ultralytics-style `images/` + `labels/` layout for the Zenodo UAV dataset.

Ultralytics resolves dataset paths with Path.resolve(), which follows **directory**
symlinks to `data/Images/Train`. Label paths are derived by replacing `/images/` with
`/labels/` in each image path; resolved paths contain `/Images/` (capital I), so
no labels match. This script uses **hard links** (or copies if needed) so resolved
paths stay under `.../yolo_ultralytics/images/...` and labels are found.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def _remove_tree_if_exists(p: Path) -> None:
    if p.is_symlink():
        p.unlink()
    elif p.is_dir():
        shutil.rmtree(p)
    elif p.is_file():
        p.unlink()


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _populate_mirror(src_dir: Path, dst_dir: Path, *, labels: bool) -> int:
    """Hardlink (or copy) every image or .txt label file from src_dir into dst_dir."""
    n = 0
    for src in sorted(src_dir.iterdir()):
        if not src.is_file():
            continue
        if labels:
            if src.suffix.lower() != ".txt":
                continue
        elif not _is_image(src):
            continue
        dst = dst_dir / src.name
        _link_or_copy(src, dst)
        n += 1
    return n


def _clear_stale_caches(data: Path) -> None:
    """Remove Ultralytics caches created when labels were missing (wrong paths)."""
    for pattern in ("Images/*.cache", "Images/*/*.cache", "yolo_ultralytics/images/*.cache"):
        for p in data.glob(pattern):
            if p.is_file():
                p.unlink()
                print(f"Removed stale cache: {p}")


def main() -> Path:
    parser = argparse.ArgumentParser(description="Prepare YOLO layout + drone.yaml for RT-DETR training.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory containing Images/ and Annotations/ (default: ./data next to this script)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild yolo_ultralytics/images and labels even if they already exist",
    )
    args = parser.parse_args()
    hw3 = Path(__file__).resolve().parent
    data = (args.data_root or (hw3 / "data")).resolve()

    root = data / "yolo_ultralytics"
    root.mkdir(parents=True, exist_ok=True)

    # Old versions used directory symlinks; Ultralytics resolve() breaks label discovery.
    legacy = root / "images" / "train"
    if legacy.is_symlink():
        print("Replacing symlink-based layout with hard links (required for Ultralytics).")
        args.force = True

    splits = [
        ("images/train", "labels/train", data / "Images" / "Train", data / "Annotations" / "Yolo" / "Train"),
        ("images/val", "labels/val", data / "Images" / "Valid", data / "Annotations" / "Yolo" / "Valid"),
        ("images/test", "labels/test", data / "Images" / "Test", data / "Annotations" / "Yolo" / "Test"),
    ]

    if args.force:
        for rel_img, rel_lb, _, _ in splits:
            _remove_tree_if_exists(root / rel_img)
            _remove_tree_if_exists(root / rel_lb)

    _clear_stale_caches(data)

    for rel_img, rel_lb, src_img, src_lb in splits:
        if not src_img.is_dir():
            raise FileNotFoundError(f"Missing: {src_img}")
        if not src_lb.is_dir():
            raise FileNotFoundError(f"Missing: {src_lb}")
        dst_img = root / rel_img
        dst_lb = root / rel_lb
        ni = _populate_mirror(src_img, dst_img, labels=False)
        nl = _populate_mirror(src_lb, dst_lb, labels=True)
        print(f"{rel_img}: {ni} image files, {rel_lb}: {nl} label files")

    yaml_path = root / "drone.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 1",
                "names:",
                "  0: drone",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {yaml_path}")
    print(f"Dataset root: {root}")
    return yaml_path


if __name__ == "__main__":
    main()
