#!/usr/bin/env python3
"""
Pack per-frame JSON files into JSON Lines (one object per line) for Hugging Face.

Each line: {"video": 1, "frame": 12345, "detections": [ ... ]}

This keeps uploads to 1–2 commits instead of tens of thousands (HF free tier: ~128 commits/hour).

Filter examples (optional):
  python3 pack_detections_for_hf.py --video 2 --min-frame 8219
  # -> hf_packed/drone_video_2_from_008219.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FRAME_PAT = re.compile(r"^drone_video_(\d+)_frame(\d+)\.json$")


def collect_files(
    detections_dir: Path,
    *,
    video: int | None = None,
    min_frame: int | None = None,
    max_frame: int | None = None,
) -> list[tuple[int, int, Path]]:
    out: list[tuple[int, int, Path]] = []
    for p in detections_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".json":
            continue
        m = FRAME_PAT.match(p.name)
        if not m:
            continue
        vid = int(m.group(1))
        frame = int(m.group(2))
        if video is not None and vid != video:
            continue
        if min_frame is not None and frame < min_frame:
            continue
        if max_frame is not None and frame > max_frame:
            continue
        out.append((vid, frame, p))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Pack HW3/detections JSON → JSONL for HF")
    p.add_argument(
        "--detections-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "detections",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .jsonl path or directory for --per-video (default: auto from filters or ./hf_packed/detections.jsonl)",
    )
    p.add_argument(
        "--per-video",
        action="store_true",
        help="Write drone_video_1.jsonl and drone_video_2.jsonl under output directory instead of one file",
    )
    p.add_argument(
        "--video",
        type=int,
        choices=(1, 2),
        default=None,
        help="Only include this video (e.g. 2 when resuming video 2)",
    )
    p.add_argument(
        "--min-frame",
        type=int,
        default=None,
        metavar="N",
        help="Only include frames with index >= N (e.g. 8219 after uploading up through frame 8218)",
    )
    p.add_argument(
        "--max-frame",
        type=int,
        default=None,
        metavar="N",
        help="Only include frames with index <= N",
    )
    args = p.parse_args()

    root = args.detections_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    if args.per_video and (args.video is not None or args.min_frame is not None or args.max_frame is not None):
        raise SystemExit("Use either --per-video OR filtered packing (--video / --min-frame / --max-frame), not both.")

    pack_root = Path(__file__).resolve().parent / "hf_packed"
    filtered = args.video is not None or args.min_frame is not None or args.max_frame is not None
    if args.output is None:
        if args.per_video:
            args.output = pack_root
        elif filtered:
            parts: list[str] = []
            if args.video is not None:
                parts.append(f"drone_video_{args.video}")
            else:
                parts.append("detections")
            if args.min_frame is not None:
                parts.append(f"from_{args.min_frame:06d}")
            if args.max_frame is not None:
                parts.append(f"to_{args.max_frame:06d}")
            args.output = pack_root / ("_".join(parts) + ".jsonl")
        else:
            args.output = pack_root / "detections.jsonl"

    rows = collect_files(
        root,
        video=args.video,
        min_frame=args.min_frame,
        max_frame=args.max_frame,
    )
    if not rows:
        raise SystemExit(f"No drone_video_*_frame*.json files in {root} (check filters)")

    if args.per_video:
        out_dir = args.output if args.output.suffix != ".jsonl" else args.output.parent
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        by_vid: dict[int, list[tuple[int, Path]]] = {}
        for vid, frame, path in rows:
            by_vid.setdefault(vid, []).append((frame, path))
        for vid, items in sorted(by_vid.items()):
            target = out_dir / f"drone_video_{vid}.jsonl"
            n = 0
            with target.open("w", encoding="utf-8") as f:
                for frame, path in items:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    rec = {"video": vid, "frame": frame, "detections": data}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
            print(f"Wrote {n} lines -> {target}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with args.output.open("w", encoding="utf-8") as f:
            for vid, frame, path in rows:
                data = json.loads(path.read_text(encoding="utf-8"))
                rec = {"video": vid, "frame": frame, "detections": data}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        print(f"Wrote {n} lines -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
