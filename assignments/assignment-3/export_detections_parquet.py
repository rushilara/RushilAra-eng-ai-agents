#!/usr/bin/env python3
"""
Export per-frame detection JSON files to a single Parquet file (video, frame, detections).

Uses PyArrow (columnar, standard for Hugging Face datasets).

  pip install pyarrow

Examples:
  python3 export_detections_parquet.py
  python3 export_detections_parquet.py --output hf_packed/detections.parquet
  python3 export_detections_parquet.py --video 2 --min-frame 8219
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pack_detections_for_hf import collect_files


def main() -> None:
    p = argparse.ArgumentParser(description="Export detections/ → Parquet")
    p.add_argument(
        "--detections-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "detections",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .parquet path (default: ./hf_packed/detections.parquet or filtered name)",
    )
    p.add_argument("--video", type=int, choices=(1, 2), default=None)
    p.add_argument("--min-frame", type=int, default=None, metavar="N")
    p.add_argument("--max-frame", type=int, default=None, metavar="N")
    p.add_argument(
        "--compression",
        default="zstd",
        choices=("snappy", "zstd", "gzip", "none"),
        help="Parquet compression codec (default: zstd)",
    )
    args = p.parse_args()

    root = args.detections_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    pack_root = Path(__file__).resolve().parent / "hf_packed"
    filtered = args.video is not None or args.min_frame is not None or args.max_frame is not None
    if args.output is None:
        if filtered:
            parts: list[str] = []
            if args.video is not None:
                parts.append(f"drone_video_{args.video}")
            else:
                parts.append("detections")
            if args.min_frame is not None:
                parts.append(f"from_{args.min_frame:06d}")
            if args.max_frame is not None:
                parts.append(f"to_{args.max_frame:06d}")
            out_path = pack_root / ("_".join(parts) + ".parquet")
        else:
            out_path = pack_root / "detections.parquet"
    else:
        out_path = args.output.resolve()

    rows = collect_files(
        root,
        video=args.video,
        min_frame=args.min_frame,
        max_frame=args.max_frame,
    )
    if not rows:
        raise SystemExit("No JSON files matched (check --detections-dir and filters).")

    records: list[dict] = []
    for vid, frame, path in rows:
        det = json.loads(path.read_text(encoding="utf-8"))
        records.append({"video": vid, "frame": frame, "detections": det})

    table = pa.Table.from_pylist(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comp = None if args.compression == "none" else args.compression
    pq.write_table(table, out_path, compression=comp)
    print(f"Wrote {len(records)} rows -> {out_path}")


if __name__ == "__main__":
    main()
