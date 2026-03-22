#!/usr/bin/env python3
"""
Upload HW3 detection JSON to a Hugging Face *dataset* repo.

**Recommended:** pack all per-frame JSON into one ``.jsonl`` or ``.parquet`` file, then upload
with **1 commit** for the file. Parquet matches common HF dataset / assignment expectations.

  python3 upload_detections_to_hf.py --repo-id USER/detections-final --mode parquet

Requires ``pyarrow`` for Parquet: ``pip install pyarrow``

The old ``upload_large_folder`` mode caused many commits + 429 / commit rate limits.

Prereqs:
  1. Token: https://huggingface.co/settings/tokens (write access)
  2. export HF_TOKEN=hf_...

Usage:
  export HF_TOKEN=hf_...
  python3 upload_detections_to_hf.py --repo-id rushilara/uav-drone-detections

Resume video 2 after frame 8218 (packs frames >= 8219 only, one commit):
  python3 upload_detections_to_hf.py --repo-id rushilara/uav-drone-detections --video 2 --min-frame 8219

Pack only (no upload):
  python3 pack_detections_for_hf.py --video 2 --min-frame 8219
  python3 upload_detections_to_hf.py --repo-id USER/DS --pack-only

Large-folder mode (not recommended; hits commit limits):
  python3 upload_detections_to_hf.py --repo-id USER/DS --mode large-folder
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    p = argparse.ArgumentParser(description="Upload HW3/detections to Hugging Face Datasets")
    p.add_argument(
        "--repo-id",
        required=True,
        help='Dataset repo id, e.g. "rushilara/uav-drone-detections"',
    )
    p.add_argument(
        "--detections-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "detections",
        help="Folder of per-frame JSON (default: ./detections)",
    )
    p.add_argument(
        "--packed-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "hf_packed",
        help="Where detections.jsonl / per-video jsonl are written (default: ./hf_packed)",
    )
    p.add_argument(
        "--per-video",
        action="store_true",
        help="Pack to drone_video_1.jsonl + drone_video_2.jsonl (2 uploads) instead of one detections.jsonl",
    )
    p.add_argument(
        "--min-frame",
        type=int,
        default=None,
        metavar="N",
        help="Only pack/upload frames with index >= N (e.g. 8219 if Hub already has video 2 through 8218)",
    )
    p.add_argument(
        "--max-frame",
        type=int,
        default=None,
        metavar="N",
        help="Only pack/upload frames with index <= N",
    )
    p.add_argument(
        "--pack-output",
        type=Path,
        default=None,
        help="Override packed .jsonl path passed to pack_detections_for_hf --output",
    )
    p.add_argument(
        "--mode",
        choices=("jsonl", "parquet", "large-folder"),
        default="jsonl",
        help="jsonl / parquet = one file upload; large-folder = many commits (not recommended)",
    )
    p.add_argument(
        "--pack-only",
        action="store_true",
        help="Only run pack_detections_for_hf; do not upload",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create dataset repo as private if it does not exist yet",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="(large-folder only) parallel workers",
    )
    p.add_argument(
        "--video",
        type=int,
        choices=(1, 2),
        default=None,
        help="jsonl: only this video; large-folder: only this video's JSON glob",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HF token (default: HF_TOKEN env or huggingface-cli login cache)",
    )
    args = p.parse_args()

    root = args.detections_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    api = HfApi(token=args.token)

    if args.mode == "parquet":
        import subprocess
        import sys

        if args.per_video:
            raise SystemExit("--per-video is only supported with --mode jsonl.")

        packed_dir = args.packed_dir.resolve()
        packed_dir.mkdir(parents=True, exist_ok=True)
        export_script = Path(__file__).resolve().parent / "export_detections_parquet.py"
        cmd = [sys.executable, str(export_script), "--detections-dir", str(root)]

        filtered = args.video is not None or args.min_frame is not None or args.max_frame is not None
        if args.pack_output is not None:
            cmd += ["--output", str(args.pack_output.resolve())]
        elif filtered:
            parts_pq: list[str] = []
            if args.video is not None:
                parts_pq.append(f"drone_video_{args.video}")
            else:
                parts_pq.append("detections")
            if args.min_frame is not None:
                parts_pq.append(f"from_{args.min_frame:06d}")
            if args.max_frame is not None:
                parts_pq.append(f"to_{args.max_frame:06d}")
            out_fp = packed_dir / ("_".join(parts_pq) + ".parquet")
            cmd += ["--output", str(out_fp)]
        else:
            cmd += ["--output", str(packed_dir / "detections.parquet")]

        if args.video is not None:
            cmd += ["--video", str(args.video)]
        if args.min_frame is not None:
            cmd += ["--min-frame", str(args.min_frame)]
        if args.max_frame is not None:
            cmd += ["--max-frame", str(args.max_frame)]

        print("Export Parquet:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        if args.pack_only:
            print("Pack-only: done.")
            return

        if args.pack_output is not None:
            fp = args.pack_output.resolve()
        elif filtered:
            parts2: list[str] = []
            if args.video is not None:
                parts2.append(f"drone_video_{args.video}")
            else:
                parts2.append("detections")
            if args.min_frame is not None:
                parts2.append(f"from_{args.min_frame:06d}")
            if args.max_frame is not None:
                parts2.append(f"to_{args.max_frame:06d}")
            fp = packed_dir / ("_".join(parts2) + ".parquet")
        else:
            fp = packed_dir / "detections.parquet"
        if not fp.is_file():
            raise SystemExit(f"Missing {fp}")

        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        remote = f"data/{fp.name}"
        print(f"Uploading {fp} -> {remote} (single commit) ...")
        api.upload_file(
            path_or_fileobj=str(fp),
            path_in_repo=remote,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"Add {fp.name} (Parquet)",
        )
        print(f"Done: https://huggingface.co/datasets/{args.repo_id}")
        return

    if args.mode == "jsonl":
        import subprocess
        import sys

        packed_dir = args.packed_dir.resolve()
        packed_dir.mkdir(parents=True, exist_ok=True)
        pack_script = Path(__file__).resolve().parent / "pack_detections_for_hf.py"
        cmd = [sys.executable, str(pack_script), "--detections-dir", str(root)]

        filtered = args.video is not None or args.min_frame is not None or args.max_frame is not None
        if args.per_video and filtered:
            raise SystemExit("Cannot use --per-video together with --video / --min-frame / --max-frame.")

        if args.per_video:
            cmd += ["--per-video", "--output", str(packed_dir)]
        elif args.pack_output is not None:
            cmd += ["--output", str(args.pack_output.resolve())]
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
            out_fp = packed_dir / ("_".join(parts) + ".jsonl")
            cmd += ["--output", str(out_fp)]
        else:
            cmd += ["--output", str(packed_dir / "detections.jsonl")]

        if args.video is not None:
            cmd += ["--video", str(args.video)]
        if args.min_frame is not None:
            cmd += ["--min-frame", str(args.min_frame)]
        if args.max_frame is not None:
            cmd += ["--max-frame", str(args.max_frame)]

        print("Packing:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        if args.pack_only:
            print("Pack-only: done.")
            return

        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )

        if args.per_video:
            files = sorted(packed_dir.glob("drone_video_*.jsonl"))
            if not files:
                raise SystemExit(f"No drone_video_*.jsonl in {packed_dir}")
            for fp in files:
                print(f"Uploading {fp.name} ...")
                api.upload_file(
                    path_or_fileobj=str(fp),
                    path_in_repo=fp.name,
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    commit_message=f"Add {fp.name}",
                )
        else:
            if args.pack_output is not None:
                fp = args.pack_output.resolve()
            elif filtered:
                parts2: list[str] = []
                if args.video is not None:
                    parts2.append(f"drone_video_{args.video}")
                else:
                    parts2.append("detections")
                if args.min_frame is not None:
                    parts2.append(f"from_{args.min_frame:06d}")
                if args.max_frame is not None:
                    parts2.append(f"to_{args.max_frame:06d}")
                fp = packed_dir / ("_".join(parts2) + ".jsonl")
            else:
                fp = packed_dir / "detections.jsonl"
            if not fp.is_file():
                raise SystemExit(f"Missing packed file {fp}")
            print(f"Uploading {fp} (single commit) ...")
            msg = f"Add {fp.name}"
            if filtered:
                msg = f"Add remaining detections ({fp.name})"
            api.upload_file(
                path_or_fileobj=str(fp),
                path_in_repo=fp.name,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=msg,
            )
        print(f"Done: https://huggingface.co/datasets/{args.repo_id}")
        return

    # large-folder (legacy)
    allow_patterns: str | None = None
    if args.video is not None:
        allow_patterns = f"drone_video_{args.video}_*.json"

    label = f" ({allow_patterns})" if allow_patterns else ""
    print(f"Uploading {root}{label} -> datasets/{args.repo_id} ...")
    print("WARNING: many commits; free tier often hits 128 commits/hour.")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_large_folder(
        folder_path=str(root),
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        allow_patterns=allow_patterns,
        num_workers=args.num_workers,
        print_report=True,
    )
    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
