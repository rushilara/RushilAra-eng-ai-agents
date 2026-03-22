#!/usr/bin/env python3
"""
Run RT-DETR on every MP4 in a directory: extract frames, run detection, save JSON per frame.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from ultralytics import RTDETR


DISPLAY_CLASS = "drone"


def _cv2_peek_first_frame(video_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok and frame is not None and frame.size > 0)
    finally:
        cap.release()


def _ffprobe_video_size(video_path: Path) -> tuple[int, int]:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {video_path}: {r.stderr.strip() or 'unknown error'}\n"
            "Install ffmpeg/ffprobe or re-encode the video to H.264, e.g.:\n"
            f"  ffmpeg -i input.mp4 -c:v libx264 -crf 18 -c:a copy out_h264.mp4"
        )
    data = json.loads(r.stdout)
    if not data.get("streams"):
        raise RuntimeError(f"No video stream in {video_path}")
    s = data["streams"][0]
    return int(s["width"]), int(s["height"])


def _iter_frames_ffmpeg(video_path: Path):
    w, h = _ffprobe_video_size(video_path)
    frame_n = w * h * 3
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-an",
            "-sn",
            "-dn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    err = b""
    try:
        while True:
            raw = proc.stdout.read(frame_n)
            if len(raw) < frame_n:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
    finally:
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            err = proc.stderr.read()
            proc.stderr.close()
        if proc.poll() is None:
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        rc = proc.returncode
        if rc not in (0, None, -13, -15):  # SIGPIPE/SIGTERM if pipe closed early
            raise RuntimeError(
                f"ffmpeg failed decoding {video_path} (exit {rc}): {err.decode(errors='replace')[:2000]}\n"
                "Re-encode to H.264: ffmpeg -i in.mp4 -c:v libx264 -crf 18 -c:a copy out.mp4"
            )


def _iter_frames_cv2(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def boxes_to_json(
    result,
    conf_threshold: float,
    class_name: str = DISPLAY_CLASS,
) -> list[dict]:
    """Convert one ultralytics Results object to the user's JSON schema."""
    out: list[dict] = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    for i in range(len(xyxy)):
        c = float(confs[i])
        if c < conf_threshold:
            continue
        x1, y1, x2, y2 = xyxy[i].tolist()
        out.append(
            {
                "class": class_name,
                "confidence": round(c, 6),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            }
        )
    return out


def process_video(
    video_path: Path,
    model: RTDETR,
    out_dir: Path,
    conf: float,
    frame_stride: int,
    max_frames: int | None,
    skip_empty: bool,
    imgsz: int,
    device: str | None,
    decoder: str,
) -> int:
    use_ffmpeg = decoder == "ffmpeg" or (
        decoder == "auto" and not _cv2_peek_first_frame(video_path)
    )
    if use_ffmpeg:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            raise RuntimeError(
                "OpenCV could not read frames (common for AV1). Install ffmpeg and ffprobe, "
                "or pass --decoder ffmpeg after installing them."
            )
        frame_iter = _iter_frames_ffmpeg(video_path)
        backend = "ffmpeg"
    else:
        frame_iter = _iter_frames_cv2(video_path)
        backend = "cv2"

    stem = video_path.stem
    written = 0
    frame_idx = 0
    inference_runs = 0
    decoded_any = False

    for frame in frame_iter:
        decoded_any = True
        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        pred_kw = dict(source=frame, conf=conf, imgsz=imgsz, verbose=False)
        if device:
            pred_kw["device"] = device
        results = model.predict(**pred_kw)
        detections = boxes_to_json(results[0], conf_threshold=conf)
        inference_runs += 1

        if skip_empty and not detections:
            frame_idx += 1
            if max_frames is not None and inference_runs >= max_frames:
                break
            continue

        out_path = out_dir / f"{stem}_frame{frame_idx:06d}.json"
        out_path.write_text(json.dumps(detections, indent=2), encoding="utf-8")
        written += 1
        frame_idx += 1

        if max_frames is not None and inference_runs >= max_frames:
            break

    if not decoded_any:
        raise RuntimeError(
            f"No frames decoded from {video_path} (backend={backend}). "
            "Try --decoder ffmpeg (needs ffmpeg+ffprobe), or re-encode to H.264."
        )
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RT-DETR per-frame JSON detections for MP4 files.")
    p.add_argument(
        "--videos-dir",
        type=Path,
        required=True,
        help="Directory containing .mp4 files",
    )
    p.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Trained weights (e.g. runs/rtdetr/drone_finetune/weights/best.pt)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder for JSON (default: videos-dir/detections)",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--frame-stride", type=int, default=1, help="Run every Nth frame (1 = all frames)")
    p.add_argument("--max-frames", type=int, default=None, help="Max frames per video (debug)")
    p.add_argument(
        "--skip-empty",
        action="store_true",
        help="Only write JSON when at least one detection passes --conf",
    )
    p.add_argument("--device", type=str, default="", help="e.g. 0 or cpu; empty = auto")
    p.add_argument(
        "--decoder",
        choices=("auto", "cv2", "ffmpeg"),
        default="auto",
        help="Frame decode: auto tries OpenCV then ffmpeg (AV1 / odd codecs need ffmpeg)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    videos_dir = args.videos_dir.resolve()
    out_dir = (args.output_dir or (videos_dir / "detections")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(videos_dir.glob("*.mp4")) + sorted(videos_dir.glob("*.MP4"))
    if not mp4s:
        raise SystemExit(f"No .mp4 files in {videos_dir}")

    weights = args.weights.resolve()
    if not weights.is_file():
        raise SystemExit(f"Weights not found: {weights}")

    model = RTDETR(str(weights))

    total = 0
    for vid in mp4s:
        n = process_video(
            vid,
            model,
            out_dir,
            conf=args.conf,
            frame_stride=max(1, args.frame_stride),
            max_frames=args.max_frames,
            skip_empty=args.skip_empty,
            imgsz=args.imgsz,
            device=args.device or None,
            decoder=args.decoder,
        )
        print(f"{vid.name}: wrote {n} JSON file(s) -> {out_dir}")
        total += n
    print(f"Done. Total JSON files: {total}")


if __name__ == "__main__":
    main()
