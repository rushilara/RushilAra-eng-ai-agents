#!/usr/bin/env python3
"""
Kalman-filter tracking over Task 1 per-frame JSON detections; export trimmed overlay videos.

Uses filterpy (constant-velocity model on bbox center). Predicts every frame; updates when
the detector fires; coasts for a limited number of missed frames.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter


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
        raise RuntimeError(f"ffprobe failed: {r.stderr}")
    data = json.loads(r.stdout)
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
        if rc not in (0, None, -13, -15):
            raise RuntimeError(
                f"ffmpeg decode failed ({rc}): {err.decode(errors='replace')[:2000]}"
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


def _ffprobe_fps(video_path: Path) -> float:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    s = r.stdout.strip()
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


def load_detection_json(path: Path) -> list[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def best_drone_detection(
    detections: list[dict],
) -> tuple[np.ndarray | None, tuple[float, float, float, float] | None]:
    """Return measurement z = [[cx],[cy]] and bbox (x1,y1,x2,y2) for highest-confidence drone."""
    candidates = [d for d in detections if d.get("class") == "drone" and "bbox" in d]
    if not candidates:
        candidates = [d for d in detections if "bbox" in d]
    if not candidates:
        return None, None
    best = max(candidates, key=lambda d: float(d.get("confidence", 0.0)))
    x1, y1, x2, y2 = (float(x) for x in best["bbox"])
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    z = np.array([[cx], [cy]], dtype=np.float64)
    return z, (x1, y1, x2, y2)


def build_detection_index(detections_dir: Path, video_stem: str) -> dict[int, list[dict]]:
    """Map frame index -> detection list from files named {stem}_frame{idx:06d}.json"""
    out: dict[int, list[dict]] = {}
    pat = re.compile(re.escape(video_stem) + r"_frame(\d+)\.json$", re.I)
    for p in detections_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".json":
            continue
        m = pat.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        out[idx] = load_detection_json(p)
    return out


def make_kalman_filter(dt: float = 1.0) -> KalmanFilter:
    """Constant velocity: state [cx, cy, vxc, vyc]."""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    kf.P *= 500.0
    kf.R = np.eye(2, dtype=np.float64) * 20.0
    kf.Q = np.eye(4, dtype=np.float64) * 0.05
    kf.Q[2, 2] = kf.Q[3, 3] = 0.5
    return kf


def init_kf_from_measurement(cx: float, cy: float) -> KalmanFilter:
    kf = make_kalman_filter()
    kf.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float64)
    return kf


def draw_overlay(
    frame: np.ndarray,
    bbox_det: tuple[float, float, float, float] | None,
    trajectory: list[tuple[float, float]],
    color_bbox: tuple[int, int, int] = (40, 165, 255),
    color_traj: tuple[int, int, int] = (0, 255, 180),
    color_kf: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    if bbox_det is not None:
        x1, y1, x2, y2 = bbox_det
        p1 = (int(round(np.clip(x1, 0, w - 1))), int(round(np.clip(y1, 0, h - 1))))
        p2 = (int(round(np.clip(x2, 0, w - 1))), int(round(np.clip(y2, 0, h - 1))))
        cv2.rectangle(out, p1, p2, color_bbox, 2, lineType=cv2.LINE_AA)
    if len(trajectory) >= 2:
        pts = np.array(
            [
                (int(round(np.clip(x, 0, w - 1))), int(round(np.clip(y, 0, h - 1))))
                for x, y in trajectory
            ],
            dtype=np.int32,
        )
        cv2.polylines(out, [pts], isClosed=False, color=color_traj, thickness=2, lineType=cv2.LINE_AA)
    if trajectory:
        cx, cy = trajectory[-1]
        cv2.circle(
            out,
            (int(round(np.clip(cx, 0, w - 1))), int(round(np.clip(cy, 0, h - 1)))),
            6,
            color_kf,
            -1,
            lineType=cv2.LINE_AA,
        )
    return out


def write_video_bgr_ffmpeg(
    frames: list[np.ndarray],
    out_path: Path,
    fps: float,
    w: int,
    h: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    for fr in frames:
        if fr.shape[1] != w or fr.shape[0] != h:
            fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_LINEAR)
        proc.stdin.write(fr.astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encoding failed with exit {proc.returncode}")


def iter_video_frames(video_path: Path, decoder: str):
    use_ffmpeg = decoder == "ffmpeg" or (
        decoder == "auto" and not _cv2_peek_first_frame(video_path)
    )
    if use_ffmpeg:
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg required for this video; install ffmpeg or use --decoder cv2 with H.264 input.")
        yield from _iter_frames_ffmpeg(video_path)
    else:
        yield from _iter_frames_cv2(video_path)


def process_one_video(
    video_path: Path,
    detections_dir: Path,
    out_path: Path,
    max_missed: int,
    decoder: str,
) -> int:
    stem = video_path.stem
    det_index = build_detection_index(detections_dir, stem)
    if not det_index:
        raise RuntimeError(f"No JSON files matching '{stem}_frame*.json' in {detections_dir}")

    fps = _ffprobe_fps(video_path)
    vw, vh = _ffprobe_video_size(video_path)

    kf: KalmanFilter | None = None
    track_active = False
    miss = 0
    traj: list[tuple[float, float]] = []
    out_frames: list[np.ndarray] = []

    for frame_idx, frame in enumerate(iter_video_frames(video_path, decoder)):
        dets = det_index.get(frame_idx, [])
        z, bbox_det = best_drone_detection(dets)

        if not track_active:
            if z is None:
                continue
            kf = init_kf_from_measurement(float(z[0, 0]), float(z[1, 0]))
            track_active = True
            miss = 0
            traj = [(float(kf.x[0, 0]), float(kf.x[1, 0]))]
            out_frames.append(draw_overlay(frame, bbox_det, traj))
            continue

        assert kf is not None
        kf.predict()
        bbox_draw: tuple[float, float, float, float] | None = None
        if z is not None:
            kf.update(z)
            miss = 0
            bbox_draw = bbox_det
        else:
            miss += 1
            if miss > max_missed:
                track_active = False
                kf = None
                traj = []
                continue

        traj.append((float(kf.x[0, 0]), float(kf.x[1, 0])))
        out_frames.append(draw_overlay(frame, bbox_draw, traj))

    if not out_frames:
        raise RuntimeError(f"No track produced for {video_path.name} (no detections to initialize).")

    write_video_bgr_ffmpeg(out_frames, out_path, fps, vw, vh)
    return len(out_frames)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kalman track drone from JSON detections; write overlay videos.")
    p.add_argument("--videos-dir", type=Path, required=True, help="Folder with input .mp4 files")
    p.add_argument(
        "--detections-dir",
        type=Path,
        required=True,
        help="Folder with Task 1 JSON files ({video_stem}_frameXXXXXX.json)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder (default: videos-dir/tracked)",
    )
    p.add_argument(
        "--max-missed",
        type=int,
        default=8,
        help="Max consecutive frames without detector update while coasting on prediction",
    )
    p.add_argument("--decoder", choices=("auto", "cv2", "ffmpeg"), default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    videos_dir = args.videos_dir.resolve()
    det_dir = args.detections_dir.resolve()
    out_dir = (args.output_dir or (videos_dir / "tracked")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = sorted(videos_dir.glob("*.mp4")) + sorted(videos_dir.glob("*.MP4"))
    if not vids:
        raise SystemExit(f"No .mp4 in {videos_dir}")

    for vp in vids:
        outp = out_dir / f"{vp.stem}_tracked.mp4"
        n = process_one_video(vp, det_dir, outp, args.max_missed, args.decoder)
        print(f"{vp.name} -> {outp} ({n} frames)")


if __name__ == "__main__":
    main()
