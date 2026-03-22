#!/usr/bin/env python3
"""
Fine-tune Ultralytics RT-DETR on the UAV dataset (train/val; test in yaml for final metrics).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import RTDETR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune RT-DETR for single-class drone detection.")
    p.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Ultralytics data yaml (default: data/yolo_ultralytics/drone.yaml after prepare_yolo_dataset.py)",
    )
    p.add_argument(
        "--model",
        type=str,
        default="rtdetr-l.pt",
        help="Pretrained weights: rtdetr-l.pt or rtdetr-x.pt",
    )
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Per-device batch (RT-DETR-L is VRAM-heavy; use 2 or 1 if CUDA OOM). -1 = autobatch",
    )
    p.add_argument("--device", type=str, default="", help="e.g. 0, cpu, or mps; empty = auto")
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader workers (lower if shared GPU or RAM constrained)",
    )
    p.add_argument("--patience", type=int, default=30, help="Early stopping patience (epochs without val improvement)")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--lr0", type=float, default=1e-3, help="Initial learning rate")
    p.add_argument("--lrf", type=float, default=0.01, help="Final LR fraction (cosine end = lr0 * lrf)")
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "Adam", "SGD", "RMSProp", "auto"])
    p.add_argument("--cos-lr", action="store_true", default=True)
    p.add_argument("--no-cos-lr", action="store_false", dest="cos_lr")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--project", type=str, default="runs/rtdetr")
    p.add_argument("--name", type=str, default="drone_finetune")
    p.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default="", help="Path to last.pt to resume")
    p.add_argument("--val-test-after", action="store_true", help="Run model.val(split='test') after training")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hw3 = Path(__file__).resolve().parent
    data_yaml = args.data_yaml or (hw3 / "data/yolo_ultralytics/drone.yaml")
    data_yaml = data_yaml.resolve()
    if not data_yaml.is_file():
        raise SystemExit(
            f"Missing {data_yaml}. Run: python prepare_yolo_dataset.py\n"
        )

    train_kw = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        workers=args.workers,
        patience=args.patience,
        weight_decay=args.weight_decay,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        optimizer=args.optimizer,
        cos_lr=args.cos_lr,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
        verbose=True,
        plots=True,
        amp=True,
    )
    if args.resume:
        model = RTDETR(args.resume)
        train_kw["resume"] = True
    else:
        model = RTDETR(args.model)

    model.train(**train_kw)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if args.val_test_after and best.is_file():
        print("Validating on test split...")
        model = RTDETR(str(best))
        model.val(data=str(data_yaml), split="test", plots=True)


if __name__ == "__main__":
    main()
