#!/usr/bin/env python3
"""Train the texture-only QALF video classifier on FF++."""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.config import load_config, save_json
from qalf.data.dataset import TEXTURE_MODES, QALFVideoDataset
from qalf.data.sbi import resolve_sbi_config, stratum_sampling_weights
from qalf.engine import aggregate_predictions, predict, train_epoch
from qalf.ema import ModelEMA
from qalf.metrics import compute_metrics, select_threshold
from qalf.models import SUPPORTED_TEXTURE_BACKBONES, QALFModel
from qalf.reporting import save_training_history_plot


def _seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(_: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _make_grad_scaler(enabled: bool):
    grad_scaler = getattr(torch.amp, "GradScaler", None)
    if grad_scaler is not None:
        return grad_scaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _loader(
    dataset: QALFVideoDataset,
    batch_size: int,
    workers: int,
    training: bool,
    balanced: bool,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = _seed_worker
    sampler = None
    shuffle = training
    if training and dataset.sbi_enabled:
        weights = stratum_sampling_weights(dataset.sampling_strata, dataset.sbi_config["mixture"])
        sampler = WeightedRandomSampler(
            weights,
            int(dataset.samples_per_epoch),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    elif training and balanced:
        counts = np.bincount(dataset.labels, minlength=2)
        if np.any(counts == 0):
            raise ValueError(f"Training manifest must contain both classes: counts={counts}")
        weights = [1.0 / counts[label] for label in dataset.labels]
        sampler = WeightedRandomSampler(
            weights, len(weights), replacement=True, generator=generator
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def _create_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("qalf.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _optimizer_groups(
    model: QALFModel,
    learning_rate: float,
    backbone_learning_rate: float,
) -> list[dict[str, object]]:
    backbone_parameters = list(model.texture_encoder.features.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    return [
        {"params": head_parameters, "lr": learning_rate, "name": "head"},
        {
            "params": backbone_parameters,
            "lr": backbone_learning_rate,
            "name": "texture_backbone",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--frame-root")
    parser.add_argument("--landmark-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--texture-frames", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--eval-clips-per-video", type=int)
    parser.add_argument("--texture-mode", choices=tuple(sorted(TEXTURE_MODES)))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--backbone-learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--early-stop-patience", type=int)
    parser.add_argument("--ema-decay", type=float)
    parser.add_argument("--validation-weights", choices=("raw", "ema"))
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--fake-methods", nargs="+")
    parser.add_argument(
        "--texture-backbone", choices=tuple(sorted(SUPPORTED_TEXTURE_BACKBONES))
    )
    parser.add_argument(
        "--srm-preprocess",
        action="store_true",
        help="Inject a fixed three-kernel SRM residual before EfficientNet-B0.",
    )
    parser.add_argument("--no-texture-augmentation", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument("--sbi", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data, model_config, training = config["data"], config["model"], config["training"]
    if args.texture_mode:
        data["texture_mode"] = args.texture_mode
    if args.texture_backbone:
        model_config["texture_backbone"] = args.texture_backbone
    if args.srm_preprocess:
        model_config["srm_preprocess"] = True
    for key, value in {
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
    }.items():
        if value is not None:
            model_config[key] = value
    if args.no_texture_augmentation:
        data["texture_augmentation"] = {}
    if args.fake_methods is not None:
        data["fake_methods"] = args.fake_methods
    if args.sbi:
        sbi_config = dict(data.get("sbi", {}))
        sbi_config["enabled"] = True
        data["sbi"] = sbi_config
    for key, value in {
        "num_frames": args.num_frames,
        "texture_frames": args.texture_frames,
        "image_size": args.image_size,
        "eval_clips_per_video": args.eval_clips_per_video,
    }.items():
        if value is not None:
            data[key] = value
    for key, value in {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "weight_decay": args.weight_decay,
        "early_stop_patience": args.early_stop_patience,
        "ema_decay": args.ema_decay,
        "validation_weights": args.validation_weights,
    }.items():
        if value is not None:
            training[key] = value
    if args.seed is not None:
        config["seed"] = args.seed
    if args.no_amp:
        training["amp"] = False
    if args.no_balanced_sampler:
        training["balanced_sampler"] = False
    if args.deterministic:
        training["deterministic"] = True
    data["sbi"] = resolve_sbi_config(data.get("sbi"))
    if data["sbi"]["enabled"] and not training.get("balanced_sampler", True):
        parser.error("SBI requires the explicit three-stratum sampler")
    if not 1 <= int(data["texture_frames"]) <= int(data["num_frames"]):
        parser.error("texture_frames must be in [1, num_frames]")
    if int(training.get("num_workers", 4)) < 0:
        parser.error("num_workers cannot be negative")
    if float(training["learning_rate"]) <= 0 or float(training["backbone_learning_rate"]) <= 0:
        parser.error("learning rates must be positive")
    if int(training.get("early_stop_patience", 0)) < 0:
        parser.error("early_stop_patience cannot be negative")
    ema_decay = float(training.get("ema_decay", 0.0))
    validation_weights = str(
        training.get("validation_weights", "ema" if ema_decay > 0.0 else "raw")
    )
    if not 0.0 <= ema_decay < 1.0:
        parser.error("ema_decay must be in [0, 1)")
    if validation_weights not in {"raw", "ema"}:
        parser.error("validation_weights must be raw or ema")
    if validation_weights == "ema" and ema_decay <= 0.0:
        parser.error("validation_weights=ema requires ema_decay > 0")

    seed = int(config.get("seed", 42))
    deterministic = bool(training.get("deterministic", False))
    _seed_everything(seed, deterministic)
    train_manifest = args.train_manifest or data["train_manifest"]
    val_manifest = args.val_manifest or data["val_manifest"]
    frame_root = args.frame_root or data["train_frame_root"]
    landmark_root = args.landmark_root or data["train_landmark_root"]
    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/qalf"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _create_logger(output_dir / "train.log")
    data.update(
        {
            "train_manifest": str(train_manifest),
            "val_manifest": str(val_manifest),
            "train_frame_root": str(frame_root),
            "train_landmark_root": str(landmark_root),
        }
    )
    config["output_dir"] = str(output_dir)
    data.setdefault("eval_clips_per_video", 3)
    data.setdefault("video_aggregation", "mean")
    data.setdefault("top_k", 1)
    save_json(config, output_dir / "config.json")

    dataset_args = {
        "frame_root": frame_root,
        "landmark_root": landmark_root,
        "num_frames": int(data["num_frames"]),
        "texture_frames": int(data["texture_frames"]),
        "image_size": int(data["image_size"]),
        "texture_mode": str(data.get("texture_mode", "full_face")),
        "texture_augmentation": data.get("texture_augmentation"),
        "fake_methods": data.get("fake_methods"),
    }
    train_dataset = QALFVideoDataset(
        train_manifest, training=True, clips_per_video=1, sbi_config=data["sbi"], **dataset_args
    )
    val_dataset = QALFVideoDataset(
        val_manifest,
        training=False,
        clips_per_video=int(data["eval_clips_per_video"]),
        **dataset_args,
    )
    batch_size = int(training["batch_size"])
    workers = int(training.get("num_workers", 4))
    train_loader = _loader(
        train_dataset,
        batch_size,
        workers,
        True,
        bool(training.get("balanced_sampler", True)),
        seed if deterministic else None,
    )
    val_loader = _loader(
        val_dataset, batch_size, workers, False, False, seed + 1 if deterministic else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QALFModel(
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        dropout=float(model_config.get("dropout", 0.2)),
        texture_pretrained=bool(model_config.get("texture_pretrained", True)),
        texture_backbone=str(model_config.get("texture_backbone", "efficientnet_b0")),
        srm_preprocess=bool(model_config.get("srm_preprocess", False)),
    ).to(device)
    optimizer = AdamW(
        _optimizer_groups(
            model,
            float(training.get("learning_rate", 3e-4)),
            float(training.get("backbone_learning_rate", 3e-5)),
        ),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(training["epochs"]))
    )
    scaler = _make_grad_scaler(bool(training.get("amp", True)) and device.type == "cuda")
    criterion = nn.BCEWithLogitsLoss()
    ema = ModelEMA(model, ema_decay) if ema_decay > 0.0 else None
    history: list[dict[str, object]] = []
    best_auc = -float("inf")
    best_epoch = 0
    best_threshold = float("nan")
    best_path = output_dir / "best.pt"
    stale_epochs = 0
    patience = int(training.get("early_stop_patience", 0))
    epochs = int(training["epochs"])
    run_started = time.perf_counter()
    logger.info("Training started")
    training_plot_path = output_dir / "plots" / "training_history.png"
    plot_enabled = True
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler, ema=ema
        )
        raw_state = None
        if validation_weights == "ema":
            if ema is None:
                raise RuntimeError("EMA validation requested without EMA state")
            raw_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            ema.copy_to(model)
        try:
            validation = aggregate_predictions(
                predict(model, val_loader, device),
                method=str(data["video_aggregation"]),
                top_k=int(data["top_k"]),
            )
        finally:
            if raw_state is not None:
                model.load_state_dict(raw_state, strict=True)
        labels = np.asarray(validation["label"], dtype=np.int64)
        scores = np.asarray(validation["score"], dtype=np.float64)
        threshold = select_threshold(labels, scores)
        val_metrics = compute_metrics(labels, scores, threshold)
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        if plot_enabled:
            try:
                save_training_history_plot(history, training_plot_path)
            except Exception as error:
                plot_enabled = False
                logger.warning("Training history plots disabled: %s", error)
        logger.info(
            "[%03d/%03d] %.0fs  loss=%.4f  auc=%.4f ap=%.4f eer=%.4f bal=%.4f "
            "thr=%.4f  lr=%.2e  sbi=%.3f",
            epoch,
            epochs,
            time.perf_counter() - epoch_started,
            train_metrics["loss"],
            val_metrics["auc"],
            val_metrics["average_precision"],
            val_metrics["eer"],
            val_metrics["balanced_accuracy"],
            threshold,
            float(optimizer.param_groups[0]["lr"]),
            train_metrics["sbi_fraction"],
        )
        scheduler.step()
        checkpoint = {
            "epoch": epoch,
            "model": ema.state_dict() if validation_weights == "ema" and ema else {
                name: value.detach().clone() for name, value in model.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "config": config,
            "threshold": threshold,
            "validation_metrics": val_metrics,
            "model_weights": validation_weights,
            "ema_decay": ema_decay,
            "label_convention": "real=0,fake=1",
            "score_target": "fake",
            "threshold_selection": "youden_j_ffpp_validation",
            "best_metric": "val_auc",
            "architecture": "texture_only",
        }
        if val_metrics["auc"] > best_auc:
            best_auc = float(val_metrics["auc"])
            best_epoch = epoch
            best_threshold = threshold
            stale_epochs = 0
            torch.save(checkpoint, best_path)
            logger.info(
                "  ★ best  epoch=%03d  val_auc=%.4f  thr=%.4f  → %s",
                best_epoch,
                best_auc,
                best_threshold,
                best_path,
            )
        else:
            stale_epochs += 1
            if patience > 0:
                logger.info(
                    "  patience %d/%d  best_epoch=%03d  best_auc=%.4f",
                    stale_epochs,
                    patience,
                    best_epoch,
                    best_auc,
                )
            if patience > 0 and stale_epochs >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    history_path = output_dir / "history.json"
    save_json(history, history_path)
    best_metrics = next(
        (row["validation"] for row in history if int(row["epoch"]) == best_epoch), {}
    )
    save_json(
        {
            "status": "complete",
            "completed_epochs": len(history),
            "best_epoch": best_epoch,
            "best_metric": "val_auc",
            "best_value": best_auc,
            "best_threshold": best_threshold,
            "best_validation_metrics": best_metrics,
            "ema_decay": ema_decay,
            "validation_weights": validation_weights,
            "duration_seconds": time.perf_counter() - run_started,
            "best_model": str(best_path),
            "history": str(history_path),
            "training_plot": str(training_plot_path) if training_plot_path.is_file() else None,
        },
        output_dir / "training_summary.json",
    )
    logger.info(
        "Done  epochs=%d  best_epoch=%03d  best_auc=%.4f  thr=%.4f  model=%s",
        len(history),
        best_epoch,
        best_auc,
        best_threshold,
        best_path,
    )


if __name__ == "__main__":
    main()
