#!/usr/bin/env python3
"""Train QALF on FF++ manifests."""

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
from qalf.data.geometry import DEFAULT_GEOMETRY_FEATURE_MODE, GEOMETRY_FEATURE_MODES
from qalf.data.sbi import resolve_sbi_config, stratum_sampling_weights
from qalf.engine import aggregate_predictions, predict, train_epoch
from qalf.metrics import compute_metrics, select_threshold
from qalf.models import SUPPORTED_TEXTURE_BACKBONES, QALFModel
from qalf.models.geometry import SUPPORTED_GEOMETRY_ARCHITECTURES
from qalf.reporting import collect_run_metadata, save_training_history_plot


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
    """Use the current AMP API with a fallback for older development environments."""

    grad_scaler = getattr(torch.amp, "GradScaler", None)
    if grad_scaler is not None:
        return grad_scaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _loader(dataset, batch_size, workers, training, balanced, seed: int | None = None):
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = _seed_worker
    sampler = None
    shuffle = training
    if training and getattr(dataset, "sbi_enabled", False):
        strata = dataset.sampling_strata
        mixture = dataset.sbi_config["mixture"]
        weights = stratum_sampling_weights(strata, mixture)
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
            weights,
            len(weights),
            replacement=True,
            generator=generator,
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
    backbone_parameters: list[nn.Parameter] = []
    if model.texture_encoder is not None:
        backbone_parameters = list(model.texture_encoder.features.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    groups: list[dict[str, object]] = []
    if head_parameters:
        groups.append({"params": head_parameters, "lr": learning_rate, "name": "head"})
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": backbone_learning_rate,
                "name": "texture_backbone",
            }
        )
    return groups


def _epoch_messages(
    epoch: int,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float | int],
    duration_seconds: float,
) -> list[str]:
    lr = float(next(iter(optimizer.param_groups))["lr"])
    return [
        f"[{epoch:03d}/{epochs}] {duration_seconds:.0f}s  "
        f"loss={train_metrics['loss']:.4f} (geo={train_metrics['geometry']:.4f} tex={train_metrics['texture']:.4f})  "
        f"auc={float(validation_metrics['auc']):.4f} "
        f"ap={float(validation_metrics['average_precision']):.4f} "
        f"eer={float(validation_metrics['eer']):.4f} "
        f"bal={float(validation_metrics['balanced_accuracy']):.4f}  "
        f"geo_auc={float(validation_metrics['geometry_auc']):.4f} "
        f"tex_auc={float(validation_metrics['texture_auc']):.4f}  "
        f"thr={float(validation_metrics['threshold']):.4f}  lr={lr:.2e}",
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
    parser.add_argument(
        "--texture-mode",
        choices=tuple(sorted(TEXTURE_MODES)),
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--backbone-learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--geometry-loss-weight", type=float)
    parser.add_argument("--texture-loss-weight", type=float)
    parser.add_argument("--early-stop-patience", type=int)
    parser.add_argument("--geometry-hidden", type=int)
    parser.add_argument("--geometry-layers", type=int)
    parser.add_argument(
        "--geometry-architecture",
        choices=tuple(sorted(SUPPORTED_GEOMETRY_ARCHITECTURES)),
    )
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument(
        "--texture-gate-bias",
        type=float,
        help="Initial trainable logit bias toward the texture fusion branch.",
    )
    parser.add_argument(
        "--fake-methods",
        nargs="+",
        help="FF++ fake methods to retain; real/original records are always retained.",
    )
    parser.add_argument(
        "--geometry-mode",
        choices=tuple(sorted(GEOMETRY_FEATURE_MODES)),
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("texture", "quality"),
    )
    parser.add_argument(
        "--texture-backbone",
        choices=tuple(sorted(SUPPORTED_TEXTURE_BACKBONES)),
    )
    parser.add_argument("--modality-dropout-probability", type=float)
    parser.add_argument(
        "--exclude-sbi-from-modality-dropout",
        action="store_true",
        default=None,
        help=(
            "Do not modality-drop SBI samples; consequently exclude them from "
            "reliability-gate supervision."
        ),
    )
    parser.add_argument("--reliability-gate-loss-weight", type=float)
    parser.add_argument("--no-geometry-augmentation", action="store_true")
    parser.add_argument("--no-texture-augmentation", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument(
        "--sbi",
        action="store_true",
        help="Enable the locked full-face SBI hybrid training mixture.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic CUDA algorithms and explicitly seeded data workers.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data = config["data"]
    training = config["training"]
    model_config = config["model"]
    if args.geometry_mode:
        data["geometry_mode"] = args.geometry_mode
    if args.fusion_mode:
        model_config["fusion_mode"] = args.fusion_mode
    if args.texture_backbone:
        model_config["texture_backbone"] = args.texture_backbone
    if args.texture_mode:
        data["texture_mode"] = args.texture_mode
    model_overrides = {
        "geometry_hidden": args.geometry_hidden,
        "geometry_layers": args.geometry_layers,
        "geometry_architecture": args.geometry_architecture,
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
        "texture_gate_bias": args.texture_gate_bias,
        "modality_dropout_probability": args.modality_dropout_probability,
        "exclude_sbi_from_modality_dropout": args.exclude_sbi_from_modality_dropout,
    }
    model_config.update({key: value for key, value in model_overrides.items() if value is not None})
    if args.no_geometry_augmentation:
        data["geometry_augmentation"] = {}
    if args.no_texture_augmentation:
        data["texture_augmentation"] = {}
    if args.fake_methods is not None:
        data["fake_methods"] = args.fake_methods
    if args.sbi:
        sbi_config = dict(data.get("sbi", {}))
        sbi_config["enabled"] = True
        data["sbi"] = sbi_config
    data_overrides = {
        "num_frames": args.num_frames,
        "texture_frames": args.texture_frames,
        "image_size": args.image_size,
        "eval_clips_per_video": args.eval_clips_per_video,
    }
    training_overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "backbone_learning_rate": args.backbone_learning_rate,
        "weight_decay": args.weight_decay,
        "geometry_loss_weight": args.geometry_loss_weight,
        "texture_loss_weight": args.texture_loss_weight,
        "early_stop_patience": args.early_stop_patience,
        "reliability_gate_loss_weight": args.reliability_gate_loss_weight,
    }
    data.update({key: value for key, value in data_overrides.items() if value is not None})
    training.update({key: value for key, value in training_overrides.items() if value is not None})
    if args.seed is not None:
        config["seed"] = args.seed
    if args.no_amp:
        training["amp"] = False
    if args.no_balanced_sampler:
        training["balanced_sampler"] = False
    if args.deterministic:
        training["deterministic"] = True
    data["sbi"] = resolve_sbi_config(data.get("sbi"))
    if bool(data["sbi"]["enabled"]) and not bool(training.get("balanced_sampler", True)):
        parser.error("SBI requires the explicit three-stratum sampler")

    positive_integer_fields = {
        "data.num_frames": data["num_frames"],
        "data.texture_frames": data["texture_frames"],
        "data.image_size": data["image_size"],
        "data.eval_clips_per_video": data.get("eval_clips_per_video", 2),
        "training.epochs": training["epochs"],
        "training.batch_size": training["batch_size"],
    }
    for name, value in positive_integer_fields.items():
        if int(value) < 1:
            parser.error(f"{name} must be at least one")
    if int(data["num_frames"]) < 2:
        parser.error("data.num_frames must be at least two")
    if int(data["texture_frames"]) > int(data["num_frames"]):
        parser.error("data.texture_frames cannot exceed data.num_frames")
    if int(training.get("num_workers", 4)) < 0:
        parser.error("training.num_workers cannot be negative")
    if float(training.get("learning_rate", 0.0)) <= 0:
        parser.error("training.learning_rate must be positive")
    if float(training.get("backbone_learning_rate", training["learning_rate"])) <= 0:
        parser.error("training.backbone_learning_rate must be positive")
    for name in ("weight_decay", "geometry_loss_weight", "texture_loss_weight"):
        if float(training.get(name, 0.0)) < 0:
            parser.error(f"training.{name} cannot be negative")
    if int(training.get("early_stop_patience", 0)) < 0:
        parser.error("training.early_stop_patience cannot be negative")
    for name in ("geometry_hidden", "geometry_layers", "embedding_dim"):
        if int(model_config.get(name, 0)) < 1:
            parser.error(f"model.{name} must be at least one")
    if not 0.0 <= float(model_config.get("dropout", 0.0)) < 1.0:
        parser.error("model.dropout must be in [0, 1)")
    modality_dropout_probability = float(model_config.get("modality_dropout_probability", 0.0))
    if not 0.0 <= modality_dropout_probability <= 0.5:
        parser.error("model.modality_dropout_probability must be in [0, 0.5]")
    reliability_gate_loss_weight = float(training.get("reliability_gate_loss_weight", 0.0))
    if reliability_gate_loss_weight < 0.0:
        parser.error("training.reliability_gate_loss_weight cannot be negative")
    if reliability_gate_loss_weight > 0.0 and modality_dropout_probability == 0.0:
        parser.error("Reliability gate loss requires positive modality dropout")
    exclude_sbi_dropout = bool(
        model_config.get("exclude_sbi_from_modality_dropout", False)
    )
    if exclude_sbi_dropout and modality_dropout_probability == 0.0:
        parser.error("SBI-aware routing requires positive modality dropout")
    if exclude_sbi_dropout and not bool(data["sbi"]["enabled"]):
        parser.error("SBI-aware routing requires enabled SBI training")
    if (
        modality_dropout_probability > 0.0
        and str(model_config.get("fusion_mode", "quality")) != "quality"
    ):
        parser.error("Modality dropout requires quality fusion")
    seed = int(config.get("seed", 42))
    deterministic = bool(training.get("deterministic", False))
    _seed_everything(seed, deterministic=deterministic)
    train_manifest = args.train_manifest or data["train_manifest"]
    val_manifest = args.val_manifest or data["val_manifest"]
    frame_root = args.frame_root or data["train_frame_root"]
    landmark_root = args.landmark_root or data["train_landmark_root"]
    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/qalf"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _create_logger(output_dir / "train.log")
    data["train_manifest"] = str(train_manifest)
    data["val_manifest"] = str(val_manifest)
    data["train_frame_root"] = str(frame_root)
    data["train_landmark_root"] = str(landmark_root)
    config["output_dir"] = str(output_dir)
    data.setdefault("eval_clips_per_video", 2)
    data.setdefault("video_aggregation", "mean")
    data.setdefault("top_k", 1)
    save_json(config, output_dir / "config.json")
    save_json(
        collect_run_metadata(sys.argv, config, PROJECT_ROOT),
        output_dir / "run_metadata.json",
    )

    dataset_args = {
        "frame_root": frame_root,
        "landmark_root": landmark_root,
        "num_frames": int(data["num_frames"]),
        "texture_frames": int(data["texture_frames"]),
        "image_size": int(data["image_size"]),
        "geometry_mode": str(data.get("geometry_mode", DEFAULT_GEOMETRY_FEATURE_MODE)),
        "texture_mode": str(data.get("texture_mode", "full_face")),
        "geometry_augmentation": data.get("geometry_augmentation", {}),
        "texture_augmentation": data.get("texture_augmentation"),
        "fake_methods": data.get("fake_methods"),
    }
    train_dataset = QALFVideoDataset(
        train_manifest,
        training=True,
        clips_per_video=1,
        sbi_config=data["sbi"],
        **dataset_args,
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
        val_dataset,
        batch_size,
        workers,
        False,
        False,
        seed + 1 if deterministic else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QALFModel(
        geometry_input_dim=train_dataset.geometry_input_dim,
        geometry_hidden=int(model_config.get("geometry_hidden", 96)),
        geometry_layers=int(model_config.get("geometry_layers", 3)),
        geometry_architecture=str(model_config.get("geometry_architecture", "tcn_mean")),
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        dropout=float(model_config.get("dropout", 0.2)),
        texture_pretrained=bool(model_config.get("texture_pretrained", True)),
        texture_backbone=str(model_config.get("texture_backbone", "efficientnet_b0")),
        geometry_quality_dim=train_dataset.geometry_quality_dim,
        texture_quality_dim=train_dataset.texture_quality_dim,
        fusion_mode=str(model_config.get("fusion_mode", "quality")),
        texture_gate_bias=float(model_config.get("texture_gate_bias", 0.0)),
        modality_dropout_probability=modality_dropout_probability,
        exclude_sbi_from_modality_dropout=bool(
            model_config.get("exclude_sbi_from_modality_dropout", False)
        ),
    ).to(device)
    optimizer = AdamW(
        _optimizer_groups(
            model,
            float(training.get("learning_rate", 3e-4)),
            float(
                training.get(
                    "backbone_learning_rate",
                    training.get("learning_rate", 3e-4),
                )
            ),
        ),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(training["epochs"]))
    )
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = _make_grad_scaler(amp_enabled)
    criterion = nn.BCEWithLogitsLoss()
    geometry_loss_weight = float(training.get("geometry_loss_weight", 0.25))
    texture_loss_weight = float(training.get("texture_loss_weight", 0.25))
    fusion_mode = str(model_config.get("fusion_mode", "quality"))
    if fusion_mode == "texture":
        geometry_loss_weight = 0.0

    history: list[dict[str, object]] = []
    best_auc = -float("inf")
    best_epoch = 0
    best_threshold = float("nan")
    best_path = output_dir / "best.pt"
    stale_epochs = 0
    patience = int(training.get("early_stop_patience", 0))
    epochs = int(training["epochs"])
    run_started = time.perf_counter()
    logger.info(
        "Training started  output=%s modality_dropout=%.2f reliability=%.2f "
        "exclude_sbi_from_modality_dropout=%s",
        output_dir,
        modality_dropout_probability,
        reliability_gate_loss_weight,
        bool(model_config.get("exclude_sbi_from_modality_dropout", False)),
    )
    history_plot_enabled = True
    training_plot_path = output_dir / "plots" / "training_history.png"
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            geometry_loss_weight,
            texture_loss_weight,
            reliability_gate_weight=reliability_gate_loss_weight,
        )
        validation_clips = predict(model, val_loader, device)
        validation = aggregate_predictions(
            validation_clips,
            method=str(data["video_aggregation"]),
            top_k=int(data["top_k"]),
        )
        labels = np.asarray(validation["label"], dtype=np.int64)
        scores = np.asarray(validation["score"], dtype=np.float64)
        threshold = select_threshold(labels, scores)
        val_metrics = compute_metrics(labels, scores, threshold)
        geometry_scores = np.asarray(validation["geometry_score"], dtype=np.float64)
        texture_scores = np.asarray(validation["texture_score"], dtype=np.float64)
        val_metrics["geometry_auc"] = compute_metrics(labels, geometry_scores, threshold)["auc"]
        val_metrics["texture_auc"] = compute_metrics(labels, texture_scores, threshold)["auc"]
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        epoch_duration = time.perf_counter() - epoch_started
        if history_plot_enabled:
            try:
                save_training_history_plot(
                    history,
                    training_plot_path,
                )
            except Exception as error:
                history_plot_enabled = False
                logger.warning("Training history plots disabled: %s", error)
        for message in _epoch_messages(
            epoch,
            epochs,
            optimizer,
            train_metrics,
            val_metrics,
            epoch_duration,
        ):
            logger.info(message)
        scheduler.step()

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "geometry_input_dim": train_dataset.geometry_input_dim,
            "geometry_quality_dim": train_dataset.geometry_quality_dim,
            "texture_quality_dim": train_dataset.texture_quality_dim,
            "threshold": threshold,
            "validation_metrics": val_metrics,
            "model_weights": "raw",
            "label_convention": "real=0,fake=1",
            "score_target": "fake",
            "threshold_selection": "youden_j_ffpp_validation",
            "best_metric": "val_auc",
        }
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            best_threshold = threshold
            stale_epochs = 0
            torch.save(checkpoint, best_path)
            logger.info("  ★ best  epoch=%03d  val_auc=%.4f  thr=%.4f  → %s", best_epoch, best_auc, best_threshold, best_path)
        else:
            stale_epochs += 1
            if patience > 0:
                logger.info("  patience %d/%d  best_epoch=%03d  best_auc=%.4f", stale_epochs, patience, best_epoch, best_auc)
            if patience > 0 and stale_epochs >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break
    logger.info(
        "Done  epochs=%d  best_epoch=%03d  best_auc=%.4f  thr=%.4f  model=%s",
        len(history),
        best_epoch,
        best_auc,
        best_threshold,
        best_path,
    )
    history_path = output_dir / "history.json"
    save_json(history, history_path)
    best_metrics = next(
        (row["validation"] for row in history if int(row["epoch"]) == best_epoch),
        {},
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
            "duration_seconds": time.perf_counter() - run_started,
            "best_model": str(best_path),
            "history": str(history_path),
            "training_plot": (
                str(training_plot_path) if training_plot_path.is_file() else None
            ),
        },
        output_dir / "training_summary.json",
    )


if __name__ == "__main__":
    main()
