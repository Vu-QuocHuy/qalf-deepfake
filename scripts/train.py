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
from qalf.data.dataset import TEXTURE_MODES, QALFVideoDataset, texture_view_count
from qalf.data.geometry import DEFAULT_GEOMETRY_FEATURE_MODE, GEOMETRY_FEATURE_MODES
from qalf.data.sbi import resolve_sbi_config, stratum_sampling_weights
from qalf.engine import EMAModel, aggregate_predictions, predict, train_epoch
from qalf.metrics import compute_metrics, select_threshold
from qalf.models import SUPPORTED_TEXTURE_BACKBONES, SUPPORTED_TEXTURE_POOLING, QALFModel
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
    learning_rates = {
        str(group.get("name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
    rate_text = " ".join(f"{name}_lr={rate:.4e}" for name, rate in learning_rates.items())
    return [
        f"epoch={epoch:03d}/{epochs:03d} duration={duration_seconds:.1f}s {rate_text}",
        "  train | "
        f"total={train_metrics['loss']:.4f} fused={train_metrics['fused']:.4f} "
        f"geometry={train_metrics['geometry']:.4f} texture={train_metrics['texture']:.4f} "
        f"sbi_fraction={train_metrics.get('sbi_fraction', 0.0):.3f} "
        f"geometry_supervision={train_metrics.get('geometry_supervision_fraction', 1.0):.3f}",
        "  val   | "
        f"auc={float(validation_metrics['auc']):.4f} "
        f"ap={float(validation_metrics['average_precision']):.4f} "
        f"eer={float(validation_metrics['eer']):.4f} "
        f"balanced_acc={float(validation_metrics['balanced_accuracy']):.4f} "
        f"accuracy={float(validation_metrics['accuracy']):.4f} "
        f"f1={float(validation_metrics['f1']):.4f} "
        f"threshold={float(validation_metrics['threshold']):.4f}",
        "  branch| "
        f"geometry_auc={float(validation_metrics['geometry_auc']):.4f} "
        f"texture_auc={float(validation_metrics['texture_auc']):.4f} "
        f"weights_geometry/texture="
        f"{float(validation_metrics['mean_geometry_weight']):.4f}/"
        f"{float(validation_metrics['mean_texture_weight']):.4f}",
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
    parser.add_argument(
        "--ema-decay",
        type=float,
        help="EMA decay in (0, 1); omit or use 0 to validate raw weights.",
    )
    parser.add_argument("--early-stop-patience", type=int)
    parser.add_argument("--geometry-hidden", type=int)
    parser.add_argument("--geometry-layers", type=int)
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
        choices=(
            "geometry",
            "texture",
            "average",
            "concat",
            "content_gate",
            "quality_only",
            "quality",
        ),
    )
    parser.add_argument(
        "--texture-backbone",
        choices=tuple(sorted(SUPPORTED_TEXTURE_BACKBONES)),
    )
    parser.add_argument(
        "--texture-temporal-pooling",
        choices=tuple(sorted(SUPPORTED_TEXTURE_POOLING)),
    )
    parser.add_argument("--texture-mixstyle-probability", type=float)
    parser.add_argument("--texture-mixstyle-alpha", type=float)
    parser.add_argument("--texture-mixstyle-layers", type=int, nargs="+")
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
    if args.texture_temporal_pooling:
        model_config["texture_temporal_pooling"] = args.texture_temporal_pooling
    if args.texture_mode:
        data["texture_mode"] = args.texture_mode
    model_overrides = {
        "geometry_hidden": args.geometry_hidden,
        "geometry_layers": args.geometry_layers,
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
        "texture_gate_bias": args.texture_gate_bias,
        "texture_mixstyle_probability": args.texture_mixstyle_probability,
        "texture_mixstyle_alpha": args.texture_mixstyle_alpha,
        "texture_mixstyle_layers": args.texture_mixstyle_layers,
    }
    model_config.update(
        {key: value for key, value in model_overrides.items() if value is not None}
    )
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
        "ema_decay": args.ema_decay,
        "early_stop_patience": args.early_stop_patience,
    }
    data.update({key: value for key, value in data_overrides.items() if value is not None})
    training.update(
        {key: value for key, value in training_overrides.items() if value is not None}
    )
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
    ema_decay = float(training.get("ema_decay", 0.0))
    if ema_decay != 0.0 and not 0.0 < ema_decay < 1.0:
        parser.error("training.ema_decay must be zero or in (0, 1)")
    for name in ("geometry_hidden", "geometry_layers", "embedding_dim"):
        if int(model_config.get(name, 0)) < 1:
            parser.error(f"model.{name} must be at least one")
    if not 0.0 <= float(model_config.get("dropout", 0.0)) < 1.0:
        parser.error("model.dropout must be in [0, 1)")
    mixstyle_probability = float(model_config.get("texture_mixstyle_probability", 0.0))
    mixstyle_alpha = float(model_config.get("texture_mixstyle_alpha", 0.1))
    mixstyle_layers = tuple(
        int(index) for index in model_config.get("texture_mixstyle_layers", [])
    )
    if not 0.0 <= mixstyle_probability <= 1.0:
        parser.error("model.texture_mixstyle_probability must be in [0, 1]")
    if mixstyle_alpha <= 0.0:
        parser.error("model.texture_mixstyle_alpha must be positive")
    if mixstyle_probability > 0.0 and not mixstyle_layers:
        parser.error("model.texture_mixstyle_layers cannot be empty when MixStyle is enabled")
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
        "texture_mode": str(data.get("texture_mode", "canonical_skin")),
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
    train_counts = np.bincount(
        [record.label for record in train_dataset.records], minlength=2
    )
    val_counts = np.bincount(
        [record.label for record in val_dataset.records], minlength=2
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
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        dropout=float(model_config.get("dropout", 0.2)),
        texture_pretrained=bool(model_config.get("texture_pretrained", True)),
        texture_backbone=str(model_config.get("texture_backbone", "efficientnet_b0")),
        texture_temporal_pooling=str(model_config.get("texture_temporal_pooling", "mean")),
        texture_views=texture_view_count(
            str(data.get("texture_mode", "canonical_skin"))
        ),
        texture_mixstyle_probability=mixstyle_probability,
        texture_mixstyle_alpha=mixstyle_alpha,
        texture_mixstyle_layers=mixstyle_layers,
        geometry_quality_dim=train_dataset.geometry_quality_dim,
        texture_quality_dim=train_dataset.texture_quality_dim,
        fusion_mode=str(model_config.get("fusion_mode", "quality")),
        texture_gate_bias=float(model_config.get("texture_gate_bias", 0.0)),
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
    ema = EMAModel(model, decay=ema_decay) if ema_decay > 0 else None
    criterion = nn.BCEWithLogitsLoss()
    geometry_loss_weight = float(training.get("geometry_loss_weight", 0.25))
    texture_loss_weight = float(training.get("texture_loss_weight", 0.25))
    fusion_mode = str(model_config.get("fusion_mode", "quality"))
    if fusion_mode == "geometry":
        texture_loss_weight = 0.0
    elif fusion_mode == "texture":
        geometry_loss_weight = 0.0

    history: list[dict[str, object]] = []
    best_auc = -float("inf")
    best_epoch = 0
    best_threshold = float("nan")
    best_path = output_dir / "best.pt"
    stale_epochs = 0
    patience = int(training.get("early_stop_patience", 0))
    epochs = int(training["epochs"])
    logger.info("=" * 72)
    logger.info("QALF TRAINING RUN")
    logger.info(
        "  model | device=%s backbone=%s texture_mode=%s pooling=%s "
        "frames=%d image_size=%d parameters=%d trainable=%d",
        device,
        model_config.get("texture_backbone", "efficientnet_b0"),
        data.get("texture_mode", "canonical_skin"),
        model_config.get("texture_temporal_pooling", "mean"),
        int(data["texture_frames"]),
        int(data["image_size"]),
        sum(parameter.numel() for parameter in model.parameters()),
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    )
    logger.info(
        "  data  | train_videos=%d (real=%d fake=%d) "
        "val_videos=%d (real=%d fake=%d) val_samples=%d "
        "num_frames=%d clips_per_video=%d aggregation=%s",
        len(train_dataset.records),
        int(train_counts[0]),
        int(train_counts[1]),
        len(val_dataset.records),
        int(val_counts[0]),
        int(val_counts[1]),
        len(val_dataset),
        int(data["num_frames"]),
        int(data["eval_clips_per_video"]),
        data["video_aggregation"],
    )
    logger.info(
        "  train | amp=%s deterministic=%s ema_decay=%.4f validation_weights=%s "
        "loss_weights=fused:1.000 geometry:%.3f texture:%.3f patience=%d",
        amp_enabled,
        deterministic,
        ema_decay,
        "ema" if ema is not None else "raw",
        geometry_loss_weight,
        texture_loss_weight,
        patience,
    )
    logger.info(
        "  extra | mixstyle_probability=%.3f mixstyle_alpha=%.3f "
        "mixstyle_layers=%s texture_gate_bias=%.3f sbi_enabled=%s sbi_mixture=%s",
        mixstyle_probability,
        mixstyle_alpha,
        list(mixstyle_layers),
        float(model_config.get("texture_gate_bias", 0.0)),
        train_dataset.sbi_enabled,
        data["sbi"]["mixture"] if train_dataset.sbi_enabled else "disabled",
    )
    logger.info("  select| best_metric=val_auc threshold=Youden-J on FF++ validation")
    logger.info("  files | output_dir=%s", output_dir)
    logger.info("=" * 72)
    run_started = time.perf_counter()
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
            ema,
        )
        eval_model = ema.shadow if ema is not None else model
        validation_clips = predict(eval_model, val_loader, device)
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
        val_metrics["geometry_auc"] = compute_metrics(
            labels, geometry_scores, threshold
        )["auc"]
        val_metrics["texture_auc"] = compute_metrics(labels, texture_scores, threshold)["auc"]
        val_metrics["mean_geometry_weight"] = float(
            np.mean(np.asarray(validation["geometry_weight"], dtype=np.float64))
        )
        val_metrics["mean_texture_weight"] = float(
            np.mean(np.asarray(validation["texture_weight"], dtype=np.float64))
        )
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        save_json(history, output_dir / "history.json")
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
            "model": eval_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "geometry_input_dim": train_dataset.geometry_input_dim,
            "geometry_quality_dim": train_dataset.geometry_quality_dim,
            "texture_quality_dim": train_dataset.texture_quality_dim,
            "threshold": threshold,
            "validation_metrics": val_metrics,
            "ema_decay": ema_decay,
            "model_weights": "ema" if ema is not None else "raw",
            "label_convention": "real=0,fake=1",
            "score_target": "fake",
            "threshold_selection": "youden_j_ffpp_validation",
            "best_metric": "val_auc",
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            best_threshold = threshold
            stale_epochs = 0
            torch.save(checkpoint, best_path)
            logger.info(
                "best_model_saved epoch=%03d val_auc=%.4f threshold=%.4f weights=%s path=%s",
                best_epoch,
                best_auc,
                best_threshold,
                "ema" if ema is not None else "raw",
                best_path,
            )
        else:
            stale_epochs += 1
            if patience > 0:
                logger.info(
                    "early_stop_wait stale_epochs=%d/%d best_epoch=%03d best_val_auc=%.4f",
                    stale_epochs,
                    patience,
                    best_epoch,
                    best_auc,
                )
            if patience > 0 and stale_epochs >= patience:
                logger.info("Early stopping after %d epochs", epoch)
                break
    best_metrics = next(
        (
            row["validation"]
            for row in history
            if int(row["epoch"]) == best_epoch
        ),
        {},
    )
    summary = {
        "status": "complete",
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_metric": "val_auc",
        "best_value": best_auc,
        "best_threshold": best_threshold,
        "best_validation_metrics": best_metrics,
        "duration_seconds": time.perf_counter() - run_started,
        "best_model": str(best_path),
        "last_model": str(output_dir / "last.pt"),
        "history": str(output_dir / "history.json"),
        "training_plot": str(training_plot_path) if training_plot_path.is_file() else None,
    }
    save_json(summary, output_dir / "training_summary.json")
    logger.info("=" * 72)
    logger.info(
        "training_complete epochs=%d duration=%.1fs best_epoch=%03d best_val_auc=%.4f",
        len(history),
        float(summary["duration_seconds"]),
        best_epoch,
        best_auc,
    )
    logger.info(
        "  result| auc=%.4f ap=%.4f eer=%.4f balanced_acc=%.4f threshold=%.4f",
        float(best_metrics.get("auc", best_auc)),
        float(best_metrics.get("average_precision", float("nan"))),
        float(best_metrics.get("eer", float("nan"))),
        float(best_metrics.get("balanced_accuracy", float("nan"))),
        best_threshold,
    )
    logger.info(
        "  model | weights=%s best_model=%s",
        "ema" if ema is not None else "raw",
        best_path,
    )
    logger.info("  files | summary=%s", output_dir / "training_summary.json")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
