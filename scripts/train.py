#!/usr/bin/env python3
"""Train QALF on FF++ manifests."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qalf.config import load_config, save_json
from qalf.data.dataset import QALFVideoDataset
from qalf.data.geometry import DEFAULT_GEOMETRY_FEATURE_MODE, GEOMETRY_FEATURE_MODES
from qalf.engine import aggregate_predictions, predict, train_epoch
from qalf.metrics import compute_metrics, select_threshold
from qalf.models import MethodDiscriminator, QALFModel


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_grad_scaler(enabled: bool):
    """Use the current AMP API with a fallback for older development environments."""

    grad_scaler = getattr(torch.amp, "GradScaler", None)
    if grad_scaler is not None:
        return grad_scaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _loader(dataset, batch_size, workers, training, balanced):
    sampler = None
    shuffle = training
    if training and balanced:
        counts = np.bincount(dataset.labels, minlength=2)
        if np.any(counts == 0):
            raise ValueError(f"Training manifest must contain both classes: counts={counts}")
        weights = [1.0 / counts[label] for label in dataset.labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _dataset_summary(dataset: QALFVideoDataset) -> dict[str, object]:
    methods = sorted({record.method for record in dataset.records})
    return {
        "videos": len(dataset.records),
        "real": sum(record.label == 0 for record in dataset.records),
        "fake": sum(record.label == 1 for record in dataset.records),
        "methods": {
            method: sum(record.method == method for record in dataset.records)
            for method in methods
        },
    }


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
    auxiliary_module: nn.Module | None = None,
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
    if auxiliary_module is not None:
        head_parameters.extend(
            parameter for parameter in auxiliary_module.parameters() if parameter.requires_grad
        )
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


def _epoch_message(
    epoch: int,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> str:
    learning_rates = {
        str(group.get("name", index)): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
    rate_text = " ".join(f"{name}_lr={rate:.4e}" for name, rate in learning_rates.items())
    auxiliary_text = ""
    if "self_blend" in train_metrics:
        auxiliary_text += f" self_blend={train_metrics['self_blend']:.4f}"
    if "method_adversarial" in train_metrics:
        auxiliary_text += f" method_adv={train_metrics['method_adversarial']:.4f}"
    return (
        f"epoch={epoch:03d}/{epochs:03d} {rate_text} | "
        f"train loss={train_metrics['loss']:.4f} fused={train_metrics['fused']:.4f} "
        f"geometry={train_metrics['geometry']:.4f} texture={train_metrics['texture']:.4f}"
        f"{auxiliary_text} | "
        f"val auc={validation_metrics['auc']:.4f} ap={validation_metrics['average_precision']:.4f} "
        f"geometry_auc={validation_metrics['geometry_auc']:.4f} "
        f"texture_auc={validation_metrics['texture_auc']:.4f} "
        f"geometry_weight={validation_metrics['mean_geometry_weight']:.4f} "
        f"texture_weight={validation_metrics['mean_texture_weight']:.4f} "
        f"balanced_acc={validation_metrics['balanced_accuracy']:.4f} "
        f"accuracy={validation_metrics['accuracy']:.4f} f1={validation_metrics['f1']:.4f} "
        f"eer={validation_metrics['eer']:.4f} threshold={validation_metrics['threshold']:.4f}"
    )


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
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--backbone-learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--geometry-loss-weight", type=float)
    parser.add_argument("--texture-loss-weight", type=float)
    parser.add_argument("--self-blend-loss-weight", type=float)
    parser.add_argument("--method-adversarial-weight", type=float)
    parser.add_argument("--method-grl-strength", type=float)
    parser.add_argument("--early-stop-patience", type=int)
    parser.add_argument("--geometry-hidden", type=int)
    parser.add_argument("--geometry-layers", type=int)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument(
        "--fake-methods",
        nargs="+",
        help="FF++ fake methods to retain; real/original records are always retained.",
    )
    parser.add_argument(
        "--val-fake-methods",
        nargs="+",
        help="Optional FF++ validation fake methods for leave-one-manipulation-out runs.",
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
    parser.add_argument("--no-geometry-augmentation", action="store_true")
    parser.add_argument("--no-texture-augmentation", action="store_true")
    parser.add_argument("--no-self-blend", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-balanced-sampler", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data = config["data"]
    training = config["training"]
    model_config = config["model"]
    if args.geometry_mode:
        data["geometry_mode"] = args.geometry_mode
    if args.fusion_mode:
        model_config["fusion_mode"] = args.fusion_mode
    model_overrides = {
        "geometry_hidden": args.geometry_hidden,
        "geometry_layers": args.geometry_layers,
        "embedding_dim": args.embedding_dim,
        "dropout": args.dropout,
    }
    model_config.update(
        {key: value for key, value in model_overrides.items() if value is not None}
    )
    if args.no_geometry_augmentation:
        data["geometry_augmentation"] = {}
    if args.no_texture_augmentation:
        data["texture_augmentation"] = {}
    if args.no_self_blend:
        data["self_blend_augmentation"] = {}
    if args.fake_methods is not None:
        data["fake_methods"] = args.fake_methods
    if args.val_fake_methods is not None:
        data["val_fake_methods"] = args.val_fake_methods
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
        "self_blend_loss_weight": args.self_blend_loss_weight,
        "method_adversarial_weight": args.method_adversarial_weight,
        "method_grl_strength": args.method_grl_strength,
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
    for name in (
        "weight_decay",
        "geometry_loss_weight",
        "texture_loss_weight",
        "self_blend_loss_weight",
        "method_adversarial_weight",
        "method_grl_strength",
    ):
        if float(training.get(name, 0.0)) < 0:
            parser.error(f"training.{name} cannot be negative")
    if int(training.get("early_stop_patience", 0)) < 0:
        parser.error("training.early_stop_patience cannot be negative")
    for name in ("geometry_hidden", "geometry_layers", "embedding_dim"):
        if int(model_config.get(name, 0)) < 1:
            parser.error(f"model.{name} must be at least one")
    if not 0.0 <= float(model_config.get("dropout", 0.0)) < 1.0:
        parser.error("model.dropout must be in [0, 1)")
    seed = int(config.get("seed", 42))
    _seed_everything(seed)
    train_manifest = args.train_manifest or data["train_manifest"]
    val_manifest = args.val_manifest or data["val_manifest"]
    frame_root = args.frame_root or data["train_frame_root"]
    landmark_root = args.landmark_root or data["train_landmark_root"]
    output_dir = Path(args.output_dir or config.get("output_dir", "outputs/qalf"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = _create_logger(output_dir / "train_log.txt")
    data["train_manifest"] = str(train_manifest)
    data["val_manifest"] = str(val_manifest)
    data["train_frame_root"] = str(frame_root)
    data["train_landmark_root"] = str(landmark_root)
    config["output_dir"] = str(output_dir)
    data.setdefault("eval_clips_per_video", 2)
    data.setdefault("video_aggregation", "mean")
    data.setdefault("top_k", 1)
    save_json(config, output_dir / "config.json")

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
    }
    self_blend_augmentation = (
        data.get("self_blend_augmentation")
        if float(training.get("self_blend_loss_weight", 0.0)) > 0.0
        and str(model_config.get("fusion_mode", "quality")) != "geometry"
        else None
    )
    train_dataset = QALFVideoDataset(
        train_manifest,
        training=True,
        clips_per_video=1,
        fake_methods=data.get("fake_methods"),
        self_blend_augmentation=self_blend_augmentation,
        **dataset_args,
    )
    val_dataset = QALFVideoDataset(
        val_manifest,
        training=False,
        clips_per_video=int(data["eval_clips_per_video"]),
        fake_methods=data.get("val_fake_methods", data.get("fake_methods")),
        **dataset_args,
    )
    train_protocol = (
        sorted({record.dataset for record in train_dataset.records}),
        sorted({record.split for record in train_dataset.records}),
    )
    val_protocol = (
        sorted({record.dataset for record in val_dataset.records}),
        sorted({record.split for record in val_dataset.records}),
    )
    if train_protocol != (["ffpp"], ["train"]):
        raise ValueError(
            "Training is restricted to the official FF++ train split; "
            f"got datasets={train_protocol[0]}, splits={train_protocol[1]}"
        )
    if val_protocol != (["ffpp"], ["val"]):
        raise ValueError(
            "Threshold selection is restricted to the official FF++ validation split; "
            f"got datasets={val_protocol[0]}, splits={val_protocol[1]}"
        )
    logger.info(
        "Training protocol:\n%s",
        json.dumps(
            {
                "protocol": "FF++ filtered training protocol",
                "train_fake_methods": data.get("fake_methods", "all"),
                "validation_fake_methods": data.get(
                    "val_fake_methods", data.get("fake_methods", "all")
                ),
                "train": _dataset_summary(train_dataset),
                "validation": _dataset_summary(val_dataset),
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    batch_size = int(training["batch_size"])
    workers = int(training.get("num_workers", 4))
    train_loader = _loader(
        train_dataset,
        batch_size,
        workers,
        True,
        bool(training.get("balanced_sampler", True)),
    )
    val_loader = _loader(val_dataset, batch_size, workers, False, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QALFModel(
        geometry_input_dim=train_dataset.geometry_input_dim,
        geometry_hidden=int(model_config.get("geometry_hidden", 96)),
        geometry_layers=int(model_config.get("geometry_layers", 3)),
        embedding_dim=int(model_config.get("embedding_dim", 128)),
        dropout=float(model_config.get("dropout", 0.2)),
        texture_pretrained=bool(model_config.get("texture_pretrained", True)),
        geometry_quality_dim=train_dataset.geometry_quality_dim,
        texture_quality_dim=train_dataset.texture_quality_dim,
        fusion_mode=str(model_config.get("fusion_mode", "quality")),
    ).to(device)
    geometry_loss_weight = float(training.get("geometry_loss_weight", 0.25))
    texture_loss_weight = float(training.get("texture_loss_weight", 0.25))
    self_blend_loss_weight = float(training.get("self_blend_loss_weight", 0.0))
    method_adversarial_weight = float(training.get("method_adversarial_weight", 0.0))
    method_grl_strength = float(training.get("method_grl_strength", 1.0))
    fusion_mode = str(model_config.get("fusion_mode", "quality"))
    if fusion_mode == "geometry":
        texture_loss_weight = 0.0
        self_blend_loss_weight = 0.0
        method_adversarial_weight = 0.0
    elif fusion_mode == "texture":
        geometry_loss_weight = 0.0
    if self_blend_loss_weight > 0.0 and not train_dataset.self_blend_augmentation:
        raise ValueError(
            "training.self_blend_loss_weight is positive but self-blend augmentation is disabled"
        )
    fake_method_names = sorted(
        {record.method for record in train_dataset.records if record.label == 1}
    )
    method_to_index = {method: index for index, method in enumerate(fake_method_names)}
    method_discriminator = None
    if method_adversarial_weight > 0.0:
        if model.texture_encoder is None:
            raise ValueError("Method-adversarial training requires the texture branch")
        if len(method_to_index) < 2:
            raise ValueError("Method-adversarial training requires at least two fake methods")
        method_discriminator = MethodDiscriminator(
            embedding_dim=int(model_config.get("embedding_dim", 128)),
            num_methods=len(method_to_index),
            dropout=float(model_config.get("dropout", 0.2)),
        ).to(device)
    model_config["method_adversarial_methods"] = (
        fake_method_names if method_discriminator is not None else []
    )
    save_json(config, output_dir / "config.json")
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
            method_discriminator,
        ),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(training["epochs"]))
    )
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = _make_grad_scaler(amp_enabled)
    criterion = nn.BCEWithLogitsLoss()

    history: list[dict[str, object]] = []
    best_auc = -float("inf")
    stale_epochs = 0
    patience = int(training.get("early_stop_patience", 0))
    epochs = int(training["epochs"])
    logger.info(
        "device=%s backbone=%s inference_parameters=%d trainable=%d "
        "training_auxiliary_parameters=%d amp=%s self_blend_weight=%.4f "
        "method_adversarial_weight=%.4f method_grl_strength=%.4f",
        device,
        model_config.get("texture_backbone", "efficientnet_b0"),
        sum(parameter.numel() for parameter in model.parameters()),
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        sum(parameter.numel() for parameter in method_discriminator.parameters())
        if method_discriminator is not None
        else 0,
        amp_enabled,
        self_blend_loss_weight,
        method_adversarial_weight,
        method_grl_strength,
    )
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            geometry_loss_weight,
            texture_loss_weight,
            self_blend_loss_weight,
            method_discriminator,
            method_to_index,
            method_adversarial_weight,
            method_grl_strength,
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
        val_metrics["geometry_auc"] = compute_metrics(
            labels, geometry_scores, threshold
        )["auc"]
        val_metrics["texture_auc"] = compute_metrics(
            labels, texture_scores, threshold
        )["auc"]
        val_metrics["mean_geometry_weight"] = float(
            np.mean(validation["geometry_weight"])
        )
        val_metrics["mean_texture_weight"] = float(np.mean(validation["texture_weight"]))
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        save_json(history, output_dir / "history.json")
        logger.info(_epoch_message(epoch, epochs, optimizer, train_metrics, val_metrics))
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
            "training_auxiliary": {
                "method_to_index": method_to_index,
                "method_discriminator": (
                    method_discriminator.state_dict()
                    if method_discriminator is not None
                    else None
                ),
            },
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            stale_epochs = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            stale_epochs += 1
            if patience > 0 and stale_epochs >= patience:
                logger.info("Early stopping after %d epochs", epoch)
                break
    logger.info(
        "training_complete best_val_auc=%.4f output_dir=%s",
        best_auc,
        output_dir,
    )


if __name__ == "__main__":
    main()
