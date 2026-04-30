"""
SonarVision Training Script

Complete training loop with:
- AdamW + cosine LR schedule
- Gradient accumulation + mixed precision + gradient clipping
- Checkpointing, TensorBoard logging, EMA
- Evaluation: PSNR/SSIM vs camera GT, depth MAE
"""

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from sonar_vision.pipeline import SonarVision
from sonar_vision.data.sonar_dataset import SonarVideoDataset, create_training_split

# Optional TensorBoard — fail gracefully if missing
_tensorboard_available = False
try:
    from torch.utils.tensorboard import SummaryWriter
    _tensorboard_available = True
except ImportError:
    SummaryWriter = None  # type: ignore


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "cuda") -> torch.device:
    """Pick the best available device."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Custom collate (variable-length cameras / detections)
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate a list of SonarVideoDataset samples."""
    B = len(batch)

    # Fixed-size tensors
    sonar_intensity = torch.stack([b["sonar_intensity"] for b in batch])
    turbidity = torch.stack([b["turbidity"] for b in batch])

    # Variable-length cameras
    max_cams = max(b["camera_frames"].shape[0] for b in batch)
    C, H, W = batch[0]["camera_frames"].shape[1:]
    camera_frames = torch.zeros(B, max_cams, C, H, W)
    camera_depths = torch.full((B, max_cams), 1e6, dtype=torch.float32)
    depth_weights = torch.zeros(B, max_cams, dtype=torch.float32)

    for i, b in enumerate(batch):
        n = b["camera_frames"].shape[0]
        camera_frames[i, :n] = b["camera_frames"]
        camera_depths[i, :n] = b["camera_depths"]
        depth_weights[i, :n] = b["depth_weights"]

    # Sonar detections — keep only depth column for loss
    max_dets = max(b["sonar_detections"].shape[0] for b in batch)
    sonar_detections = torch.zeros(B, max_dets, dtype=torch.float32)
    for i, b in enumerate(batch):
        n = b["sonar_detections"].shape[0]
        if n > 0:
            sonar_detections[i, :n] = b["sonar_detections"][:, 0]

    return {
        "sonar_intensity": sonar_intensity,
        "camera_frames": camera_frames,
        "camera_depths": camera_depths,
        "sonar_detections": sonar_detections,
        "depth_weights": depth_weights,
        "turbidity": turbidity,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """PSNR in dB. Both tensors in [0, 1]."""
    mse = F.mse_loss(pred, target, reduction="mean")
    if mse.item() < 1e-10:
        return 100.0
    return -10.0 * math.log10(mse.item())


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """Simple box-filter SSIM. Both tensors in [0, 1], shape (N, 3, H, W)."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu1 = F.avg_pool2d(pred, window_size, 1, padding=window_size // 2)
    mu2 = F.avg_pool2d(target, window_size, 1, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(pred ** 2, window_size, 1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target ** 2, window_size, 1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, 1, padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


def compute_depth_mae(
    pred_depth_map: torch.Tensor,
    sonar_detections: torch.Tensor,
    max_depth: int,
) -> float:
    """Mean predicted depth vs mean sonar detection depth (meters)."""
    pred_depth_m = pred_depth_map.mean(dim=[1, 2, 3]) * max_depth  # (B,)

    mask = sonar_detections > 0
    if mask.any():
        gt_depth_m = sonar_detections.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        gt_depth_m = torch.zeros_like(pred_depth_m)

    return F.l1_loss(pred_depth_m, gt_depth_m, reduction="mean").item()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup.clear()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    ema: EMA,
    step: int,
    epoch: int,
    path: Path,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.shadow,
            "step": step,
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    ema: EMA,
    device: torch.device,
) -> Tuple[int, int]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    ema.shadow = ckpt["ema"]
    return ckpt["step"], ckpt["epoch"]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def build_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    train_ds, val_ds = create_training_split(
        root_dir=args.data_dir,
        train_ratio=0.8,
        bearing_bins=args.bearing_bins,
        max_depth=args.max_depth,
        depth_sigma=3.0,
        min_cameras=1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_step(
    model: SonarVision,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    ema: EMA,
    device: torch.device,
    grad_accum_steps: int,
    is_boundary: bool,
) -> Dict[str, float]:
    model.train()

    sonar = batch["sonar_intensity"].to(device)
    cam_frames = batch["camera_frames"].to(device)
    cam_depths = batch["camera_depths"].to(device)
    sonar_dets = batch["sonar_detections"].to(device)

    use_amp = device.type == "cuda"

    with autocast(enabled=use_amp):
        out = model(
            sonar_intensity=sonar,
            camera_frames=cam_frames,
            camera_depths=cam_depths,
            sonar_detections=sonar_dets,
        )
        loss = out["loss"] / grad_accum_steps

    scaler.scale(loss).backward()

    metrics: Dict[str, float] = {}
    if out.get("loss_dict"):
        for k, v in out["loss_dict"].items():
            metrics[k] = v / grad_accum_steps

    # Optimizer step on accumulation boundary
    if is_boundary:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)
        scheduler.step()

    return metrics


def optimizer_step_boundary(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    ema: EMA,
):
    """Force an optimizer step (used at epoch end for leftover gradients)."""
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    ema.update(model)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: SonarVision,
    dataloader: DataLoader,
    device: torch.device,
    writer: Optional[object],
    step: int,
    max_depth: int,
    log_images: bool = True,
) -> Dict[str, float]:
    model.eval()
    ema.apply_shadow(model)

    psnr_list = []
    ssim_list = []
    depth_mae_list = []

    for batch in dataloader:
        sonar = batch["sonar_intensity"].to(device)
        cam_frames = batch["camera_frames"].to(device)
        cam_depths = batch["camera_depths"].to(device)
        sonar_dets = batch["sonar_detections"].to(device)

        use_amp = device.type == "cuda"
        with autocast(enabled=use_amp):
            out = model(
                sonar_intensity=sonar,
                camera_frames=cam_frames,
                camera_depths=cam_depths,
                sonar_detections=sonar_dets,
            )

        pred = out["frame"]            # (B, 3, H, W) in [-1, 1]
        depth_map = out["depth_map"]   # (B, 1, H, W) in [0, 1]
        B = pred.shape[0]

        # Pick ground-truth camera closest to mean sonar detection depth
        mean_det = sonar_dets.mean(dim=1)  # (B,)
        diffs = (cam_depths - mean_det.unsqueeze(1)).abs()  # (B, max_cams)
        best_idx = diffs.argmin(dim=1)  # (B,)
        gt = torch.stack([cam_frames[i, best_idx[i]] for i in range(B)])  # (B, 3, H, W)

        # Normalize to [0, 1]
        pred_01 = (pred + 1.0) / 2.0
        gt_01 = gt.clamp(0.0, 1.0)

        psnr_list.append(compute_psnr(pred_01, gt_01))
        ssim_list.append(compute_ssim(pred_01, gt_01))
        depth_mae_list.append(compute_depth_mae(depth_map, sonar_dets, max_depth))

        if log_images and writer is not None:
            n = min(B, 4)
            writer.add_images("val/pred", pred_01[:n], step)
            writer.add_images("val/gt", gt_01[:n], step)
            writer.add_images("val/depth", depth_map[:n], step)
            log_images = False  # log only first batch

    ema.restore(model)

    return {
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
        "depth_mae": float(np.mean(depth_mae_list)),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    set_seed(42)
    device = get_device()
    print(f"[train] Using device: {device}")

    if not _tensorboard_available:
        print("[train] Warning: tensorboard not installed. Logging disabled.")

    # Dataloaders
    train_loader, val_loader = build_dataloaders(args)
    print(f"[train] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = SonarVision(
        max_depth=args.max_depth,
        bearing_bins=args.bearing_bins,
        embed_dim=args.embed_dim,
        pretrained_encoder=args.pretrained_encoder,
    ).to(device)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps, 1), eta_min=args.lr * 1e-2
    )

    scaler = GradScaler(enabled=(device.type == "cuda"))
    ema = EMA(model, decay=0.999)

    start_step = 0
    start_epoch = 0

    if args.resume_checkpoint:
        print(f"[train] Resuming from {args.resume_checkpoint}")
        start_step, start_epoch = load_checkpoint(
            args.resume_checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
            ema,
            device,
        )
        print(f"[train] Resumed at epoch {start_epoch}, step {start_step}")

    # TensorBoard
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir / "logs") if _tensorboard_available else None

    global_step = start_step
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_metrics: Dict[str, float] = {}

        for batch_idx, batch in enumerate(train_loader):
            is_boundary = (batch_idx + 1) % args.gradient_accumulation_steps == 0
            metrics = train_one_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                device=device,
                grad_accum_steps=args.gradient_accumulation_steps,
                is_boundary=is_boundary,
            )

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v

            # On optimizer step boundary
            if is_boundary:
                global_step += 1

                # Log training metrics
                if writer is not None:
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    for k, v in metrics.items():
                        writer.add_scalar(f"train/{k}", v * args.gradient_accumulation_steps, global_step)

                # Log sample training images every 500 steps
                if global_step % 500 == 0 and writer is not None:
                    with torch.no_grad():
                        sonar = batch["sonar_intensity"][:4].to(device)
                        cam_frames = batch["camera_frames"][:4].to(device)
                        cam_depths = batch["camera_depths"][:4].to(device)
                        sonar_dets = batch["sonar_detections"][:4].to(device)
                        model.eval()
                        out = model(
                            sonar_intensity=sonar,
                            camera_frames=cam_frames,
                            camera_depths=cam_depths,
                            sonar_detections=sonar_dets,
                        )
                        pred_01 = (out["frame"] + 1.0) / 2.0
                        writer.add_images("train/pred", pred_01, global_step)
                        writer.add_images("train/depth", out["depth_map"], global_step)
                        writer.add_images("train/gt", cam_frames[:, 0].clamp(0, 1), global_step)
                        model.train()

                # Checkpoint every 1000 steps
                if global_step % 1000 == 0:
                    ckpt_path = output_dir / f"checkpoint_step_{global_step:07d}.pt"
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        ema,
                        global_step,
                        epoch,
                        ckpt_path,
                    )
                    print(f"[train] Saved checkpoint: {ckpt_path}")

                # Evaluation every 500 steps
                if global_step % 500 == 0:
                    val_metrics = evaluate(
                        model,
                        val_loader,
                        device,
                        writer,
                        global_step,
                        args.max_depth,
                        log_images=True,
                    )
                    print(
                        f"[val] step {global_step} | "
                        f"PSNR {val_metrics['psnr']:.3f} | "
                        f"SSIM {val_metrics['ssim']:.4f} | "
                        f"DepthMAE {val_metrics['depth_mae']:.3f}"
                    )
                    if writer is not None:
                        for k, v in val_metrics.items():
                            writer.add_scalar(f"val/{k}", v, global_step)

        # End-of-epoch: step any leftover gradients
        if (len(train_loader) % args.gradient_accumulation_steps) != 0:
            optimizer_step_boundary(model, optimizer, scaler, ema)
            global_step += 1

        # End-of-epoch summary
        print(
            f"[train] Epoch {epoch + 1}/{args.epochs} | "
            f"step {global_step} | "
            + " | ".join(f"{k} {v:.4f}" for k, v in epoch_metrics.items())
        )

    # Final checkpoint
    final_path = output_dir / "checkpoint_final.pt"
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        ema,
        global_step,
        args.epochs,
        final_path,
    )
    print(f"[train] Final checkpoint: {final_path}")
    if writer is not None:
        writer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SonarVision")

    # Data
    parser.add_argument("--data_dir", type=str, required=True, help="Root dataset directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")

    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")

    # Model
    parser.add_argument("--embed_dim", type=int, default=1024, help="Token embedding dimension")
    parser.add_argument("--max_depth", type=int, default=200, help="Max sonar depth (meters)")
    parser.add_argument("--bearing_bins", type=int, default=128, help="Sonar bearing resolution")
    parser.add_argument("--pretrained_encoder", type=str, default="",
                        help="Path to pretrained encoder checkpoint")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Path to checkpoint to resume from")

    # Data loading
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
