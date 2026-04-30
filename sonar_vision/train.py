"""
SonarVision Training Script.

Usage:
    python -m sonar_vision.train --data_dir /data/sonar --output_dir /output --epochs 100
    python -m sonar_vision.train --data_dir /data/sonar --resume checkpoint.pt
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def build_model(args) -> nn.Module:
    """Build SonarVision model from args."""
    from sonar_vision.pipeline import SonarVision
    return SonarVision(
        max_depth=args.max_depth,
        bearing_bins=args.bearing_bins,
        embed_dim=args.embed_dim,
        output_resolution=(384, 512),
        depth_sigma=args.depth_sigma,
        pretrained_encoder=args.pretrained_encoder or "",
    )


def build_dataloader(args, train: bool = True) -> DataLoader:
    """Build training or validation dataloader."""
    from sonar_vision.data.sonar_dataset import SonarVideoDataset

    ds = SonarVideoDataset(
        root_dir=args.data_dir,
        bearing_bins=args.bearing_bins,
        max_depth=args.max_depth,
        depth_sigma=args.depth_sigma,
        augment=train,
        train=train,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=train,
    )


class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    ema: EMA,
    writer: SummaryWriter,
    args,
    global_step: int,
) -> int:
    """Train for one epoch. Returns updated global_step."""
    model.train()
    accum_steps = args.gradient_accumulation_steps
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        sonar = batch["sonar_intensity"].cuda()
        cam_frames = batch["camera_frames"].cuda()
        cam_depths = batch["camera_depths"].cuda()
        det = batch["sonar_detections"].cuda()
        weights = batch["depth_weights"].cuda()

        with autocast():
            output = model(
                sonar_intensity=sonar,
                camera_frames=cam_frames,
                camera_depths=cam_depths,
                sonar_detections=det,
            )

            loss = output["loss"] / accum_steps
            loss_dict = output["loss_dict"]

        scaler.scale(loss).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            ema.update(model)
            global_step += 1

            # Logging
            if global_step % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/loss", loss_dict["total"], global_step)
                writer.add_scalar("train/l1", loss_dict["weighted_l1"], global_step)
                writer.add_scalar("train/depth", loss_dict["depth_consistency"], global_step)
                writer.add_scalar("train/cam_weight", loss_dict["mean_cam_weight"], global_step)
                writer.add_scalar("train/lr", lr, global_step)
                print(
                    f"  step {global_step:6d} | loss {loss_dict['total']:.4f} | "
                    f"l1 {loss_dict['weighted_l1']:.4f} | depth {loss_dict['depth_consistency']:.4f} | "
                    f"cam_w {loss_dict['mean_cam_weight']:.3f} | lr {lr:.2e}"
                )

            # Checkpoint
            if global_step % args.checkpoint_every == 0:
                save_checkpoint(model, ema, optimizer, global_step, args.output_dir)

            # Sample images
            if global_step % 500 == 0:
                log_sample_images(model, batch, writer, global_step)

    return global_step


@torch.no_grad()
def log_sample_images(model, batch, writer, step):
    """Log generated vs ground truth images."""
    model.eval()
    sonar = batch["sonar_intensity"][:1].cuda()
    cam_frames = batch["camera_frames"][:1].cuda()
    cam_depths = batch["camera_depths"][:1]
    weights = batch["depth_weights"][:1]

    with autocast():
        output = model(sonar_intensity=sonar)

    pred = output["frame"][0].cpu()  # (3, H, W)
    pred = (pred + 1) / 2  # [-1, 1] → [0, 1]
    pred = pred.clamp(0, 1)

    # Find best camera (highest weight)
    best_idx = weights.argmax().item()
    best_cam = cam_frames[0, best_idx].cpu()  # (3, H, W)

    writer.add_image("samples/predicted", pred, step)
    writer.add_image("samples/best_camera", best_cam, step)

    # Depth map
    depth_map = output["depth_map"][0, 0].cpu()
    writer.add_image("samples/depth_map", depth_map, step)

    # Sonar input
    sonar_img = sonar[0, 0].cpu()
    writer.add_image("samples/sonar", sonar_img, step)

    model.train()


@torch.no_grad()
def evaluate(model, loader, writer, step):
    """Run evaluation."""
    model.eval()
    total_loss = 0
    count = 0

    for batch in loader:
        sonar = batch["sonar_intensity"].cuda()
        cam_frames = batch["camera_frames"].cuda()
        cam_depths = batch["camera_depths"].cuda()
        det = batch["sonar_detections"].cuda()

        with autocast():
            output = model(
                sonar_intensity=sonar,
                camera_frames=cam_frames,
                camera_depths=cam_depths,
                sonar_detections=det,
            )

        if "loss" in output:
            total_loss += output["loss"].item()
            count += 1

    avg_loss = total_loss / max(count, 1)
    writer.add_scalar("val/loss", avg_loss, step)
    print(f"  val_loss {avg_loss:.4f}")
    model.train()
    return avg_loss


def save_checkpoint(model, ema, optimizer, step, output_dir):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"checkpoint_{step}.pt")
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema": ema.shadow,
        "optimizer": optimizer.state_dict(),
    }, path)
    print(f"  saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Train SonarVision")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--embed_dim", type=int, default=1024)
    parser.add_argument("--max_depth", type=int, default=200)
    parser.add_argument("--bearing_bins", type=int, default=128)
    parser.add_argument("--depth_sigma", type=float, default=3.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--pretrained_encoder", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model = build_model(args).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * 1000  # approximate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Mixed precision
    scaler = GradScaler(enabled=(device.type == "cuda"))
    ema = EMA(model, decay=0.999)

    # Resume
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("step", 0)
        if "ema" in ckpt:
            ema.shadow = ckpt["ema"]
        print(f"Resumed from step {global_step}")

    # Data
    print("Loading training data...")
    train_loader = build_dataloader(args, train=True)
    print(f"  {len(train_loader)} batches")

    # TensorBoard
    writer = SummaryWriter(args.output_dir)

    # Training loop
    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        t0 = time.time()
        global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, ema, writer, args, global_step
        )
        dt = time.time() - t0
        print(f"Epoch {epoch + 1}/{args.epochs} done in {dt:.1f}s ({global_step} steps)")

    # Save final
    ema.apply(model)
    save_checkpoint(model, ema, optimizer, global_step, args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
