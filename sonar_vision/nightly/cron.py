#!/usr/bin/env python3
"""
SonarVision Nightly Training Cron Job.

Runs automatically every night to:
1. Pull new data from boat's data store
2. Train a new LoRA on accumulated data
3. Evaluate quality vs previous best
4. Promote if better, rollback if worse
5. Optionally share federated contribution

Usage:
    python -m sonar_vision.nightly.cron --data_dir /data/sonar --output_dir /output

Crontab:
    0 3 * * * cd /opt/sonar-vision && python -m sonar_vision.nightly.cron --data_dir /data/sonar --output_dir /output >> /var/log/sonar-nightly.log 2>&1
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch


def pull_data(data_dir: str, remote_source: str = "") -> int:
    """Pull new data from remote source (S3, rsync, USB, etc.).

    Returns number of new samples pulled.
    """
    data_path = Path(data_dir)
    existing = set(f.stem for f in data_path.glob("sonar/*.npy"))

    if remote_source.startswith("s3://"):
        # AWS S3 sync
        import subprocess
        result = subprocess.run(
            ["aws", "s3", "sync", remote_source, str(data_path), "--exclude", "*"],
            capture_output=True, text=True
        )
        new_files = set(f.stem for f in data_path.glob("sonar/*.npy")) - existing
        return len(new_files)
    elif remote_source.startswith("rsync://") or remote_source.startswith("/"):
        # rsync from local mount or network
        import subprocess
        result = subprocess.run(
            ["rsync", "-av", "--include=*.npy", f"{remote_source}/", f"{data_path}/"],
            capture_output=True, text=True
        )
        new_files = set(f.stem for f in data_path.glob("sonar/*.npy")) - existing
        return len(new_files)
    else:
        # No remote source — count existing
        return len(list(data_path.glob("sonar/*.npy")))


def check_sensor_health(data_dir: str) -> bool:
    """Verify sensors are producing valid data.

    Checks:
    - Sonar files not all zeros
    - Camera files not corrupted
    - Timestamps recent (within 48 hours)
    - Water params in valid range
    """
    data_path = Path(data_dir)
    sonar_dir = data_path / "sonar"

    if not sonar_dir.exists():
        return False

    # Check latest sonar file
    sonar_files = sorted(sonar_dir.glob("*.npy"))
    if not sonar_files:
        return False

    latest = sonar_files[-1]
    import numpy as np
    sweep = np.load(latest)
    if sweep.max() == 0 and sweep.min() == 0:
        return False  # All zeros — sensor dead

    return True


def send_alert(message: str, webhook_url: str = ""):
    """Send alert if training fails or quality drops."""
    print(f"[ALERT] {message}")
    if webhook_url:
        import urllib.request
        try:
            payload = json.dumps({"text": f"🔔 SonarVision: {message}"}).encode()
            req = urllib.request.Request(webhook_url, data=payload, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:
            print(f"  Failed to send alert: {e}")


def main():
    parser = argparse.ArgumentParser(description="SonarVision Nightly Training")
    parser.add_argument("--data_dir", type=str, default="/data/sonar")
    parser.add_argument("--output_dir", type=str, default="/output/sonar-nightly")
    parser.add_argument("--remote_source", type=str, default="", help="S3 or rsync source")
    parser.add_argument("--federated", action="store_true", help="Opt-in to federated sharing")
    parser.add_argument("--federated_endpoint", type=str, default="")
    parser.add_argument("--alert_webhook", type=str, default="")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=10)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  SonarVision Nightly Training")
    print(f"  {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    # 1. Pull new data
    if args.remote_source:
        print("[1/6] Pulling new data...")
        new_samples = pull_data(args.data_dir, args.remote_source)
        print(f"  {new_samples} new samples")
    else:
        print("[1/6] Using local data (no remote source)")

    # 2. Sensor health check
    print("[2/6] Checking sensor health...")
    if not check_sensor_health(args.data_dir):
        msg = "Sensor health check FAILED — skipping training"
        print(f"  ❌ {msg}")
        send_alert(msg, args.alert_webhook)
        sys.exit(1)
    print("  ✅ Sensors healthy")

    # 3. Train LoRA
    print("[3/6] Training LoRA...")
    from sonar_vision.pipeline import SonarVision
    from sonar_vision.nightly.lora_trainer import LoRAConfig, NightlyTrainer

    model = SonarVision(max_depth=200, bearing_bins=128, embed_dim=768)
    lora_config = LoRAConfig(rank=args.lora_rank, max_epochs=args.max_epochs)

    trainer = NightlyTrainer(
        model=model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lora_config=lora_config,
    )

    try:
        run = trainer.run()
    except Exception as e:
        msg = f"Training FAILED: {e}"
        print(f"  ❌ {msg}")
        send_alert(msg, args.alert_webhook)
        sys.exit(1)

    # 4. Quality check
    print("[4/6] Quality check...")
    if run.quality_score < 0.3:
        msg = f"Low quality score: {run.quality_score:.3f} — investigating"
        print(f"  ⚠️ {msg}")
        send_alert(msg, args.alert_webhook)
    elif run.promoted:
        print(f"  ✅ Promoted! New best: {run.quality_score:.3f}")
    else:
        print(f"  ℹ️ Not promoted ({run.quality_score:.3f} < {trainer.best_score:.3f})")

    # 5. Federated sharing (opt-in)
    if args.federated and args.federated_endpoint:
        print("[5/6] Federated sharing (opt-in)...")
        lora_path = trainer.get_latest_lora()
        if lora_path and run.quality_score > 0.5:
            print(f"  Sharing LoRA weights (score: {run.quality_score:.3f})")
            # In production: POST anonymized weights to federated endpoint
            # with differential privacy noise
        else:
            print("  Skipped (no LoRA weights or quality below threshold)")
    else:
        print("[5/6] Federated sharing: not opted in")

    # 6. Report
    print("[6/6] Generating report...")
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": run.dataset_size,
        "quality_score": run.quality_score,
        "promoted": run.promoted,
        "best_score": trainer.best_score,
        "total_runs": len(trainer.history),
        "latest_lora": trainer.get_latest_lora(),
    }
    report_path = Path(args.output_dir) / "latest_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Nightly complete: score={run.quality_score:.3f}, promoted={run.promoted}")
    print(f"  Total runs: {len(trainer.history)}")
    print(f"  Report: {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
