#!/usr/bin/env python3
"""SonarVision CLI — underwater video prediction from depth sounder data."""

import argparse
import sys
import json
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sonar-vision")

try:
    import numpy as np
    from sonar_vision.config import load_config
except ImportError as e:
    log.error(f"Missing dependency: {e}")
    log.error("Run: pip install -e .")
    sys.exit(1)


def cmd_predict(args):
    """Run inference on depth data and generate video."""
    from sonar_vision.water.physics import UnderwaterPhysics
    from sonar_vision.encoder import DepthEncoder
    from sonar_vision.decoder import VideoDecoder

    log.info(f"Loading config: {args.config}")
    config = load_config(args.config)

    log.info(f"Loading depth data: {args.depth}")
    depth_data = np.load(args.depth)
    log.info(f"Depth shape: {depth_data.shape}, range: [{depth_data.min():.1f}, {depth_data.max():.1f}]")

    log.info("Running physics simulation...")
    physics = UnderwaterPhysics(config.water)
    signal = physics.simulate(depth_data)

    log.info("Encoding depth features...")
    encoder = DepthEncoder(config.encoder)
    latent = encoder.encode(signal)

    log.info("Decoding video frames...")
    decoder = VideoDecoder(config.decoder)
    frames = decoder.decode(latent)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, frames=np.array(frames))
    log.info(f"Saved {len(frames)} frames -> {output}")
    return frames


def cmd_train(args):
    """Train a SonarVision model on depth data."""
    from sonar_vision.train import Trainer
    from sonar_vision.config import load_config

    log.info(f"Loading config: {args.config}")
    config = load_config(args.config)
    config.trainer.epochs = args.epochs
    config.trainer.batch_size = args.batch_size

    log.info(f"Training: {args.epochs} epochs, batch {args.batch_size}, lr {args.learning_rate}")
    trainer = Trainer(config)
    trainer.train()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(output))
    log.info(f"Model saved -> {output}")


def cmd_visualize(args):
    """Generate comparison plots and visualizations."""
    import matplotlib.pyplot as plt

    log.info(f"Loading results: {args.input}")
    data = np.load(args.input)
    frames = data["frames"]

    n = min(args.frames, len(frames), 16)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i in range(n):
        axes[i].imshow(frames[i], cmap=args.cmap)
        axes[i].set_title(f"Frame {i+1}")
        axes[i].axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output), dpi=args.dpi, bbox_inches="tight")
    log.info(f"Visualization saved -> {output}")
    if args.show:
        plt.show()
    plt.close()


def cmd_serve(args):
    """Start the SonarVision REST API server."""
    from sonar_vision.deploy import create_app

    log.info(f"Starting SonarVision API on {args.host}:{args.port}")
    app = create_app(args.config)

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


def cmd_benchmark(args):
    """Run performance benchmarks."""
    from benchmarks.run_benchmarks import run_all

    log.info("Running SonarVision benchmark suite...")
    results = run_all()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([r._asdict() for r in results], indent=2))
    log.info(f"Benchmark results saved -> {output}")


def cmd_config(args):
    """Show or validate configuration."""
    from sonar_vision.config import load_config

    config = load_config(args.config)
    if args.show:
        import yaml
        print(yaml.dump(json.loads(config.model_dump_json()), default_flow_style=False))
    else:
        log.info("Configuration valid")
        log.info(f"  Water type: {config.water.water_type}")
        log.info(f"  Encoder dim: {config.encoder.latent_dim}")
        log.info(f"  Decoder fps: {config.decoder.fps}")
        log.info(f"  Trainer epochs: {config.trainer.epochs}")


def main():
    parser = argparse.ArgumentParser(
        description="SonarVision -- Underwater video from depth sounder data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  predict --depth data/sample.npy --output results/prediction.npz
  train --epochs 100 --batch 32 --learning-rate 1e-4
  visualize --input results/prediction.npz --cmap plasma
  serve --port 8000 --reload
  benchmark
  config --show
        """,
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    sub = parser.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="Run inference on depth data")
    p_predict.add_argument("--depth", required=True, help="Path to .npy depth data")
    p_predict.add_argument("--output", default="output/prediction.npz", help="Output path")

    p_train = sub.add_parser("train", help="Train a model")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--learning-rate", type=float, default=1e-4)
    p_train.add_argument("--output", default="output/model.pt", help="Checkpoint path")

    p_viz = sub.add_parser("visualize", help="Generate visualizations")
    p_viz.add_argument("--input", required=True, help="Predictions .npz file")
    p_viz.add_argument("--output", default="output/viz.png", help="Output image path")
    p_viz.add_argument("--frames", type=int, default=8, help="Number of frames to show")
    p_viz.add_argument("--cmap", default="plasma", help="Matplotlib colormap")
    p_viz.add_argument("--dpi", type=int, default=150)
    p_viz.add_argument("--show", action="store_true", help="Display interactively")

    p_serve = sub.add_parser("serve", help="Start API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    p_bench = sub.add_parser("benchmark", help="Run benchmarks")
    p_bench.add_argument("--output", default="benchmark-results.json")

    p_config = sub.add_parser("config", help="Inspect configuration")
    p_config.add_argument("--show", action="store_true", help="Dump full config")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    commands = {
        "predict": cmd_predict,
        "train": cmd_train,
        "visualize": cmd_visualize,
        "serve": cmd_serve,
        "benchmark": cmd_benchmark,
        "config": cmd_config,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        log.info("Interrupted. Exiting.")
    except Exception as e:
        log.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
