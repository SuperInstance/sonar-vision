#!/usr/bin/env python3
"""SonarVision benchmark suite — performance baselines and regression detection."""

import time
import json
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from sonar_vision.config import load_config
from sonar_vision.water.physics import UnderwaterPhysics
from sonar_vision.encoder import DepthEncoder
from sonar_vision.decoder import VideoDecoder


BENCHMARK_RESULTS = Path("benchmark-results.json")


@dataclass
class BenchmarkResult:
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    samples: int
    units: str = "ms"


def benchmark(fn: Callable, name: str, warmup: int = 3, runs: int = 30) -> BenchmarkResult:
    """Run a benchmark with warmup and statistical sampling."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    
    return BenchmarkResult(
        name=name,
        mean_ms=float(np.mean(times)),
        std_ms=float(np.std(times)),
        min_ms=float(np.min(times)),
        max_ms=float(np.max(times)),
        samples=runs,
    )


def bench_inference():
    """Benchmark end-to-end inference pipeline."""
    config = load_config("configs/default.yaml")
    physics = UnderwaterPhysics(config.water)
    depth_data = np.random.randn(100, 64).astype(np.float32)
    
    def run():
        physics.simulate(depth_data)
    
    return benchmark(run, "end-to-end-inference")


def bench_encoding():
    """Benchmark depth data encoding."""
    config = load_config("configs/default.yaml")
    encoder = DepthEncoder(config.encoder)
    depth_data = np.random.randn(100, 64).astype(np.float32)
    
    def run():
        encoder.encode(depth_data)
    
    return benchmark(run, "depth-encoding")


def bench_decoding():
    """Benchmark video frame decoding."""
    config = load_config("configs/default.yaml")
    decoder = VideoDecoder(config.decoder)
    latent = np.random.randn(16, 32, 32).astype(np.float32)
    
    def run():
        decoder.decode(latent)
    
    return benchmark(run, "video-decoding")


def bench_water_physics():
    """Benchmark underwater physics simulation."""
    config = load_config("configs/default.yaml")
    physics = UnderwaterPhysics(config.water)
    
    def run():
        physics.compute_absorption(450, 10.0)
        physics.compute_scattering(450, 10.0)
        physics.jerlov_water_type(10.0)
    
    return benchmark(run, "water-physics")


def run_all():
    """Run all benchmarks and save results."""
    print("=" * 60)
    print("  SonarVision Benchmark Suite")
    print("=" * 60)
    
    benchmarks_to_run = [
        bench_inference,
        bench_encoding,
        bench_decoding,
        bench_water_physics,
    ]
    
    results = []
    for b in benchmarks_to_run:
        print(f"\n  Running: {b.__name__}...", end=" ", flush=True)
        result = b()
        results.append(result)
        print(f"{result.mean_ms:.2f}ms ± {result.std_ms:.2f}ms")
    
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for r in results:
        print(f"  {r.name:30s} {r.mean_ms:8.2f}ms ± {r.std_ms:.2f}ms  (n={r.samples})")
    
    BENCHMARK_RESULTS.write_text(
        json.dumps([asdict(r) for r in results], indent=2)
    )
    print(f"\n  Results saved to {BENCHMARK_RESULTS}")
    
    return results


if __name__ == "__main__":
    run_all()
