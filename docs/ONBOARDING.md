# Onboarding — Contributing to SonarVision

Welcome. Here's how to get set up and start contributing.

## Project Philosophy

SonarVision solves a real problem: **underwater imagery is scarce and expensive to annotate.** By leveraging the physics of the water column and the sensors already on fishing boats, we can train vision models without any human labels.

This project was built by the Cocapn fleet — a group of AI agents coordinated through shared infrastructure. Contributions from humans and agents are equally welcome.

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/sonar-vision.git
cd sonar-vision

# Add upstream
git remote add upstream https://github.com/SuperInstance/sonar-vision.git
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install --upgrade pip
pip install torch numpy pillow pyyaml einops pytest
pip install -e .  # Editable install
```

### 3. Verify Installation

```bash
python -c "import sonar_vision; print('✅ SonarVision ready')"
pytest tests/ -v  # Run test suite
```

### 4. Run a Quick Test

```python
import torch
from sonar_vision.pipeline import SonarVision

model = SonarVision(max_depth=50, bearing_bins=32, embed_dim=256)
model.eval()

sonar = torch.randn(1, 32, 50)
output = model.generate(sonar)
print(f"Frame: {output['frame'].shape}")  # Should be (1, 3, 384, 512)
```

## Code Structure

```
sonar_vision/
├── pipeline.py              # Main SonarVision model (encoder → GCT → decoder)
├── config.py                # YAML-based experiment configuration
├── deploy.py                # TorchScript export, quantization, Jetson inference
├── train.py                 # Training loop with AMP, EMA, gradient accumulation
├── encoder/
│   └── sonar_encoder.py     # SonarSweepEmbedding + SonarEncoder (ViT-adapted)
├── aggregator/
│   └── gct.py               # Streaming GCT with KV cache, 3D RoPE, GQA
├── decoder/
│   └── video_decoder.py     # DPT head + UnderwaterColorHead + temporal refinement
├── supervision/
│   └── depth_weighted_loss.py  # Self-supervision losses
├── water/
│   └── physics.py           # Underwater physics models
├── data/
│   ├── sonar_dataset.py     # PyTorch dataset
│   ├── augmentation.py      # Sonar/camera augmentation
│   └── preprocessing.py     # NMEA parsing, detection extraction
├── nightly/
│   ├── lora_trainer.py      # LoRA training + quality scoring
│   └── cron.py              # Production cron job
├── federated/
│   └── __init__.py          # Data anonymization, DP, FedAvg, global model
└── utils/
    └── visualization.py     # Sonar heatmap, detection overlay
```

## Architecture at a Glance

```
Sonar → NMEAInterpreter → SonarEncoder (4-ch ViT) → GCT (streaming) → VideoDecoder → RGB Frame
                                                            ↓
                                              DepthWeightedLoss ← cameras at known depths
```

**Key design decisions:**
- **Feed-forward** (not recurrent) — enables streaming, no hidden state explosion
- **KV cache** with sliding window — bounded memory for edge deployment
- **3D RoPE** — positional encoding on (depth, bearing, time) axes
- **Grouped-Query Attention** — 4 KV heads per 16 query heads, 75% KV reduction
- **Physics-based self-supervision** — no manual labels, depth weighting from sensor geometry

## Adding a New Module

### Example: Add a Turbidity Estimator

1. **Create the module** in the appropriate subpackage:

```python
# sonar_vision/water/turbidity.py
import torch
import torch.nn as nn

class TurbidityEstimator(nn.Module):
    """Estimate water turbidity from sonar returns."""
    def __init__(self, embed_dim=1024):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Estimate turbidity from encoder tokens."""
        pooled = tokens.mean(dim=1)  # (B, embed_dim)
        return self.head(pooled)     # (B, 1)
```

2. **Export from `__init__.py`:**

```python
# sonar_vision/water/__init__.py
from sonar_vision.water.turbidity import TurbidityEstimator
```

3. **Write tests:**

```python
# tests/test_turbidity.py
import torch
from sonar_vision.water.turbidity import TurbidityEstimator

def test_output_shape():
    est = TurbidityEstimator(embed_dim=128)
    tokens = torch.randn(2, 16, 128)
    turb = est(tokens)
    assert turb.shape == (2, 1)
```

4. **Run tests:**

```bash
pytest tests/test_turbidity.py -v
```

5. **Commit and push:**

```bash
git add -A
git commit -m "feat: turbidity estimator from sonar tokens"
git push origin main
```

## Testing Guidelines

- **Every module needs tests.** No exceptions.
- **Test with synthetic data** — no external files, no network calls.
- **Test shapes and types**, not exact values (model outputs vary).
- **Test edge cases:** empty inputs, single samples, mismatched sizes.

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_pipeline.py::TestSonarVisionPipeline::test_model_creation -v

# Run with coverage
pytest tests/ --cov=sonar_vision --cov-report=term-missing
```

## PR Process

1. **Branch from main:** `git checkout -b feat/your-feature`
2. **Write code + tests:** See "Adding a New Module" above
3. **Update docs:** If you change behavior, update the relevant doc
4. **Commit convention:**
   - `feat:` New feature or module
   - `fix:` Bug fix
   - `docs:` Documentation only
   - `test:` Tests only
   - `refactor:` Code cleanup
5. **Push and create PR** targeting `main`
6. **One logical change per commit.** Push after every feature.

## Config System

All experiments use YAML configs:

```yaml
name: my-experiment
encoder:
  embed_dim: 768
gct:
  num_layers: 4
train:
  epochs: 50
  lr: 0.0001
```

```python
from sonar_vision.config import SonarVisionConfig
cfg = SonarVisionConfig.from_yaml("configs/my-experiment.yaml")
```

## Communication

- **GitHub Issues:** Bug reports, feature requests, questions
- **Discussions:** Architecture decisions, research ideas
- **PRs:** Code changes

## Fleet Integration

SonarVision is part of the Cocapn fleet ecosystem:
- **PLATO tiles** in `.plato/tiles/` — knowledge fragments for the fleet
- **I2I bottles** in `.plato/` — async fleet communication
- **Flux programs** in `.plato/flux-bytecode/` — NMEA preprocessing
- **Git-native protocol** in `.plato/GIT_NATIVE.md`

See `.plato/README.md` for full integration details.

## Good First Issues

Looking for somewhere to start? These are high-impact contributions:

1. **Add more NMEA sentence parsers** — `$SDDBT`, `$SDVHW`, `$SDMTW`
2. **Improve augmentation** — simulated turbidity, fish school patterns
3. **Real data benchmarks** — test on actual sonar recordings
4. **Jetson TensorRT export** — INT8 quantization, latency benchmarks
5. **Documentation** — add examples, fix typos, translate to other languages

## License

MIT — use it however you want. Attribution appreciated but not required.
