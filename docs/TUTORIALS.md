# Tutorials — SonarVision

Step-by-step guides from first prediction to federated deployment.

---

## Tutorial 1: Your First Sonar Prediction

Generate an underwater video frame from synthetic sonar data.

```python
import torch
import numpy as np
from sonar_vision.pipeline import SonarVision

# Create model
model = SonarVision(max_depth=50, bearing_bins=32, embed_dim=256)
model.eval()

# Simulate a sonar detection: strong return at bearing 45°, depth 15m
sonar = np.zeros((1, 32, 50), dtype=np.float32)
# Add a "fish" echo at bearing bin 16 (45°), depth bin 15 (15m)
for b in range(14, 19):
    for d in range(13, 18):
        sonar[0, b, d] = 0.8 + np.random.uniform(-0.1, 0.1)

# Add ambient noise
sonar += np.random.uniform(0, 0.05, sonar.shape).astype(np.float32)

# Predict
sonar_tensor = torch.from_numpy(sonar)
with torch.no_grad():
    output = model.generate(sonar_tensor)

frame = output["frame"][0].cpu().numpy()    # (3, H, W)
depth = output["depth_map"][0, 0].cpu().numpy()  # (H, W)

# Save the result
from PIL import Image
frame_img = ((frame.transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
Image.fromarray(frame_img).save("predicted_frame.jpg")
print("Saved predicted_frame.jpg")

# Visualize depth
depth_img = (depth * 255).astype(np.uint8)
Image.fromarray(depth_img).save("predicted_depth.jpg")
print("Saved predicted_depth.jpg")
```

**What you'll see:** An underwater-style image with blue-green color cast. The depth map should show higher values where the sonar detection was placed.

---

## Tutorial 2: Setting Up Your Boat's Data Pipeline

Connect your fish finder to SonarVision via NMEA 0183.

### What You Need

- Fish finder with NMEA 0183 output (Garmin, Humminbird, Simrad, Lowrance)
- USB-to-serial adapter (FTDI USB-RS485, ~$15)
- Jetson Orin NX or laptop running Linux
- 4 GoPro cameras on a downrigger cable

### Step 1: Wire the Sonar

```
Fish Finder NMEA OUT ──→ RS485 ──→ USB-RS485 adapter ──→ Jetson USB port
```

Most fish finders output NMEA 0183 at 4800 baud on a 2-wire RS485 connection.

### Step 2: Read NMEA Sentences

```python
import serial
from sonar_vision.water.physics import NMEAInterpreter
from sonar_vision.data.preprocessing import sonar_to_image

# Open serial port
ser = serial.Serial('/dev/ttyUSB0', baudrate=4800, timeout=1)

# Read and parse
while True:
    line = ser.readline().decode('ascii', errors='ignore').strip()
    if line.startswith('$SDDBT'):
        # $SDDBT,12.5,f,3.81,M,2.07,F*2A
        result = NMEAInterpreter.parse_sonar_return(line)
        print(f"Depth: {result['depth']}m")

    if line.startswith('$PSDVS'):
        # Proprietary: $PSDVS,15.2,45.0,-30.5,3.0*4A
        result = NMEAInterpreter.parse_sonar_return(line)
        print(f"Target: {result['depth']}m @ {result['bearing']}° "
              f"(intensity: {result['intensity']} dB)")
```

### Step 3: Store Training Data

```python
from sonar_vision.data.preprocessing import save_sample
from datetime import datetime
import numpy as np

data_dir = "./boat_data"

# Collect one second of sonar pings
sweeps = []
for _ in range(10):  # ~10 pings per second at 4800 baud
    line = ser.readline().decode('ascii', errors='ignore').strip()
    if line.startswith('$PSDVS'):
        result = NMEAInterpreter.parse_sonar_return(line)
        sweeps.append(result)

# Convert to sonar image
pings = [{"depth": s["depth"], "bearing": s["bearing"], "intensity": s["intensity"]}
         for s in sweeps]
sonar_image = sonar_to_image(pings, bearing_bins=128, max_depth=200)

# Save with timestamp
timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

# Camera frames (from GoPro USB mount or WiFi)
cameras = {
    5.0: np.array(Image.open(f"/cameras/cam1/{timestamp}.jpg")),
    15.0: np.array(Image.open(f"/cameras/cam2/{timestamp}.jpg")),
}

save_sample(data_dir, timestamp, sonar_image, cameras,
            sweeps, {"temperature": 12.0, "salinity": 35.0, "turbidity": 0.3})
print(f"Saved sample {timestamp}")
```

### Step 4: Auto-Collect Script

```bash
# Run on Jetson at boot
python -c "
from sonar_vision.data import auto_collect
auto_collect(
    serial_port='/dev/ttyUSB0',
    camera_mount='/mnt/cameras',
    output_dir='./boat_data',
    max_samples=10000,
)
"
```

---

## Tutorial 3: Multi-Camera Self-Supervision

Mount cameras at known depths and let the physics do the labeling.

### Camera Rig Design

```
              Boat Deck
                 |
            Downrigger Cable
                 |
    ┌────────────┼────────────┐
    │            │            │
  [5m Cam]   [15m Cam]   [25m Cam]
  GoPro 12    GoPro 12    GoPro 12
    │            │            │
    └────────────┼────────────┘
                 |
              Weight (keeps cable vertical)
```

### Mounting Details

- **Cameras:** GoPro HERO12 Black, ~$400 each, 4 cameras = ~$1,600
- **Housing:** Dive rated to 50m, ~$100 each
- **Downrigger cable:** 200lb test monofilament or coated wire
- **Interval:** Every 5 meters of depth
- **Synchronization:** GPS time (NMEA GGA sentence) on all cameras
- **Resolution:** 1080p at 30fps (sufficient for training, manageable storage)

### Why 5m Intervals?

The depth weighting function has σ=3.0m. At 5m spacing, each target detection falls within 3m of exactly one camera. The weighting is:
- Camera at exact depth: w = exp(0) = 1.0 (100% weight)
- Camera 5m away: w = exp(-25/18) = 0.25 (25% weight)
- Camera 10m away: w = exp(-100/18) = 0.004 (0.4% weight)

---

## Tutorial 4: Training Your First LoRA

Set up automated nightly training that gets better every day.

### Step 1: Prepare Data

```bash
# Your boat data should be in this structure:
ls ./boat_data/
# sonar/   cameras/   detections/   water/

# Count samples
ls ./boat_data/sonar/*.npy | wc -l
# Should be at least 100 for meaningful training
```

### Step 2: Configure Nightly Training

```python
from sonar_vision.nightly.lora_trainer import LoRAConfig, NightlyTrainer
from sonar_vision.pipeline import SonarVision

# Load base model
model = SonarVision(max_depth=200, bearing_bins=128, embed_dim=768)

# Configure LoRA
lora_config = LoRAConfig(
    rank=16,           # LoRA rank (higher = more capacity)
    alpha=16.0,        # Scaling factor
    dropout=0.1,       # Regularization
    lr=1e-4,           # Learning rate
    max_epochs=10,     # Max epochs (progressive will use fewer)
    batch_size=4,
    gradient_accumulation=4,  # Effective batch = 16
)

# Create trainer
trainer = NightlyTrainer(
    model=model,
    data_dir="./boat_data",
    output_dir="./nightly_output",
    lora_config=lora_config,
)
```

### Step 3: Run Training

```python
# Single run
run = trainer.run()
print(f"Quality: {run.quality_score:.3f}, Promoted: {run.promoted}")

# Check history
report = trainer.get_training_report(last_n=7)
for r in report:
    print(f"  {r['timestamp']}: score={r['quality_score']:.3f} promoted={r['promoted']}")
```

### Step 4: Deploy Trained LoRA to Jetson

```python
from sonar_vision.nightly.lora_trainer import apply_lora, extract_lora_weights
from sonar_vision.deploy import export_torchscript

# Load base model and apply LoRA
model = SonarVision(max_depth=200, bearing_bins=128, embed_dim=768)
model, lora_layers = apply_lora(model, ["q_proj", "k_proj", "v_proj", "out_proj"], rank=16)

# Load trained LoRA weights
import torch
weights = torch.load(trainer.get_latest_lora())
load_lora_weights(model, lora_layers, weights)

# Merge LoRA into base model
for layer in lora_layers:
    parent_name = [n for n, _ in model.named_modules() if _ is layer][0]
    # Merge and replace...

# Export for Jetson
export_torchscript(model, "sonarvision_with_lora.pt")
```

### Step 5: Set Up Cron Job

```bash
# Add to crontab
crontab -e

# Run nightly at 3:00 AM Alaska time
0 3 * * * cd /opt/sonar-vision && python -m sonar_vision.nightly.cron \
    --data_dir /data/boat_data \
    --output_dir /data/nightly_output \
    --lora_rank 16 \
    --max_epochs 10 \
    >> /var/log/sonar-nightly.log 2>&1
```

---

## Tutorial 5: Deploying to Jetson Orin NX

### Step 1: Flash Jetson

```bash
# Use NVIDIA JetPack 6.0
# Follow: https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nx-devkit
```

### Step 2: Install Dependencies

```bash
sudo apt update
sudo apt install python3-pip python3-venv
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/jetson
pip3 install numpy pillow pyyaml einops
```

### Step 3: Copy Model

```bash
scp user@dev-machine:/opt/sonar-vision/sonarvision_with_lora.pt ~/models/
```

### Step 4: Run Inference

```python
from sonar_vision.deploy import JetsonInference

engine = JetsonInference(
    model_path="/home/user/models/sonarvision_with_lora.pt",
    bearing_bins=128,
    max_depth=200,
    device="cuda",
    precision="fp16",
)

# Process one sweep
import numpy as np
sweep = np.random.randn(128, 200).astype(np.float32)
frame, depth, latency = engine.process_sweep(sweep)

print(f"Frame: {frame.shape}, Latency: {latency:.1f}ms")
print(f"Stats: {engine.get_stats()}")
```

### Step 5: Connect to Sonar

```bash
# USB serial
python -c "
import serial
from sonar_vision.deploy import JetsonInference
import numpy as np

engine = JetsonInference('models/sonarvision.pt', precision='fp16')
ser = serial.Serial('/dev/ttyUSB0', 4800)

while True:
    line = ser.readline().decode('ascii', errors='ignore').strip()
    if line.startswith('\$PSDVS'):
        result = NMEAInterpreter.parse_sonar_return(line)
        sweep = np.zeros((128, 200), dtype=np.float32)
        # Fill sweep from parsed data...
        frame, depth, ms = engine.process_sweep(sweep)
        # Display on HDMI or stream to shore...
"
```

---

## Tutorial 6: Joining the Federated Network

Opt in to share anonymized data for global model improvement.

### Step 1: Create Consent Record

```python
from sonar_vision.federated import ConsentRecord

consent = ConsentRecord(
    user_id="your-email@example.com",  # Will be hashed
    vessel_type="longliner",
    region="GOA",                       # Gulf of Alaska
    opt_in=True,
    share_depth_data=True,
    share_camera_data=True,
    share_location=False,               # No GPS — just region
    share_detection_data=True,
    date_consented="2026-04-29",
    min_quality_psnr=20.0,             # Only share good data
    differential_privacy_epsilon=1.0,   # Standard DP budget
)
```

### Step 2: Anonymize and Share

```python
from sonar_vision.federated import DataAnonymizer, DifferentialPrivacy

anonymizer = DataAnonymizer(salt="your-secret-salt")
dp = DifferentialPrivacy(epsilon=1.0)

# Anonymize your data
contributor_hash = anonymizer.hash_contributor("your-email@example.com")
time_bucket = anonymizer.bucket_timestamp("2026-04-29T14:30:00")  # → "2026-04"

# Anonymize detections (strip GPS, keep physical measurements)
clean_detections = anonymizer.anonymize_detections(raw_detections)

# Add differential privacy noise to sonar sweeps
noisy_sweep = anonymizer.anonymize_sonar_sweep(raw_sweep)

# Share LoRA weights (not raw data — more private, smaller upload)
lora_weights = trainer.get_latest_lora()
noisy_weights = dp.add_noise_to_weights(lora_weights, sensitivity=0.01)
# POST to federated endpoint...
```

### How Federated Averaging Works

```
Your Boat          Boat B            Boat C           Global Model
    │                  │                 │                  │
    ├─ Train LoRA ──→  │                 │                  │
    │                  ├─ Train LoRA ──→  │                  │
    │                  │                 ├─ Train LoRA ──→  │
    │                  │                 │                  │
    └─ Share (DP) ────→└─ Share (DP) ──→└─ Share (DP) ──→ │
                                                                 │
                                              Federated Average  │
                                              (weighted by data) │
                                                                 │
                                              Global v1.0 ←──────┘
```

No raw images or GPS data ever leaves your boat. Only noise-injected LoRA weights.
