# GETTING STARTED — Sonar Vision

> *Estimated time: 5 minutes*

## Prerequisites

- Python 3.10+

## Installation

```bash
pip install sonar-vision
```

## Your First 5 Minutes

### 1. Generate a Sonar Ping

```python
from sonar_vision import Sonar

sonar = Sonar(frequency=50000, max_range=500)
ping = sonar.generate_ping(sample_rate=44100)
print(f"Ping signal: {len(ping.samples)} samples")
```

### 2. Ping a Target

```python
result = sonar.ping(distance=100)
print(f"Distance: {result.distance:.1f}m")
print(f"Travel time: {result.travel_time*1000:.1f}ms")
print(f"SNR: {result.snr_db:.1f} dB")
```

### 3. Track Multiple Objects

```python
from sonar_vision import ObjectTracker, Detection

tracker = ObjectTracker()
tracker.update([Detection(x=10, y=20)])
tracker.update([Detection(x=12, y=22)])
tracker.update([Detection(x=14, y=19)])
print(f"Tracks: {len(tracker.tracks)}")
for t in tracker.tracks:
    print(f"  ID {t.track_id}: ({t.x:.1f}, {t.y:.1f}), v=({t.vx:.1f}, {t.vy:.1f})")
```

### 4. Filter a Signal

```python
from sonar_vision import Signal

sig = Signal.chirp(f0=1000, f1=10000, duration=0.5)
lowpassed = sig.lowpass(cutoff=5000)
print(f"Original: {len(sig.samples)} samples, Filtered: {len(lowpassed.samples)}")
```

## Next Steps

- [ARCHITECTURE.md](./ARCHITECTURE.md) — Design overview
- [API_REFERENCE.md](./API_REFERENCE.md) — Full API
- [LOW_LEVEL.md](./LOW_LEVEL.md) — Internal details
