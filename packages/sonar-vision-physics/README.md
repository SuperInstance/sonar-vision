# SonarVision Physics

Real-time underwater sonar physics engine. Deterministic FLUX 9-opcode model.

## Install

```bash
pip install sonar-vision-physics
```

## Usage

```python
from sonar_vision_physics import compute_physics

# Physics at 15m depth
result = compute_physics(depth=15.0, chlorophyll=4.0)
print(f"Water: {result['water_type_name']}")
print(f"Temperature: {result['temperature']}C")
print(f"Visibility: {result['visibility']}m")
print(f"Sound speed: {result['sound_speed']} m/s")
```

### CLI

```bash
sonar-ping --depth 30 --format json
sonar-ping --dive --start 0 --end 100 --step 5
sonar-ping --serve --port 8081
```

### WebSocket Streaming

```bash
sonar-ping --serve --port 8081
# Connect: ws://localhost:8081
# Commands: reset, pause, resume, goto:50
```

## Physics Model

| Op | Function | Model |
|----|----------|-------|
| B0 | Absorption | Francois-Garrison |
| B1 | Scattering | Rayleigh-like |
| B2 | Jerlov Type | Chlorophyll classification |
| B3 | Thermocline | Gaussian gradient |
| B4 | Seabed | Sediment reflectivity |
| B5 | Attenuation | Absorption + Scattering |
| B6 | Visibility | Secchi depth |
| B7 | Sound Speed | Mackenzie equation |
| B8 | Refraction | Snell's law |

Deterministic: same inputs → bit-identical outputs across all instances.
