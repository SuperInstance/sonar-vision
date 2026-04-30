# Hardware Guide — Setting Up SonarVision on Your Boat

Complete guide to hardware selection, wiring, and installation.

---

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Boat Deck / Wheelhouse                │
│                                                         │
│  ┌──────────┐    RS485    ┌──────────────────────┐     │
│  │ Fish     │────────────→│  Jetson Orin NX      │     │
│  │ Finder   │  (NMEA)     │  ┌────────────────┐  │     │
│  │ (Sonar)  │             │  │ SonarVision    │  │     │
│  └──────────┘             │  │ - Encode sonar │  │     │
│                           │  │ - Predict video│  │     │
│  ┌──────────┐   USB/WiFi  │  │ - LoRA trained │  │     │
│  │ Cameras  │────────────→│  └────────────────┘  │     │
│  │ (4x)     │             │                      │     │
│  └──────────┘             │  Output:             │     │
│                           │  - HDMI display      │     │
│  ┌──────────┐   USB       │  - 4G/WiFi stream    │     │
│  │ GPS      │────────────→│  - NMEA log         │     │
│  └──────────┘             └──────────────────────┘     │
│                                                         │
└─────────────────────────────────────────────────────────┘
                          │
                    Downrigger Cable
                          │
              ┌───────────┼───────────┐
              │           │           │
           [5m Cam]   [15m Cam]   [25m Cam]
              │           │           │
              └───────────┼───────────┘
                          │
                       Weight
```

---

## Sonar Selection

### Recommended: Garmin Panoptix LiveScope
- **Type:** Real-time scanning sonar
- **Frequency:** 1.2 MHz (narrow beam, high detail)
- **Range:** 0-60m forward/down
- **NMEA:** 0183 and 2000 output
- **Price:** ~$1,500 (transducer) + $800 (display)
- **Why:** Best real-time imaging, clean NMEA output, widely available

### Alternative: Humminbird HELIX 10
- **Type:** CHIRP sonar + MEGA Imaging
- **Frequency:** 455/800/1200 kHz
- **Range:** 0-120m
- **NMEA:** 0183 output
- **Price:** ~$1,200 (unit) + $400 (transducer)
- **Why:** Excellent CHIRP resolution, good NMEA support

### Budget: Any NMEA 0183 Depth Sounder
- **Minimum:** $SDDPT or $SDDBT sentence output
- **Baud rate:** 4800
- **Connector:** RS485 (2-wire)
- **Price:** Used units from $100-300
- **Why:** Even a basic depth sounder provides training data

### NMEA Sentences We Use

| Sentence | Description | Example |
|----------|-------------|---------|
| `$SDDBT` | Depth Below Transducer | `$SDDBT,12.5,f,3.81,M,2.07,F*2A` |
| `$SDDPT` | Depth (with offset) | `$SDDPT,3.81,0.0,0.0*4F` |
| `$SDMTW` | Water Temperature | `$SDMTW,12.5,C*28` |
| `$PSDVS` | Proprietary sonar return | `$PSDVS,15.2,45.0,-30.5,3.0*4A` |

---

## Camera Selection

### Recommended: GoPro HERO12 Black
- **Resolution:** 1080p @ 30fps (sufficient, manageable storage)
- **Housing:** Dive rated to 50m (built-in)
- **Power:** 90 min @ 1080p
- **Sync:** GPS time via NMEA GGA sentence
- **Price:** ~$400 each
- **Qty needed:** 3-4 (at 5m depth intervals)

### Alternative: Axis M3115-LVE
- **Resolution:** 1080p
- **Housing:** IP66 (not submersible — use custom housing)
- **Power:** PoE (Power over Ethernet)
- **Price:** ~$500 each
- **Why:** Industrial grade, easier long-term deployment

### Budget: Any action camera
- **Minimum:** 720p, any frame rate
- **Waterproof housing required below 10m
- **Price:** $50-100 each
- **Sync:** Manual timestamp alignment (within 1-2 sec is fine)

---

## Camera Rig Design

### Downrigger Setup

```
Cable attachment point (on downrigger boom)
           │
     ┌─────┴─────┐
     │ 5m marker  │ ← Camera 1 (surface waters)
     │            │
     │ 10m marker │
     │            │
     │ 15m marker │ ← Camera 2 (mid-water — best coverage)
     │            │
     │ 20m marker │
     │            │
     │ 25m marker │ ← Camera 3 (deep water)
     │            │
     └─────┬─────┘
           │
       Cannonball
       (10-15 lbs)
```

### Mounting Hardware

| Item | Qty | Price | Notes |
|------|-----|-------|-------|
| Downrigger boom | 1 | $100-200 | Must support 30m cable |
| Monofilament cable | 200m | $50 | 200lb test |
| Camera clamps | 4 | $20 each | Stainless steel |
| Camera housings | 4 | $100 each | For cameras without built-in |
| Depth markers | 6 | $5 each | Cable clips at 5m intervals |
| Cannonball weight | 1 | $30 | 10-15 lbs |
| USB hub (waterproof) | 1 | $50 | For camera data transfer |
| **Total rig** | | **~$850** | |

---

## Compute Platform

### Recommended: Jetson Orin NX (16GB)
- **GPU:** 1024 CUDA cores, 40 TOPS
- **CPU:** 8-core ARM Cortex-A78AE
- **RAM:** 16GB LPDDR5
- **Storage:** 256GB NVMe (recommended)
- **Power:** 15-25W
- **Price:** ~$500 (dev kit)
- **Performance:** 10-15 fps inference (FP16)

### Alternative: Jetson AGX Orin (64GB)
- **GPU:** 2048 CUDA cores, 275 TOPS
- **RAM:** 64GB
- **Power:** 15-60W
- **Price:** ~$2,000 (dev kit)
- **Performance:** 20-30 fps inference (FP16)

### Development: Any CUDA GPU
- **Minimum:** GTX 1060 (6GB)
- **Recommended:** RTX 4050, RTX 4090
- **Performance:** 30-60+ fps inference

### Power Budget (Boat Installation)

| Component | Power |
|-----------|-------|
| Jetson Orin NX | 15-25W |
| USB hub | 5W |
| 4 cameras (charging) | 20W |
| USB-RS485 adapter | 2W |
| **Total** | **42-52W** |

Can run off boat's 12V system with a 100W DC-DC converter.

---

## Wiring Diagram

### Sonar → Jetson

```
Fish Finder                  Jetson Orin NX
┌──────────┐                ┌──────────────┐
│ NMEA OUT │                │              │
│ (TX+) ───┼── RS485 ──────┤ USB-RS485    │
│ (TX-) ───┼── twisted ────┤ Adapter ──────┤ USB
│ GND ──────┼── pair ───────┤              │
└──────────┘                └──────────────┘

Pinout (standard NMEA 0183):
  TX+ (A) → RS485 A (or D+)
  TX- (B) → RS485 B (or D-)
  GND     → GND

Baud rate: 4800 (default) or 38400 (high-speed)
```

### Cameras → Jetson

```
Option A: USB (simple)
  Camera → USB cable → Jetson USB port (needs hub for 4 cameras)

Option B: WiFi (wireless)
  Camera → GoPro WiFi → Jetson WiFi → HTTP download

Option C: Ethernet (reliable)
  Camera → PoE splitter → Ethernet switch → Jetson Ethernet
```

---

## Storage

### Training Data (per day)

| Data type | Size per hour | Size per day (12h) |
|-----------|--------------|-------------------|
| Sonar sweeps (NMEA text) | 5 MB | 60 MB |
| Sonar images (128×200 float32) | 50 MB | 600 MB |
| Camera frames (1080p, 1 fps) | 2 GB | 24 GB |
| **Total** | **2 GB** | **25 GB** |

### Recommended Storage

- **512GB NVMe SSD** on Jetson (keeps 20 days of data)
- **External 2TB SSD** for archival (USB 3.0)
- **Sync to shore** via 4G/WiFi when available (rsync to S3)

---

## Cost Summary

| Configuration | Items | Total |
|---------------|-------|-------|
| **Entry** ($500) | Used sonar + 2 action cameras + Jetson Nano | ~$500 |
| **Standard** ($2,500) | Garmin sonar + 4 GoPros + Jetson Orin NX | ~$2,500 |
| **Professional** ($5,000) | Premium sonar + 4 GoPros + Jetson Orin NX + rig | ~$5,000 |
| **Research** ($10,000) | Full system + Simrad + Jetson AGX + ROV mount | ~$10,000 |

---

## Installation Checklist

- [ ] Mount sonar transducer (through-hull or transom)
- [ ] Run NMEA cable from sonar to wheelhouse
- [ ] Install USB-RS485 adapter
- [ ] Test NMEA output: `screen /dev/ttyUSB0 4800`
- [ ] Mount downrigger with camera clamps
- [ ] Attach cameras at 5m intervals
- [ ] Run USB/ethernet from cameras to Jetson
- [ ] Install Jetson (JetPack 6.0, Python, PyTorch)
- [ ] Install SonarVision: `pip install -e .`
- [ ] Run test: `python -c "from sonar_vision.pipeline import SonarVision; print('OK')"`
- [ ] Test data collection: 1 hour, verify files in data directory
- [ ] Train first LoRA: `python -m sonar_vision.nightly.cron --data_dir ./data`
- [ ] Set up nightly cron: `crontab -e`
- [ ] Set up data sync to cloud (S3/rsync)
- [ ] (Optional) Join federated network

---

## Maintenance

### Daily
- Check camera batteries/storage
- Verify NMEA data is flowing
- Review overnight training report

### Weekly
- Sync training data to cloud
- Check LoRA quality trend
- Clean camera housings

### Monthly
- Update SonarVision (`git pull && pip install -e .`)
- Review and rotate training data
- Check for sensor drift
