# Use Cases — SonarVision in the Real World

> **Landing page:** [superinstance.github.io/sonar-vision-landing](https://superinstance.github.io/sonar-vision-landing/)

## The Problem

Underwater imagery is **scarce and expensive**. Annotation is nearly impossible at depth. Existing AI systems require thousands of labeled images that don't exist. Meanwhile, every fishing boat has a sonar and could have cameras — producing data 24/7 with no labels needed.

SonarVision bridges this gap. The sonar IS the label.

---

## Commercial Fishing 🐟

### Target Species Identification
**Pipeline:** Sonar ping → Pattern match → Species: Pollock → Set gear

Sonar detects a school at depth D. The camera at depth D captures the visual appearance. Over time, the system learns to identify species from sonar signatures alone — "this acoustic pattern means pollock at 80m."

**Impact:** Fishermen identify target species before setting gear. Reduces wasted tows. Current systems (Smartrawl by FiS) cost $30K-$80K and require cameras in the net. SonarVision uses the sonar you already have.

### Bycatch Reduction
**Pipeline:** Sonar sweep → Species predict → Non-target? → Move vessel

Real-time prediction from sonar shows what's below before gear enters the water. If the prediction shows non-target species, the captain moves on.

**Impact:** The Nature Conservancy's Edge AI project received funding in 2025-2026 specifically to combat IUU fishing through automated monitoring. SonarVision adds the species-level intelligence these systems lack.

### Catch Forecasting
**Pipeline:** 24h sonar log → 3D biomass map → + SST/chlorophyll → Forecast

Accumulated sonar data builds a 3D picture of what's below. Combined with oceanographic data (SST, chlorophyll), the system predicts where fish will be tomorrow.

**Impact:** Predictive AI platforms like GreenFish achieve 75-92% accuracy in forecasting productive fishing areas up to 8 days in advance. SonarVision adds the underwater visual layer these surface-only systems lack.

---

## Aquaculture 🐠

> **Market:** AI-powered fish farming projected to reach **$910M by 2026**, growing at **15.4% CAGR**.

### Fish Counting & Biomass Estimation
**Pipeline:** Fixed sonar → Detections → Camera verify → Count estimate

Fixed sonar + camera installation on a pen. Nightly LoRA training adapts to local conditions, species, and pen geometry. Count fish from sonar detections, verify counts with camera predictions.

**Impact:** Replaces $5K+ per manual dive survey. Continuous monitoring instead of quarterly snapshots. More accurate, less stressful for fish. Feed is 50-60% of operating cost — accurate biomass means accurate feed.

### Health Monitoring
**Pipeline:** Behavior baseline → Real-time predict → Anomaly detect → Alert crew

Predicted video shows fish behavior patterns — schooling density, swimming speed, vertical distribution. Deviations indicate stress or disease.

**Impact:** AI models predict disease outbreaks before visible symptoms (SeafoodSource, 2026). Early detection saves entire pen stocks worth millions. Replaces visual observation that misses subclinical signs.

### Feeding Optimization
**Pipeline:** Feed deployed → Sonar: fish at feed? → Camera: feeding? → Adjust rate

Sonar shows fish distribution relative to feed pellets. Camera confirms feeding activity. System adjusts feed timing and quantity in real time.

**Impact:** AI-driven feeding systems are "a defining innovation of 2026" — precise feed delivery based on real-time fish behavior, reducing waste by 20-30% and improving feed conversion ratios.

---

## Marine Research 🔬

### Habitat Mapping
**Pipeline:** Sonar survey → Camera stations → Model extrapolate → Full habitat map

Research vessel with sonar + cameras maps seabed habitat. Sonar provides broad coverage; cameras provide ground truth at sample points. The model extrapolates camera-quality imagery to full sonar coverage.

**Impact:** Scalable habitat mapping without expensive ROV surveys. Coverage area increases 100x compared to camera-only surveys. ROV surveys cost $10K-50K/day for 1km²/day. SonarVision covers 100km²/day at $2.5K total hardware cost.

### Species Surveys
**Pipeline:** Passive sonar → Species classify → Population data → Trend analysis

Fixed stations with sonar + camera record species presence. The model learns to identify species from sonar alone, enabling passive acoustic surveys.

**Impact:** Population assessments without netting or diver surveys. Non-invasive, continuous, and cheaper than annual survey cruises. Enables real-time trend analysis instead of annual snapshots.

### Environmental Monitoring
**Pipeline:** Baseline capture → Continuous sonar → Predict changes → Trend report

Long-term stations track changes in the underwater environment. Predicted video provides visual baseline for before/after comparison across seasons and years.

**Impact:** Automated, continuous environmental monitoring. Detect changes in kelp forest health, coral bleaching, invasive species spread in real time. Replaces expensive periodic surveys.

---

## Autonomous Underwater Vehicles (AUVs) 🤖

### Obstacle Avoidance
**Pipeline:** Sonar ping → Video predict → Path plan → Avoid obstacle

AUV sonar feeds SonarVision in real-time. Predicted video shows what's ahead. Navigation system uses the prediction for path planning.

**Impact:** AUVs navigate complex underwater terrain without cameras (which fail in turbid water). Sonar works in zero visibility. Visual SLAM fails below 5m visibility — SonarVision bridges the gap.

### SLAM & 3D Reconstruction
**Pipeline:** Sonar sweep → Predict frames → Depth + scale → 3D map

Sonar sweeps + predicted video frames build 3D maps. Depth predictions from the model provide metric scale.

**Impact:** Robust SLAM that works in conditions where visual SLAM fails (dark, turbid, deep water). Enables autonomous inspection of underwater infrastructure (pipelines, wind farms, oil rigs) without tethered ROVs.

### Search & Inspection
**Pipeline:** Wide sonar scan → Predict content → High confidence? → Dive confirm

AUV searches with sonar, predicts what it's seeing, surfaces to confirm. Reduces search time by focusing dives on high-confidence detections.

**Impact:** Faster underwater search and rescue, infrastructure inspection, archaeological surveys. Sonar covers 100x the area of cameras per unit time.

---

## Sustainability & Compliance 🌍

### IUU Fishing Detection
**Pipeline:** Sonar detect → Vessel classify → + AIS check → Flag illegal

Shore-based stations or patrol vessels detect fishing activity via sonar. Predicted video identifies vessel type and gear. Cross-referenced with AIS data to flag illegal operations.

**Impact:** IUU fishing accounts for **20% of global catch ($23B/year)**. AI-powered monitoring is being deployed by The Nature Conservancy (2025-2026). SonarVision adds species-level intelligence that current sonar-only systems lack.

### Catch Monitoring
**Pipeline:** On-board sonar → Species classify → Auto report → Regulatory submit

On-board sonar records catch composition automatically. Nightly LoRA training improves species identification accuracy over time. Reports generated without manual input.

**Impact:** Transparent catch reporting. Reduces misreporting and underreporting. Meets electronic monitoring requirements being adopted by NOAA, EU, and Pacific fisheries.

### Traceability
**Pipeline:** Catch record → Species verify → Blockchain anchor → Consumer trust

Sonar + camera records provide evidence of where, when, and what was caught. Federated LoRA models share learning across vessels while preserving privacy.

**Impact:** Consumer confidence in sustainable seafood. Premium pricing for verified sustainable catch (15-30% premium). Meets EU/US import traceability requirements. Blockchain-anchored predictions provide tamper-proof records.

---

## How SonarVision Fits

| Application | Input | Output | Training Data | Nightly LoRA | Federated |
|---|---|---|---|---|---|
| Species ID | Sonar pings | Species label | Camera at detection depth | ✅ Adapts to local species | ✅ Cross-vessel learning |
| Bycatch reduction | Sonar pings | Species prediction | Multi-camera self-supervision | ✅ Improves accuracy | ✅ Fleet-wide models |
| Fish counting | Sonar pings | Count estimate | Camera verification | ✅ Pen-specific adaptation | ✅ Multi-pen knowledge |
| AUV navigation | Sonar sweep | Obstacle map | Self-supervised depth | ✅ Environment adaptation | ✅ Multi-mission learning |
| Habitat mapping | Sonar survey | Predicted video | Sample camera stations | ✅ Seasonal adaptation | ✅ Regional models |
| IUU detection | Sonar + AIS | Activity flag | Fleet-wide federated model | ✅ Pattern refinement | ✅ Fleet intelligence |

**The common thread:** sonar is everywhere on the water. Cameras are cheap. The physics of depth alignment provides the labels. No human annotation ever needed.

---

## Market Context

| Metric | Value | Source |
|---|---|---|
| AI aquaculture market | $910M by 2026 | Coherent Market Insights |
| Annual IUU fishing losses | $23B | FAO / World Bank |
| Aquaculture CAGR | 15.4% | Coherent Market Insights |
| Entry-level AI fishing systems | $8,000-$15,000 | Industry reports |
| Premium AI fishing systems | $30,000-$80,000 | Industry reports |
| **SonarVision complete system** | **~$2,500** | Uses existing sonar + commodity cameras |
| Edge AI hardware (Jetson Orin NX) | ~$500 | NVIDIA |
| 4 GoPro cameras | ~$1,600 | GoPro |
| Sustainable catch premium | 15-30% | SeafoodSource 2026 |
| GreenFish forecast accuracy | 75-92% | GreenFish |

---

## Research References

1. **Self-Supervised Learning for Sonar Image Classification** — CVPR 2022 Workshop
2. **SonarSweep: Cross-Modal Plane Sweeping for Underwater Depth** — arXiv 2210.03206
3. **Self-Supervised Monocular Depth Underwater** — ResearchGate 2022
4. **Camera-Sonar Combination for Underwater SLAM** — ResearchGate 2023
5. **NMEA 0183 Protocol Specification** — Actisense
6. **AI Transforming Fishing into Sustainable Industry** — Logistics Handling, 2025
7. **5 AI Trends in Seafood 2026** — SeafoodSource
8. **Edge AI for IUU Fishing Detection** — The Nature Conservancy, 2025
9. **Francois-Garrison Absorption Model** — JASA, 1982
10. **Jerlov Water Classification** — Jerlov, 1976
11. **Mackenzie Sound Speed Equation** — JASA, 1981
