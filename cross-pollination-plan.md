# SonarVision Cross-Pollination Plan

## Target Repos & Integration Points

### 1. marine-gpu-edge (CUDA sensor fusion)
**Integration:** SonarVision's Python pipeline as the inference layer for marine-gpu-edge's CUDA sensor fusion.
- PR: Add `sonar_vision.integrations.marine_gpu` module
  - `MarineGPUBridge` — reads MEP protocol packets, feeds depth data to SonarVision encoder
  - `CUDASonarPipeline` — wraps marine-gpu-edge's CUDA beamformer output as SonarVision input
  - `AdaptivePrecisionController` — passes precision hints from marine-gpu-edge to SonarVision
- Files to create:
  - `sonar_vision/integrations/marine_gpu/__init__.py`
  - `sonar_vision/integrations/marine_gpu/bridge.py`
  - `sonar_vision/integrations/marine_gpu/cuda_pipeline.py`
  - `tests/test_marine_gpu_integration.py`

### 2. holodeck-rust (Multi-agent MUD)
**Integration:** SonarVision as an underwater sensor room in the MUD.
- PR: Add SonarVision plugin
  - `plugins/sonar_vision.rs` — depth sounder room that generates underwater video descriptions
  - Players explore ocean depths, SonarVision renders what they'd see
  - Physics-engine integration (thermocline, water types affect visibility)

### 3. cocapn-dashboard (Fleet telemetry)
**Integration:** Live sonar telemetry visualization.
- PR: Add SonarVision widgets
  - Depth-to-video stream panel (WebSocket → canvas)
  - Fleet sonar coverage heatmap
  - Real-time inference metrics gauge
  - Files: `dashboard/src/widgets/sonar/`

### 4. fleet-simulator (Multi-agent simulation)
**Integration:** SonarVision as a sensor for simulated agents.
- PR: Add sensor plugin
  - `plugins/sonar_sensor.py` — simulated depth sounder feeding SonarVision
  - Agents can "see" underwater via sonar predictions
  - Configurable water conditions affect prediction quality

### 5. businesslog-ai (Business ops)
**Integration:** SonarVision inference logging + billing.
- PR: Add metrics logging
  - Log inference events, model versions, latency, confidence scores
  - Usage metering for SonarVision API endpoints
  - Business metrics (surveys completed, area mapped)

### 6. TrendRadar (AI trend monitor)
**Integration:** Monitor underwater video prediction research trends.
- PR: Add marine AI feeds
  - Track arXiv papers on sonar→video, underwater perception
  - Cross-reference with SonarVision roadmap
  - Auto-generate research briefs for SonarVision maintainers

### 7. jepa-perception-lab (JEPA models)
**Integration:** JEPA-based video prediction as alternative decoder.
- PR: Add JEPA decoder
  - `sonar_vision/decoder/jepa_decoder.py` — replace deterministic decoder with JEPA
  - Joint embedding predictive architecture for better temporal consistency
  - Benchmark vs existing decoder
