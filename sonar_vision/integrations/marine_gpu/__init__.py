"""Marine GPU Edge integration — bridges SonarVision inference with marine-gpu-edge's CUDA pipeline.

This module connects SonarVision (Python depth→video prediction) with
marine-gpu-edge (CUDA sensor fusion, MEP protocol, constraint-aware scheduling).

Usage:
    from sonar_vision.integrations.marine_gpu import MarineGPUBridge, CUDASonarPipeline

    bridge = MarineGPUBridge(host="jetsonclaw1", port=9734)
    depth_frame = bridge.read_depth_frame()
    video = CUDASonarPipeline(depth_frame).infer()
"""

from .bridge import MarineGPUBridge, CUDASonarPipeline, MEPSonarPacket
from .cuda_pipeline import CUDASonarPipeline

__all__ = ["MarineGPUBridge", "CUDASonarPipeline", "MEPSonarPacket"]
