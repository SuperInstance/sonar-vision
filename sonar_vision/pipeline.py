"""
SonarVision — Main pipeline combining encoder, aggregator, and decoder.

The full inference pipeline:
1. Sonar ping stream → SonarEncoder → feature tokens
2. Feature tokens → Streaming GCT Aggregator → aggregated tokens
3. Aggregated tokens → VideoDecoder → underwater video frames
4. Camera frames + sonar detections → DepthWeightedLoss → self-supervision
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from sonar_vision.encoder.sonar_encoder import SonarEncoder
from sonar_vision.decoder.video_decoder import VideoDecoder, VideoDecoderSequence
from sonar_vision.aggregator.gct import StreamingGCTAggregator, KVCache
from sonar_vision.supervision.depth_weighted_loss import (
    DepthWeightedLoss,
    TemporalConsistencyLoss,
)
from sonar_vision.water.physics import WaterColumnModel, SonarBeamModel


class SonarVision(nn.Module):
    """Complete SonarVision pipeline.
    
    Converts depth sounder returns into predicted underwater video,
    with self-supervised learning from multi-camera arrays.
    """
    
    def __init__(
        self,
        # Encoder
        max_depth: int = 200,
        bearing_bins: int = 128,
        patch_size: int = 14,
        embed_dim: int = 1024,
        pretrained_encoder: str = "",
        # Decoder
        output_resolution: Tuple[int, int] = (384, 512),
        # Supervision
        depth_sigma: float = 3.0,
        temporal_weight: float = 0.1,
        # Water physics
        sonar_frequency_khz: float = 200.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Core components
        self.encoder = SonarEncoder(
            max_depth=max_depth,
            bearing_bins=bearing_bins,
            patch_size=patch_size,
            embed_dim=embed_dim,
            pretrained=pretrained_encoder,
        )
        
        self.decoder = VideoDecoder(
            embed_dim=embed_dim,
            output_resolution=output_resolution,
        )
        
        # Self-supervision
        self.supervision_loss = DepthWeightedLoss(sigma=depth_sigma)
        self.temporal_loss = TemporalConsistencyLoss(weight=temporal_weight)
        
        # Water physics (for feature conditioning)
        self.water_model = WaterColumnModel()
        self.beam_model = SonarBeamModel(frequency_khz=sonar_frequency_khz)
        
        # GCT Streaming Aggregator (adapted from LingBot-Map)
        self.aggregator = StreamingGCTAggregator(
            embed_dim=embed_dim,
            num_heads=16,
            num_layers=6,
            gqa_ratio=4,
            window_size=32,
            max_depth=max_depth,
            max_bearing=bearing_bins,
        )

        # Feature adapter: encoder tokens → decoder input
        self.feature_adapter = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Output resolution
        self.output_resolution = output_resolution
    
    def forward(
        self,
        sonar_intensity: torch.Tensor,    # (B, bearing_bins, max_depth)
        camera_frames: Optional[torch.Tensor] = None,  # (B, C, 3, H, W) for training
        camera_depths: Optional[torch.Tensor] = None,  # (B, C) for training
        sonar_detections: Optional[torch.Tensor] = None,  # (B, D) for training
        accumulated: Optional[torch.Tensor] = None,
        depth_axis: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.
        
        Training mode: computes self-supervision loss
        Inference mode: returns generated video frame
        
        Args:
            sonar_intensity: Raw sonar returns
            camera_frames: Ground truth camera frames (training only)
            camera_depths: Camera depths (training only)
            sonar_detections: Sonar-detected target depths (training only)
            accumulated: Time-accumulated sonar returns
            depth_axis: Custom depth axis
        
        Returns:
            Training: {'loss': scalar, 'loss_dict': dict, ...}
            Inference: {'frame': tensor, 'depth_map': tensor, ...}
        """
        # Encode sonar
        tokens, enc_info = self.encoder(sonar_intensity, depth_axis, accumulated)
        
        # Stream through GCT aggregator
        # Note: for training (non-streaming), we pass all tokens at once
        # For streaming inference, use generate_stream() method
        aggregated, _ = self.aggregator(tokens)
        adapted_tokens = self.feature_adapter(aggregated)
        
        # Estimate average depth from sonar for color conditioning
        if sonar_detections is not None and sonar_detections.shape[1] > 0:
            avg_depth = sonar_detections.mean(dim=-1)  # (B,)
        else:
            avg_depth = torch.full((sonar_intensity.shape[0],), 10.0, 
                                   device=sonar_intensity.device)
        
        # Decode to video
        result = self.decoder(adapted_tokens, target_depth=avg_depth)
        
        output = {
            "frame": result["frame"],
            "depth_map": result["depth_map"],
            "features": result["features"],
        }
        
        # Training: compute self-supervision loss
        if self.training and camera_frames is not None and camera_depths is not None:
            if sonar_detections is None:
                # Use depth map prediction as pseudo-detections
                sonar_detections = result["depth_map"].mean(dim=[1, 2, 3]) * self.encoder.sweep_embed.max_depth
            
            loss, loss_dict = self.supervision_loss(
                predicted_frames=result["frame"],
                camera_frames=camera_frames,
                camera_depths=camera_depths,
                sonar_detections=sonar_detections,
                predicted_depth_map=result["depth_map"],
            )
            
            output["loss"] = loss
            output["loss_dict"] = loss_dict
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        sonar_intensity: torch.Tensor,
        accumulated: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate underwater video from sonar (inference mode).
        
        Args:
            sonar_intensity: (1, bearing_bins, max_depth) sonar sweep
        
        Returns:
            Generated frame and depth map
        """
        self.eval()
        return self.forward(sonar_intensity, accumulated=accumulated)
    
    def generate_stream(self, sonar_intensity: torch.Tensor, cache: Optional[KVCache] = None):
        """Generate one frame at a time (streaming inference).
        
        Use for real-time sonar processing where sweeps arrive continuously.
        
        Args:
            sonar_intensity: (1, bearing_bins, max_depth) single sonar sweep
            cache: KV cache from previous call (None for first call)
        
        Returns:
            (frame, depth_map, cache) — cache must be passed to next call
        """
        self.eval()
        with torch.no_grad():
            tokens, _ = self.encoder(sonar_intensity)
            aggregated, cache = self.aggregator(tokens, cache=cache)
            adapted = self.feature_adapter(aggregated)
            result = self.decoder(adapted)
            return result["frame"], result["depth_map"], cache

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs):
        """Load a pretrained SonarVision model."""
        model = cls(**kwargs)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        return model
