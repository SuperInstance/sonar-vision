"""
Video Decoder — Generates underwater video from aggregated features.

Adapts the DPT head from LingBot-Map for video generation:
- Multi-scale feature fusion (DPT architecture)
- Temporal diffusion decoder for smooth video
- Underwater-specific output (low light, blue-green color cast, particulate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


class UnderwaterColorHead(nn.Module):
    """Post-processing head that applies underwater color characteristics.
    
    Underwater imagery has:
    - Blue-green color cast (red light absorbed first)
    - Reduced contrast with depth
    - Scattering from particulates (haze effect)
    
    This head learns to apply these characteristics to generated frames
    for photorealistic underwater appearance.
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, 1),
            nn.Tanh(),  # Output in [-1, 1] range
        )
        
        # Depth-dependent color shift parameters
        self.red_attenuation = nn.Parameter(torch.tensor(0.7))  # How much red fades
        self.blue_boost = nn.Parameter(torch.tensor(0.2))  # Blue-green cast
    
    def forward(
        self,
        features: torch.Tensor,  # (B, C, H, W) decoded features
        depth: Optional[torch.Tensor] = None,  # (B,) average depth in meters
    ) -> torch.Tensor:
        """Apply underwater color characteristics."""
        x = self.net(features)
        
        if depth is not None:
            # Scale color shift with depth
            d_norm = (depth / 50.0).clamp(0, 1).view(-1, 1, 1, 1)
            
            # Attenuate red channel with depth
            x[:, 0:1] = x[:, 0:1] * (1.0 - self.red_attenuation * d_norm)
            # Boost blue channel
            x[:, 2:3] = x[:, 2:3] + self.blue_boost * d_norm
        
        return x


class TemporalRefinementModule(nn.Module):
    """Refines frame sequences for temporal consistency.
    
    Uses 3D convolutions over short temporal windows to smooth
    generated video and remove flickering artifacts.
    """
    
    def __init__(self, channels: int = 64, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.temporal_conv = nn.Conv3d(
            channels, channels,
            kernel_size=(kernel_size, 3, 3),
            padding=(padding, 1, 1),
        )
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine temporal sequence.
        
        Args:
            x: (B, T, C, H, W) frame features
        Returns:
            Refined: (B, T, C, H, W)
        """
        B, T, C, H, W = x.shape
        # Rearrange for 3D conv: (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.temporal_conv(x)
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        return x + self.act(self.norm(x))


class VideoDecoder(nn.Module):
    """Full video decoder pipeline.
    
    Takes aggregated feature tokens and generates underwater video frames.
    
    Pipeline:
    1. Token-to-patch unfolding
    2. Multi-scale DPT fusion
    3. Upsampling to target resolution
    4. Underwater color head
    5. Temporal refinement (optional, for multi-frame)
    """
    
    def __init__(
        self,
        embed_dim: int = 1024,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        output_resolution: Tuple[int, int] = (384, 512),  # (H, W)
        num_temporal_refinement_layers: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_resolution = output_resolution
        
        # Reassemble projection (tokens → 2D feature map)
        self.reassemble = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, features * 14 * 14),  # Unpatch to 14x14 spatial
        )
        
        # Multi-scale feature processing (simplified DPT)
        self.feature_proj = nn.ModuleList([
            nn.Conv2d(features, oc, 1) for oc in out_channels
        ])
        
        # Upsampling stages
        self.upsample_stages = nn.ModuleList()
        in_ch = out_channels[-1]
        for i in range(4):
            out_ch = features if i < 3 else 64
            scale = 4 if i == 0 else 2
            self.upsample_stages.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, scale, stride=scale),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GELU(),
            ))
            in_ch = out_ch
        
        # Final output: features → RGB
        self.to_rgb = nn.Conv2d(64, 3, 3, padding=1)
        
        # Underwater color head
        self.color_head = UnderwaterColorHead()
        
        # Depth map prediction head (for self-supervision)
        self.depth_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),  # Normalize to [0, 1]
        )
        
        # Temporal refinement
        self.temporal_refine = nn.ModuleList([
            TemporalRefinementModule(channels=64)
            for _ in range(num_temporal_refinement_layers)
        ])
    
    def forward(
        self,
        tokens: torch.Tensor,          # (B, N, embed_dim) aggregated tokens
        prev_frame: Optional[torch.Tensor] = None,  # (B, 3, H, W) for temporal blending
        target_depth: Optional[torch.Tensor] = None,  # (B,) for color adjustment
    ) -> Dict[str, torch.Tensor]:
        """Decode tokens to underwater video frame.
        
        Args:
            tokens: Aggregated feature tokens from GCT
            prev_frame: Previous frame for temporal blending
            target_depth: Depth for underwater color adjustment
        
        Returns:
            dict with 'frame' (B, 3, H, W) and 'depth_map' (B, 1, H, W)
        """
        B = tokens.shape[0]
        H, W = self.output_resolution
        
        # Unpatch tokens to 2D feature map
        x = self.reassemble(tokens)  # (B, features * 14 * 14)
        x = x.view(B, -1, 14, 14)  # (B, features, 14, 14)
        
        # Multi-scale feature extraction
        features = [proj(x) for proj in self.feature_proj]
        
        # Progressive upsampling
        x = features[-1]
        for i, stage in enumerate(self.upsample_stages):
            x = stage(x)
            # Skip connection from corresponding feature level
            if i < len(features) - 1:
                fi = len(features) - 2 - i
                # Resize feature to match current spatial size
                skip = features[fi]
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
                x = x + skip
        
        # RGB output
        rgb = self.to_rgb(x)
        rgb = torch.tanh(rgb)  # [-1, 1]
        
        # Underwater color adjustment
        rgb = self.color_head(rgb, target_depth)
        
        # Depth map prediction
        depth_map = self.depth_head(x)
        
        # Temporal blending with previous frame
        if prev_frame is not None:
            alpha = 0.7  # Weight for new prediction
            rgb = alpha * rgb + (1 - alpha) * prev_frame
        
        return {
            "frame": rgb,       # (B, 3, H, W) in [-1, 1]
            "depth_map": depth_map,  # (B, 1, H, W) in [0, 1]
            "features": x,      # (B, 64, H, W) intermediate features
        }


class VideoDecoderSequence(nn.Module):
    """Generates a sequence of video frames from streaming sonar tokens.
    
    Extends VideoDecoder with temporal refinement across frames.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.decoder = VideoDecoder(**kwargs)
        self.temporal_refine = TemporalRefinementModule(channels=64)
    
    def forward(
        self,
        token_sequence: torch.Tensor,  # (B, T, N, embed_dim)
        target_depths: Optional[torch.Tensor] = None,  # (B, T)
    ) -> Dict[str, torch.Tensor]:
        """Generate video sequence from token sequence.
        
        Returns:
            frames: (B, T, 3, H, W)
            depth_maps: (B, T, 1, H, W)
        """
        B, T, N, E = token_sequence.shape
        
        # Decode each frame
        frame_list = []
        depth_list = []
        feat_list = []
        
        prev = None
        for t in range(T):
            depth_t = target_depths[:, t] if target_depths is not None else None
            result = self.decoder(token_sequence[:, t], prev_frame=prev, target_depth=depth_t)
            frame_list.append(result["frame"])
            depth_list.append(result["depth_map"])
            feat_list.append(result["features"])
            prev = result["frame"]
        
        # Stack: (B, T, ...)
        frames = torch.stack(frame_list, dim=1)
        depth_maps = torch.stack(depth_list, dim=1)
        features = torch.stack(feat_list, dim=1)
        
        # Temporal refinement
        refined_features = self.temporal_refine(features)
        
        return {
            "frames": frames,
            "depth_maps": depth_maps,
            "features": refined_features,
        }
