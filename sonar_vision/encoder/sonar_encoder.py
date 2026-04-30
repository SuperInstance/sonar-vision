"""
Sonar Ping Encoder — Converts raw sonar returns to feature tokens.

Treats sonar pings as a "1D image" where:
  - Y-axis = depth (0 to max_range)
  - X-axis = bearing angle
  - Channel = intensity (dB) + derived features

Uses a ViT patch embedding approach adapted for acoustic data.
Each sonar sweep becomes a sequence of tokens for the GCT aggregator.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple


class SonarSweepEmbedding(nn.Module):
    """Convert sonar ping sweeps into patch tokens.
    
    Input: Raw sonar data (depth x bearing x channels)
    Output: Sequence of patch tokens (N_patches x embed_dim)
    
    Unlike visual ViT which uses 2D patches on RGB images, this uses
    patches over the depth-bearing space with acoustic-specific features:
    - Raw intensity (dB)
    - Intensity gradient (rate of change = target detection cue)
    - Depth-normalized intensity (accounts for spherical spreading)
    - Accumulated returns (time-integrated for slow targets)
    """
    
    def __init__(
        self,
        max_depth: int = 200,       # meters
        bearing_bins: int = 128,     # angular resolution
        patch_size: int = 14,        # patch size in pixels
        in_channels: int = 4,        # intensity, gradient, depth_norm, accumulated
        embed_dim: int = 1024,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.bearing_bins = bearing_bins
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        self.num_patches_h = bearing_bins // patch_size
        self.num_patches_v = max_depth // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_v
        
        # Patch embedding: project acoustic patches to embed_dim
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
            bias=True
        )
        
        # Acoustic-specific normalization
        self.intensity_norm = nn.LayerNorm(embed_dim)
        
        # Positional encoding for depth-bearing space
        # Depth has different physical meaning than bearing (meters vs degrees)
        self.depth_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches_v, embed_dim)
        )
        self.bearing_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches_h, embed_dim)
        )
        
        # Learnable scale token (for estimating scene scale, adapted from lingbot-map)
        self.scale_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.depth_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.bearing_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.scale_token, std=0.02)
    
    def preprocess_sonar(
        self,
        intensity: torch.Tensor,       # (B, bearing_bins, max_depth)
        depth_axis: Optional[torch.Tensor] = None,  # (B, max_depth) in meters
    ) -> torch.Tensor:
        """Preprocess raw sonar intensity into multi-channel input.
        
        Channel 0: Raw intensity (dB normalized)
        Channel 1: Depth gradient (vertical derivative — detects targets)
        Channel 2: Depth-normalized intensity (corrects spherical spreading)
        Channel 3: Accumulated returns (running sum — detects slow movers)
        """
        # Normalize intensity to [0, 1]
        intensity_norm = (intensity - intensity.min(dim=-1, keepdim=True)[0]) / \
                        (intensity.max(dim=-1, keepdim=True)[0] - intensity.min(dim=-1, keepdim=True)[0] + 1e-6)
        
        # Channel 0: raw normalized
        c0 = intensity_norm
        
        # Channel 1: vertical gradient (depth derivative)
        c1 = torch.zeros_like(c0)
        c1[:, :, 1:] = c0[:, :, 1:] - c0[:, :, :-1]
        
        # Channel 2: depth-normalized (correct for spherical spreading loss)
        if depth_axis is None:
            depth_axis = torch.linspace(0, 1, intensity.shape[-1], device=intensity.device)
            depth_axis = depth_axis.unsqueeze(0).expand(intensity.shape[0], -1)
        
        # Spherical spreading: intensity ∝ 1/r²
        # Compensate: multiply by r² to normalize across depths
        depth_comp = (depth_axis[:, None, :] ** 2).clamp(min=0.01)
        c2 = intensity_norm * depth_comp
        
        # Channel 3: accumulated (temporal integration)
        # This is computed externally and passed in; default to zeros for single-frame
        c3 = torch.zeros_like(c0)
        
        # Stack: (B, bearing_bins, max_depth, 4)
        multi_channel = torch.stack([c0, c1, c2, c3], dim=-1)
        # Rearrange to (B, 4, bearing_bins, max_depth)
        multi_channel = multi_channel.permute(0, 3, 1, 2)
        
        return multi_channel
    
    def forward(
        self,
        intensity: torch.Tensor,       # (B, bearing_bins, max_depth)
        depth_axis: Optional[torch.Tensor] = None,
        accumulated: Optional[torch.Tensor] = None,  # (B, bearing_bins, max_depth)
    ) -> Tuple[torch.Tensor, Dict]:
        """Convert sonar sweep to patch tokens.
        
        Returns:
            tokens: (B, num_patches + 1, embed_dim) — patch tokens + scale token
            info: dict with preprocessing metadata
        """
        B = intensity.shape[0]
        
        # Preprocess
        x = self.preprocess_sonar(intensity, depth_axis)
        
        # Add accumulated returns if provided
        if accumulated is not None:
            acc_norm = (accumulated - accumulated.min(dim=-1, keepdim=True)[0]) / \
                      (accumulated.max(dim=-1, keepdim=True)[0] - accumulated.min(dim=-1, keepdim=True)[0] + 1e-6)
            x[:, 3, :, :] = acc_norm
        
        # Patch embedding: (B, embed_dim, num_patches_v, num_patches_h)
        tokens = self.proj(x)
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Normalize
        tokens = self.intensity_norm(tokens)
        
        # Reshape for separable positional encoding
        # tokens: (B, num_patches_v * num_patches_h, embed_dim)
        tokens_2d = tokens.view(B, self.num_patches_v, self.num_patches_h, -1)
        
        # Add depth and bearing position embeddings
        tokens_2d = tokens_2d + self.depth_pos_embed[:, :, None, :]
        tokens_2d = tokens_2d + self.bearing_pos_embed[:, None, :, :]
        
        # Flatten back
        tokens = tokens_2d.view(B, self.num_patches, -1)
        
        # Prepend scale token
        scale_tokens = self.scale_token.expand(B, -1, -1)
        tokens = torch.cat([scale_tokens, tokens], dim=1)
        
        info = {
            "num_patches": self.num_patches,
            "num_patches_v": self.num_patches_v,
            "num_patches_h": self.num_patches_h,
            "has_scale_token": True,
        }
        
        return tokens, info


class SonarEncoder(nn.Module):
    """Full sonar encoder pipeline.
    
    Wraps SonarSweepEmbedding with optional DINOv2 backbone initialization
    for transfer learning from visual features.
    """
    
    def __init__(
        self,
        max_depth: int = 200,
        bearing_bins: int = 128,
        patch_size: int = 14,
        in_channels: int = 4,
        embed_dim: int = 1024,
        pretrained: str = "",  # DINOv2 checkpoint path
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.sweep_embed = SonarSweepEmbedding(
            max_depth=max_depth,
            bearing_bins=bearing_bins,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.embed_dim = embed_dim
        self.num_patches = self.sweep_embed.num_patches
        
        # If pretrained DINOv2 path given, load patch embedding weights
        # (transfer visual patch embedding → acoustic, freeze if specified)
        if pretrained:
            self._load_pretrained(pretrained, freeze_backbone)
    
    def _load_pretrained(self, path: str, freeze: bool):
        """Load DINOv2 patch embedding as initialization."""
        try:
            state = torch.load(path, map_location="cpu")
            # Map DINOv2 patch_embed to our projection
            if "patch_embed.proj.weight" in state:
                w = state["patch_embed.proj.weight"]
                # DINOv2 has 3 input channels (RGB), we have 4 (acoustic)
                # Average the RGB weights and replicate for 4th channel
                w_rgb = w.mean(dim=1, keepdim=True)  # (1, C_out, kH, kW)
                w_acoustic = w_rgb.expand(-1, 4, -1, -1)
                self.sweep_embed.proj.weight.data.copy_(w_acoustic[:, :3, :, :])
                self.sweep_embed.proj.weight.data[:, 3, :, :] = w_rgb.squeeze(1)
                
                if "patch_embed.proj.bias" in state:
                    self.sweep_embed.proj.bias.data.copy_(state["patch_embed.proj.bias"])
                
            if freeze:
                self.sweep_embed.proj.requires_grad_(False)
                print(f"[SonarEncoder] Loaded DINOv2 patch embedding (frozen)")
            else:
                print(f"[SonarEncoder] Loaded DINOv2 patch embedding (fine-tuning)")
        except Exception as e:
            print(f"[SonarEncoder] Could not load pretrained: {e}")
    
    def forward(
        self,
        intensity: torch.Tensor,
        depth_axis: Optional[torch.Tensor] = None,
        accumulated: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Encode sonar sweep into tokens.
        
        Args:
            intensity: (B, bearing_bins, max_depth) raw sonar returns
            depth_axis: (B, max_depth) depth values in meters
            accumulated: (B, bearing_bins, max_depth) time-accumulated returns
        
        Returns:
            tokens: (B, num_patches + 1, embed_dim)
            info: metadata dict
        """
        return self.sweep_embed(intensity, depth_axis, accumulated)
