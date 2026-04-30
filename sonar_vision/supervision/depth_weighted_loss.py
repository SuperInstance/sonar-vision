"""
Self-Supervision Loss — Automatic depth-weighted supervision from multi-camera array.

The key insight: when the sonar detects a target at depth D, the camera at depth D
captures the ground truth. The loss weight for each camera exponentially decays with
the distance between camera depth and sonar-detected depth.

This creates an automatic supervision signal:
- No manual labeling needed
- Model learns sonar→visual mapping from natural correspondence
- Cameras at wrong depths contribute low-weight negative examples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class DepthWeightedLoss(nn.Module):
    """Self-supervised loss for sonar-to-video prediction.
    
    Computes weighted reconstruction loss between predicted video frames
    and camera captures, with weights based on depth correspondence.
    
    The weight function:
        w(d_cam, d_sonar) = exp(-|d_cam - d_sonar|² / (2 * sigma²))
    
    where sigma controls how sharply the weight falls off with depth mismatch.
    """
    
    def __init__(
        self,
        sigma: float = 3.0,           # Depth matching tolerance (meters)
        l1_weight: float = 1.0,       # Pixel-level reconstruction
        perceptual_weight: float = 0.1,  # Perceptual quality
        depth_consistency_weight: float = 0.5,  # Depth prediction accuracy
        negative_weight: float = 0.01,  # Weight for "wrong depth" cameras
        min_camera_weight: float = 0.01,  # Minimum weight even for far cameras
    ):
        super().__init__()
        self.sigma = sigma
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.depth_consistency_weight = depth_consistency_weight
        self.negative_weight = negative_weight
        self.min_camera_weight = min_camera_weight
        
        # Learnable depth scaling (adapts to sonar calibration)
        self.depth_scale = nn.Parameter(torch.tensor(1.0))
        self.depth_bias = nn.Parameter(torch.tensor(0.0))
    
    def compute_depth_weights(
        self,
        camera_depths: torch.Tensor,   # (B, num_cameras) depth of each camera in meters
        sonar_detections: torch.Tensor, # (B, num_detections) detected depths from sonar
    ) -> torch.Tensor:
        """Compute per-camera supervision weights based on sonar detections.
        
        For each camera, find the closest sonar detection and compute weight.
        If no detection is close, use negative_weight.
        
        Args:
            camera_depths: (B, C) depth of each camera
            sonar_detections: (B, D) sonar-detected target depths
        
        Returns:
            weights: (B, C) per-camera loss weights
        """
        B, C = camera_depths.shape
        D = sonar_detections.shape[1]
        
        # Scale sonar detections (learnable calibration)
        scaled_detections = sonar_detections * self.depth_scale + self.depth_bias
        
        # Compute pairwise depth differences: (B, C, D)
        cam_d = camera_depths.unsqueeze(-1)  # (B, C, 1)
        son_d = scaled_detections.unsqueeze(1)  # (B, 1, D)
        depth_diff = (cam_d - son_d) ** 2  # (B, C, D)
        
        # Weight for each (camera, detection) pair
        weights_per_det = torch.exp(-depth_diff / (2 * self.sigma ** 2))  # (B, C, D)
        
        # Take max weight across detections (closest detection wins)
        weights, _ = weights_per_det.max(dim=-1)  # (B, C)
        
        # Ensure minimum weight for negative examples
        weights = weights.clamp(min=self.min_camera_weight)
        
        # If no sonar detections, all cameras get negative weight
        if D == 0:
            weights = torch.full_like(camera_depths, self.negative_weight)
        
        return weights
    
    def l1_loss(
        self,
        predicted: torch.Tensor,  # (B, C, H, W)
        target: torch.Tensor,     # (B, C, H, W)
        weights: torch.Tensor,    # (B,) per-sample weights
    ) -> torch.Tensor:
        """Weighted L1 reconstruction loss."""
        B = predicted.shape[0]
        per_sample = F.l1_loss(predicted, target, reduction='none')  # (B, C, H, W)
        per_sample = per_sample.mean(dim=[1, 2, 3])  # (B,)
        return (per_sample * weights).mean()
    
    def depth_consistency_loss(
        self,
        predicted_depth: torch.Tensor,  # (B, H, W) predicted depth map
        sonar_depth: torch.Tensor,      # (B,) or (B, D) ground truth from sonar
    ) -> torch.Tensor:
        """L2 loss between predicted depth and sonar-measured depth."""
        if sonar_depth.dim() == 1:
            # Single depth value per sample
            target_depth = sonar_depth.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        else:
            # Multiple detections — use mean
            target_depth = sonar_depth.mean(dim=-1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        
        return F.mse_loss(predicted_depth, target_depth.expand_as(predicted_depth))
    
    def forward(
        self,
        predicted_frames: torch.Tensor,    # (B, 3, H, W) predicted video frames
        camera_frames: torch.Tensor,       # (B, num_cameras, 3, H, W) camera captures
        camera_depths: torch.Tensor,       # (B, num_cameras) camera depths in meters
        sonar_detections: torch.Tensor,     # (B, num_detections) sonar-detected depths
        predicted_depth_map: Optional[torch.Tensor] = None,  # (B, H, W)
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute total self-supervised loss.
        
        Args:
            predicted_frames: Model's predicted underwater video
            camera_frames: Ground truth from camera array
            camera_depths: Physical depth of each camera
            sonar_detections: Depths of targets detected by sonar
            predicted_depth_map: Optional depth map prediction for consistency loss
        
        Returns:
            total_loss: scalar
            loss_dict: individual loss components
        """
        B, num_cams = camera_depths.shape
        
        # Compute depth-based weights for each camera
        cam_weights = self.compute_depth_weights(camera_depths, sonar_detections)
        # cam_weights: (B, num_cams)
        
        # For each camera, compute loss against the SAME predicted frame
        # (the prediction is for the sonar-detected depth, not per-camera)
        # Expand predicted to match cameras: (B, num_cams, 3, H, W)
        pred_expanded = predicted_frames.unsqueeze(1).expand(-1, num_cams, -1, -1, -1)
        
        # Per-camera L1 losses
        per_cam_l1 = F.l1_loss(pred_expanded, camera_frames, reduction='none')
        per_cam_l1 = per_cam_l1.mean(dim=[2, 3, 4])  # (B, num_cams)
        
        # Weight by depth correspondence
        weighted_l1 = (per_cam_l1 * cam_weights).mean()
        
        # Depth consistency (if depth map predicted)
        depth_loss = torch.tensor(0.0, device=predicted_frames.device)
        if predicted_depth_map is not None and sonar_detections.shape[1] > 0:
            depth_loss = self.depth_consistency_loss(predicted_depth_map, sonar_detections)
        
        # Total loss
        total = (
            self.l1_weight * weighted_l1 +
            self.depth_consistency_weight * depth_loss
        )
        
        loss_dict = {
            "total": total.item(),
            "weighted_l1": weighted_l1.item(),
            "depth_consistency": depth_loss.item(),
            "mean_cam_weight": cam_weights.mean().item(),
            "max_cam_weight": cam_weights.max().item(),
            "sigma": self.sigma,
        }
        
        return total, loss_dict


class TemporalConsistencyLoss(nn.Module):
    """Ensures smooth transitions between generated frames.
    
    Underwater scenes change slowly — the generated video should not flicker.
    Uses temporal L1 between consecutive frames.
    """
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss.
        
        Args:
            frames: (B, T, C, H, W) sequence of generated frames
        """
        if frames.shape[1] < 2:
            return torch.tensor(0.0, device=frames.device)
        
        diff = frames[:, 1:] - frames[:, :-1]
        return self.weight * diff.abs().mean()


class TurbidityAwareLoss(nn.Module):
    """Adjusts loss weighting based on water turbidity.
    
    In turbid water, fine details are invisible — the model shouldn't be
    penalized for not predicting them. In clear water, details matter.
    """
    
    def __init__(self, base_sigma: float = 3.0, turbidity_range: Tuple[float, float] = (0.0, 1.0)):
        super().__init__()
        self.base_sigma = base_sigma
        self.turbidity_min, self.turbidity_max = turbidity_range
    
    def forward(
        self,
        turbidity: torch.Tensor,  # (B,) turbidity score [0=clear, 1=murky]
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute turbidity-adjusted reconstruction loss.
        
        Higher turbidity → larger sigma → more forgiving loss.
        """
        # Scale sigma with turbidity
        sigma = self.base_sigma * (1.0 + 2.0 * turbidity)  # 1x to 3x
        
        # Per-pixel weight based on turbidity
        # In clear water, weight all pixels equally
        # In turbid water, downweight high-frequency details
        weights = torch.exp(-turbidity.view(-1, 1, 1, 1) * 0.5)
        
        loss = F.l1_loss(predicted, target, reduction='none')
        loss = (loss * weights).mean()
        
        return loss
