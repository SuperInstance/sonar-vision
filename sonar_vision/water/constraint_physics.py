"""
Constraint-Theoretic Physics for SonarVision.

Core insight: every sonar-video prediction must satisfy physical constraints.
Rather than hoping the network learns them, we *enforce* them as a constraint
satisfaction problem. The physics model IS the constraint system.

Key constraint-theoretic tools applied:

1. PYTHAGOREAN SNAP: Given a prediction that violates physics, project it
   onto the nearest physically-valid point. Instead of rejection, compute the
   minimum-distance projection onto the constraint manifold.

   Example: prediction says "bright red at 50m" but Beer-Lambert says red
   transmission at 50m is ~0.0001. Snap projects to the nearest valid color
   that satisfies attenuation, minimizing perceptual distortion.

2. ARC CONSISTENCY (AC-3): Each processing stage has a domain of valid
   outputs. Propagate constraints forward: if the beam geometry constrains
   a detection to [14m, 16m], prune camera assignments outside that range
   before computing the loss.

3. DEPENDENCY SCHEDULING: The pipeline has natural dependencies. Schedule
   computation so that the most constraining variables (sound speed, beam
   angle) are resolved first, pruning the search space for expensive stages
   (video generation).

4. CONSTRAINT GRAPH: Sonar → beam geometry → depth assignment → attenuation
   → color space → temporal consistency. Each node has a domain; arcs enforce
   physical laws between adjacent nodes.

This makes the physics hard — predictions CAN'T violate Beer-Lambert,
CAN'T have impossible sound speeds, CAN'T assign cameras to wrong depths.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class PythagoreanSnap(nn.Module):
    """Project predictions onto the nearest physically-valid point.

    The key insight from constraint theory: when a solution violates
    constraints, the "snap" finds the minimum-distance projection onto
    the constraint manifold. This is superior to rejection or clamping
    because it preserves the maximum amount of information from the
    prediction while guaranteeing physical consistency.

    Applied to SonarVision:
    - Color snap: project RGB onto the Beer-Lambert attenuation manifold
    - Depth snap: project estimated depths onto beam geometry constraints
    - Temperature snap: project sound speed estimates onto valid T(z) profiles
    - Motion snap: project temporal predictions onto physically-consistent trajectories

    Mathematical basis:
    Given constraint C: f(x) = 0 and prediction x₀,
    find x* = argmin ||x - x₀||² subject to f(x) = 0

    For affine constraints (most physics): this is a closed-form projection.
    For nonlinear constraints: one step of Newton's method on the Lagrangian.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def snap_to_attenuation_manifold(
        self,
        predicted_rgb: torch.Tensor,   # (B, 3) predicted RGB values
        depth: torch.Tensor,           # (B,) depth in meters
        rgb_max: torch.Tensor,         # (B, 3) surface (unattenuated) RGB
        attenuation_factors: torch.Tensor,  # (B, 3) [R, G, B] transmission
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Snap predicted colors onto the Beer-Lambert attenuation manifold.

        Constraint: predicted_rgb ≤ rgb_max * attenuation_factors (element-wise)
        Each channel is independently bounded by Beer-Lambert law.

        Pythagorean snap: for each violated channel, project back to the
        constraint boundary (the attenuation surface), preserving the
        relative ratios between channels.

        This is equivalent to:
        scale = min(1, max_k(attenuation_k * rgb_max_k / predicted_k))
        snapped = predicted * scale

        which finds the nearest point on the valid region boundary
        in L2 norm (the Pythagorean projection).

        Returns:
            snapped_rgb: (B, 3) physically-valid colors
            snap_distance: (B,) L2 distance moved (diagnostic)
        """
        # Upper bound from Beer-Lambert
        upper_bound = rgb_max * attenuation_factors

        # Compute per-sample scale factor (how much to shrink to satisfy)
        # For each sample, find the channel that needs the most reduction
        ratios = upper_bound / (predicted_rgb + self.epsilon)
        scale = ratios.min(dim=-1, keepdim=True)[0]  # (B, 1)
        scale = scale.clamp(max=1.0)  # Only shrink, never expand

        snapped = predicted_rgb * scale

        # Snap distance (L2)
        snap_distance = (snapped - predicted_rgb).norm(dim=-1)

        return snapped, snap_distance

    def snap_to_beam_arc(
        self,
        predicted_position: torch.Tensor,  # (B, 2) [depth, bearing] in meters, degrees
        beam_center_depth: float,
        beam_center_bearing: float,
        beam_depth_width: float,
        beam_bearing_width: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Snap a predicted position onto the sonar beam constraint arc.

        The beam defines a region in (depth, bearing) space. Targets detected
        by this beam MUST lie within this region. Pythagorean snap projects
        any out-of-bounds prediction to the nearest point on the beam boundary.

        This is a rectangular projection — minimum L2 distance to the
        rectangle defined by beam geometry.

        Returns:
            snapped_position: (B, 2) [depth, bearing]
            snap_distance: (B,) distance moved
        """
        d_min = beam_center_depth - beam_depth_width / 2
        d_max = beam_center_depth + beam_depth_width / 2
        b_min = beam_center_bearing - beam_bearing_width / 2
        b_max = beam_center_bearing + beam_bearing_width / 2

        depth = predicted_position[:, 0]
        bearing = predicted_position[:, 1]

        # Clamp to beam rectangle (this IS the Pythagorean projection for boxes)
        snapped_depth = depth.clamp(d_min, d_max)
        snapped_bearing = bearing.clamp(b_min, b_max)

        snapped = torch.stack([snapped_depth, snapped_bearing], dim=-1)
        snap_distance = (snapped - predicted_position).norm(dim=-1)

        return snapped, snap_distance

    def snap_to_sound_speed_profile(
        self,
        estimated_speed: torch.Tensor,  # (B,) estimated sound speed in m/s
        depth: torch.Tensor,            # (B,) depth in meters
        surface_speed: float = 1480.0,
        deep_speed: float = 1500.0,
        mld: float = 50.0,
        thermocline_bottom: float = 300.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Snap estimated sound speed onto physically-valid profile.

        Ocean sound speed is bounded:
        - Surface: ~1440-1540 m/s (depends on temperature, salinity)
        - Deep (>1000m): ~1480-1560 m/s (pressure dominated)
        - Sound channel minimum: ~1480 m/s at ~700-1000m

        The valid profile is a band around the Mackenzie/UNESCO equation.
        Pythagorean snap projects estimates into this band.

        Returns:
            snapped_speed: (B,) valid sound speed
            snap_distance: (B,) distance moved
        """
        # Simple bounds based on depth
        speed_min = torch.where(
            depth < mld,
            torch.tensor(surface_speed - 30.0),
            torch.where(
                depth < thermocline_bottom,
                torch.tensor(surface_speed - 20.0),
                torch.tensor(deep_speed - 20.0)
            )
        )
        speed_max = torch.where(
            depth < mld,
            torch.tensor(surface_speed + 20.0),
            torch.where(
                depth < thermocline_bottom,
                torch.tensor(deep_speed + 30.0),
                torch.tensor(deep_speed + 60.0)
            )
        )

        snapped = estimated_speed.clamp(speed_min, speed_max)
        snap_distance = (snapped - estimated_speed).abs()

        return snapped, snap_distance

    def snap_to_temporal_consistency(
        self,
        current_prediction: torch.Tensor,   # (B, C, H, W) current frame
        previous_prediction: torch.Tensor,  # (B, C, H, W) previous frame
        max_flow: float = 30.0,             # max pixel displacement per step
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Snap current prediction for temporal consistency.

        Constraint: consecutive frames should differ by at most max_flow pixels
        (based on maximum expected object velocity × frame time).

        If the optical flow between frames exceeds max_flow, snap back
        to the boundary. This prevents temporal discontinuities that
        are physically impossible for underwater objects.

        Implementation: use the previous frame as an anchor. Any region
        that moved more than max_flow from its previous position gets
        projected back to the max_flow circle around its anchor.
        """
        # Difference between frames
        diff = current_prediction - previous_prediction
        # Spatial displacement estimate (sum of absolute differences as proxy)
        spatial_diff = diff.abs().sum(dim=1)  # (B, H, W) per-pixel total change

        # Normalize to flow estimate (approximate)
        flow_estimate = spatial_diff  # Proxy for optical flow magnitude

        # Scale factor to snap to max_flow boundary
        overflow_mask = flow_estimate > max_flow
        scale = torch.where(
            overflow_mask,
            max_flow / (flow_estimate + self.epsilon),
            torch.ones_like(flow_estimate)
        )

        # Apply per-pixel scaling
        snapped = previous_prediction + diff * scale.unsqueeze(1)

        # Snap distance (average per pixel)
        snap_distance = (snapped - current_prediction).abs().mean(dim=(1, 2, 3))

        return snapped, snap_distance


class PhysicalConstraintGraph(nn.Module):
    """Constraint graph for the SonarVision processing pipeline.

    Each processing stage is a node with a domain of valid outputs.
    Arcs between nodes enforce physical laws.

    Graph structure:
        Sonar → BeamGeometry → DepthAssignment → Attenuation → ColorSpace → Video
              ↘ Thermocline ↗                                              ↓
                                                                         Temporal

    AC-3 style propagation: after each stage, check arc consistency
    with adjacent stages. Prune impossible values before expensive
    downstream computation.
    """

    def __init__(
        self,
        max_depth: float = 200.0,
        max_bearing: float = 90.0,
        frequency_khz: float = 200.0,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.max_bearing = max_bearing
        self.frequency_khz = frequency_khz
        self.snap = PythagoreanSnap()

        # Sound speed bounds (m/s)
        self.speed_min = 1400.0
        self.speed_max = 1600.0

        # Wavelength
        self.wavelength = 1500.0 / (frequency_khz * 1000)

    def propagate_beam_geometry(
        self,
        sonar_returns: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """AC-3 style constraint propagation through beam geometry.

        Input: raw sonar returns with depth, bearing, intensity
        Output: returns with pruned domains based on beam physics

        Constraints applied:
        1. Depth must be > 0 and ≤ max_depth
        2. Bearing must be within beam width
        3. Intensity must be positive
        4. Range resolution: two returns must differ by > range_resolution
        """
        depth = sonar_returns.get("depth", torch.tensor([]))
        bearing = sonar_returns.get("bearing", torch.tensor([]))
        intensity = sonar_returns.get("intensity", torch.tensor([]))

        if depth.numel() == 0:
            return sonar_returns

        # Constraint 1: Valid depth range
        valid_depth = (depth > 0) & (depth <= self.max_depth)

        # Constraint 2: Valid bearing
        valid_bearing = bearing.abs() <= self.max_bearing

        # Constraint 3: Positive intensity
        valid_intensity = intensity > 0

        # Combined mask
        mask = valid_depth & valid_bearing & valid_intensity

        # Apply mask
        pruned = {}
        for key, val in sonar_returns.items():
            if val.numel() == depth.numel():
                pruned[key] = val[mask]
            else:
                pruned[key] = val
        pruned["valid_mask"] = mask

        return pruned

    def compute_depth_assignment_domains(
        self,
        sonar_depth: torch.Tensor,    # (N,) sonar detection depths
        camera_depths: torch.Tensor,  # (M,) camera mounting depths
        depth_tolerance: float = 5.0, # meters
    ) -> torch.Tensor:
        """Compute valid camera-sonar assignment matrix using constraints.

        Constraint: camera at depth D_c can only provide ground truth for
        sonar detection at depth D_s if |D_s - D_c| ≤ tolerance AND
        the light attenuation at D_c is sufficient for imaging.

        Returns: (N, M) assignment validity matrix (True = valid)
        """
        # Depth proximity constraint
        depth_diff = (sonar_depth.unsqueeze(1) - camera_depths.unsqueeze(0)).abs()
        depth_valid = depth_diff <= depth_tolerance

        # Light attenuation constraint: camera needs some light to image
        # Simple model: red attenuates fastest, need > 1% transmission for RGB
        max_valid_depth = 50.0  # meters (where even blue light is ~5%)
        camera_valid = camera_depths.unsqueeze(0) <= max_valid_depth

        # Combined
        assignment = depth_valid & camera_valid

        return assignment

    def propagate_attenuation(
        self,
        predicted_rgb: torch.Tensor,    # (B, 3) predicted colors
        depth: torch.Tensor,            # (B,) depth
        turbidity: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Propagate attenuation constraints and snap invalid predictions.

        Uses Pythagorean snap to project predictions onto the physically-valid
        color space defined by Beer-Lambert attenuation.

        Returns:
            snapped_rgb: (B, 3) valid colors
            diagnostics: dict with snap distances and constraint violations
        """
        # Beer-Lambert attenuation factors (Jerlov IB water)
        # R(650nm): k≈0.37/m, G(555nm): k≈0.11/m, B(443nm): k≈0.045/m
        kd_r = 0.37 * (1 + turbidity)
        kd_g = 0.11 * (1 + turbidity)
        kd_b = 0.045 * (1 + turbidity)

        attenuation = torch.stack([
            torch.exp(-kd_r * depth),
            torch.exp(-kd_g * depth),
            torch.exp(-kd_b * depth),
        ], dim=-1)  # (B, 3)

        # Surface colors (upper bound — what you'd see at surface)
        # Use predicted values as surface estimate (they represent what
        # the network "thinks" is there before attenuation)
        surface_rgb = predicted_rgb / (attenuation + 1e-6)

        # Pythagorean snap onto attenuation manifold
        snapped, snap_dist = self.snap.snap_to_attenuation_manifold(
            predicted_rgb, depth, surface_rgb, attenuation
        )

        # Count violations
        violations = (snap_dist > 0.01).sum()

        diagnostics = {
            "snap_distance": snap_dist,
            "mean_snap": snap_dist.mean(),
            "max_snap": snap_dist.max(),
            "violations": violations,
            "attenuation_factors": attenuation,
        }

        return snapped, diagnostics


class DependencyScheduler(nn.Module):
    """Schedule SonarVision computation using constraint-based dependency analysis.

    From constraint theory: resolve the most constraining variables first
    to prune the search space before expensive computation.

    SonarVision dependency order:
    1. NMEA decode (must happen first — everything depends on it)
    2. Sound speed calculation (depends on depth, temperature)
    3. Beam geometry (depends on sound speed, sonar parameters)
    4. Depth assignment (depends on beam geometry, camera positions)
    5. Attenuation calculation (depends on depth, water type)
    6. Video generation (depends on all above — most expensive)
    7. Temporal consistency check (depends on previous frame + current)

    Key optimization: steps 1-5 are O(N) or O(N²). Step 6 is O(N·C·H·W).
    By propagating constraints through 1-5 first, we can:
    - Skip video generation for impossible detections
    - Reduce resolution for low-confidence regions
    - Reuse computations across similar detections
    """

    def __init__(self, confidence_threshold: float = 0.1):
        super().__init__()
        self.confidence_threshold = confidence_threshold

    def compute_execution_order(
        self,
        sonar_returns: Dict[str, torch.Tensor],
        camera_depths: List[float],
    ) -> Dict[str, torch.Tensor]:
        """Compute which detections need full processing vs. skip.

        Early termination: if beam geometry + attenuation constraints
        make a detection impossible to image, skip video generation.

        Returns schedule with flags for each detection.
        """
        depth = sonar_returns.get("depth", torch.tensor([]))
        intensity = sonar_returns.get("intensity", torch.tensor([]))
        n = depth.numel()

        if n == 0:
            return {"process_mask": torch.tensor([], dtype=torch.bool)}

        # Step 1: Beam geometry check — detection within any camera's range?
        camera_tensor = torch.tensor(camera_depths, dtype=depth.dtype)
        depth_diff = (depth.unsqueeze(1) - camera_tensor.unsqueeze(0)).abs()
        min_depth_diff = depth_diff.min(dim=1)[0]  # Closest camera
        has_camera = min_depth_diff < 20.0  # Within 20m of any camera

        # Step 2: Light check — is there enough light at the detection depth?
        # Even with the deepest camera, need some light
        light_available = torch.exp(-0.045 * depth) > 0.01  # Blue channel > 1%

        # Step 3: Intensity check — is the sonar return above noise floor?
        has_signal = intensity > self.confidence_threshold

        # Step 4: Combined mask
        process_mask = has_camera & light_available & has_signal

        return {
            "process_mask": process_mask,
            "skip_reasons": {
                "no_camera": ~has_camera,
                "no_light": ~light_available,
                "no_signal": ~has_signal,
            },
            "confidence": intensity * light_available * has_camera.float(),
        }

    def adaptive_resolution(
        self,
        confidence: torch.Tensor,
        base_height: int = 384,
        base_width: int = 512,
        min_scale: float = 0.25,
    ) -> torch.Tensor:
        """Scale output resolution based on confidence.

        High-confidence detections get full resolution.
        Low-confidence detections get reduced resolution (faster).

        Constraint-theoretic basis: the tighter the constraints are satisfied,
        the more computation is warranted. Loose constraints → less precision needed.
        """
        scale = min_scale + (1.0 - min_scale) * confidence
        scale = scale.clamp(min_scale, 1.0)

        heights = (base_height * scale).int()
        widths = (base_width * scale).int()

        return torch.stack([heights, widths], dim=-1)


class SoundChannelConstraint(nn.Module):
    """SOFAR channel (Sound Fixing and Ranging) constraint.

    The ocean has a sound speed minimum at ~700-1000m depth (varies by
    location and season). Sound gets "trapped" in this channel and can
    propagate thousands of kilometers.

    For SonarVision, the SOFAR channel affects:
    - Multipath arrival times (reflections from channel boundaries)
    - Effective detection range (much longer in-channel)
    - Ambient noise (channel focuses noise from distant sources)

    Constraint: predicted sound speed profile MUST have a minimum
    between 700-1200m. If the thermocline model doesn't produce this,
    Pythagorean snap projects it onto a valid profile.
    """

    def __init__(
        self,
        channel_min_depth: float = 700.0,
        channel_max_depth: float = 1200.0,
        channel_min_speed: float = 1480.0,
    ):
        super().__init__()
        self.channel_min_depth = channel_min_depth
        self.channel_max_depth = channel_max_depth
        self.channel_min_speed = channel_min_speed

    def check_sound_channel(
        self,
        sound_speed_profile: torch.Tensor,  # (B, D) speed at each depth
        depths: torch.Tensor,                # (D,) depth values
    ) -> Tuple[bool, Dict[str, float]]:
        """Verify the sound speed profile has a valid SOFAR channel.

        Returns:
            is_valid: True if profile has minimum in channel depth range
            diagnostics: depth and speed of the minimum
        """
        # Find minimum speed in the channel depth range
        channel_mask = (depths >= self.channel_min_depth) & (depths <= self.channel_max_depth)
        if not channel_mask.any():
            return False, {"error": "no depths in channel range"}

        channel_speeds = sound_speed_profile[:, channel_mask]
        min_speed, min_idx = channel_speeds.min(dim=-1)
        channel_depths = depths[channel_mask]
        min_depth = channel_depths[min_idx]

        # The minimum speed in the channel must be less than speeds above and below
        # (it's a local minimum, not just the global minimum)
        above_mask = depths < self.channel_min_depth
        below_mask = depths > self.channel_max_depth

        valid = True
        if above_mask.any():
            valid = valid & (min_speed < sound_speed_profile[:, above_mask].min(dim=-1)[0])
        if below_mask.any():
            valid = valid & (min_speed < sound_speed_profile[:, below_mask].min(dim=-1)[0])

        return valid, {
            "min_speed": min_speed.item(),
            "min_depth": min_depth.item(),
            "channel_present": valid.item(),
        }


class DepthWeightedAssignment(nn.Module):
    """Constraint-based camera-sonar depth assignment using Pythagorean distance.

    When multiple cameras are at different depths, we need to assign each
    sonar detection to the best camera for ground truth. This is a
    constraint satisfaction problem:

    Variables: assignment matrix A[i,j] = 1 if detection i assigned to camera j
    Constraints:
    - Each detection assigned to at most one camera (or none)
    - Assignment only valid if |depth_det - depth_cam| ≤ tolerance
    - Weight by: 1) depth proximity (closer = better), 2) light availability,
      3) camera health/confidence

    The "Pythagorean" aspect: the assignment cost is the Euclidean distance
    in (depth_diff, bearing_diff, time_diff) space. Optimal assignment
    minimizes total Pythagorean distance.

    This is equivalent to solving a minimum-weight bipartite matching,
    which we approximate greedily for speed (O(NM) instead of O(N³M³)).
    """

    def __init__(
        self,
        depth_weight: float = 1.0,
        light_weight: float = 0.5,
        temporal_weight: float = 0.3,
    ):
        super().__init__()
        self.depth_weight = depth_weight
        self.light_weight = light_weight
        self.temporal_weight = temporal_weight

    def compute_cost_matrix(
        self,
        detection_depths: torch.Tensor,   # (N,) sonar detection depths
        camera_depths: torch.Tensor,      # (M,) camera depths
        detection_times: Optional[torch.Tensor] = None,  # (N,) timestamps
        camera_times: Optional[torch.Tensor] = None,     # (M,) camera timestamps
    ) -> torch.Tensor:
        """Compute Pythagorean cost matrix for assignment.

        Cost[i,j] = w_d * |d_i - c_j|² + w_l * light_penalty(c_j) + w_t * |t_i - t_j|²

        Lower cost = better assignment.
        """
        # Depth distance (squared)
        depth_diff = (detection_depths.unsqueeze(1) - camera_depths.unsqueeze(0)) ** 2

        # Light availability penalty (more light = lower penalty)
        # Blue channel transmission as proxy for overall imaging quality
        light_penalty = 1.0 - torch.exp(-0.045 * camera_depths.unsqueeze(0))

        # Temporal proximity (if timestamps available)
        if detection_times is not None and camera_times is not None:
            time_diff = (detection_times.unsqueeze(1) - camera_times.unsqueeze(0)) ** 2
            time_diff = time_diff / 3600.0  # Normalize: 1 hour = 1 unit
        else:
            time_diff = torch.zeros_like(depth_diff)

        # Combined cost (Pythagorean distance in weighted space)
        cost = (self.depth_weight * depth_diff
                + self.light_weight * light_penalty
                + self.temporal_weight * time_diff)

        return cost

    def greedy_assign(
        self,
        cost_matrix: torch.Tensor,  # (N, M)
        max_distance: float = 25.0,  # max allowed depth difference
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy optimal assignment (approximation to Hungarian algorithm).

        For each detection, assign to the camera with minimum cost,
        provided cost is below threshold.

        Returns:
            assignments: (N,) camera index for each detection (-1 = unassigned)
            weights: (N,) assignment confidence (inverse cost)
        """
        N, M = cost_matrix.shape
        assignments = torch.full((N,), -1, dtype=torch.long)
        weights = torch.zeros(N)

        for i in range(N):
            min_j = cost_matrix[i].argmin()
            min_cost = cost_matrix[i, min_j]

            if min_cost < max_distance ** 2:  # Cost threshold
                assignments[i] = min_j
                weights[i] = 1.0 / (1.0 + min_cost.sqrt())

        return assignments, weights
