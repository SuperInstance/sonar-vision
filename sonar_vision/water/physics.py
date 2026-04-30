"""
Underwater Physics — Sound propagation, attenuation, and environmental modeling.

Models the physical properties of the underwater channel that affect
both sonar sensing and camera imaging.
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Optional


class WaterColumnModel(nn.Module):
    """Models the underwater environment for realistic signal propagation.
    
    Accounts for:
    - Sound speed profile (temperature + salinity + depth dependent)
    - Absorption (frequency-dependent)
    - Scattering (particulate + biological)
    - Light attenuation (for camera prediction)
    """
    
    def __init__(self):
        super().__init__()
        
        # Typical values (learnable for adaptation to local conditions)
        self.surface_temp = nn.Parameter(torch.tensor(15.0))  # °C
        self.temp_gradient = nn.Parameter(torch.tensor(-0.01))  # °C/m (thermocline)
        self.salinity = nn.Parameter(torch.tensor(35.0))  # PSU
        self.ph = nn.Parameter(torch.tensor(8.1))
    
    def sound_speed(
        self,
        depth: torch.Tensor,  # (B,) or (B, D) depth in meters
        temperature: Optional[torch.Tensor] = None,  # (B,) surface temp override
    ) -> torch.Tensor:
        """Calculate sound speed using Mackenzie equation (1981).
        
        c = 1448.96 + 4.591T - 5.304e-2 T² + 2.374e-4 T³
            + 1.340(S - 35) + 1.630e-2 D + 1.675e-7 D²
            - 1.025e-2 T(S - 35) - 7.139e-13 T D³
        
        where T=temp(°C), S=salinity(PSU), D=depth(m)
        
        Returns sound speed in m/s.
        """
        if temperature is None:
            # Use surface temp + gradient
            T = self.surface_temp + self.temp_gradient * depth
        else:
            T = temperature + self.temp_gradient * depth
        
        S = self.salinity
        D = depth
        
        c = (1448.96
             + 4.591 * T
             - 5.304e-2 * T ** 2
             + 2.374e-4 * T ** 3
             + 1.340 * (S - 35)
             + 1.630e-2 * D
             + 1.675e-7 * D ** 2
             - 1.025e-2 * T * (S - 35)
             - 7.139e-13 * T * D ** 3)
        
        return c
    
    def absorption_coefficient(
        self,
        frequency_khz: float = 200.0,  # Typical fish finder frequency
        depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate absorption coefficient using Francois-Garrison equation.
        
        Simplified for typical sonar frequencies.
        Returns absorption in dB/km.
        """
        # Simplified: absorption increases with frequency squared
        # At 200 kHz in seawater: ~60 dB/km
        freq_factor = (frequency_khz / 200.0) ** 2
        base_absorption = 60.0  # dB/km at 200 kHz
        
        return torch.tensor(base_absorption * freq_factor)
    
    def light_attenuation(
        self,
        depth: torch.Tensor,  # (B,) or scalar
        wavelength_nm: float = 550.0,  # Green light (best penetration)
        turbidity: float = 0.5,  # 0=clear, 1=very turbid
    ) -> torch.Tensor:
        """Calculate light attenuation at given depth.
        
        Beer-Lambert law: I = I₀ * exp(-k * d)
        where k = k_water + k_particulate
        
        Different wavelengths attenuate at different rates:
        - Red (>600nm): absorbed in first 5-10m
        - Green (500-550nm): penetrates to 20-30m
        - Blue (450-500nm): penetrates to 50m+
        """
        # Base attenuation by wavelength (per meter)
        if wavelength_nm > 600:  # Red
            k_water = 0.4
        elif wavelength_nm > 500:  # Green
            k_water = 0.1
        else:  # Blue
            k_water = 0.05
        
        # Turbidity adds scattering loss
        k_turbidity = turbidity * 0.3
        
        k_total = k_water + k_turbidity
        
        # Transmission fraction
        transmission = torch.exp(-k_total * depth)
        
        return transmission
    
    def color_attenuation_vector(
        self,
        depth: torch.Tensor,  # (B,) meters
        turbidity: float = 0.5,
    ) -> torch.Tensor:
        """Get RGB attenuation factors for given depth.
        
        Returns: (B, 3) with [R, G, B] transmission fractions.
        """
        r = self.light_attenuation(depth, wavelength_nm=650, turbidity=turbidity)
        g = self.light_attenuation(depth, wavelength_nm=530, turbidity=turbidity)
        b = self.light_attenuation(depth, wavelength_nm=470, turbidity=turbidity)
        
        return torch.stack([r, g, b], dim=-1)


class SonarBeamModel(nn.Module):
    """Models sonar beam geometry and return characteristics.
    
    Converts between sonar measurements and physical coordinates.
    """
    
    def __init__(
        self,
        beam_width_deg: float = 12.0,
        max_range: float = 200.0,
        frequency_khz: float = 200.0,
    ):
        super().__init__()
        self.beam_width = math.radians(beam_width_deg)
        self.max_range = max_range
        self.frequency_khz = frequency_khz
    
    def range_resolution(self, pulse_length_us: float = 50.0) -> float:
        """Calculate range resolution from pulse length.
        
        ΔR = c * τ / 2
        """
        c = 1500.0  # m/s (approximate sound speed)
        return c * pulse_length_us * 1e-6 / 2
    
    def beam_footprint(self, depth: torch.Tensor) -> torch.Tensor:
        """Calculate beam footprint diameter at given depth.
        
        footprint = 2 * depth * tan(beam_width/2)
        """
        return 2 * depth * torch.tan(self.beam_width / 2)
    
    def target_strength_to_intensity(
        self,
        ts_db: torch.Tensor,   # Target strength in dB
        depth: torch.Tensor,   # (B,) depth in meters
    ) -> torch.Tensor:
        """Convert target strength to received intensity.
        
        Sonar equation: RL = SL - 2TL + TS
        where TL = 20*log10(R) + α*R (transmission loss)
        """
        tl = 20 * torch.log10(depth.clamp(min=1)) + 0.06 * depth  # Simplified TL
        intensity = ts_db - 2 * tl  # Received level
        return intensity


class NMEAInterpreter(nn.Module):
    """Parses NMEA sonar sentences into structured data.
    
    Handles common sonar NMEA formats:
    - SDDBT (Depth Below Transducer)
    - SDMTW (Water Temperature)
    - VHW (Speed Through Water)
    - Sonar-specific proprietary sentences
    """
    
    @staticmethod
    def parse_sonar_return(nmea_string: str) -> Dict:
        """Parse a sonar return NMEA sentence.
        
        Expected proprietary format:
        $PSDVS,depth,bearing,intensity,beam_width*checksum
        
        Returns dict with parsed values.
        """
        try:
            parts = nmea_string.strip().split('*')[0].split(',')
            if len(parts) < 5:
                return {}
            
            return {
                "depth": float(parts[1]),
                "bearing": float(parts[2]),
                "intensity": float(parts[3]),
                "beam_width": float(parts[4]),
                "timestamp": parts[0].split('-')[-1] if '-' in parts[0] else None,
            }
        except (ValueError, IndexError):
            return {}
    
    @staticmethod
    def depth_to_sonar_image(
        returns: list,  # List of parsed sonar returns
        bearing_bins: int = 128,
        max_depth: int = 200,
    ) -> torch.Tensor:
        """Convert sonar returns to a 2D sonar image.
        
        Returns: (bearing_bins, max_depth) intensity image
        """
        image = torch.zeros(bearing_bins, max_depth)
        
        for ret in returns:
            if not ret:
                continue
            depth_idx = int(ret["depth"] / max_depth * max_depth)
            bearing_idx = int((ret["bearing"] + 90) / 180 * bearing_bins)
            
            if 0 <= depth_idx < max_depth and 0 <= bearing_idx < bearing_bins:
                image[bearing_idx, depth_idx] = ret["intensity"]
        
        return image
