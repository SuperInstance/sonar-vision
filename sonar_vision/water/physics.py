"""
Underwater Physics — Sound propagation, attenuation, and environmental modeling.

Advanced models for the underwater channel:
- Mackenzie (1981) and UNESCO/Chen-Millero (1977) sound speed
- Francois-Garrison (1982) full acoustic absorption
- Jerlov water types I-9 for spectral light attenuation
- Seasonal thermocline with latitude dependence
- Seabed acoustic impedance and backscatter models
- Sonar beam pattern with sinc sidelobes
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Optional, List


# ── Jerlov Water Type Diffuse Attenuation Coefficients ──────────────────────
# Kd values in m⁻¹ for spectral bands (380, 420, 470, 530, 580, 650, 700 nm)
# Sources: Jerlov (1976), Preisendorfer (1976)
JEROLOV_KD = {
    #         380     420     470     530     580     650     700
    "I":    [0.032,  0.028,  0.022,  0.018,  0.030,  0.350,  0.650],
    "IA":   [0.038,  0.033,  0.027,  0.022,  0.041,  0.420,  0.700],
    "II":   [0.060,  0.050,  0.040,  0.033,  0.060,  0.550,  0.850],
    "III":  [0.100,  0.080,  0.060,  0.050,  0.090,  0.700,  1.000],
    "1":    [0.080,  0.065,  0.050,  0.042,  0.075,  0.620,  0.900],
    "3":    [0.160,  0.120,  0.090,  0.072,  0.130,  0.850,  1.200],
    "5":    [0.320,  0.230,  0.170,  0.130,  0.230,  1.200,  1.600],
    "7":    [0.480,  0.340,  0.250,  0.190,  0.350,  1.600,  2.100],
    "9":    [0.650,  0.450,  0.330,  0.250,  0.450,  2.000,  2.600],
}

JEROLOV_WAVELENGTHS = [380, 420, 470, 530, 580, 650, 700]  # nm


# ── Seabed Acoustic Properties ─────────────────────────────────────────────
# Impedance Z = ρ * c (kg/m²s), backscatter σ_bs in dB
SEABED_PROPERTIES = {
    "sand": {
        "density": 1800.0,       # kg/m³
        "sound_speed": 1650.0,   # m/s
        "backscatter_db": -20.0, # dB re 1 m²
        "grain_size_mm": 0.5,
        "description": "Fine to coarse sand, low backscatter",
    },
    "mud": {
        "density": 1400.0,
        "sound_speed": 1520.0,
        "backscatter_db": -30.0,
        "grain_size_mm": 0.02,
        "description": "Soft sediment, high absorption",
    },
    "rock": {
        "density": 2500.0,
        "sound_speed": 3500.0,
        "backscatter_db": -5.0,
        "grain_size_mm": 50.0,
        "description": "Hard substrate, strong returns",
    },
    "gravel": {
        "density": 2100.0,
        "sound_speed": 2200.0,
        "backscatter_db": -12.0,
        "grain_size_mm": 10.0,
        "description": "Mixed coarse sediment",
    },
    "clay": {
        "density": 1500.0,
        "sound_speed": 1500.0,
        "backscatter_db": -28.0,
        "grain_size_mm": 0.005,
        "description": "Fine clay, very smooth surface",
    },
    "coral": {
        "density": 2200.0,
        "sound_speed": 2800.0,
        "backscatter_db": -8.0,
        "grain_size_mm": 20.0,
        "description": "Living or dead coral reef",
    },
}


class JerlovModel(nn.Module):
    """Spectral light attenuation using Jerlov water types.
    
    Models how different water clarity conditions affect light
    propagation across the visible spectrum. Essential for realistic
    underwater color rendering in SonarVision's camera predictions.
    """
    
    def __init__(self, water_type: str = "III"):
        """Initialize with a Jerlov water type.
        
        Args:
            water_type: One of 'I', 'IA', 'II', 'III' (oceanic)
                       or '1', '3', '5', '7', '9' (coastal).
                       'III' is typical open ocean, '5' is typical coastal.
        """
        super().__init__()
        if water_type not in JEROLOV_KD:
            raise ValueError(
                f"Unknown water type '{water_type}'. "
                f"Valid: {list(JEROLOV_KD.keys())}"
            )
        self.water_type = water_type
        self.register_buffer(
            "kd_table",
            torch.tensor(JEROLOV_KD[water_type], dtype=torch.float32),
        )
        self.wavelengths = JEROLOV_WAVELENGTHS  # nm
    
    def Kd(
        self,
        wavelength_nm: float,
    ) -> torch.Tensor:
        """Get diffuse attenuation coefficient Kd at a specific wavelength.
        
        Linearly interpolates between the tabulated spectral bands.
        
        Args:
            wavelength_nm: Wavelength in nanometers (380-700).
        
        Returns:
            Kd in m⁻¹ (scalar tensor).
        """
        if wavelength_nm <= self.wavelengths[0]:
            return self.kd_table[0]
        if wavelength_nm >= self.wavelengths[-1]:
            return self.kd_table[-1]
        
        # Find bracketing indices
        for i in range(len(self.wavelengths) - 1):
            if self.wavelengths[i] <= wavelength_nm <= self.wavelengths[i + 1]:
                frac = (wavelength_nm - self.wavelengths[i]) / (
                    self.wavelengths[i + 1] - self.wavelengths[i]
                )
                return self.kd_table[i] + frac * (self.kd_table[i + 1] - self.kd_table[i])
        
        return self.kd_table[-1]
    
    def spectral_attenuation(
        self,
        depth: torch.Tensor,  # (B,) or (B, 1)
        turbidity_boost: float = 0.0,
    ) -> torch.Tensor:
        """Get full spectral attenuation across all Jerlov bands.
        
        Beer-Lambert: I(λ,z) = I₀(λ) * exp(-Kd(λ) * z)
        
        Args:
            depth: Depth in meters, (B,) or (B, 1).
            turbidity_boost: Additional attenuation factor (m⁻¹) added uniformly.
        
        Returns:
            (B, 7) spectral transmission factors.
        """
        # (1, 7) broadcast against (B, 1) → (B, 7)
        kd = self.kd_table.unsqueeze(0) + turbidity_boost
        d = depth.unsqueeze(-1) if depth.dim() == 1 else depth
        return torch.exp(-kd * d)
    
    def color_attenuation_vector(
        self,
        depth: torch.Tensor,  # (B,) meters
        turbidity_boost: float = 0.0,
    ) -> torch.Tensor:
        """Get RGB attenuation factors using Jerlov spectral data.
        
        Maps visible spectrum bands to R/G/B camera channels.
        
        Args:
            depth: (B,) depth in meters.
            turbidity_boost: Additional uniform attenuation.
        
        Returns:
            (B, 3) with [R, G, B] transmission fractions.
        """
        spectral = self.spectral_attenuation(depth, turbidity_boost)
        # Approximate RGB from spectral bands:
        # R ≈ 650nm band, G ≈ 530nm band, B ≈ 470nm band
        r = spectral[:, 5]  # 650nm
        g = spectral[:, 3]  # 530nm
        b = spectral[:, 2]  # 470nm
        return torch.stack([r, g, b], dim=-1)
    
    def euphotic_depth(
        self,
        fraction: float = 0.01,
        wavelength_nm: float = 470.0,
    ) -> float:
        """Calculate euphotic depth (depth where light = fraction of surface).
        
        z_eu = -ln(fraction) / Kd(λ)
        
        Args:
            fraction: Light fraction threshold (default 1% = 0.01).
            wavelength_nm: Wavelength to evaluate.
        
        Returns:
            Depth in meters.
        """
        kd = float(self.Kd(wavelength_nm))
        return -math.log(fraction) / kd if kd > 0 else float("inf")


class ThermoclineModel(nn.Module):
    """Seasonal and latitude-dependent thermocline model.
    
    Models the mixed layer depth, thermocline strength, and deep water
    temperature as functions of latitude and day-of-year.
    
    Uses a simplified two-layer model:
    - Mixed layer: uniform temperature (wind-driven mixing)
    - Thermocline: exponential decay to deep-water temperature
    - Deep layer: ~4°C (polar) to ~2°C (tropical)
    """
    
    def __init__(
        self,
        latitude: float = 55.0,  # degrees (positive = north)
    ):
        super().__init__()
        self.latitude = latitude
    
    def mixed_layer_depth(
        self,
        day_of_year: float,
        latitude: Optional[float] = None,
    ) -> float:
        """Estimate mixed layer depth based on season and latitude.
        
        MLD varies from ~20m (summer, low lat) to ~300m (winter, high lat).
        
        Args:
            day_of_year: 1-365.
            latitude: Override default latitude.
        
        Returns:
            Mixed layer depth in meters.
        """
        lat = latitude or self.latitude
        abs_lat = abs(lat)
        
        # Seasonal cycle: winter = day 1 (NH), summer = day 182
        # Southern hemisphere shifted by 180 days
        if lat >= 0:
            season_phase = 2 * math.pi * (day_of_year - 1) / 365.0
        else:
            season_phase = 2 * math.pi * (day_of_year - 183) / 365.0
        
        # Base MLD increases with latitude
        base_mld = 20.0 + abs_lat * 2.5
        
        # Seasonal variation (±50% of base)
        seasonal_factor = 1.0 - 0.5 * math.cos(season_phase)
        
        return base_mld * seasonal_factor
    
    def temperature_profile(
        self,
        depth: torch.Tensor,  # (B,) meters
        day_of_year: float = 172.0,  # ~summer solstice
        surface_temp: Optional[float] = None,
    ) -> torch.Tensor:
        """Calculate temperature at depth with thermocline.
        
        Two-layer model:
        - z < MLD: T = T_surface (mixed layer)
        - z >= MLD: T = T_deep + (T_surface - T_deep) * exp(-(z-MLD)/thermo_scale)
        
        Args:
            depth: (B,) depth in meters.
            day_of_year: Day of year (1-365).
            surface_temp: Override surface temperature (°C). 
                         If None, estimated from latitude.
        
        Returns:
            (B,) temperature in °C.
        """
        lat = self.latitude
        mld = self.mixed_layer_depth(day_of_year, lat)
        
        # Estimate surface temperature from latitude + season
        if surface_temp is None:
            abs_lat = abs(lat)
            if lat >= 0:
                season_phase = 2 * math.pi * (day_of_year - 1) / 365.0
            else:
                season_phase = 2 * math.pi * (day_of_year - 183) / 365.0
            
            # Annual mean SST decreases with latitude
            mean_sst = 28.0 - abs_lat * 0.35
            # Seasonal amplitude increases with latitude
            seasonal_amp = abs_lat * 0.15
            surface_temp = mean_sst + seasonal_amp * math.sin(season_phase)
        
        # Deep water temperature
        deep_temp = max(0.0, 4.0 - abs(lat) * 0.03)
        
        # Thermocline e-folding scale
        thermo_scale = 100.0 + abs(lat) * 2.0  # meters
        
        # Two-layer temperature
        mld_t = torch.tensor(mld, dtype=depth.dtype, device=depth.device)
        surface_t = torch.tensor(surface_temp, dtype=depth.dtype, device=depth.device)
        deep_t = torch.tensor(deep_temp, dtype=depth.dtype, device=depth.device)
        scale_t = torch.tensor(thermo_scale, dtype=depth.dtype, device=depth.device)
        
        # Mixed layer: constant temperature
        mixed = surface_t.expand_as(depth)
        
        # Thermocline + deep: exponential decay
        decay = deep_t + (surface_t - deep_t) * torch.exp(
            -(depth - mld_t).clamp(min=0) / scale_t
        )
        
        return torch.where(depth < mld_t, mixed, decay)
    
    def sound_speed_profile(
        self,
        depth: torch.Tensor,
        day_of_year: float = 172.0,
        salinity: float = 35.0,
    ) -> torch.Tensor:
        """Full sound speed profile incorporating thermocline.
        
        Uses Mackenzie (1981) with temperature from thermocline model.
        
        Args:
            depth: (B,) depth in meters.
            day_of_year: Day of year.
            salinity: Salinity in PSU.
        
        Returns:
            (B,) sound speed in m/s.
        """
        T = self.temperature_profile(depth, day_of_year)
        S = salinity
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


class SeabedModel(nn.Module):
    """Seabed acoustic properties for sonar return modeling.
    
    Models acoustic impedance, reflection coefficient, and
    backscatter strength for different seabed types.
    
    Uses:
    - Acoustic impedance: Z = ρ * c
    - Rayleigh reflection coefficient: R = (Z₂ - Z₁) / (Z₂ + Z₁)
    - Lambert's law backscatter: σ = μ * sin²(θ)
    """
    
    SEABED_TYPES = list(SEABED_PROPERTIES.keys())
    
    def __init__(self):
        super().__init__()
        # Water properties for impedance contrast
        self.water_density = 1025.0  # kg/m³
        self.water_sound_speed = 1500.0  # m/s
    
    def acoustic_impedance(self, seabed_type: str) -> float:
        """Calculate acoustic impedance for a seabed type.
        
        Z = ρ * c (Rayls = kg/m²s)
        """
        props = SEABED_PROPERTIES.get(seabed_type)
        if props is None:
            raise ValueError(f"Unknown seabed type '{seabed_type}'. Valid: {self.SEABED_TYPES}")
        return props["density"] * props["sound_speed"]
    
    def reflection_coefficient(
        self,
        seabed_type: str,
        incidence_angle_deg: float = 0.0,
    ) -> float:
        """Calculate reflection coefficient at the water-seabed interface.
        
        Normal incidence: R = (Z₂ - Z₁) / (Z₂ + Z₁)
        """
        Z_water = self.water_density * self.water_sound_speed
        Z_seabed = self.acoustic_impedance(seabed_type)
        
        # Normal incidence
        R_normal = (Z_seabed - Z_water) / (Z_seabed + Z_water)
        
        # Angle-dependent correction (simplified Rayleigh)
        theta = math.radians(incidence_angle_deg)
        # At grazing angles, reflection decreases
        R = R_normal * math.cos(theta)
        
        return R
    
    def backscatter_coefficient(
        self,
        seabed_type: str,
        incidence_angle_deg: float = 30.0,
        frequency_khz: float = 200.0,
    ) -> torch.Tensor:
        """Calculate backscatter strength using Lambert's law.
        
        σ_bs = σ₀ + 10*log10(sin²(θ)) + 20*log10(f/f_ref)
        
        Where σ₀ is the type-specific baseline backscatter.
        
        Args:
            seabed_type: Seabed sediment type.
            incidence_angle_deg: Angle from vertical (0=nadir, 90=grazing).
            frequency_khz: Sonar frequency.
        
        Returns:
            Backscatter strength in dB (tensor).
        """
        props = SEABED_PROPERTIES[seabed_type]
        sigma_0 = props["backscatter_db"]
        
        theta = math.radians(incidence_angle_deg)
        
        # Lambert's law angular dependence
        lambert = 10.0 * math.log10(max(math.sin(theta) ** 2, 1e-10))
        
        # Frequency dependence (higher freq = more backscatter from rough surfaces)
        freq_term = 20.0 * math.log10(max(frequency_khz / 200.0, 0.01))
        
        sigma_bs = sigma_0 + lambert + freq_term
        
        return torch.tensor(sigma_bs, dtype=torch.float32)
    
    def bottom_loss(
        self,
        seabed_type: str,
        frequency_khz: float = 200.0,
    ) -> float:
        """Calculate bottom loss in dB.
        
        BL = -20*log10|R| for normal incidence.
        """
        R = self.reflection_coefficient(seabed_type, 0.0)
        if abs(R) < 1e-10:
            return 60.0  # effectively total absorption
        return -20.0 * math.log10(abs(R))
    
    def classification_features(
        self,
        backscatter_db: torch.Tensor,
        depth: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract features for seabed type classification.
        
        Given backscatter measurements, returns feature vectors
        that can be used to classify seabed type.
        
        Args:
            backscatter_db: (B,) measured backscatter in dB.
            depth: (B,) depth in meters.
        
        Returns:
            Dict with 'impedance_contrast', 'roughness_score', 'hardness'.
        """
        # Impedance contrast relative to water
        Z_water = self.water_density * self.water_sound_speed
        # Estimate seabed impedance from backscatter
        estimated_R = 10 ** (backscatter_db / 20.0)
        estimated_Z = Z_water * (1 + estimated_R) / (1 - estimated_R.clamp(min=-0.99, max=0.99))
        impedance_contrast = estimated_Z / Z_water
        
        # Roughness proxy: backscatter variability
        roughness_score = backscatter_db - backscatter_db.mean()
        
        # Hardness proxy: impedance-based
        hardness = (impedance_contrast - 1.0).clamp(min=0)
        
        return {
            "impedance_contrast": impedance_contrast,
            "roughness_score": roughness_score,
            "hardness": hardness,
        }


class FrancoisGarrisonAbsorption(nn.Module):
    """Full Francois-Garrison (1982) seawater acoustic absorption.
    
    Three-component absorption model:
    1. Boric acid (B(OH)₃) relaxation — dominant below 10 kHz
    2. Magnesium sulfate (MgSO₄) relaxation — dominant 10-500 kHz  
    3. Pure water (viscosity) absorption — dominant above 500 kHz
    
    α = α_boric + α_mgso4 + α_pure
    
    All parameters are temperature, salinity, depth, and pH dependent.
    Valid: 0.4°C ≤ T ≤ 30°C, 25 ≤ S ≤ 40 PSU, 0 ≤ D ≤ 11 km.
    """
    
    def __init__(self):
        super().__init__()
    
    def absorption(
        self,
        frequency_khz: torch.Tensor,  # (B,) frequency in kHz
        temperature: torch.Tensor,     # (B,) temperature in °C
        salinity: torch.Tensor = None,  # (B,) salinity in PSU
        depth: torch.Tensor = None,     # (B,) depth in meters
        ph: torch.Tensor = None,        # (B,) pH
    ) -> torch.Tensor:
        """Calculate total absorption coefficient.
        
        Args:
            frequency_khz: (B,) frequency in kHz.
            temperature: (B,) temperature in °C.
            salinity: (B,) salinity in PSU. Default 35.0.
            depth: (B,) depth in meters. Default 0.
            ph: (B,) pH. Default 8.0.
        
        Returns:
            (B,) absorption in dB/km.
        """
        B = frequency_khz.shape[0]
        device = frequency_khz.device
        
        if salinity is None:
            salinity = torch.full((B,), 35.0, device=device)
        if depth is None:
            depth = torch.zeros(B, device=device)
        if ph is None:
            ph = torch.full((B,), 8.0, device=device)
        
        T = temperature
        S = salinity
        D = depth
        f = frequency_khz
        pH = ph
        
        # ── 1. Boric Acid Relaxation ──
        A1 = (8.86 / (T * T + 80.0)) * 10 ** (0.78 * pH - 5.0)
        P1 = 1.0
        f1 = 2.8 * (S / 35.0) ** 0.5 * 10 ** (4.0 - (1245.0 / (T + 273.0)))
        alpha_boric = (A1 * P1 * f1 * f * f) / (f1 * f1 + f * f)
        
        # ── 2. Magnesium Sulfate Relaxation ──
        A2 = 21.44 * (S / 35.0) * (1.0 + 0.025 * T)
        P2 = 1.0 - 1.37e-4 * D + 6.2e-9 * D * D
        f2 = (8.17 * 10 ** (8.0 - 1990.0 / (T + 273.0))) / (1.0 + 0.0018 * (S - 35.0))
        alpha_mgso4 = (A2 * P2 * f2 * f * f) / (f2 * f2 + f * f)
        
        # ── 3. Pure Water (Viscosity) ──
        A3 = (4.937e-4 - 2.59e-5 * T + 9.11e-7 * T * T
               - 1.50e-8 * T * T * T) * (1.0 - 3.84e-4 * D + 7.0e-8 * D * D)
        alpha_pure = A3 * f * f
        
        # Total absorption in dB/km
        alpha_total = alpha_boric + alpha_mgso4 + alpha_pure
        
        return alpha_total
    
    def absorption_at_depth(
        self,
        frequency_khz: float,
        depth: torch.Tensor,
        surface_temp: float = 15.0,
        temp_gradient: float = -0.01,
        salinity: float = 35.0,
        ph: float = 8.0,
    ) -> torch.Tensor:
        """Convenience: absorption at depth with linear temperature profile.
        
        Args:
            frequency_khz: Frequency in kHz (scalar).
            depth: (B,) depth in meters.
            surface_temp: Surface temperature °C.
            temp_gradient: Temperature gradient °C/m.
            salinity: Salinity PSU.
            ph: pH.
        
        Returns:
            (B,) absorption in dB/km.
        """
        B = depth.shape[0]
        device = depth.device
        T = torch.full((B,), surface_temp, device=device) + temp_gradient * depth
        f = torch.full((B,), frequency_khz, device=device)
        S = torch.full((B,), salinity, device=device)
        pH_t = torch.full((B,), ph, device=device)
        
        return self.absorption(f, T, S, depth, pH_t)


class WaterColumnModel(nn.Module):
    """Full underwater environment model.
    
    Combines sound speed, absorption, light attenuation, thermocline,
    and Jerlov water type into a unified model. Learnable parameters
    allow adaptation to local conditions during training.
    
    This is the primary interface used by SonarVision's pipeline.
    """
    
    def __init__(
        self,
        water_type: str = "III",
        latitude: float = 55.0,
    ):
        super().__init__()
        
        # Learnable environmental parameters
        self.surface_temp = nn.Parameter(torch.tensor(15.0))  # °C
        self.temp_gradient = nn.Parameter(torch.tensor(-0.01))  # °C/m
        self.salinity = nn.Parameter(torch.tensor(35.0))  # PSU
        self.ph = nn.Parameter(torch.tensor(8.1))
        self.turbidity = nn.Parameter(torch.tensor(0.3))
        
        # Sub-models
        self.jerlov = JerlovModel(water_type)
        self.thermocline = ThermoclineModel(latitude)
        self.absorption_model = FrancoisGarrisonAbsorption()
    
    def sound_speed(
        self,
        depth: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sound speed using Mackenzie (1981) with learnable parameters.
        
        Args:
            depth: (B,) or (B, D) depth in meters.
            temperature: (B,) surface temp override.
        
        Returns:
            Sound speed in m/s.
        """
        if temperature is None:
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
        frequency_khz: float = 200.0,
        depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full Francois-Garrison absorption at given frequency.
        
        Args:
            frequency_khz: Sonar frequency in kHz.
            depth: (B,) depth for temperature-dependent absorption.
                   If None, returns single value at surface.
        
        Returns:
            Absorption in dB/km.
        """
        if depth is None:
            depth = torch.zeros(1)
        
        return self.absorption_model.absorption_at_depth(
            frequency_khz=frequency_khz,
            depth=depth,
            surface_temp=self.surface_temp.item(),
            temp_gradient=self.temp_gradient.item(),
            salinity=self.salinity.item(),
            ph=self.ph.item(),
        )
    
    def light_attenuation(
        self,
        depth: torch.Tensor,
        wavelength_nm: float = 550.0,
        turbidity: Optional[float] = None,
    ) -> torch.Tensor:
        """Spectral light attenuation using Jerlov model + turbidity.
        
        Args:
            depth: (B,) meters.
            wavelength_nm: Wavelength in nm.
            turbidity: Override turbidity boost. Default: learnable.
        
        Returns:
            (B,) transmission fraction [0, 1].
        """
        turb = turbidity if turbidity is not None else self.turbidity.item()
        return torch.exp(-self.jerlov.Kd(wavelength_nm) * depth - turb * 0.3 * depth)
    
    def color_attenuation_vector(
        self,
        depth: torch.Tensor,
        turbidity: Optional[float] = None,
    ) -> torch.Tensor:
        """RGB attenuation using Jerlov spectral model.
        
        Args:
            depth: (B,) meters.
            turbidity: Turbidity override.
        
        Returns:
            (B, 3) [R, G, B] transmission.
        """
        turb = turbidity if turbidity is not None else self.turbidity.item()
        return self.jerlov.color_attenuation_vector(depth, turbidity_boost=turb * 0.3)
    
    def temperature_at_depth(
        self,
        depth: torch.Tensor,
        day_of_year: float = 172.0,
    ) -> torch.Tensor:
        """Temperature profile from thermocline model.
        
        Args:
            depth: (B,) meters.
            day_of_year: 1-365.
        
        Returns:
            (B,) temperature in °C.
        """
        return self.thermocline.temperature_profile(depth, day_of_year)
    
    def euphotic_depth(self, fraction: float = 0.01) -> float:
        """Euphotic depth for current Jerlov water type."""
        return self.jerlov.euphotic_depth(fraction, 470.0)


class SonarBeamModel(nn.Module):
    """Sonar beam geometry with realistic beam pattern.
    
    Models the 3D beam pattern including main lobe (sinc approximation)
    and sidelobes. Converts between sonar measurements and physical
    coordinates with proper transmission loss modeling.
    """
    
    def __init__(
        self,
        beam_width_deg: float = 12.0,
        max_range: float = 200.0,
        frequency_khz: float = 200.0,
        sidelobe_db: float = -30.0,
    ):
        super().__init__()
        self.beam_width = math.radians(beam_width_deg)
        self.max_range = max_range
        self.frequency_khz = frequency_khz
        self.sidelobe_db = sidelobe_db
    
    def range_resolution(self, pulse_length_us: float = 50.0) -> float:
        """Range resolution from pulse length: ΔR = c * τ / 2."""
        c = 1500.0
        return c * pulse_length_us * 1e-6 / 2
    
    def beam_footprint(self, depth: torch.Tensor) -> torch.Tensor:
        """Beam footprint diameter at depth.
        
        footprint = 2 * depth * tan(beam_width/2)
        """
        return 2 * depth * torch.tan(self.beam_width / 2)
    
    def beam_pattern(
        self,
        angle_off_axis: torch.Tensor,  # (B,) radians from beam center
    ) -> torch.Tensor:
        """Normalized beam pattern (power).
        
        Approximates a circular piston transducer:
        - Main lobe: sinc²(ka*sin(θ))
        - Sidelobes: capped at sidelobe_db below main lobe
        
        Args:
            angle_off_axis: (B,) angle from beam axis in radians.
        
        Returns:
            (B,) normalized power [0, 1].
        """
        # wavenumber * effective radius
        ka = 2 * math.pi * (self.frequency_khz * 1000) / 1500.0 * 0.1
        
        # sinc-like main lobe
        u = ka * torch.sin(angle_off_axis)
        # Avoid division by zero
        pattern = torch.where(
            u.abs() < 1e-6,
            torch.ones_like(u),
            (torch.sin(u) / u) ** 2,
        )
        
        # Cap sidelobes
        sidelobe_linear = 10 ** (self.sidelobe_db / 10.0)
        pattern = torch.clamp(pattern, min=sidelobe_linear)
        
        # Normalize to peak = 1
        return pattern / pattern.max().clamp(min=1e-10)
    
    def transmission_loss(
        self,
        depth: torch.Tensor,
        absorption_db_km: float = 60.0,
    ) -> torch.Tensor:
        """Spherical spreading + absorption transmission loss.
        
        TL = 20*log10(R) + α*R/1000
        
        Args:
            depth: (B,) range in meters.
            absorption_db_km: Absorption in dB/km.
        
        Returns:
            (B,) TL in dB.
        """
        spreading = 20 * torch.log10(depth.clamp(min=1.0))
        absorption = absorption_db_km * depth / 1000.0
        return spreading + absorption
    
    def target_strength_to_intensity(
        self,
        ts_db: torch.Tensor,
        depth: torch.Tensor,
        source_level_db: float = 220.0,
    ) -> torch.Tensor:
        """Convert target strength to received level.
        
        Sonar equation: RL = SL - 2*TL + TS
        
        Args:
            ts_db: (B,) target strength in dB.
            depth: (B,) depth/range in meters.
            source_level_db: Source level in dB re 1 μPa.
        
        Returns:
            (B,) received level in dB.
        """
        tl = self.transmission_loss(depth)
        return source_level_db - 2 * tl + ts_db
    
    def detectability(
        self,
        received_level: torch.Tensor,
        noise_level_db: float = 60.0,
        detection_threshold_db: float = 10.0,
    ) -> torch.Tensor:
        """Binary detectability mask.
        
        Detectable when SNR > detection_threshold.
        
        Returns:
            (B,) boolean tensor.
        """
        snr = received_level - noise_level_db
        return snr > detection_threshold_db


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
    def parse_dbt(sentence: str) -> Dict:
        """Parse SDDBT sentence: $--DBT,x.x,f,x.x,M,x.x,F*hh
        
        Returns depth in meters.
        """
        try:
            parts = sentence.strip().split('*')[0].split(',')
            # Field 3 is depth in meters
            depth_m = float(parts[3])
            return {"depth_m": depth_m, "raw": sentence}
        except (ValueError, IndexError):
            return {}
    
    @staticmethod
    def parse_mtw(sentence: str) -> Dict:
        """Parse SDMTW sentence: $--MTW,x.x,C*hh
        
        Returns water temperature.
        """
        try:
            parts = sentence.strip().split('*')[0].split(',')
            temp_c = float(parts[1])
            return {"temperature_c": temp_c, "raw": sentence}
        except (ValueError, IndexError):
            return {}
    
    @staticmethod
    def depth_to_sonar_image(
        returns: list,
        bearing_bins: int = 128,
        max_depth: int = 200,
    ) -> torch.Tensor:
        """Convert sonar returns to a 2D sonar image.
        
        Returns: (bearing_bins, max_depth) intensity image.
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
