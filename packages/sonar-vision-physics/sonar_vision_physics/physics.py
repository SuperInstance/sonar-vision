"""FLUX 9-opcode marine physics engine. Deterministic underwater acoustics."""

import math

WATER_TYPES = {0: 'Coastal', 1: 'Oceanic Type II', 2: 'Oceanic Type IB', 3: 'Clear Oceanic'}
SEDIMENT_NAMES = {0: 'mud', 1: 'sand', 2: 'gravel', 3: 'rock', 4: 'seagrass'}
SEDIMENT_REFLECT = {0: 0.3, 1: 0.5, 2: 0.7, 3: 0.85, 4: 0.2}


class FluxPhysics:
    """Deterministic underwater physics engine (FLUX 9-opcode model v3.1).
    
    All 9 physics operations produce bit-identical outputs for identical inputs
    across all interpreter instances.
    """
    
    def compute(self, depth, chl=5.0, season=0, sediment=1, wl=480.0, sal=35.0):
        """Compute all 9 FLUX physics parameters for a given depth/environment."""
        # PHY_JERLOV (B2): water type
        wt = 0 if chl > 10 else 1 if chl > 1 else 2 if chl > 0.1 else 3
        
        # PHY_ABSORB (B0): Francois-Garrison absorption
        wa = wl / 1000.0
        if wt <= 1:
            absorp = 0.04 + 0.96 * math.exp(-((wa - 0.42)**2) / (2 * 0.02**2))
        elif wt == 2:
            absorp = 0.3 + 0.9 * math.exp(-((wa - 0.48)**2) / (2 * 0.03**2))
        else:
            absorp = 0.02 + 0.51 * math.exp(-((wa - 0.42)**2) / (2 * 0.015**2))
        
        # PHY_SCATTER (B1): Rayleigh-like
        ns = 0.002 * (480.0 / wl)**4.3
        scat = ns * max(0.01, 1.0 - depth * 0.003)
        
        # PHY_THERMO (B3): thermocline gradient
        tc, tw = (15.0, 5.0) if season == 0 else (40.0, 15.0)
        st, dt = (22.0, 4.0) if season == 0 else (8.0, 4.0)
        temp = dt + (st - dt) * math.exp(-((depth - tc)**2) / (2 * tw**2))
        dtdz = -(st - dt) * (depth - tc) / (tw**2) * math.exp(-((depth - tc)**2) / (2 * tw**2))
        
        # PHY_SEABED (B4): seabed reflectivity
        seabed = SEDIMENT_REFLECT[sediment] * math.exp(-depth * 0.003)
        
        # PHY_ATTEN (B5): total attenuation
        atten = absorp + scat
        
        # PHY_VISIB (B6): Secchi visibility
        vis = min(depth, 1.7 / max(atten, 0.001))
        
        # PHY_SOUNDV (B7): Mackenzie sound speed
        ss = (1449.2 + 4.6*temp - 0.055*temp**2 + 0.00029*temp**3 +
              (1.34 - 0.01*temp)*(sal - 35) + 0.016*depth)
        
        # PHY_REFRAC (B8): Snell's law refraction
        v_ratio = ss / 1480.0
        theta = math.radians(30.0)
        st2 = math.sin(theta) * (1.0 / v_ratio)
        refrac = 90.0 if st2 > 1.0 else math.degrees(math.asin(st2))
        
        return {
            'depth': round(depth, 1),
            'water_type': wt,
            'water_type_name': WATER_TYPES[wt],
            'temperature': round(temp, 2),
            'dTdz': round(dtdz, 4),
            'absorption': round(absorp, 4),
            'scattering': round(scat, 4),
            'attenuation': round(atten, 3),
            'visibility': round(vis, 2),
            'seabed_reflectivity': round(seabed, 4),
            'sound_speed': round(ss, 1),
            'refraction_deg': round(refrac, 2),
            'sediment': SEDIMENT_NAMES[sediment],
        }


def compute_physics(depth, chl=5.0, season=0, sediment=1, wl=480.0, sal=35.0):
    """Convenience function: compute all physics for one depth point."""
    return FluxPhysics().compute(depth, chl, season, sediment, wl, sal)


def dive_profile(start=0, end=100, step=5, chl=5.0, season=0, sediment=1):
    """Compute physics for a range of depths (full dive profile)."""
    depths = list(range(start, end + 1, step))
    physics = FluxPhysics()
    return [physics.compute(d, chl=max(0.05, chl - d * 0.12),
                            season=season, sediment=sediment) for d in depths]
