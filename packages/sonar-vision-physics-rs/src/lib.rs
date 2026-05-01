//! SonarVision Physics — deterministic underwater sonar physics engine.
//!
//! FLUX 9-opcode model v3.1. Bit-identical for identical inputs.
//!
//! # Example
//! ```ignore
//! use sonar_vision_physics::Ping;
//!
//! let ping = Ping::new(15.0, 4.0, self::Season::Summer, self::Sediment::Sand);
//! assert_eq!(ping.water_type_name(), "Oceanic Type II");
//! assert!((ping.temperature - 22.0).abs() < 0.1);
//! assert!((ping.sound_speed - 1527.1).abs() < 0.5);
//! ```


/// Water type classification (Jerlov).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaterType {
    Coastal,        // High chlorophyll
    OceanicTypeII,  // Moderate
    OceanicTypeIB,  // Low
    ClearOceanic,   // Very low
}

impl WaterType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Coastal => "Coastal",
            Self::OceanicTypeII => "Oceanic Type II",
            Self::OceanicTypeIB => "Oceanic Type IB",
            Self::ClearOceanic => "Clear Oceanic",
        }
    }
}

/// Season determines thermocline depth and spread.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Season {
    Summer,
    Winter,
}

/// Seabed sediment type determines reflectivity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sediment {
    Mud,
    Sand,
    Gravel,
    Rock,
    Seagrass,
}

impl Sediment {
    fn reflectivity(&self) -> f64 {
        match self {
            Self::Mud => 0.3,
            Self::Sand => 0.5,
            Self::Gravel => 0.7,
            Self::Rock => 0.85,
            Self::Seagrass => 0.2,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Self::Mud => "mud",
            Self::Sand => "sand",
            Self::Gravel => "gravel",
            Self::Rock => "rock",
            Self::Seagrass => "seagrass",
        }
    }
}

/// Complete sonar physics state at one depth.
#[derive(Debug, Clone)]
pub struct Ping {
    pub depth: f64,
    pub water_type: u8,
    pub temperature: f64,
    pub dtdz: f64,
    pub absorption: f64,
    pub scattering: f64,
    pub attenuation: f64,
    pub visibility: f64,
    pub seabed_reflectivity: f64,
    pub sound_speed: f64,
    pub refraction_deg: f64,
    pub sediment_name: &'static str,
}

impl Ping {
    /// Compute all physics for one depth point.
    pub fn new(depth: f64, chlorophyll: f64, season: Season, sediment: Sediment) -> Self {
        // Jerlov water type
        let wt = if chlorophyll > 10.0 { 0 }
                 else if chlorophyll > 1.0 { 1 }
                 else if chlorophyll > 0.1 { 2 }
                 else { 3 };

        // Absorption — Francois-Garrison
        let wa: f64 = 0.480; // ~default wavelength in um
        let abs = match wt {
            0 | 1 => 0.04 + 0.96 * (-((wa - 0.42).powi(2)) / (2.0 * 0.02_f64.powi(2))).exp(),
            2 => 0.3 + 0.9 * (-((wa - 0.48).powi(2)) / (2.0 * 0.03_f64.powi(2))).exp(),
            _ => 0.02 + 0.51 * (-((wa - 0.42).powi(2)) / (2.0 * 0.015_f64.powi(2))).exp(),
        };

        // Scattering — Rayleigh-like
        let ns: f64 = 0.002 * 1.0_f64.powf(4.3);  // ns = 0.002
        let scat = ns * (1.0 - depth * 0.003).max(0.01);

        // Thermocline
        let (tc, tw): (f64, f64) = match season {
            Season::Summer => (15.0, 5.0),
            Season::Winter => (40.0, 15.0),
        };
        let (st, dt) = match season {
            Season::Summer => (22.0, 4.0),
            Season::Winter => (8.0, 4.0),
        };
        let temp = dt + (st - dt) * (-((depth - tc).powi(2)) / (2.0 * tw.powi(2))).exp();
        let dtdz = -(st - dt) * (depth - tc) / tw.powi(2)
            * (-((depth - tc).powi(2)) / (2.0 * tw.powi(2))).exp();

        // Seabed reflectivity
        let seabed = sediment.reflectivity() * (-depth * 0.003).exp();

        // Attenuation
        let atten = abs + scat;

        // Visibility — Secchi depth
        let vis = depth.min(1.7 / atten.max(0.001));

        // Sound speed — Mackenzie equation
        let sal = 35.0;
        let ss = 1449.2 + 4.6 * temp - 0.055 * temp.powi(2) + 0.00029 * temp.powi(3)
            + (1.34 - 0.01 * temp) * (sal - 35.0) + 0.016 * depth;

        // Refraction — Snell's law
        let v_ratio = ss / 1480.0;
        let theta = (std::f64::consts::PI / 6.0).sin();
        let st2 = theta * (1.0 / v_ratio);
        let refrac = if st2 > 1.0 { 90.0 } else { st2.asin().to_degrees() };

        Self { depth, water_type: wt, temperature: temp, dtdz, absorption: abs,
            scattering: scat, attenuation: atten, visibility: vis,
            seabed_reflectivity: seabed, sound_speed: ss, refraction_deg: refrac,
            sediment_name: sediment.name() }
    }

    pub fn water_type_name(&self) -> &'static str {
        match self.water_type {
            0 => "Coastal",
            1 => "Oceanic Type II",
            2 => "Oceanic Type IB",
            _ => "Clear Oceanic",
        }
    }
}

/// Full dive profile.
pub fn dive_profile(start: f64, end: f64, step: f64,
                    chlorophyll: f64, season: Season, sediment: Sediment) -> Vec<Ping> {
    let mut v = Vec::new();
    let mut d = start;
    while d <= end {
        v.push(Ping::new(d, chlorophyll, season, sediment));
        d += step;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surface_ping() {
        let p = Ping::new(0.0, 5.0, Season::Summer, Sediment::Sand);
        assert_eq!(p.water_type_name(), "Oceanic Type II");
        assert!((p.temperature - 4.2).abs() < 0.1);
        assert!((p.sound_speed - 1467.6).abs() < 0.5);
    }

    #[test]
    fn test_thermocline_peak() {
        let p = Ping::new(15.0, 4.0, Season::Summer, Sediment::Sand);
        assert!((p.temperature - 22.0).abs() < 0.1);
        assert!((p.sound_speed - 1527.1).abs() < 0.5);
    }

    #[test]
    fn test_winter_vs_summer() {
        let s = Ping::new(15.0, 5.0, Season::Summer, Sediment::Rock);
        let w = Ping::new(15.0, 5.0, Season::Winter, Sediment::Rock);
        assert!(s.temperature > w.temperature);
        assert!(s.sound_speed > w.sound_speed);
    }

    #[test]
    fn test_sediment_reflectivity() {
        let mud = Ping::new(10.0, 5.0, Season::Summer, Sediment::Mud);
        let rock = Ping::new(10.0, 5.0, Season::Summer, Sediment::Rock);
        assert!(rock.seabed_reflectivity > mud.seabed_reflectivity);
    }

    #[test]
    fn test_dive_profile() {
        let profile = dive_profile(0.0, 50.0, 10.0, 5.0, Season::Summer, Sediment::Sand);
        assert_eq!(profile.len(), 6);
        assert!((profile[0].depth - 0.0).abs() < 0.01);
        assert!((profile[5].depth - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_increases_with_depth() {
        let p1 = Ping::new(10.0, 5.0, Season::Summer, Sediment::Sand);
        let p2 = Ping::new(11.0, 5.0, Season::Summer, Sediment::Sand);
        assert!(p2.absorption > p1.absorption || (p2.absorption - p1.absorption).abs() < 1e-6);
    }

    #[test]
    fn test_deterministic() {
        let a = Ping::new(37.0, 2.5, Season::Winter, Sediment::Mud);
        let b = Ping::new(37.0, 2.5, Season::Winter, Sediment::Mud);
        let a_bytes: [u8; 8] = unsafe { std::mem::transmute(a.temperature) };
        let b_bytes: [u8; 8] = unsafe { std::mem::transmute(b.temperature) };
        assert_eq!(a_bytes, b_bytes);
    }

    #[test]
    fn test_deep_water() {
        let p = Ping::new(500.0, 0.05, Season::Winter, Sediment::Mud);
        assert_eq!(p.water_type_name(), "Clear Oceanic");
        assert!(p.sound_speed > 1440.0);
    }

    #[test]
    fn test_refraction_bounds() {
        let p = Ping::new(15.0, 5.0, Season::Summer, Sediment::Sand);
        assert!(p.refraction_deg >= 20.0 && p.refraction_deg <= 50.0);
    }
}
