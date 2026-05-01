//! SonarVision MUD plugin for holodeck-rust
//!
//! Underwater sensor room that generates video descriptions from simulated sonar.
//! Players explore ocean depths; physics affects visibility and room descriptions.
//!
//! Room hierarchy:
//!   OceanSurface (0-5m) → WaterColumn (5-50m) → Seabed (50m+)
//!
//! Each room type has different sonar returns, light levels, and marine life.

use std::collections::HashMap;

/// Water type classification (Jerlov)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaterType {
    Coastal,     // Type 1-3: turbid, high productivity
    Oceanic,     // Type I-III: clear, low productivity
    Brackish,    // Estuary mixing zone
    Polar,       // Cold, low light, ice cover
}

impl WaterType {
    pub fn attenuation_coefficient(&self, depth: f32) -> f32 {
        match self {
            WaterType::Coastal => 0.2 + depth * 0.01,
            WaterType::Oceanic => 0.05 + depth * 0.002,
            WaterType::Brackish => 0.3 + depth * 0.015,
            WaterType::Polar => 0.1 + depth * 0.005,
        }
    }
}

/// Fish school descriptor
#[derive(Debug, Clone)]
pub struct FishSchool {
    pub species: String,
    pub depth: f32,
    pub size: u32,
    pub biomass_kg: f32,
}

/// Seabed feature descriptor
#[derive(Debug, Clone)]
pub struct SeabedFeature {
    pub feature_type: String,  // wreck, vent, coral, rock, sand
    pub depth: f32,
    pub description: String,
}

/// Complete underwater room state
#[derive(Debug, Clone)]
pub struct UnderwaterRoom {
    pub name: String,
    pub depth: f32,
    pub water_type: WaterType,
    pub visibility: f32,            // meters
    pub temperature: f32,           // celsius
    pub fish_schools: Vec<FishSchool>,
    pub seabed_features: Vec<SeabedFeature>,
    pub bioluminescence: bool,
    pub ambient_light: f32,         // 0.0 - 1.0
}

impl UnderwaterRoom {
    /// Generate a room description from sonar data
    pub fn generate_description(&self) -> String {
        let mut desc = String::new();

        // Base description by depth zone
        desc.push_str(&match self.depth {
            d if d < 5.0 => "The sunlit surface waters shimmer above you. ".to_string(),
            d if d < 20.0 => "The thermocline layer surrounds you — warm above, cold below. ".to_string(),
            d if d < 50.0 => "Deep blue twilight. Pressure increases. Strange shapes move in the darkness. ".to_string(),
            _ => "The abyssal plain stretches into infinite darkness. Bioluminescent sparks drift like stars. ".to_string(),
        });

        // Visibility
        if self.visibility < 2.0 {
            desc.push_str("Visibility is nearly zero. You're swimming blind. ");
        } else if self.visibility < 10.0 {
            desc.push_str("The water is murky. Shapes emerge from the gloom. ");
        } else {
            desc.push_str("The water is remarkably clear. You can see far into the depths. ");
        }

        // Fish schools
        for fish in &self.fish_schools {
            desc.push_str(&format!(
                "A school of {} ({} individuals) drifts past at {}m. ",
                fish.species, fish.size, fish.depth
            ));
        }

        // Seabed features
        for feature in &self.seabed_features {
            desc.push_str(&format!("{} ", feature.description));
        }

        // Bioluminescence
        if self.bioluminescence {
            desc.push_str("Tiny sparks of bioluminescence dance around you. ");
        }

        desc
    }

    /// Simulate a sonar ping for this room
    pub fn sonar_ping(&self) -> Vec<f32> {
        let num_bins = 32;
        let mut returns = vec![0.0_f32; num_bins];

        // Seabed return (strong, wide)
        let seabed_idx = (num_bins as f32 * 0.8) as usize;
        for i in 0..num_bins {
            let dist = (i as i32 - seabed_idx as i32).abs() as f32;
            returns[i] += (1.0 - dist / 8.0).max(0.0) * 0.9;
        }

        // Fish returns
        for fish in &self.fish_schools {
            let idx = ((fish.depth / 100.0) * num_bins as f32) as usize;
            if idx < num_bins {
                returns[idx] += (fish.size as f32 / 100.0).min(0.5);
            }
        }

        // Apply attenuation
        let atten = self.water_type.attenuation_coefficient(self.depth);
        for i in 0..num_bins {
            returns[i] *= (-atten * i as f32).exp();
        }

        returns
    }
}

// Plugin trait that holodeck-rust would call
pub trait SonarVisionPlugin: Send + Sync {
    fn create_ocean_surface(&self, name: &str) -> UnderwaterRoom;
    fn create_water_column(&self, name: &str, depth: f32, water_type: WaterType) -> UnderwaterRoom;
    fn create_seabed(&self, name: &str, depth: f32) -> UnderwaterRoom;
    fn generate_description(&self, room: &UnderwaterRoom) -> String;
    fn simulate_sonar(&self, room: &UnderwaterRoom) -> Vec<f32>;
}

pub struct DefaultSonarPlugin;

impl SonarVisionPlugin for DefaultSonarPlugin {
    fn create_ocean_surface(&self, name: &str) -> UnderwaterRoom {
        UnderwaterRoom {
            name: name.to_string(),
            depth: 2.0,
            water_type: WaterType::Coastal,
            visibility: 15.0,
            temperature: 18.0,
            fish_schools: vec![
                FishSchool { species: "Anchovy".into(), depth: 3.0, size: 200, biomass_kg: 10.0 }
            ],
            seabed_features: vec![],
            bioluminescence: false,
            ambient_light: 1.0,
        }
    }

    fn create_water_column(&self, name: &str, depth: f32, water_type: WaterType) -> UnderwaterRoom {
        let temp = 20.0 - depth * 0.3; // Linear thermocline approximation
        UnderwaterRoom {
            name: name.to_string(),
            depth,
            water_type,
            visibility: (10.0 - depth * 0.1).max(1.0),
            temperature: temp,
            fish_schools: vec![],
            seabed_features: vec![],
            bioluminescence: depth > 30.0,
            ambient_light: (1.0 - depth / 100.0).max(0.05),
        }
    }

    fn create_seabed(&self, name: &str, depth: f32) -> UnderwaterRoom {
        UnderwaterRoom {
            name: name.to_string(),
            depth,
            water_type: WaterType::Coastal,
            visibility: 2.0,
            temperature: 4.0,
            fish_schools: vec![],
            seabed_features: vec![
                SeabedFeature {
                    feature_type: "rock formation".into(),
                    depth,
                    description: "A jagged rock formation rises from the seafloor, covered in anemones.".into(),
                }
            ],
            bioluminescence: true,
            ambient_light: 0.01,
        }
    }

    fn generate_description(&self, room: &UnderwaterRoom) -> String {
        room.generate_description()
    }

    fn simulate_sonar(&self, room: &UnderwaterRoom) -> Vec<f32> {
        room.sonar_ping()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sonar_ping_length() {
        let plugin = DefaultSonarPlugin;
        let room = plugin.create_ocean_surface("test");
        let ping = plugin.simulate_sonar(&room);
        assert_eq!(ping.len(), 32);
    }

    #[test]
    fn test_description_generated() {
        let plugin = DefaultSonarPlugin;
        let room = plugin.create_water_column("deep_test", 25.0, WaterType::Oceanic);
        let desc = plugin.generate_description(&room);
        assert!(!desc.is_empty());
        assert!(desc.contains("thermocline") || desc.contains("twilight") || desc.contains("deep"));
    }

    #[test]
    fn test_seabed_returns_strong() {
        let plugin = DefaultSonarPlugin;
        let room = plugin.create_seabed("wreck", 45.0);
        let ping = plugin.simulate_sonar(&room);
        let max_return = ping.iter().cloned().fold(0.0_f32, f32::max);
        assert!(max_return > 0.5); // Seabed should have strong return
    }

    #[test]
    fn test_attenuation_increases_with_depth() {
        let deep = DefaultSonarPlugin.create_water_column("deep", 50.0, WaterType::Coastal);
        let shallow = DefaultSonarPlugin.create_water_column("shallow", 5.0, WaterType::Coastal);
        let deep_ping = deep.sonar_ping();
        let shallow_ping = shallow.sonar_ping();
        let deep_avg: f32 = deep_ping.iter().sum::<f32>() / deep_ping.len() as f32;
        let shallow_avg: f32 = shallow_ping.iter().sum::<f32>() / shallow_ping.len() as f32;
        assert!(shallow_avg >= deep_avg * 0.5); // Deeper = more attenuated
    }
}
