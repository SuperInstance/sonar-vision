# SonarVision Holodeck MUD Plugin

## Overview
An underwater sensor room for holodeck-rust that uses SonarVision to generate 
underwater video descriptions from simulated depth sounder data.

## Room Types

### `OceanSurface`
- Light, weather, surface conditions
- Connects to `WaterColumn` rooms

### `WaterColumn`
- Depth parameter, thermocline layers, water type (Jerlov classification)
- Contains: plankton layers, fish schools, thermal gradients
- Visibility affected by: depth, turbidity, ambient light

### `Seabed`
- Benthic terrain map, sediment type, structure
- Contains: wrecks, vents, coral formations

## MUD Integration

```rust
// Proposed plugin structure for holodeck-rust
pub struct SonarVisionPlugin {
    sonar_simulator: DepthSounderSim,
    physics_engine: UnderwaterPhysics,
    connection: SonarVisionHTTPClient,  // calls SonarVision API
}

impl Plugin for SonarVisionPlugin {
    fn on_player_enter(&self, player: &Player, room: &mut Room) {
        // Generate underwater description from sonar data
        let depth = player.depth();
        let water_type = room.get_property("water_type");
        let description = self.sonar_simulator.render_description(depth, water_type);
        player.send_description(description);
    }
}
```

## Room Descriptions

Players at different depths see different things:
- **Surface (0-5m):** Sunlight, waves, near-surface plankton
- **Thermocline (5-20m):** Temperature gradient, fish schools, visibility shift
- **Deep (20-50m):** Bioluminescence, larger predators, pressure effects
- **Benthic (50m+):** Seabed features, wrecks, hydrothermal vents

## Integration with SonarVision

The MUD room generates sonar returns based on room contents, then sends them to 
the SonarVision API for video prediction. The resulting frame is sent as a 
text description or (for WebSocket clients) as a rendered image.

## Fleet Agent Interaction

- Agents with `sonar-vision` skill can interpret underwater environments
- SonarVision predictions affect agent pathfinding (avoid obstacles)
- Federated learning: agents exploring different rooms improve model accuracy
