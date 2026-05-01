# SonarVision → FLUX: Underwater Physics as Bytecode

## Concept
Port the underwater physics engine (Francois-Garrison absorption, Jerlov water types,
thermocline, seabed models) to FLUX bytecode for deterministic, edge-deployable
underwater simulation.

## Why FLUX?
- **Deterministic**: Same bytecode → same result everywhere (critical for simulation)
- **Edge-native**: Runs on Jetson, Raspberry Pi, browser
- **A2A-native**: Agents can call physics routines via FLUX A2A protocol
- **Composable**: Physics primitives compose into larger simulations

## Proposed Bytecode Primitives

| Opcode | Name | Description |
|--------|------|-------------|
| `PHY_ABSORB` | Absorption | Francois-Garrison absorption coefficient |
| `PHY_SCATTER` | Scattering | Compute scattering at wavelength/depth |
| `PHY_JERLOV` | Jerlov Type | Classify water type from optical data |
| `PHY_THERMO` | Thermocline | Compute thermal gradient profile |
| `PHY_SEABED` | Seabed Return | Model seabed acoustic return |
| `PHY_ATTEN` | Total Attenuation | Combined absorption + scattering |

## FLUX Program Example (Pseudocode)

```asm
; Compute underwater visibility at 15m in coastal water
LOAD_R      depth, 15.0       ; depth in meters
LOAD_R      wavelength, 480   ; blue-green light (nm)
LOAD_STR    water_type, "coastal"

; Get absorption coefficient
PHY_ABSORB  R0, wavelength, water_type
; Get scattering coefficient
PHY_SCATTER R1, wavelength, depth
; Classify Jerlov type
PHY_JERLOV  R2, depth, R0, R1
; Compute total attenuation
PHY_ATTEN   R3, depth, R0, R1

; Result: R3 = total attenuation
; Used by SonarVision to predict visibility
STORE       visibility_r3, R3
```

## Integration Points
- Constraint theory snapped coefficients (no float drift)
- Federated learning: deployment-specific calibration via FLUX parameters
- PLATO tiles: store pre-computed absorption tables as bytecode constants
