# SonarVision PLATO Integration Layer

This directory contains PLATO tiles, I2I bottles, and flux programs
that integrate SonarVision with the Cocapn fleet knowledge system.

## Integration Points

1. **PLATO Tiles** — SonarVision knowledge fragments submitted to PLATO rooms
2. **I2I Bottles** — SonarVision status and deliverables shared with fleet
3. **Flux Programs** — Data pipeline bytecode for sonar preprocessing
4. **Git-Native** — All work tracked via vessel repos and fleet knowledge

## Fleet Integration

SonarVision connects to the Cocapn ecosystem through:

| System | Connection |
|--------|-----------|
| PLATO | Tiles submitted to `sonar-vision-*` rooms |
| I2I | Bottles in `for-fleet/` with sonar deliverables |
| Flux | `flux-bytecode/sonar-preprocess.flux` for NMEA→image |
| Git | `SuperInstance/sonar-vision` vessel repo |
| Fleet Knowledge | Tiles synced to `cocapn/fleet-knowledge` |
