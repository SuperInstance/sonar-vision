# SonarVision Plugins

These plugins connect SonarVision to the fleet ecosystem:

| Plugin | Target | Language | Status |
|--------|--------|----------|--------|
| marine-gpu | marine-gpu-edge (CUDA) | Python | Done |
| holodeck | holodeck-rust (MUD) | Rust proposal | Doc only |
| dashboard | cocapn-dashboard (JS) | Proposal | Doc only |

## For holodeck-rust integration:

The SonarVision MUD plugin would create an underwater sensor room:
- Depth sounder room that generates underwater video descriptions
- Players explore ocean depths, physics affects visibility
- See `plugins/sonar-vision-mud-plugin.md` for the proposal.
