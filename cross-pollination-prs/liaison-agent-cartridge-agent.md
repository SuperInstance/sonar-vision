# Cross-Pollination PR: liaison-agent ↔ cartridge-agent

## Synergy Score: 7.0

**Shared Concepts:** standalone, fleet, bridge, orchestration, handles, agent

## Recommendation
Merge standalone, fleet, bridge between liaison-agent and cartridge-agent

## Proposed Changes

### In `liaison-agent`
- Add integration module referencing cartridge-agent
- Document connection in README

### In `cartridge-agent`
- Import concept from liaison-agent
- Add usage example

## Files to Create
- `integrations/liaison-agent-cartridge-agent.md`
- Example code showing the cross-pollination
