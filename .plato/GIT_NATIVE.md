# SonarVision Git-Native Protocol

This repo follows the Cocapn fleet's git-native communication pattern.

## Vessel Repo

**Primary vessel:** `SuperInstance/sonar-vision`
- All source code, configs, tests
- Git history IS the development timeline
- Each commit = one logical change (pushed after every feature)

## Fleet Knowledge Sync

Tiles and knowledge are distributed through:

1. **PLATO Rooms** (HTTP API at `http://147.224.38.131:8847`):
   - `sonar-vision-architecture` — system design, pipeline, modules
   - `sonar-vision-physics` — underwater physics models
   - `sonar-vision-gct` — GCT streaming aggregator internals
   - `sonar-vision-self-supervision` — depth-weighted loss, augmentation

2. **Fleet Knowledge Base** (`cocapn/fleet-knowledge`):
   - Tiles synced to `tiles/sonar-vision-*.jsonl`
   - Blocked: cocapn PAT expired (401)

3. **For-Fleet Directory** (`for-fleet/`):
   - I2I bottles for fleet distribution
   - Deliverables, blockers, status

## I2I Protocol

SonarVision delivers to the fleet via I2I bottles:

```
[I2I:DELIVERY] Forgemaster → Fleet — Subject

Deliverables:
- What was shipped
- Where it lives

Blockers:
- What's blocking next steps

Status: COMPLETE/IN PROGRESS/BLOCKED
```

## Commit Convention

- `feat:` — New feature or module
- `fix:` — Bug fix or correction
- `docs:` — Documentation
- `test:` — Tests
- `refactor:` — Code cleanup

## Push Cadence

Per Casey's instruction: push after EVERY feature.
No batching commits. Each logical unit gets its own commit + push.

## Recovery

If context is lost:
1. Read `.plato/README.md` — integration overview
2. Read `.plato/tiles/` — PLATO knowledge tiles
3. Read `ARCHITECTURE.md` — system design
4. Read `configs/` — experiment configurations
5. Read `git log --oneline` — development timeline
