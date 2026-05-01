# SonarVision Published Packages

## Python

```bash
pip install sonar-vision-physics
```

```python
from sonar_vision_physics import compute_physics, dive_profile

# Single ping
result = compute_physics(depth=25.0)
print(result['temperature'], result['sound_speed'])

# Dive profile (0-100m)
profile = dive_profile(0, 100, 2)
```

### CLI
```bash
sonar-ping --depth 30 --format json
sonar-ping --dive --start 0 --end 100 --step 5
sonar-ping --serve --port 8081 --rate 5
```

## TypeScript

```bash
npm install @superinstance/sonar-vision-tool
```

```typescript
import { SonarVisionTool } from '@superinstance/sonar-vision-tool';

const tool = new SonarVisionTool();
const result = await tool.execute({ action: 'physics', depth: 15 });
console.log(result.sound_speed); // 1527.1
```

## Rust

```bash
cargo add constraint-theory-demo
```

## Docker

```bash
docker build -f Dockerfile.streaming -t sonar-vision-streaming .
docker compose -f docker-compose.streaming.yml up
```

## Registry Links

| Registry | Package | Version |
|----------|---------|---------|
| [PyPI](https://pypi.org/project/sonar-vision-physics/) | sonar-vision-physics | 1.0.1 |
| [npm](https://www.npmjs.com/package/@superinstance/sonar-vision-tool) | @superinstance/sonar-vision-tool | 1.0.0 |
| [crates.io](https://crates.io/crates/constraint-theory-demo) | constraint-theory-demo | 0.5.1 |
| [PyPI](https://pypi.org/project/constraint-theory/) | constraint-theory | 1.0.1 |
| [GitHub](https://github.com/SuperInstance/sonar-vision) | sonar-vision | v1.0.0 |
