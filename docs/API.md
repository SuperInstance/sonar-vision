# SonarVision API Reference

> **Version:** 1.0.0 | **Base URL:** `http://localhost:8000` | **Protocol:** REST + WebSocket

## Overview

SonarVision exposes a RESTful API for depth-to-video prediction, model management, and system monitoring. All responses are JSON. Authentication uses API keys via the `X-API-Key` header.

---

## Authentication

```http
X-API-Key: sk-sonar-vision-<your-key>
```

Keys are generated via `/api/v1/auth/keys` (admin only).

---

## Endpoints

### Predictions

#### `POST /api/v1/predict`
Run inference on depth sounder data.

**Request Body:**
```json
{
  "depth_data": [[1.2, 2.3, ...], [4.5, 5.6, ...]],
  "timestamps": ["2026-04-30T10:00:00Z", "2026-04-30T10:00:01Z"],
  "model_id": "sonar-vision-v1",
  "water_type": "coastal",
  "output_format": "video"
}
```

**Response:**
```json
{
  "prediction_id": "pred_a1b2c3d4",
  "video_url": "/api/v1/predictions/pred_a1b2c3d4/output.mp4",
  "confidence": 0.92,
  "metadata": {
    "model": "sonar-vision-v1",
    "inference_time_ms": 45
  }
}
```

#### `GET /api/v1/predictions/{id}`
Retrieve a prediction result.

#### `GET /api/v1/predictions/{id}/output`
Download the generated video.

#### `DELETE /api/v1/predictions/{id}`
Remove a prediction.

### Models

#### `GET /api/v1/models`
List available models.

#### `POST /api/v1/models`
Register a new model checkpoint.

#### `GET /api/v1/models/{id}`
Get model metadata and metrics.

#### `POST /api/v1/models/{id}/train`
Trigger a training run.

### Federated Learning

#### `POST /api/v1/federated/aggregate`
Trigger global model aggregation from local checkpoints.

#### `GET /api/v1/federated/status`
Get federation status and peer counts.

### System

#### `GET /api/v1/health`
Health check endpoint.

#### `GET /api/v1/metrics`
Prometheus-compatible metrics.

#### `GET /api/v1/config`
Current system configuration.

---

## WebSocket API

### `ws://localhost:8000/ws/stream`
Stream real-time predictions.

**Message Format:**
```json
{
  "type": "depth_frame",
  "data": [[1.2, 2.3, ...]],
  "timestamp": "2026-04-30T10:00:00Z"
}
```

---

## SDK Examples

### Python
```python
from sonar_vision import SonarVisionClient

client = SonarVisionClient(api_key="sk-...")
result = client.predict(depth_data=my_data)
result.save_video("output.mp4")
```

### curl
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-..." \
  -d '{"depth_data": [[1.2, 2.3]], "model_id": "sonar-vision-v1"}'
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `400` | Bad request — invalid parameters |
| `401` | Unauthorized — missing or invalid API key |
| `404` | Resource not found |
| `429` | Rate limit exceeded |
| `500` | Internal server error |

---

## Rate Limiting

- Free tier: 100 requests/hour
- Pro tier: 10,000 requests/hour
- Enterprise: Custom

---

## Versioning

The API is versioned via URL path (`/api/v1/`). Breaking changes increment the major version. Deprecated endpoints return a `Sunset` header with the deprecation date.
