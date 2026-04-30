"""Usage metering for SonarVision — logs inference events for BusinessLog.ai."""

import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict

log = logging.getLogger("sonar-vision.businesslog")


@dataclass
class InferenceEvent:
    """A single SonarVision inference event."""
    timestamp: float
    model_version: str
    api_key: str
    depth_shape: str
    inference_time_ms: float
    confidence: float
    water_type: str
    output_format: str
    status: str  # "success", "error"
    error_message: Optional[str] = None


class InferenceMeter:
    """Tracks and logs inference usage for billing and analytics.

    Stores events as JSONL files for BusinessLog.ai consumption.
    """

    def __init__(self, log_dir: str = "data/usage"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_log = None

    def log_inference(self, event: InferenceEvent):
        """Record a single inference event."""
        log_path = self.log_dir / f"{time.strftime('%Y-%m-%d')}.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(asdict(event)) + "\n")
        log.debug(f"Inference logged: {event.model_version} ({event.inference_time_ms:.1f}ms)")

    def get_daily_stats(self, date_str: Optional[str] = None) -> Dict:
        """Get inference statistics for a given date."""
        if date_str is None:
            date_str = time.strftime("%Y-%m-%d")
        log_path = self.log_dir / f"{date_str}.jsonl"
        if not log_path.exists():
            return {"date": date_str, "total": 0, "avg_latency_ms": 0.0}

        latencies = []
        statuses = {}
        water_types = {}
        with open(log_path) as f:
            for line in f:
                event = json.loads(line)
                latencies.append(event.get("inference_time_ms", 0))
                status = event.get("status", "unknown")
                statuses[status] = statuses.get(status, 0) + 1
                wt = event.get("water_type", "unknown")
                water_types[wt] = water_types.get(wt, 0) + 1

        return {
            "date": date_str,
            "total": sum(statuses.values()),
            "success": statuses.get("success", 0),
            "error": statuses.get("error", 0),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "water_types": water_types,
        }
