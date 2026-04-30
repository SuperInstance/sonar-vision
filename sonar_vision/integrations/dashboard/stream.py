"""SonarTelemetryStream — WebSocket broadcast for fleet dashboard."""

import json
import asyncio
import logging
from typing import Set, Optional
import numpy as np

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

log = logging.getLogger("sonar-vision.dashboard")


class SonarTelemetryStream:
    """Broadcasts SonarVision inference frames to connected dashboard clients."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9090):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self._server = None
        self._running = False

    async def _handler(self, websocket):
        self.clients.add(websocket)
        log.info(f"Dashboard client connected ({len(self.clients)} total)")
        try:
            async for _ in websocket:
                pass  # Keep connection open
        finally:
            self.clients.remove(websocket)

    def start(self):
        """Start the WebSocket broadcast server."""
        if not HAS_WEBSOCKETS:
            log.warning("websockets not installed — dashboard stream unavailable")
            return

        async def _start():
            self._server = await websockets.serve(self._handler, self.host, self.port)
            self._running = True
            log.info(f"Dashboard stream on ws://{self.host}:{self.port}")

        asyncio.create_task(_start())

    def broadcast_frame(self, frame: np.ndarray, metadata: Optional[dict] = None):
        """Broadcast a video frame to all connected dashboard clients."""
        if not self.clients:
            return

        message = {
            "type": "sonar_frame",
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "timestamp": __import__("time").time(),
        }
        if metadata:
            message.update(metadata)

        # Send metadata as JSON
        coros = [client.send(json.dumps(message)) for client in self.clients]
        asyncio.create_task(asyncio.gather(*coros, return_exceptions=True))

    def stop(self):
        """Stop the broadcast server."""
        self._running = False
        if self._server:
            self._server.close()
