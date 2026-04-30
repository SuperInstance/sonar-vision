"""Cocapn Dashboard integration — live sonar telemetry for fleet monitoring.

Provides WebSocket-streaming depth-to-video for cocapn-dashboard widgets.

Usage:
    from sonar_vision.integrations.dashboard import SonarTelemetryStream

    stream = SonarTelemetryStream(port=9090)
    stream.start()
    stream.broadcast_frame(video_frame)  # every inference tick
"""
