"""MarineGPUBridge — connects SonarVision to the MEP protocol from marine-gpu-edge.

Receives depth sounder data via the Marine Edge Protocol (MEP),
feeds it through SonarVision's pipeline, and returns video frames.

MEP header format (16 bytes):
  - magic: uint32_t (0x4D4550)
  - type: MEPType enum
  - seq: uint32_t (monotonic)
  - len: uint32_t (payload length)
"""

import struct
import socket
import logging
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

log = logging.getLogger("sonar-vision.marine-gpu")

MEP_MAGIC = 0x4D4550  # "MEP"
MEP_PORT = 9734

# MEP type codes (matching marine-gpu-edge's MEPType enum)
MEP_SONAR_WATERFALL = 6    # Sonar waterfall data
MEP_DEPTH_PROFILE = 7      # Depth sounder profile
MEP_SENSOR_FUSED = 8       # Fused sensor data
MEP_GPU_KERNEL = 9         # GPU kernel offload request
MEP_CONSTRAINT_ALERT = 10  # Constraint violation alert


@dataclass
class MEPSonarPacket:
    """Decoded MEP packet containing sonar/depth data."""
    seq: int
    mep_type: int
    depth_data: np.ndarray      # (time_steps, bearing_bins)
    timestamps: List[float]
    water_type: str = "coastal"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MarineGPUBridge:
    """Bridge between marine-gpu-edge MEP protocol and SonarVision inference.

    Can operate in two modes:
      1. Server mode — workstation listens for Jetson connections
      2. Client mode — connects to marine-gpu-edge server
    """

    def __init__(self, host: str = "0.0.0.0", port: int = MEP_PORT, mode: str = "server"):
        self.host = host
        self.port = port
        self.mode = mode
        self.sock: Optional[socket.socket] = None
        self.conn: Optional[socket.socket] = None
        self._seq = 0

    def start(self) -> bool:
        """Start the bridge in configured mode."""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.settimeout(30.0)

            if self.mode == "server":
                self.sock.bind((self.host, self.port))
                self.sock.listen(1)
                log.info(f"MEP bridge listening on {self.host}:{self.port}")
                self.conn, addr = self.sock.accept()
                log.info(f"Connected to marine-gpu-edge node: {addr}")
                return True
            else:
                self.sock.connect((self.host, self.port))
                log.info(f"Connected to MEP server at {self.host}:{self.port}")
                self.conn = self.sock
                return True
        except socket.timeout:
            log.warning("MEP bridge connection timed out")
            return False
        except Exception as e:
            log.error(f"MEP bridge error: {e}")
            return False

    def read_packet(self) -> Optional[MEPSonarPacket]:
        """Read and decode a single MEP packet."""
        if not self.conn:
            log.error("No MEP connection")
            return None

        try:
            # Read 16-byte header
            header = self._recv_all(16)
            if not header or len(header) < 16:
                return None

            magic, mep_type, seq, length = struct.unpack("<IIII", header)

            if magic != MEP_MAGIC:
                log.warning(f"Invalid MEP magic: 0x{magic:08X}")
                return None

            # Read payload
            payload = self._recv_all(length) if length > 0 else b""
            if length > 0 and (not payload or len(payload) < length):
                return None

            return self._decode_payload(mep_type, seq, payload)

        except (socket.timeout, ConnectionResetError, BrokenPipeError) as e:
            log.warning(f"MEP connection lost: {e}")
            self.conn = None
            return None

    def send_frame(self, frame: np.ndarray, confidence: float = 0.0) -> bool:
        """Send a video frame back over MEP."""
        if not self.conn:
            return False

        try:
            payload = frame.astype(np.float32).tobytes()
            # Prepend metadata: frame shape + confidence
            meta = struct.pack("<HHHf", *frame.shape[:3], confidence)
            full_payload = meta + payload

            header = struct.pack(
                "<IIII",
                MEP_MAGIC,
                8,  # MEP_VIDEO_FRAME
                self._next_seq(),
                len(full_payload),
            )
            self._send_all(header + full_payload)
            return True
        except Exception as e:
            log.error(f"Failed to send frame: {e}")
            return False

    def close(self):
        """Close all connections."""
        if self.conn:
            self.conn.close()
        if self.sock:
            self.sock.close()
        log.info("MEP bridge closed")

    def _decode_payload(self, mep_type: int, seq: int, payload: bytes) -> MEPSonarPacket:
        """Decode payload bytes into a SonarPacket based on MEP type."""
        if mep_type == MEP_SONAR_WATERFALL:
            # Waterfall format: n_time x n_bearing float32 matrix
            arr = np.frombuffer(payload, dtype=np.float32)
            if len(arr) >= 2:
                n_time = int(arr[0])
                n_bearing = int(arr[1])
                data = arr[2:2 + n_time * n_bearing].reshape(n_time, n_bearing)
            else:
                data = arr.reshape(1, -1)
            n_time = data.shape[0]
            timestamps = [float(i) for i in range(n_time)]
            return MEPSonarPacket(seq=seq, mep_type=mep_type, depth_data=data, timestamps=timestamps)

        elif mep_type == MEP_DEPTH_PROFILE:
            arr = np.frombuffer(payload, dtype=np.float32)
            if len(arr) >= 1:
                data = arr.reshape(1, -1)
            else:
                data = np.array([[0.0]])
            return MEPSonarPacket(seq=seq, mep_type=mep_type, depth_data=data, timestamps=[0.0])

        elif mep_type == MEP_SENSOR_FUSED:
            # Fused data includes water type metadata in first 4 chars
            water_type = payload[:8].decode("ascii", errors="replace").strip("\x00")
            arr = np.frombuffer(payload[8:], dtype=np.float32)
            if len(arr) >= 2:
                n_time = int(arr[0])
                data = arr[1:1 + n_time * 3].reshape(-1, 3)  # depth, temp, salinity
            else:
                data = np.array([[0.0, 0.0, 0.0]])
            return MEPSonarPacket(
                seq=seq, mep_type=mep_type, depth_data=data,
                timestamps=[float(i) for i in range(data.shape[0])],
                water_type=water_type,
                metadata={"sensor_fused": True},
            )

        return MEPSonarPacket(
            seq=seq, mep_type=mep_type,
            depth_data=np.array([[]]),
            timestamps=[],
        )

    def _recv_all(self, n: int) -> Optional[bytes]:
        """Receive exactly n bytes."""
        chunks = []
        received = 0
        while received < n:
            try:
                chunk = self.conn.recv(n - received)
            except Exception:
                return None
            if not chunk:
                return None
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _send_all(self, data: bytes) -> bool:
        """Send all bytes."""
        try:
            self.conn.sendall(data)
            return True
        except Exception:
            return False

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq


class CUDASonarPipeline:
    """Bridge between marine-gpu-edge's CUDA output and SonarVision's inference.

    Handles the transformation from CUDA-resident sonar data to
    SonarVision's native tensor format.
    """

    def __init__(self, depth_data: np.ndarray):
        self.depth_data = depth_data

    def to_sonar_vision_tensor(self) -> np.ndarray:
        """Convert MEP-format depth data to SonarVision input tensor.

        Returns shape: (1, bearing_bins, depth_samples) normalized to [-1, 1].
        """
        data = self.depth_data.astype(np.float32)

        # Normalize to [-1, 1]
        data_min = data.min()
        data_max = data.max()
        if data_max - data_min > 1e-8:
            data = 2.0 * (data - data_min) / (data_max - data_min) - 1.0

        # Add batch dimension
        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        return data

    def infer(self) -> np.ndarray:
        """Run SonarVision inference on CUDA pipeline data.

        Returns generated video frame as (H, W, 3) uint8 array.
        """
        tensor = self.to_sonar_vision_tensor()
        # Placeholder: actual inference when model is loaded
        return tensor
