"""WebSocket streaming server. Pure stdlib, 0 external deps."""

import asyncio
import json
import math
import struct
import signal

from .physics import FluxPhysics


class DiveSimulator:
    def __init__(self, max_depth=100.0, rate_hz=5.0, descent_speed=3.0):
        self.physics = FluxPhysics()
        self.max_depth = max_depth
        self.rate_hz = rate_hz
        self.descent_speed = descent_speed
        self.current_depth = 0.0
        self.descending = True
        self.paused = False
        self.samples = 0

    def step(self):
        if not self.paused:
            if self.descending:
                self.current_depth += self.descent_speed / self.rate_hz
                if self.current_depth >= self.max_depth:
                    self.current_depth = self.max_depth
                    self.descending = False
            else:
                self.current_depth -= self.descent_speed / self.rate_hz
                if self.current_depth <= 0:
                    self.current_depth = 0.0
                    self.descending = True

        self.current_depth = max(0.0, min(self.max_depth, self.current_depth))
        self.samples += 1
        chl = max(0.05, 8.0 - self.current_depth * 0.12)
        frame = self.physics.compute(self.current_depth, chl=chl)
        frame['frame_id'] = self.samples
        return frame

    def reset(self):
        self.current_depth = 0.0
        self.descending = True
        self.paused = False
        self.samples = 0

    def goto(self, depth):
        self.current_depth = max(0.0, min(self.max_depth, depth))


class WebSocketConnection:
    WS_GUID = b'258EAFA5-E914-47DA-95CA-5AB9DC11B85B'

    def __init__(self, reader, writer):
        self.reader = reader
        self.writer = writer
        self.open = True

    @classmethod
    async def handshake(cls, reader, writer):
        data = await reader.readuntil(b'\r\n\r\n')
        request = data.decode('utf-8', errors='replace')
        key = None
        for line in request.split('\r\n'):
            if line.lower().startswith('sec-websocket-key:'):
                key = line.split(':', 1)[1].strip()
                break
        if not key:
            raise ValueError('Missing Sec-WebSocket-Key')

        import hashlib, base64
        accept = base64.b64encode(hashlib.sha1(key.encode() + cls.WS_GUID).digest()).decode()
        response = (
            'HTTP/1.1 101 Switching Protocols\r\n'
            'Upgrade: websocket\r\nConnection: Upgrade\r\n'
            f'Sec-WebSocket-Accept: {accept}\r\n'
            '\r\n'
        )
        writer.write(response.encode())
        await writer.drain()
        return cls(reader, writer)

    async def recv(self):
        try:
            header = await self.reader.readexactly(2)
            b0, b1 = header[0], header[1]
            opcode = b0 & 0x0F
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack('!H', await self.reader.readexactly(2))[0]
            elif length == 127:
                length = struct.unpack('!Q', await self.reader.readexactly(8))[0]
            payload = await self.reader.readexactly(length)
            if opcode == 0x08:
                self.open = False
                return None
            if opcode == 0x01:
                return payload.decode('utf-8')
        except (asyncio.IncompleteReadError, ConnectionResetError):
            self.open = False
            return None

    async def send(self, message):
        if not self.open:
            return
        try:
            payload = message.encode('utf-8')
            header = bytearray([0x81])
            length = len(payload)
            if length < 126:
                header.append(length)
            elif length < 65536:
                header.extend([126, *struct.pack('!H', length)])
            else:
                header.extend([127, *struct.pack('!Q', length)])
            self.writer.write(bytes(header) + payload)
            await self.writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            self.open = False

    async def close(self):
        if self.open:
            self.open = False
            try:
                self.writer.close()
            except:
                pass


class StreamingServer:
    def __init__(self, host='0.0.0.0', port=8081, rate_hz=5.0, max_depth=100.0):
        self.host = host
        self.port = port
        self.rate_hz = rate_hz
        self.simulator = DiveSimulator(max_depth=max_depth, rate_hz=rate_hz)
        self.connections = []
        self.running = True

    async def handle_client(self, reader, writer):
        conn = None
        try:
            conn = await WebSocketConnection.handshake(reader, writer)
            self.connections.append(conn)
            frame = self.simulator.step()
            await conn.send(json.dumps(frame))
            while self.running and conn.open:
                msg = await conn.recv()
                if msg is None:
                    break
                msg = msg.strip().lower()
                if msg == 'reset':
                    self.simulator.reset()
                elif msg == 'pause':
                    self.simulator.paused = True
                elif msg in ('resume', 'start'):
                    self.simulator.paused = False
                elif msg.startswith('goto:'):
                    try:
                        self.simulator.goto(float(msg.split(':')[1]))
                    except ValueError:
                        pass
        except Exception:
            pass
        finally:
            if conn and conn in self.connections:
                self.connections.remove(conn)
            try:
                writer.close()
            except:
                pass

    async def stream_data(self):
        interval = 1.0 / self.rate_hz
        while self.running:
            frame = self.simulator.step()
            payload = json.dumps(frame)
            dead = []
            for conn in self.connections[:]:
                if conn.open:
                    await conn.send(payload)
                else:
                    dead.append(conn)
            for conn in dead:
                if conn in self.connections:
                    self.connections.remove(conn)
            await asyncio.sleep(max(0, interval - 0.001))

    async def start(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        print(f'[*] SonarVision WebSocket on ws://{self.host}:{self.port}')
        print(f'[*] Rate: {self.rate_hz} Hz | Commands: reset, pause, resume, goto:<depth>')
        stream_task = asyncio.create_task(self.stream_data())
        stop = asyncio.Future()
        def shutdown():
            if not stop.done():
                stop.set_result(True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                asyncio.get_event_loop().add_signal_handler(sig, shutdown)
            except NotImplementedError:
                pass
        async with server:
            await stop
        self.running = False
        stream_task.cancel()
        for conn in self.connections[:]:
            await conn.close()
        self.connections.clear()
