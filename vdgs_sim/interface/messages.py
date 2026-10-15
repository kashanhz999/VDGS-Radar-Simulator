"""Sensor-to-VDGS messages.

The JSON layout follows the detection format reported for a commercial
radar VDGS in prior work (apiVersion / data / tracks, with position, speed,
size and SNR quality per track), extended with a frame counter and a
sensor timestamp. Position x is range along the stand centreline.
"""
from __future__ import annotations

import json
import socket
import time

from ..processing.range_doppler import Detection


def encode(frame_id: int, t_sensor: float, dets: list[Detection]) -> bytes:
    tracks = [{
        "id": i,
        "position": {"x": round(d.range_m, 4), "y": 0.0, "z": 0.0},
        "speed": {"x": round(d.velocity_mps, 4), "y": 0.0, "z": 0.0},
        "size": {"x": 0, "y": 0, "z": 0},
        "quality": {"SNR": round(d.snr_db, 2)},
    } for i, d in enumerate(dets)]
    msg = {"apiVersion": "0.1", "frame": frame_id, "timestamp": round(t_sensor, 4),
           "data": {"msgType": "reply", "status": "running", "tracks": tracks}}
    return json.dumps(msg, separators=(",", ":")).encode()


def decode(payload: bytes) -> tuple[int, float, list[Detection]]:
    m = json.loads(payload)
    dets = [Detection(t["position"]["x"], t["speed"]["x"], t["quality"]["SNR"])
            for t in m["data"]["tracks"]]
    return m["frame"], m["timestamp"], dets


class UdpLink:
    """Loopback UDP link; measures wall-clock send-to-receive latency."""

    def __init__(self, port: int = 0):
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.bind(("127.0.0.1", port))
        self.rx.settimeout(1.0)
        self.addr = self.rx.getsockname()
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.latencies_ms: list[float] = []
        self.sizes: list[int] = []

    def roundtrip(self, payload: bytes) -> bytes:
        t0 = time.perf_counter()
        self.tx.sendto(payload, self.addr)
        data, _ = self.rx.recvfrom(65535)
        self.latencies_ms.append((time.perf_counter() - t0) * 1000)
        self.sizes.append(len(payload))
        return data

    def close(self):
        self.rx.close()
        self.tx.close()
