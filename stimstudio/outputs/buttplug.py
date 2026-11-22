"""Intiface Central / Buttplug output: drive vibrators (and similar toys).

Speaks the Buttplug v3 JSON protocol over a local WebSocket (Intiface Central
listens on ``ws://127.0.0.1:12345`` by default). Every 0.1 s the current pulse
level of channel A / B — after power, soft start, mute and limits — becomes a
``ScalarCmd`` for each actuator assigned to that channel (A, B, the louder of
the two, or off). Commands are only sent when the value changes by at least
one of the device's steps. Stop, PANIC, a dropped connection and closing the
app all send ``StopAllDevices``.

Vibrators have no notion of e-stim pulse frequency, so only the level is
used. Not tested with real devices (``verified = False``).
"""

from __future__ import annotations

import itertools
import json
import threading
import time
from dataclasses import dataclass, field

from .base import (EV_DEVICES, EV_ERROR, POWER_MAX, STATUS_CONNECTED, STATUS_CONNECTING,
                   STATUS_DISCONNECTED, ClockedSink, Controls, FrequencyMap, OutputPulse)
from .wsclient import ConnectionClosed, WebSocket, WebSocketError

DEFAULT_URL = "ws://127.0.0.1:12345"
CLIENT_NAME = "StimStudio"
MESSAGE_VERSION = 3
ASSIGNMENTS = ("max", "A", "B", "off")
ASSIGN_LABELS = {"max": "A / B (louder)", "A": "Channel A", "B": "Channel B", "off": "Off"}
HANDSHAKE_TIMEOUT = 3.0


@dataclass
class Actuator:
    index: int
    kind: str              # Vibrate, Oscillate, Constrict, Inflate, Rotate…
    steps: int


@dataclass
class Device:
    index: int
    name: str
    actuators: list[Actuator] = field(default_factory=list)

    @classmethod
    def from_message(cls, d: dict) -> "Device":
        scalars = (d.get("DeviceMessages") or {}).get("ScalarCmd") or []
        acts = [Actuator(i, str(a.get("ActuatorType", "Vibrate")), int(a.get("StepCount", 20)))
                for i, a in enumerate(scalars)]
        return cls(int(d.get("DeviceIndex", 0)), str(d.get("DeviceName", "device")), acts)

    def to_dict(self) -> dict:
        return {"index": self.index, "name": self.name,
                "actuators": [(a.kind, a.steps) for a in self.actuators]}


def quantise(value: float, steps: int) -> float:
    steps = max(1, int(steps))
    return round(min(1.0, max(0.0, value)) * steps) / steps


class ButtplugSink(ClockedSink):
    label = "Intiface / Buttplug (vibrators)"
    requires = ()
    verified = False
    needs_link = True
    frame_interval = 0.1
    pulse_batch_size = 1
    default_freq_map = FrequencyMap(1, 200, 0.05, 0.5)   # unused: level only

    def __init__(self, source, events, connect=None):
        super().__init__(source, events)
        self.url = DEFAULT_URL
        self.assign: dict[str, str] = {}           # device name → ASSIGNMENTS
        self.devices: dict[int, Device] = {}
        self._connect = connect                    # test seam: socket factory
        self._ws: WebSocket | None = None
        self._ids = itertools.count(1)
        self._waiting: dict[int, threading.Event] = {}
        self._replies: dict[int, dict] = {}
        self._reader: threading.Thread | None = None
        self._pinger: threading.Thread | None = None
        self._ping_s = 0.0
        self._sent: dict[tuple[int, int], float] = {}
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self.server_name = ""

    # ── settings ────────────────────────────────────────────────────────────
    def configure(self, settings: dict):
        self.url = str(settings.get("buttplug_url") or DEFAULT_URL)
        assign = settings.get("buttplug_assign")
        if isinstance(assign, dict):
            self.assign = {str(k): str(v) for k, v in assign.items() if v in ASSIGNMENTS}

    def assignment(self, device: Device) -> str:
        return self.assign.get(device.name, "max")

    # ── connection ──────────────────────────────────────────────────────────
    def connect(self):
        if self.status != STATUS_DISCONNECTED:
            return
        self._set_status(STATUS_CONNECTING)
        threading.Thread(target=self._connect_worker, name="buttplug-connect",
                         daemon=True).start()

    def _connect_worker(self):
        try:
            kwargs = {"connect": self._connect} if self._connect else {}
            self._ws = WebSocket(self.url, timeout=HANDSHAKE_TIMEOUT, **kwargs)
            self._stopping.clear()
            self._reader = threading.Thread(target=self._read_loop, name="buttplug-reader",
                                            daemon=True)
            self._reader.start()
            info = self.request({"RequestServerInfo": {"ClientName": CLIENT_NAME,
                                                       "MessageVersion": MESSAGE_VERSION}})
            if "ServerInfo" not in info:
                raise WebSocketError(f"unexpected reply: {info}")
            si = info["ServerInfo"]
            self.server_name = str(si.get("ServerName", ""))
            self._ping_s = float(si.get("MaxPingTime", 0)) / 1000.0
            if self._ping_s > 0:
                self._pinger = threading.Thread(target=self._ping_loop, name="buttplug-ping",
                                                daemon=True)
                self._pinger.start()
            reply = self.request({"RequestDeviceList": {}})
            for d in (reply.get("DeviceList") or {}).get("Devices", []):
                dev = Device.from_message(d)
                self.devices[dev.index] = dev
            self._post_devices()
            self._set_status(STATUS_CONNECTED)
        except Exception as exc:  # noqa: BLE001
            self._post(EV_ERROR, f"Intiface: {exc}")
            self._teardown()
            self._set_status(STATUS_DISCONNECTED)

    def disconnect(self):
        self._stop_all()
        super().disconnect()
        self._teardown()

    def _teardown(self):
        self._stopping.set()
        ws, self._ws = self._ws, None
        if ws is not None:
            ws.close()
        for ev in list(self._waiting.values()):
            ev.set()
        self.devices.clear()
        self._sent.clear()
        self._post_devices()

    # ── messages ────────────────────────────────────────────────────────────
    def send(self, name: str, body: dict) -> int:
        """Send one message; returns its Id."""
        ws = self._ws
        if ws is None:
            raise ConnectionClosed("not connected")
        msg_id = next(self._ids)
        ws.send_text(json.dumps([{name: {"Id": msg_id, **body}}]))
        return msg_id

    def request(self, message: dict, timeout: float = HANDSHAKE_TIMEOUT) -> dict:
        """Send and wait for the reply with the same Id."""
        (name, body), = message.items()
        event = threading.Event()
        ws = self._ws
        if ws is None:
            raise ConnectionClosed("not connected")
        msg_id = next(self._ids)
        self._waiting[msg_id] = event
        ws.send_text(json.dumps([{name: {"Id": msg_id, **body}}]))
        if not event.wait(timeout):
            self._waiting.pop(msg_id, None)
            raise WebSocketError(f"no reply to {name}")
        self._waiting.pop(msg_id, None)
        reply = self._replies.pop(msg_id, None)
        if reply is None:
            raise ConnectionClosed("connection lost")
        if "Error" in reply:
            raise WebSocketError(reply["Error"].get("ErrorMessage", "error"))
        return reply

    def scan(self, on: bool = True) -> None:
        try:
            self.send("StartScanning" if on else "StopScanning", {})
        except (OSError, WebSocketError) as exc:
            self._post(EV_ERROR, f"Intiface: {exc}")

    def _read_loop(self):
        ws = self._ws
        while ws is not None and not self._stopping.is_set():
            try:
                text = ws.recv()
            except (OSError, WebSocketError):
                if not self._stopping.is_set():
                    self._lost("the connection to Intiface was lost")
                return
            try:
                messages = json.loads(text)
            except ValueError:
                continue
            for m in messages if isinstance(messages, list) else [messages]:
                self._handle(m)

    def _handle(self, m: dict):
        if not isinstance(m, dict) or not m:
            return
        (name, body), = list(m.items())[:1]
        body = body or {}
        if name == "DeviceAdded":
            dev = Device.from_message(body)
            self.devices[dev.index] = dev
            self._post_devices()
        elif name == "DeviceRemoved":
            self.devices.pop(int(body.get("DeviceIndex", -1)), None)
            self._post_devices()
        elif name == "ScanningFinished":
            pass
        msg_id = int(body.get("Id", 0)) if isinstance(body, dict) else 0
        if msg_id and msg_id in self._waiting:
            self._replies[msg_id] = m
            self._waiting[msg_id].set()
        elif name == "Error" and msg_id:
            self._post(EV_ERROR, f"Intiface: {body.get('ErrorMessage', 'error')}")

    def _ping_loop(self):
        interval = max(0.2, self._ping_s / 2)
        while not self._stopping.wait(interval):
            try:
                self.send("Ping", {})
            except (OSError, WebSocketError):
                return

    def _lost(self, reason: str):
        self._playing = False
        self._post(EV_ERROR, f"Intiface: {reason}")
        self._clock.stop()
        self._teardown()
        self._set_status(STATUS_DISCONNECTED)

    def _post_devices(self):
        self._post(EV_DEVICES, [d.to_dict() for d in sorted(self.devices.values(),
                                                              key=lambda d: d.index)])

    # ── output ──────────────────────────────────────────────────────────────
    def levels(self, pulse: OutputPulse) -> tuple[float, float]:
        """Channel A / B intensity 0..1 (power scaled to the 0..200 range)."""
        return (pulse.amp_a * pulse.power_a / POWER_MAX, pulse.amp_b * pulse.power_b / POWER_MAX)

    def commands(self, pulse: OutputPulse, force: bool = False) -> list[dict]:
        """``ScalarCmd`` bodies for the actuators whose value changed."""
        a, b = self.levels(pulse)
        out = []
        with self._lock:
            for dev in self.devices.values():
                mode = self.assignment(dev)
                if mode == "off":
                    level = 0.0
                else:
                    level = {"A": a, "B": b}.get(mode, max(a, b))
                scalars = []
                for act in dev.actuators:
                    q = quantise(level, act.steps)
                    key = (dev.index, act.index)
                    if force or self._sent.get(key) != q:
                        self._sent[key] = q
                        scalars.append({"Index": act.index, "Scalar": q,
                                        "ActuatorType": act.kind})
                if scalars:
                    out.append({"DeviceIndex": dev.index, "Scalars": scalars})
        return out

    def _emit(self, frame: list[OutputPulse], controls: Controls):
        if self._ws is None:
            return
        for body in self.commands(frame[-1]):
            self.send("ScalarCmd", body)

    def _silence(self):
        super()._silence()               # zero ScalarCmds through _emit
        self._stop_all()

    def _stop_all(self):
        if self._ws is None:
            return
        try:
            self.send("StopAllDevices", {})
        except (OSError, WebSocketError):
            pass
        with self._lock:
            for key in self._sent:
                self._sent[key] = 0.0

    def stop(self, *, panic: bool = False):
        super().stop(panic=panic)
        if panic:
            self._stop_all()


def wait_connected(sink: ButtplugSink, timeout: float = 3.0) -> bool:
    """Block until *sink* is connected (for tests and scripts)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if sink.status == STATUS_CONNECTED:
            return True
        time.sleep(0.01)
    return sink.status == STATUS_CONNECTED
