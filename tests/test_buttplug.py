"""Intiface / Buttplug output against a fake Intiface server (stdlib WebSocket)."""

import json
import queue
import socket
import threading
import time

import numpy as np
import pytest

from stimstudio.outputs import EV_DEVICES, EV_ERROR, PulseSource, STATUS_CONNECTED, \
    STATUS_DISCONNECTED
from stimstudio.outputs.base import OutputPulse
from stimstudio.outputs.buttplug import ButtplugSink, Device, quantise, wait_connected
from stimstudio.outputs.wsclient import (OP_CLOSE, OP_TEXT, WebSocket, WebSocketError,
                                         accept_key, encode_frame, read_frame)

DEVICES = [
    {"DeviceName": "Toy A", "DeviceIndex": 0,
     "DeviceMessages": {"ScalarCmd": [{"StepCount": 20, "ActuatorType": "Vibrate"}]}},
    {"DeviceName": "Toy B", "DeviceIndex": 1,
     "DeviceMessages": {"ScalarCmd": [{"StepCount": 10, "ActuatorType": "Vibrate"},
                                      {"StepCount": 10, "ActuatorType": "Oscillate"}]}},
]


class FakeIntiface:
    """Accepts one client; answers like Intiface Central; records every message."""

    def __init__(self, ping_ms=0):
        self.ping_ms = ping_ms
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.url = f"ws://127.0.0.1:{self.port}"
        self.received: list[tuple[str, dict]] = []
        self.conn = None
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def names(self):
        return [n for n, _b in self.received]

    def wait_for(self, name, timeout=2.0, count=1):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.names().count(name) >= count:
                return True
            time.sleep(0.01)
        return False

    def send(self, obj):
        try:
            self.conn.sendall(encode_frame(OP_TEXT, json.dumps([obj]).encode(), mask=False))
        except OSError:
            pass                              # the client already hung up

    def kill(self):
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.conn.close()

    def close(self):
        self.srv.close()
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass

    def _serve(self):
        try:
            conn, _ = self.srv.accept()
        except OSError:
            return
        self.conn = conn
        data = b""
        while b"\r\n\r\n" not in data:
            data += conn.recv(1024)
        key = next(line.split(":", 1)[1].strip() for line in data.decode().split("\r\n")
                   if line.lower().startswith("sec-websocket-key"))
        conn.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                      "Connection: Upgrade\r\n"
                      f"Sec-WebSocket-Accept: {accept_key(key)}\r\n\r\n").encode())
        while True:
            try:
                _fin, op, payload = read_frame(conn)
            except Exception:  # noqa: BLE001
                return
            if op == OP_CLOSE:
                self.received.append(("<close>", {}))
                return
            for msg in json.loads(payload.decode()):
                (name, body), = msg.items()
                self.received.append((name, body))
                mid = body.get("Id", 0)
                if name == "RequestServerInfo":
                    self.send({"ServerInfo": {"Id": mid, "ServerName": "Fake Intiface",
                                              "MessageVersion": 3, "MaxPingTime": self.ping_ms}})
                elif name == "RequestDeviceList":
                    self.send({"DeviceList": {"Id": mid, "Devices": DEVICES}})
                elif name == "StartScanning":
                    self.send({"Ok": {"Id": mid}})
                    self.send({"DeviceAdded": {"Id": 0, "DeviceName": "Toy C", "DeviceIndex": 2,
                                               "DeviceMessages": {"ScalarCmd": [
                                                   {"StepCount": 5,
                                                    "ActuatorType": "Vibrate"}]}}})
                else:
                    self.send({"Ok": {"Id": mid}})


@pytest.fixture
def server():
    s = FakeIntiface()
    yield s
    s.close()


def connected_sink(server, **settings):
    src = PulseSource()
    events = queue.Queue()
    sink = ButtplugSink(src, events)
    sink.configure({"buttplug_url": server.url, **settings})
    sink.connect()
    assert wait_connected(sink)
    return sink, src, events


def drain(events):
    out = []
    while True:
        try:
            out.append(events.get_nowait())
        except queue.Empty:
            return out


# ── the WebSocket layer ─────────────────────────────────────────────────────
def test_accept_key_rfc_vector():
    assert accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


@pytest.mark.parametrize("n", [0, 5, 125, 126, 300, 70000])
@pytest.mark.parametrize("mask", [False, True])
def test_frame_roundtrip(n, mask):
    payload = bytes(range(256)) * (n // 256) + bytes(range(n % 256))
    frame = encode_frame(OP_TEXT, payload, mask)

    class Buf:
        def __init__(self, data):
            self.data = data

        def recv(self, k):
            out, self.data = self.data[:k], self.data[k:]
            return out

    fin, op, got = read_frame(Buf(frame))
    assert fin and op == OP_TEXT and got == payload


def test_bad_urls_and_refusal():
    with pytest.raises(WebSocketError):
        WebSocket("http://127.0.0.1:1")
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def refuse():
        c, _ = srv.accept()
        c.recv(1024)
        c.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        c.close()

    threading.Thread(target=refuse, daemon=True).start()
    with pytest.raises(WebSocketError):
        WebSocket(f"ws://127.0.0.1:{port}")
    srv.close()


# ── the sink ────────────────────────────────────────────────────────────────
def test_connect_lists_devices(server):
    sink, _src, events = connected_sink(server)
    assert server.names()[:2] == ["RequestServerInfo", "RequestDeviceList"]
    assert server.received[0][1]["ClientName"] == "StimStudio"
    assert server.received[0][1]["MessageVersion"] == 3
    assert sorted(d.name for d in sink.devices.values()) == ["Toy A", "Toy B"]
    devs = [p for k, p in drain(events) if k == EV_DEVICES][-1]
    assert [d["name"] for d in devs] == ["Toy A", "Toy B"]
    assert devs[1]["actuators"] == [("Vibrate", 10), ("Oscillate", 10)]
    sink.disconnect()


def test_scan_adds_devices(server):
    sink, _src, _events = connected_sink(server)
    sink.scan(True)
    deadline = time.time() + 2
    while 2 not in sink.devices and time.time() < deadline:
        time.sleep(0.01)
    assert sink.devices[2].name == "Toy C"
    sink.disconnect()


def test_commands_follow_channels_and_dedupe(server):
    sink, _src, _e = connected_sink(server, buttplug_assign={"Toy A": "A", "Toy B": "max"})
    p = OutputPulse(amp_a=0.8, amp_b=0.3, power_a=100, power_b=200)
    cmds = {c["DeviceIndex"]: c["Scalars"] for c in sink.commands(p)}
    assert cmds[0] == [{"Index": 0, "Scalar": quantise(0.4, 20), "ActuatorType": "Vibrate"}]
    assert [s["Scalar"] for s in cmds[1]] == [quantise(0.4, 10)] * 2   # max(0.4, 0.3)
    assert [s["ActuatorType"] for s in cmds[1]] == ["Vibrate", "Oscillate"]
    assert sink.commands(p) == []                    # nothing changed → nothing sent
    tiny = OutputPulse(amp_a=0.81, amp_b=0.3, power_a=100, power_b=200)
    assert sink.commands(tiny) == []                 # below one step of Toy A
    sink.configure({"buttplug_url": server.url, "buttplug_assign": {"Toy A": "off"}})
    cmds = {c["DeviceIndex"]: c["Scalars"] for c in sink.commands(p)}
    assert cmds[0][0]["Scalar"] == 0.0 and 1 not in cmds
    sink.disconnect()


def test_playing_sends_scalars_and_panic_stops_all(server):
    sink, src, _e = connected_sink(server)
    pulses = np.zeros((400, 4), np.float32)
    pulses[:, 0] = pulses[:, 1] = 0.9
    src.set_window(pulses, loop=True)
    src.set_controls(power_a=200, power_b=200, soft_start=0.0)
    sink.start()
    assert server.wait_for("ScalarCmd")
    body = next(b for n, b in server.received if n == "ScalarCmd")
    assert body["Scalars"][0]["Scalar"] == pytest.approx(0.9, abs=0.06)
    sink.stop(panic=True)
    assert server.wait_for("StopAllDevices")
    assert not sink.playing
    sink.disconnect()
    assert server.wait_for("StopAllDevices", count=2) and server.wait_for("<close>")
    assert sink.status == STATUS_DISCONNECTED and not sink.devices


def test_ping_keeps_the_server_happy():
    s = FakeIntiface(ping_ms=400)
    try:
        sink, _src, _e = connected_sink(s)
        assert s.wait_for("Ping", timeout=2.0)
        sink.disconnect()
    finally:
        s.close()


def test_lost_connection_reports_and_silences(server):
    sink, src, events = connected_sink(server)
    src.set_window(np.full((400, 4), 0.5, np.float32), loop=True)
    src.set_controls(power_a=100, power_b=100, soft_start=0.0)
    sink.start()
    server.kill()
    deadline = time.time() + 2
    while sink.status != STATUS_DISCONNECTED and time.time() < deadline:
        time.sleep(0.01)
    assert sink.status == STATUS_DISCONNECTED and not sink.playing
    assert any(k == EV_ERROR and "lost" in str(p) for k, p in drain(events))


def test_connect_failure_is_reported():
    src, events = PulseSource(), queue.Queue()
    sink = ButtplugSink(src, events)
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()                                    # nothing listens there now
    sink.configure({"buttplug_url": f"ws://127.0.0.1:{port}"})
    sink.connect()
    deadline = time.time() + 3
    while sink.status != STATUS_DISCONNECTED and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)
    assert sink.status == STATUS_DISCONNECTED
    assert any(k == EV_ERROR for k, _p in drain(events))


def test_device_from_message():
    d = Device.from_message(DEVICES[1])
    assert d.index == 1 and [a.kind for a in d.actuators] == ["Vibrate", "Oscillate"]
    assert quantise(0.44, 10) == 0.4 and quantise(1.7, 5) == 1.0 and quantise(-1, 5) == 0.0


# ── the panel ───────────────────────────────────────────────────────────────
def test_panel_rows_and_assignment(window, qapp):
    panel = window.device_panel
    panel.device.setCurrentIndex(panel.device.findData("buttplug"))
    assert panel.rows["bpurl"].isVisibleTo(panel) and not panel.rows["freq"].isVisibleTo(panel)
    assert panel.connect_btn.isVisibleTo(panel)
    window.playback.devicesReported.emit([{"index": 0, "name": "Toy A",
                                           "actuators": [("Vibrate", 20)]}])
    assert panel.bp_table.rowCount() == 1
    combo = panel.bp_table.cellWidget(0, 1)
    combo.setCurrentIndex(combo.findData("B"))
    assert panel.settings_dict()["buttplug_assign"] == {"Toy A": "B"}
    assert window.playback.sink.assign == {"Toy A": "B"}
    panel.device.setCurrentIndex(panel.device.findData("dryrun"))
