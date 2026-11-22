"""A small blocking WebSocket client (RFC 6455) on the standard library.

Enough for local JSON servers such as Intiface Central: ``ws://`` only (no TLS),
text frames, ping / pong, fragmented messages, close. Sending is thread-safe
(the output clock and the GUI may both send); receiving belongs to one reader
thread. The frame helpers are public so a test can play the server side.
"""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import struct
import threading
from urllib.parse import urlparse

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BINARY, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA


class WebSocketError(Exception):
    pass


class ConnectionClosed(WebSocketError):
    pass


def accept_key(key: str) -> str:
    return base64.b64encode(hashlib.sha1((key + GUID).encode("ascii")).digest()).decode("ascii")


def encode_frame(opcode: int, payload: bytes, mask: bool, fin: bool = True) -> bytes:
    head = bytearray([(0x80 if fin else 0) | opcode])
    n = len(payload)
    mbit = 0x80 if mask else 0
    if n < 126:
        head.append(mbit | n)
    elif n < 65536:
        head.append(mbit | 126)
        head += struct.pack(">H", n)
    else:
        head.append(mbit | 127)
        head += struct.pack(">Q", n)
    if not mask:
        return bytes(head) + payload
    key = os.urandom(4)
    body = bytes(b ^ key[i % 4] for i, b in enumerate(payload))
    return bytes(head) + key + body


def _read_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionClosed("connection closed by the peer")
        buf += chunk
    return bytes(buf)


def read_frame(sock) -> tuple[bool, int, bytes]:
    """``(fin, opcode, payload)`` of the next frame (unmasking if needed)."""
    b0, b1 = _read_exact(sock, 2)
    fin, opcode = bool(b0 & 0x80), b0 & 0x0F
    n = b1 & 0x7F
    if n == 126:
        n = struct.unpack(">H", _read_exact(sock, 2))[0]
    elif n == 127:
        n = struct.unpack(">Q", _read_exact(sock, 8))[0]
    key = _read_exact(sock, 4) if b1 & 0x80 else None
    payload = _read_exact(sock, n) if n else b""
    if key is not None:
        payload = bytes(b ^ key[i % 4] for i, b in enumerate(payload))
    return fin, opcode, payload


class _Raw:
    """Byte reader over the socket that first drains what the handshake over-read."""

    def __init__(self, ws: "WebSocket"):
        self.recv = ws._recv


class WebSocket:
    """``WebSocket("ws://127.0.0.1:12345")`` — connects and handshakes at once."""

    def __init__(self, url: str, timeout: float = 3.0, connect=socket.create_connection):
        u = urlparse(url)
        if u.scheme != "ws":
            raise WebSocketError(f"only ws:// URLs are supported, got {url!r}")
        host, port = u.hostname or "127.0.0.1", u.port or 80
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        self.sock = connect((host, port), timeout)
        self._send_lock = threading.Lock()
        self.closed = False
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                   f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                   "Sec-WebSocket-Version: 13\r\n\r\n")
        self.sock.sendall(request.encode("ascii"))
        response = bytearray()
        while b"\r\n\r\n" not in response:
            chunk = self.sock.recv(1024)
            if not chunk:
                raise WebSocketError("the server closed the connection during the handshake")
            response += chunk
            if len(response) > 16384:
                raise WebSocketError("handshake response too long")
        head, _, rest = bytes(response).partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        if " 101 " not in lines[0] + " ":
            raise WebSocketError(f"handshake refused: {lines[0]}")
        headers = {k.strip().lower(): v.strip() for k, _, v in
                   (line.partition(":") for line in lines[1:])}
        if headers.get("sec-websocket-accept") != accept_key(key):
            raise WebSocketError("bad Sec-WebSocket-Accept in the handshake")
        self._pending = rest                     # bytes that arrived after the handshake
        self._raw = _Raw(self)
        self.sock.settimeout(None)

    # ── sending ─────────────────────────────────────────────────────────────
    def _send(self, opcode: int, payload: bytes) -> None:
        if self.closed:
            raise ConnectionClosed("the WebSocket is closed")
        frame = encode_frame(opcode, payload, mask=True)
        with self._send_lock:
            self.sock.sendall(frame)

    def send_text(self, text: str) -> None:
        self._send(OP_TEXT, text.encode("utf-8"))

    def ping(self, data: bytes = b"") -> None:
        self._send(OP_PING, data)

    # ── receiving ───────────────────────────────────────────────────────────
    def _recv(self, n: int) -> bytes:
        if self._pending:
            out, self._pending = self._pending[:n], self._pending[n:]
            return out
        return self.sock.recv(n)

    def recv(self) -> str:
        """The next text (or binary, decoded) message. Blocks; answers pings."""
        parts: list[bytes] = []
        while True:
            fin, opcode, payload = read_frame(self._raw)
            if opcode == OP_PING:
                self._send(OP_PONG, payload)
                continue
            if opcode == OP_PONG:
                continue
            if opcode == OP_CLOSE:
                try:
                    self._send(OP_CLOSE, payload[:2])
                except (OSError, WebSocketError):
                    pass
                self.closed = True
                raise ConnectionClosed("the server closed the WebSocket")
            parts.append(payload)
            if fin:
                return b"".join(parts).decode("utf-8", "replace")

    def close(self, linger_s: float = 0.5) -> None:
        """Send a close frame and half-close; the socket itself closes after
        *linger_s* so the server can still read the frame (an immediate close
        with unread data turns into a reset on Windows and the frame is lost)."""
        if self.closed:
            return
        try:
            self._send(OP_CLOSE, struct.pack(">H", 1000))
        except (OSError, WebSocketError):
            pass
        self.closed = True
        try:
            self.sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        timer = threading.Timer(linger_s, self.sock.close)
        timer.daemon = True
        timer.start()
