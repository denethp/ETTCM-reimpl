"""Reader/writer for the Event Stream (.es) format used by the official code (sepia).

Format (sepia, Event Stream v2, type "dvs"):
  header : b"Event Stream" (12 bytes) + version (3 bytes: major, minor, patch)
           + type (1 byte, dvs = 1) + width (uint16 LE) + height (uint16 LE)
  events : byte state machine
           idle byte 0xFF -> t += 127 (overflow, no event)
           idle byte 0xFE -> reset marker, ignored
           otherwise      -> t += byte >> 1 ; polarity = byte & 1
                             then 4 bytes: x (uint16 LE), y (uint16 LE)
  timestamps are in microseconds.

Returned events are a structured array with fields t (uint64, us), x, y (uint16)
and p (uint8, 1 = ON / increase, 0 = OFF).
"""
import numpy as np

from ._jit import njit

SIGNATURE = b"Event Stream"
EVENT_DTYPE = np.dtype([("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")])


@njit
def _decode_dvs(buf, t_out, x_out, y_out, p_out):
    n = 0
    t = 0
    state = 0
    x = 0
    y = 0
    p = 0
    for i in range(buf.shape[0]):
        b = int(buf[i])
        if state == 0:
            if b == 0xFF:
                t += 127
            elif b != 0xFE:
                t += b >> 1
                p = b & 1
                state = 1
        elif state == 1:
            x = b
            state = 2
        elif state == 2:
            x |= b << 8
            state = 3
        elif state == 3:
            y = b
            state = 4
        else:
            y |= b << 8
            t_out[n] = t
            x_out[n] = x
            y_out[n] = y
            p_out[n] = p
            n += 1
            state = 0
    return n


def read_es(path):
    """Read a DVS .es file. Returns (events, width, height)."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw[:12].tobytes() != SIGNATURE:
        raise ValueError(f"{path}: not an Event Stream file")
    major = int(raw[12])
    if major != 2:
        raise ValueError(f"{path}: unsupported Event Stream version {raw[12:15]}")
    ev_type = int(raw[15])
    if ev_type != 1:
        raise ValueError(f"{path}: event type {ev_type} is not dvs (1)")
    width = int(raw[16]) | (int(raw[17]) << 8)
    height = int(raw[18]) | (int(raw[19]) << 8)
    buf = raw[20:]
    cap = buf.shape[0] // 5 + 1  # each event takes at least 5 bytes
    t = np.empty(cap, np.uint64)
    x = np.empty(cap, np.uint16)
    y = np.empty(cap, np.uint16)
    p = np.empty(cap, np.uint8)
    n = _decode_dvs(buf, t, x, y, p)
    ev = np.empty(n, EVENT_DTYPE)
    ev["t"], ev["x"], ev["y"], ev["p"] = t[:n], x[:n], y[:n], p[:n]
    return ev, width, height


def write_es(path, events, width, height):
    """Write a DVS .es file (used by tests/test_io.py)."""
    out = bytearray(SIGNATURE)
    out += bytes([2, 0, 0, 1])
    out += int(width).to_bytes(2, "little") + int(height).to_bytes(2, "little")
    t_prev = 0
    for t, x, y, p in zip(events["t"], events["x"], events["y"], events["p"]):
        dt = int(t) - t_prev
        if dt < 0:
            raise ValueError("timestamps must be non-decreasing")
        while dt >= 127:
            out.append(0xFF)
            dt -= 127
        out.append((dt << 1) | (int(p) & 1))
        out += int(x).to_bytes(2, "little") + int(y).to_bytes(2, "little")
        t_prev = int(t)
    with open(path, "wb") as f:
        f.write(out)
