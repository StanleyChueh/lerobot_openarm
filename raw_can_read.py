#!/usr/bin/env python
"""Read-only raw CAN listener that independently decodes Damiao motor feedback.

Why this exists: openarm_can's Python bindings (self-described as
"EXPERIMENTAL - UNSTABLE API") appear to be misreporting position via
get_position() -- on 2026-07-01, safe_probe.py reported -12.4676 rad for
several left-arm joints and exactly 0.0 for others, none of which matched a
manual decode of the actual CAN bytes captured via candump at the same time
(which showed distinct, physically plausible small angles per joint). This
script decodes the standard Damiao MIT feedback frame directly from the
SocketCAN raw socket, bypassing openarm_can's Python layer entirely, to get
an independently trustworthy reading.

This script NEVER transmits anything on the bus -- it only listens (a plain
SocketCAN RAW socket bound read-only, with CAN_RAW_FD_FRAMES enabled since this
codebase always runs the bus in CAN-FD mode -- without that option the socket
receives zero frames of any kind, silently, indistinguishable from a timing
mismatch). Zero risk to the motors. It relies on
some other process (e.g. safe_probe.py in read-only mode, or the follower's
get_observation()) to actually poll the motors for state -- Damiao motors
only reply when queried, they don't broadcast on their own. Run this in one
terminal and the querying script in another, same as you did with candump.

Damiao feedback frame layout (8-byte payload), for response arbitration IDs
0x11-0x17 (arm joints 1-7) and 0x18 (gripper):
  byte 0: [error(4bit) | motor_id(4bit)]
  byte 1-2: position, big-endian uint16 -> maps to [P_MIN, P_MAX] = [-12.5, 12.5] rad
  byte 3: velocity high 8 bits (of 12-bit)
  byte 4: [velocity low 4bit | torque high 4bit]
  byte 5: torque low 8 bits
  byte 6-7: MOS / rotor temperature

Usage:
  Terminal A: python raw_can_read.py --port can3 --duration 8
  Terminal B (while A is running): python safe_probe.py --side left
"""

import argparse
import socket
import struct
import time

P_MIN, P_MAX = -12.5, 12.5
RECV_ID_NAMES = {
    0x11: "J1", 0x12: "J2", 0x13: "J3", 0x14: "J4",
    0x15: "J5", 0x16: "J6", 0x17: "J7", 0x18: "J8(gripper)",
}

CAN_FRAME_FMT = "=IB3x8s"
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FMT)

CANFD_FRAME_FMT = "=IBBBB64s"
CANFD_FRAME_SIZE = struct.calcsize(CANFD_FRAME_FMT)


def decode_frame(data: bytes) -> dict:
    """error/motor_id share byte 0's nibbles; error != 0 means the motor itself is
    reporting a fault (over-current, over-temp, comm loss, etc. -- see Damiao MIT
    protocol docs for the exact code table) and will ignore position commands
    regardless of kp, independent of anything the sending side does correctly."""
    error = (data[0] >> 4) & 0x0F
    motor_id = data[0] & 0x0F
    raw = (data[1] << 8) | data[2]
    pos = P_MIN + (raw / 65535.0) * (P_MAX - P_MIN)
    return {"error": error, "motor_id": motor_id, "pos": pos}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=str, required=True, help="e.g. can3 (left) or can2 (right)")
    parser.add_argument("--duration", type=float, default=8.0, help="seconds to listen")
    args = parser.parse_args()

    sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    # This codebase always runs the arm bus in CAN-FD mode (enable_fd=True everywhere else,
    # e.g. mirror_bridge.py). Without this option a CAN_RAW socket silently receives ZERO
    # frames of any kind on an FD-only bus -- not an error, just total silence, indistinguishable
    # from a timing mismatch unless you know to look for it.
    sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FD_FRAMES, 1)
    sock.bind((args.port,))
    sock.settimeout(0.5)

    latest = {}
    history_counts = {rid: 0 for rid in RECV_ID_NAMES}
    total_frames = 0
    unmatched_ids = set()
    deadline = time.time() + args.duration
    print(f"Listening on {args.port} for {args.duration:.0f}s (read-only, transmits nothing)...")
    print("(trigger motor state reads in another terminal now, e.g. `python safe_probe.py --side left`, a few times if you can)")
    while time.time() < deadline:
        try:
            # Read with the larger (FD) buffer size -- the kernel returns exactly
            # CAN_FRAME_SIZE bytes for a classic frame or CANFD_FRAME_SIZE for an FD
            # frame; len(frame) tells you which one you got.
            frame = sock.recv(CANFD_FRAME_SIZE)
        except socket.timeout:
            continue
        total_frames += 1
        if len(frame) == CANFD_FRAME_SIZE:
            can_id, dlc, _flags, _res0, _res1, data = struct.unpack(CANFD_FRAME_FMT, frame)
        elif len(frame) == CAN_FRAME_SIZE:
            can_id, dlc, data = struct.unpack(CAN_FRAME_FMT, frame)
        else:
            continue
        can_id &= socket.CAN_EFF_MASK if (can_id & socket.CAN_EFF_FLAG) else socket.CAN_SFF_MASK
        if can_id in RECV_ID_NAMES and dlc >= 3:
            latest[can_id] = decode_frame(data)
            history_counts[can_id] += 1
        else:
            unmatched_ids.add(can_id)

    print(f"\nTotal raw CAN frames seen on {args.port} during the window: {total_frames}")
    if unmatched_ids:
        print(f"Frames seen with IDs not in our recv-ID table: {sorted(hex(i) for i in unmatched_ids)}")
    print("\nIndependently decoded positions (raw CAN bytes, bypassing openarm_can bindings):")
    for rid, name in RECV_ID_NAMES.items():
        if rid in latest:
            f = latest[rid]
            flag = "  <-- NONZERO ERROR CODE" if f["error"] != 0 else ""
            print(
                f"  {name:14s} (id 0x{rid:02X}): {f['pos']:+.4f} rad   error={f['error']}"
                f"   motor_id_in_payload={f['motor_id']}   [{history_counts[rid]} frames seen]{flag}"
            )
        else:
            print(f"  {name:14s} (id 0x{rid:02X}): NO RESPONSE SEEN")

    if total_frames == 0:
        print(
            "\nZero frames of ANY kind were seen -- this points to the listener itself (wrong --port,"
            " socket/permission issue), not a timing mismatch. Re-check --port matches the interface"
            " candump used successfully."
        )


if __name__ == "__main__":
    main()
