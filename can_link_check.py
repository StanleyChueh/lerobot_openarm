#!/usr/bin/env python
"""Live per-motor CAN link monitor -- for finding an intermittent connector with your hands.

Sends nothing but the DM state query (0x7FF, data[2]=0xCC), which asks a motor to report itself.
It never enables a motor and never sends a torque, gain or position, so the arm cannot move no
matter what you do to it while this is running. That is the point: you need both hands free for
the harness.

Deliberately standalone -- a raw SocketCAN socket and nothing else. No openarm_can, no lerobot,
no pinocchio. This is the tool you want to still work when the reason you are running it is that
something else does not.

Each line is one second. A channel prints its short name when it answered and `--` when it did
not, and every change of state is called out with a timestamp, so a connector you wiggle back to
life announces itself while you are looking at the connector rather than at the screen.

  python can_link_check.py --port can0                 # right arm, J1..J8
  python can_link_check.py --port can1 --duration 120  # left arm, two minutes

Reading it: the motors are a daisy chain, J1 nearest the base. A CONTIGUOUS TAIL going dark
(e.g. J5-J8 silent, J1-J4 fine) puts the fault at the link feeding the first silent one -- the
motors past a broken link cannot answer, and the ones before it are unaffected. A single channel
dropping out on its own is that motor or its own connector.
"""

import argparse
import select
import socket
import struct
import time

CANFD_MTU = 72
CAN_MTU = 16
CAN_RAW_FD_FRAMES = 5
CAN_RAW_ERR_FILTER = 2
CANFD_BRS = 0x01
CAN_ERR_FLAG = 0x20000000
CAN_EFF_MASK = 0x1FFFFFFF
REPLY_ID_BASE = 0x10          # motor n answers on 0x10 + n


def open_socket(port: str, fd: bool):
    sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    if fd:
        sock.setsockopt(socket.SOL_CAN_RAW, CAN_RAW_FD_FRAMES, 1)
    sock.setsockopt(socket.SOL_CAN_RAW, CAN_RAW_ERR_FILTER, CAN_EFF_MASK)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind((port,))
    sock.setblocking(False)
    return sock


# Gap between the individual queries in a round. Without it the FIRST motor's answer -- which
# arrives ~0.13 ms after its request -- lands while this process is still handing the adapter
# frames 2..8 of the same burst, and the adapter loses it. That shows up as J1 sitting at 60-95%
# while everything else reads 100%, which is a lie told by the measurement, and a diagnostic that
# invents a fault on the very joint you are trying to clear is worse than no diagnostic.
QUERY_GAP_S = 0.0004


def query(sock, motor_id: int, fd: bool) -> bool:
    """Send one state query. False if the adapter would not take it.

    A CAN raw socket does not block when the interface queue is full -- it fails the write with
    ENOBUFS (txqueuelen on these gs_usb channels is 10 frames). That is a normal, transient
    condition on a USB adapter, not a reason to stop: this tool exists to be left running for
    minutes at a time while somebody works on the harness, and it used to die with a traceback
    after a few minutes, throwing away the session.
    """
    data = bytes([motor_id & 0xFF, (motor_id >> 8) & 0xFF, 0xCC, 0, 0, 0, 0, 0])
    if fd:
        frame = struct.pack("=IBBBB", 0x7FF, 8, CANFD_BRS, 0, 0) + data.ljust(64, b"\0")
    else:
        frame = struct.pack("=IB3x", 0x7FF, 8) + data.ljust(8, b"\0")
    try:
        sock.send(frame)
        return True
    except OSError:
        time.sleep(0.001)   # let the interface queue drain before the next one
        return False


def drain(sock, until: float, seen: set, errors: dict):
    while time.perf_counter() < until:
        r, _, _ = select.select([sock], [], [], 0.001)
        if not r:
            continue
        try:
            buf = sock.recv(CANFD_MTU)
        except BlockingIOError:
            continue
        if len(buf) < CAN_MTU:
            continue
        raw = struct.unpack("=I", buf[:4])[0]
        if raw & CAN_ERR_FLAG:
            cls = raw & CAN_EFF_MASK
            errors[cls] = errors.get(cls, 0) + 1
            continue
        seen.add(raw & CAN_EFF_MASK)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="can0", help="can0 is the right arm, can1 the left")
    ap.add_argument("--ids", default="1,2,3,4,5,6,7,8", help="motor send ids to query")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds; 0 runs until Ctrl-C")
    ap.add_argument("--hz", type=float, default=20.0, help="queries per second per motor")
    ap.add_argument("--no-fd", action="store_true", help="query with classic CAN 2.0 frames")
    args = ap.parse_args()

    ids = [int(x, 0) for x in args.ids.split(",")]
    fd = not args.no_fd
    sock = open_socket(args.port, fd)
    period = 1.0 / args.hz

    print(f"\n{args.port}: querying J{ids[0]}..J{ids[-1]} at {args.hz:g} Hz. Read-only -- nothing"
          f" is enabled and no motion can be commanded.\nWiggle connectors freely. Ctrl-C to stop.\n")
    print(f"  {'t':>6s}  " + " ".join(f"{f'J{i}':>4s}" for i in ids) + "   reply rate this second")

    alive = {i: None for i in ids}
    errors: dict[int, int] = {}
    tx_fail = 0
    attempts = 0
    t0 = time.perf_counter()
    try:
        while args.duration <= 0 or time.perf_counter() - t0 < args.duration:
            second_start = time.perf_counter()
            hits = {i: 0 for i in ids}
            polls = 0
            while time.perf_counter() - second_start < 1.0:
                cycle = time.perf_counter()
                seen: set = set()
                for motor_id in ids:
                    attempts += 1
                    if not query(sock, motor_id, fd):
                        tx_fail += 1
                    drain(sock, time.perf_counter() + QUERY_GAP_S, seen, errors)
                drain(sock, cycle + min(period, 0.02), seen, errors)
                for motor_id in ids:
                    if REPLY_ID_BASE + motor_id in seen:
                        hits[motor_id] += 1
                polls += 1
                rest = period - (time.perf_counter() - cycle)
                if rest > 0:
                    time.sleep(rest)

            now = time.perf_counter() - t0
            cells, worst = [], 100
            for motor_id in ids:
                pct = 100 * hits[motor_id] // max(1, polls)
                cells.append(f"J{motor_id}" if pct >= 50 else "--")
                worst = min(worst, pct)
                # Announce transitions: this is what you are listening for while your hands are
                # in the harness and your eyes are not on the terminal.
                state = pct >= 50
                if alive[motor_id] is not None and state != alive[motor_id]:
                    print(f"  {now:6.1f}  *** J{motor_id} "
                          + ("CAME BACK" if state else "WENT SILENT") + f" ({pct}%)", flush=True)
                alive[motor_id] = state
            rates = " ".join(f"{100 * hits[i] // max(1, polls):3d}%" for i in ids)
            print(f"  {now:6.1f}  " + " ".join(f"{c:>4s}" for c in cells) + f"   {rates}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")

    silent = [i for i in ids if not alive[i]]
    print()

    # Before saying anything about the robot, check our own frames actually left the host. When
    # most writes are refused there is no evidence about any motor: nothing was asked.
    if attempts and tx_fail > 0.5 * attempts:
        print(f"  THE HOST NEVER TRANSMITTED. {tx_fail} of {attempts} queries were refused by the"
              f" adapter (ENOBUFS), so the silence above says NOTHING about the motors -- they"
              f" were never asked.\n"
              f"\n  FIRST: IS THE ARM POWERED ON?\n"
              f"  That is by far the most likely answer, and it is not a guess about probability"
              f" -- it is how CAN works. Every frame must be acknowledged by at least one OTHER"
              f" node on the bus. With the arm unpowered there is nobody to acknowledge, so the"
              f" controller retransmits the same frame forever, the queue behind it fills, and"
              f" every further write is refused. The link keeps reporting UP and ERROR-ACTIVE"
              f" throughout, which makes an unpowered robot look exactly like dead hardware."
              f" Confirmed 2026-08-26, when this was mistaken for a failing adapter.\n"
              f"  Power the arm on and re-run. If the interface stays jammed afterwards, clear the"
              f" retransmit backlog it built up:  sudo ./can_recover.sh\n"
              f"\n  If the arm IS powered on, then the adapter's transmit path has wedged. Check"
              f" `tc -s qdisc show dev {args.port}`: a backlog that will not drain, or a Sent"
              f" counter frozen with nothing running. Recover it with `sudo ./can_recover.sh`,"
              f" then `--reset`, then by unplugging the adapter's USB cable. Both channels live on"
              f" one USB interface, so they wedge and recover together.")
        return

    if errors:
        print(f"  CAN controller error frames seen: "
              + ", ".join(f"class {c:#x} x{n}" for c, n in sorted(errors.items())))
    if tx_fail:
        print(f"  {tx_fail} query/queries could not be handed to the adapter (ENOBUFS)."
              " Transient USB back-pressure; they were skipped, not lost readings.")
    if not silent:
        print("  All queried motors answered. This bus is healthy right now.")
        return
    print(f"  Silent: {', '.join(f'J{i}' for i in silent)}")
    tail = [i for i in ids if i >= min(silent)]
    if silent == tail:
        first = min(silent)
        print(f"  They are a CONTIGUOUS TAIL of the chain (J{first} onward), and J{first - 1} and"
              f" everything before it answered. Motors past a broken link cannot reply and motors"
              f" before it are unaffected, so the fault is at or just after J{first - 1}: check the"
              f" CAN and power connectors on the J{first - 1} -> J{first} run, and flex that"
              f" section by hand while this is running."
              if first > ids[0] else
              "  Every queried motor is silent -- suspect this arm's power or the adapter cable"
              " rather than any one joint.")
    else:
        print("  They are NOT a contiguous tail, so this is not one broken link in the chain."
              " Check each silent motor's own connector.")


if __name__ == "__main__":
    main()
