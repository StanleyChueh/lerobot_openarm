"""A passive tap on a CAN interface that COUNTS the state frames each motor actually sends.

Why this exists
---------------
Everything upstream of this file infers feedback freshness from the position value, and that
inference cannot work.  openarm_can hands us retained Motor state: if a reply never arrives,
get_position() returns the previous number, which is plausible, unchanging, and bit-for-bit
identical to what a joint that simply is not moving reports.  So "the reading did not change
while the command swept past it" -- the test reset_to_rest_pose.py used -- fires on BOTH
"the feedback died" and "the joint did not move", which are opposite diagnoses with opposite
fixes.  Measured 2026-08-26: it flagged LJ8, a gripper at kp=3.0 commanded to travel 0.0245
rad, i.e. a channel that was reporting the truth about a joint physically incapable of
responding to that command.

Counting frames replaces the inference with a measurement.  A channel that sent no state frame
during a read is stale, full stop; a channel that sent one and still reads unchanged did not
move, full stop.

Why a second socket is safe
---------------------------
SocketCAN delivers every received frame to EVERY raw socket bound to that interface, so this
tap observes openarm_can's traffic without consuming it -- the library's own recv_all() still
sees everything.  The filter is restricted to the motor reply IDs (0x11..0x18): the kernel also
loops locally-sent frames back to other sockets on the same interface, and without the filter
this would count the library's own commands (0x01..0x08, 0x7FF) as motor replies.

What the status nibble is for
-----------------------------
The high nibble of a DM state frame's data[0] is the motor's own status/error code, and
openarm_can's decoder skips data[0] entirely (see CanPacketDecoder::parse_motor_state_data,
which starts at data[1]).  That discarded nibble is the ONLY trustworthy "is this motor armed"
signal on the bus: Motor::is_enabled() is dead code -- Motor::set_enabled() is defined and never
called from anywhere in the library, so it returns False for all sixteen channels forever, which
is exactly what was observed on a run where five joints tracked to within 0.03 rad.
"""

import logging
import select
import socket
import struct

logger = logging.getLogger(__name__)

CANFD_MTU = 72
CAN_MTU = 16
CAN_RAW_FD_FRAMES = 5
CAN_RAW_ERR_FILTER = 2
CAN_EFF_MASK = 0x1FFFFFFF
CAN_ERR_FLAG = 0x20000000
CAN_ERR_MASK_ALL = 0x1FFFFFFF   # every error class the controller is willing to report

# canfd_frame.flags. ESI is the one that matters here: a CAN-FD transmitter sets it when it has
# gone ERROR-PASSIVE, i.e. it is still on the bus but has accumulated enough transmit errors to
# stop being trusted. A node that then goes bus-off disappears entirely and reports nothing --
# which from the host looks exactly like a motor that decided to stop talking. ESI on the frames
# *before* the silence is the difference between "this node was struggling on the wire" and
# "this node was fine and then stopped", and openarm_can never looks at it.
CANFD_BRS = 0x01
CANFD_ESI = 0x02

# High nibble of data[0] in a DM state frame. 0/1 are the normal states; everything else is the
# motor telling us it has cut its own output, which is a fault a control loop must not paper over.
DM_STATUS = {
    0x0: "disabled",
    0x1: "enabled",
    0x8: "OVERVOLTAGE",
    0x9: "UNDERVOLTAGE",
    0xA: "OVERCURRENT",
    0xB: "MOS-OVERTEMP",
    0xC: "ROTOR-OVERTEMP",
    0xD: "LOST-COMMS",
    0xE: "OVERLOAD",
}
DM_FAULT_CODES = frozenset(k for k in DM_STATUS if k >= 0x8)


class CanFeedbackMonitor:
    """Per-channel state-frame counter for one CAN interface.

    Deliberately fail-soft: if the socket cannot be opened, `available` is False and every method
    is a no-op returning empty data.  This is a diagnostic riding along on a robot control path,
    and it must never be the reason a session cannot run.
    """

    def __init__(self, port: str, channels: dict[int, str], fd: bool = True):
        """`channels` maps motor reply CAN id (0x11..0x18) -> observation key prefix ("RJ1")."""
        self.port = port
        self.channels = dict(channels)
        self.available = False
        self._sock = None
        self.counts: dict[str, int] = {name: 0 for name in self.channels.values()}
        # A second, independent tally with its own reset, so the read path can ask "has everyone
        # answered THIS request yet?" without disturbing whatever a caller is accumulating in
        # `counts`. Two consumers, two counters; sharing one made them clobber each other.
        self.cycle: dict[str, int] = {name: 0 for name in self.channels.values()}
        self.status: dict[str, int | None] = {name: None for name in self.channels.values()}
        self.total: dict[str, int] = {name: 0 for name in self.channels.values()}
        # Frames whose transmitter flagged itself error-passive, per channel, and the error
        # frames the CAN controller itself raised. Both are evidence about the WIRE, which is the
        # question a silent channel always ends up posing.
        self.esi: dict[str, int] = {name: 0 for name in self.channels.values()}
        self.errors: dict[int, int] = {}
        try:
            sock = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            if fd:
                sock.setsockopt(socket.SOL_CAN_RAW, CAN_RAW_FD_FRAMES, 1)
            # Only the motor reply IDs -- see the module docstring on local loopback.
            filters = b"".join(struct.pack("=II", cid, CAN_EFF_MASK) for cid in self.channels)
            sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
            # Error frames bypass CAN_RAW_FILTER and need their own opt-in. Without this the
            # controller's own verdict on the bus is invisible from here, and "no errors" is an
            # assumption rather than a reading -- note `ip -s link`'s bus-errors counter only
            # moves when berr-reporting is enabled on the link, which it is not by default.
            sock.setsockopt(socket.SOL_CAN_RAW, CAN_RAW_ERR_FILTER, CAN_ERR_MASK_ALL)
            # A read cycle can leave a few hundred frames queued here between polls; a roomy
            # buffer means the tap never becomes the thing that loses a frame.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            sock.bind((port,))
            sock.setblocking(False)
        except OSError as e:
            logger.warning(
                f"CAN feedback monitor unavailable on {port} ({type(e).__name__}: {e})."
                " Feedback-freshness counts will not be reported; nothing else is affected."
            )
            return
        self._sock = sock
        self.available = True

    def poll(self) -> None:
        """Drain whatever has arrived since the last poll into the per-channel counters."""
        if not self.available:
            return
        sock = self._sock
        while True:
            r, _, _ = select.select([sock], [], [], 0)
            if not r:
                return
            try:
                buf = sock.recv(CANFD_MTU)
            except BlockingIOError:
                return
            except OSError as e:
                logger.warning(f"CAN feedback monitor on {self.port} stopped: {e}")
                self.available = False
                return
            if len(buf) < CAN_MTU:
                continue
            raw_id = struct.unpack("=I", buf[:4])[0]
            if raw_id & CAN_ERR_FLAG:
                self.errors[raw_id & CAN_ERR_MASK_ALL] = self.errors.get(raw_id & CAN_ERR_MASK_ALL, 0) + 1
                continue
            can_id = raw_id & CAN_EFF_MASK
            name = self.channels.get(can_id)
            if name is None:
                continue
            if len(buf) == CANFD_MTU and buf[5] & CANFD_ESI:
                self.esi[name] += 1
            self.counts[name] += 1
            self.cycle[name] += 1
            self.total[name] += 1
            # canfd_frame and can_frame agree on the first 5 bytes (id, len/dlc) and both start
            # their payload at offset 8, so one decode covers a bus running either format.
            length = buf[4]
            if length >= 1:
                self.status[name] = buf[8] >> 4

    def mark_cycle(self) -> None:
        """Start a new "have they answered yet?" window (see `cycle`)."""
        for name in self.cycle:
            self.cycle[name] = 0

    def pending(self) -> list[str]:
        """Channels that have not sent a frame since the last mark_cycle()."""
        if not self.available:
            return []
        return [name for name, n in self.cycle.items() if n == 0]

    def take_counts(self) -> dict[str, int]:
        """Frames seen per channel since the previous take_counts(), then reset."""
        if not self.available:
            return {}
        taken = dict(self.counts)
        for name in self.counts:
            self.counts[name] = 0
        return taken

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
        self.available = False


def status_name(code: int | None) -> str:
    """Human-readable DM status nibble, or '?' when nothing has been observed on that channel."""
    if code is None:
        return "?"
    return DM_STATUS.get(code, f"0x{code:X}")


def is_fault(code: int | None) -> bool:
    return code in DM_FAULT_CODES
