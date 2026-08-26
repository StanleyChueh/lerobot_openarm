#!/bin/bash
# Recover the DM-USB2FDCAN adapter when its transmit path has wedged, and bring both CAN
# channels back up with this rig's verified bit timing.
#
# The symptom this fixes: every write fails with ENOBUFS ("No buffer space available") while
# `ip link` still cheerfully reports UP / ERROR-ACTIVE, so nothing reaches the bus and every
# motor looks dead. `tc -s qdisc show dev can0` shows it plainly -- a Sent counter that does not
# move with nothing running, or a backlog that will not drain.
#
# `ip link set down/up` does NOT clear it (verified 2026-08-26: both channels still refused ~94%
# of writes afterwards). The device has to be re-enumerated. This script does that in software,
# so you do not have to reach behind the machine for the USB cable.
#
# Both channels live on ONE USB interface, so they wedge together and are recovered together --
# there is no way to reset just one.
#
#   sudo ./can_recover.sh            # unbind/rebind the driver, then configure can0 + can1
#   sudo ./can_recover.sh --reset    # use usbreset instead (harder; try if the above fails)
#
# Afterwards, verify with:  python can_link_check.py --port can1 --duration 5
set -u

VID_PID="1d50:606f"
BITRATE=1000000
DBITRATE=5000000
SAMPLE_POINT=0.75
PORTS=(can0 can1)

if [[ $EUID -ne 0 ]]; then
    echo "This needs root (it rebinds a USB driver and reconfigures network links)."
    echo "  sudo $0 $*"
    exit 1
fi

# The USB path is discovered rather than hardcoded: it changes with the physical port, and a
# hardcoded "1-5" that silently stops matching would make this script quietly do nothing.
find_usb_path() {
    local dev
    for dev in /sys/bus/usb/devices/*; do
        [[ -r "$dev/idVendor" && -r "$dev/idProduct" ]] || continue
        if [[ "$(cat "$dev/idVendor"):$(cat "$dev/idProduct")" == "$VID_PID" ]]; then
            basename "$dev"
            return 0
        fi
    done
    return 1
}

USB_PATH="$(find_usb_path)" || {
    echo "No $VID_PID adapter found in /sys/bus/usb/devices."
    echo "It is not enumerated at all -- check the USB cable and 'lsusb | grep 1d50'."
    exit 1
}
echo "Adapter: $VID_PID at $USB_PATH"

if [[ "${1:-}" == "--reset" ]]; then
    echo "Re-enumerating with usbreset..."
    usbreset "$VID_PID" || { echo "usbreset failed."; exit 1; }
else
    IFACE="$USB_PATH:1.0"
    echo "Rebinding gs_usb on $IFACE..."
    # Failure here is expected and harmless if the driver already let go of the device.
    echo "$IFACE" > /sys/bus/usb/drivers/gs_usb/unbind 2>/dev/null
    sleep 1
    echo "$IFACE" > /sys/bus/usb/drivers/gs_usb/bind 2>/dev/null
fi

# The interfaces come back as fresh netdevs: DOWN, and with no bitrate set at all. Waiting for
# them to reappear beats a fixed sleep, which is either too short on a slow enumeration or wasted
# time on a fast one.
echo -n "Waiting for interfaces"
for _ in $(seq 1 20); do
    if [[ -d "/sys/class/net/${PORTS[0]}" && -d "/sys/class/net/${PORTS[1]}" ]]; then
        echo " ok"
        break
    fi
    echo -n "."
    sleep 0.5
done
sleep 1

for port in "${PORTS[@]}"; do
    if [[ ! -d "/sys/class/net/$port" ]]; then
        echo "$port never came back -- unplug and replug the adapter's USB cable."
        exit 1
    fi
    ip link set "$port" down 2>/dev/null
    ip link set "$port" type can \
        bitrate "$BITRATE" sample-point "$SAMPLE_POINT" \
        dbitrate "$DBITRATE" dsample-point "$SAMPLE_POINT" fd on || {
            echo "Failed to configure $port."; exit 1; }
    ip link set "$port" up || { echo "Failed to bring $port up."; exit 1; }
done
sleep 1

echo
for port in "${PORTS[@]}"; do
    printf '  %s: ' "$port"
    ip -details link show "$port" | awk '/bitrate/ {printf "%s ", $0}' | tr -s ' '
    echo
done

# An interface that is UP is not proof of anything: the whole failure this script exists to fix
# leaves the link UP and ERROR-ACTIVE while every write is refused. So the script verifies its
# own work rather than telling the operator to go and check -- it declared success once while the
# adapter was still dead, which is worse than failing loudly.
echo
failed=()
for port in "${PORTS[@]}"; do
    before=$(cat "/sys/class/net/$port/statistics/tx_packets")
    for _ in $(seq 1 40); do
        # A DM state query addressed to motor 1: read-only, commands no motion, and harmless
        # even if a motor is listening. We only care whether it leaves the host.
        cansend "$port" "7FF#0100CC0000000000" 2>/dev/null
    done
    sleep 0.5
    after=$(cat "/sys/class/net/$port/statistics/tx_packets")
    sent=$((after - before))
    if [[ $sent -ge 35 ]]; then
        printf '  %s: OK -- %d/40 test frames transmitted\n' "$port" "$sent"
    else
        printf '  %s: STILL WEDGED -- only %d/40 test frames left the host\n' "$port" "$sent"
        failed+=("$port")
    fi
done

if [[ ${#failed[@]} -eq 0 ]]; then
    echo
    echo "Adapter is transmitting. Now check the motors:"
    echo "  python can_link_check.py --port can1 --duration 5     # left arm, expect J1..J8 at 100%"
    echo "  python can_link_check.py --port can0 --duration 5     # right arm"
    exit 0
fi

echo
echo "NOT RECOVERED: ${failed[*]} still will not transmit."
echo
echo "IS THE ARM POWERED ON? Check that before suspecting anything else."
echo "  A CAN frame has to be acknowledged by at least one OTHER node on the bus. With the arm"
echo "  unpowered nobody acknowledges, the controller retransmits forever, the queue fills, and"
echo "  every write is refused -- while the link still reports UP and ERROR-ACTIVE. An unpowered"
echo "  robot is indistinguishable from a dead adapter from up here. Confirmed 2026-08-26, when"
echo "  it cost an afternoon of chasing the adapter and the arm's wrist harness."
echo "  Power the arm on, then run this script again to clear the backlog it built up."
echo
echo "If the arm IS powered on, escalate in this order:"
echo "  1. sudo $0 --reset          # real USB port reset, stronger than a rebind"
echo "  2. Unplug the adapter's USB cable, wait 5s, plug it back in, re-run this script."
echo "  3. Move it to a different USB port -- ideally a different root hub -- and use a"
echo "     different cable. This adapter is bus-powered, and gs_usb wedges exactly like this"
echo "     on marginal USB power or a flaky cable."
echo "  4. Only then suspect the adapter itself."
exit 1
