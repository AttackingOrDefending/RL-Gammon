#!/usr/bin/env bash
# Helper for local testing: start the gnubg bridge, run a python client, then stop the bridge.
# Usage: bash run_with_bridge.sh <python-module-or-script> [timeout-seconds]
# Must be run from the repo root on a machine with gnubg installed (e.g. WSL Debian).
#
# NOTE: do NOT use `pkill -f bridge.py` to clean up from the launching shell -- the
# pattern matches the launching command line itself and kills the parent shell.
# Always kill the gnubg bridge by the PID captured below.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

TARGET="${1:-rlgammon/environment/gnubg/smoke_client.py}"
TIMEOUT_SECS="${2:-60}"
GNUBG_BIN="${GNUBG_BIN:-/usr/games/gnubg}"
BRIDGE="rlgammon/environment/gnubg/bridge.py"
BRIDGE_LOG="/tmp/gnubg_bridge.log"

"$GNUBG_BIN" -t -q -p "$BRIDGE" > "$BRIDGE_LOG" 2>&1 &
BPID=$!

cleanup() { kill "$BPID" 2>/dev/null; }
trap cleanup EXIT

# Wait for the bridge to bind localhost:8001.
for _ in $(seq 1 30); do
    if ss -ltn 2>/dev/null | grep -q ":8001 "; then break; fi
    if ! kill -0 "$BPID" 2>/dev/null; then
        echo "BRIDGE FAILED TO START:"; cat "$BRIDGE_LOG"; exit 1
    fi
    sleep 1
done

echo "=== bridge up (pid=$BPID), running: $TARGET ==="
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
timeout "$TIMEOUT_SECS" python3 "$TARGET"
RC=$?
if [ "$RC" -eq 124 ]; then
    echo "=== TARGET TIMED OUT after ${TIMEOUT_SECS}s ==="
fi
echo "=== target rc=$RC ==="
echo "=== bridge log tail ==="
tail -25 "$BRIDGE_LOG"
exit "$RC"
