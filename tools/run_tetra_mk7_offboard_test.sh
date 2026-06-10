#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GZ_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PX4_ROOT="$(cd -- "${GZ_ROOT}/../../.." && pwd)"
SESSION="tetra_move_test"

cleanup() {
  "${SCRIPT_DIR}/stop_tetra_mk7_sitl.sh" >/dev/null 2>&1 || true
}

latest_log() {
  find "${PX4_ROOT}/build/px4_sitl_default/rootfs/log" -name "*.ulg" -printf "%T@ %p\n" 2>/dev/null \
    | sort -n \
    | tail -1 \
    | cut -d" " -f2-
}

trap cleanup EXIT INT TERM

cd "${PX4_ROOT}"
cleanup
rm -f build/px4_sitl_default/rootfs/parameters.bson build/px4_sitl_default/rootfs/parameters_backup.bson

before="$(latest_log || true)"
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "cd '${PX4_ROOT}' && HEADLESS=1 make px4_sitl gz_teTra_mk-7_EM2"

sleep 18
"${SCRIPT_DIR}/tetra_mk7_offboard_test.py" --connect udpin:0.0.0.0:14550 "$@"
rc=$?

sleep 5
cleanup
after="$(latest_log || true)"

echo "TEST_RC=${rc}"
echo "BEFORE=${before}"
echo "AFTER=${after}"
exit "${rc}"
