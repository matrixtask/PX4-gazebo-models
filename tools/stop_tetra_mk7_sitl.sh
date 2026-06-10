#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="${1:-teTra_mk-7_EM2}"
SIM_MODEL="gz_${MODEL_NAME}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GZ_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PX4_ROOT="$(cd -- "${GZ_ROOT}/../../.." && pwd)"

declare -a NEEDLES=(
  "make px4_sitl ${SIM_MODEL}"
  "PX4_SIM_MODEL=${SIM_MODEL}"
  "${PX4_ROOT}/build/px4_sitl_default/bin/px4"
  "${GZ_ROOT}/worlds/default.sdf"
)

declare -a PIDS=()

if command -v tmux >/dev/null 2>&1; then
  for session in tetra_tune tetra_mk7 tetra_mk7_sitl; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      echo "Stopping tmux session: ${session}"
      tmux kill-session -t "${session}" || true
    fi
  done
fi

for needle in "${NEEDLES[@]}"; do
  while IFS= read -r pid; do
    PIDS+=("${pid}")
  done < <(
    ps -eo pid=,args= \
      | awk -v needle="${needle}" -v self="$$" '
          index($0, needle) > 0 && $1 != self && index($0, "stop_tetra_mk7_sitl.sh") == 0 {
            print $1
          }'
  )
done

if [ "${#PIDS[@]}" -eq 0 ]; then
  echo "No matching PX4/Gazebo SITL processes found for ${SIM_MODEL}."
  exit 0
fi

mapfile -t UNIQUE_PIDS < <(printf "%s\n" "${PIDS[@]}" | sort -un)

echo "Stopping PX4/Gazebo SITL processes: ${UNIQUE_PIDS[*]}"
kill -TERM "${UNIQUE_PIDS[@]}" 2>/dev/null || true
sleep 2

declare -a ALIVE=()
for pid in "${UNIQUE_PIDS[@]}"; do
  if kill -0 "${pid}" 2>/dev/null; then
    ALIVE+=("${pid}")
  fi
done

if [ "${#ALIVE[@]}" -gt 0 ]; then
  echo "Force-stopping remaining processes: ${ALIVE[*]}"
  kill -KILL "${ALIVE[@]}" 2>/dev/null || true
fi

echo "PX4/Gazebo SITL cleanup complete."
