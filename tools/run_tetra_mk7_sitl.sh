#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="${1:-teTra_mk-7_EM2}"
SIM_MODEL="gz_${MODEL_NAME}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GZ_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PX4_ROOT="$(cd -- "${GZ_ROOT}/../../.." && pwd)"
STOP_SCRIPT="${SCRIPT_DIR}/stop_tetra_mk7_sitl.sh"

cleanup() {
  "${STOP_SCRIPT}" "${MODEL_NAME}" >/dev/null 2>&1 || true
}

trap cleanup EXIT INT TERM

cleanup
cd "${PX4_ROOT}"
make px4_sitl "${SIM_MODEL}"
