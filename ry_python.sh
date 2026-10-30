#!/usr/bin/env bash
set -e
RY_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UNITREE_RL_LAB_PATH="$RY_PROJECT_ROOT"
export PYTHONPATH="$RY_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$RY_PROJECT_ROOT"
exec /workspace/isaaclab/_isaac_sim/python.sh "$@"
