#!/usr/bin/env bash
set -e
cd "$(dirname "$(readlink -f "$0")")"
exec "$HOME/miniconda3/envs/RL/bin/python" deploy.py "$@"
