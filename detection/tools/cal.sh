#!/usr/bin/env bash
set -euo pipefail

CONFIG=$1
CHECKPOINT=$2
GPUS=${3:-1}
PORT=${PORT:-29400}

PYTHONPATH="$(dirname "$0")/..:${PYTHONPATH:-}" \
python -m torch.distributed.launch \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    "$(dirname "$0")/Cal_firing_num.py" \
    "$CONFIG" \
    "$CHECKPOINT" \
    --launcher pytorch "${@:4}"
