#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/hrh/anaconda3/envs/verl-torch280/bin:${PATH}"

python DataObs/tools/run_qwen3_base_oc_eval.py "$@"
