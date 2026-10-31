#!/usr/bin/env bash
set -euo pipefail

MODEL_FAMILY=qwen3 bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/commands.sh"
