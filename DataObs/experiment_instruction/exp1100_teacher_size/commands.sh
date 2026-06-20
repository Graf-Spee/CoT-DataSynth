#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible default: Qwen2.5 teacher-size curve.
bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/commands_qwen2_5.sh"
