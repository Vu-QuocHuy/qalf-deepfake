#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export QALF_TEXTURE_BACKBONE='efficientnet_b1'
exec "$SCRIPT_DIR/run_test_flip_consistency.sh"
