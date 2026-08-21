#!/usr/bin/env bash
set -euo pipefail

# Prepare the derived data, train seed 4, and evaluate the matching checkpoint.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ "${QALF_SKIP_32F_PREPARE:-0}" != "1" ]]; then
    ./prepare_32f_data.sh
fi
./run_train_32f_seed4.sh
./run_test_32f_seed4.sh
