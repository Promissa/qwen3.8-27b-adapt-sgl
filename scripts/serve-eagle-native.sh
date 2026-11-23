#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/runtime-env.sh
exec .venv/bin/python scripts/serve.py eagle-native "$@"
