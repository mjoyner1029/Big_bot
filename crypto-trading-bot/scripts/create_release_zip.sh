#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_NAME="${1:-crypto-trading-bot-release.zip}"
OUT_PATH="$ROOT_DIR/$OUT_NAME"

cd "$ROOT_DIR"

# Build a clean release archive that excludes local/dev/runtime artifacts.
zip -r "$OUT_PATH" . \
  -x ".git/*" \
  -x ".venv/*" \
  -x "__pycache__/*" \
  -x "*/__pycache__/*" \
  -x ".pytest_cache/*" \
  -x "logs/*" \
  -x "state/*" \
  -x "*.pyc" \
  -x "*.pyo" \
  -x "*.zip"

echo "Created: $OUT_PATH"
