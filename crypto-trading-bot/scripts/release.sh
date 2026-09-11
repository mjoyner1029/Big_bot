#!/usr/bin/env bash
# Release script — creates a clean zip that NEVER contains secrets.
#
# Excluded:
#   .env / .env.*         — API keys and secrets
#   *.pem / *.key         — certificates and private keys
#   .venv/                — virtual environment (too large)
#   __pycache__/          — Python bytecode
#   *.pyc / *.pyo         — compiled Python
#   data/*.sqlite         — live trade database
#   logs/                 — runtime logs
#   state/                — browser/session state
#   .pytest_cache/        — test artefacts
#
# Usage:
#   cd crypto-trading-bot
#   bash scripts/release.sh
#   # → ../crypto-trading-bot-<date>.zip

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_NAME="$(basename "$REPO_DIR")"
DATE="$(date +%Y%m%d)"
OUT="$REPO_DIR/../${REPO_NAME}-${DATE}.zip"

echo "Creating release zip: $OUT"
echo "Source: $REPO_DIR"

cd "$REPO_DIR/.."

zip -r "$OUT" "$REPO_NAME" \
  --exclude "${REPO_NAME}/.env" \
  --exclude "${REPO_NAME}/.env.*" \
  --exclude "${REPO_NAME}/**/.env" \
  --exclude "${REPO_NAME}/**/.env.*" \
  --exclude "${REPO_NAME}/*.pem" \
  --exclude "${REPO_NAME}/*.key" \
  --exclude "${REPO_NAME}/**/*.pem" \
  --exclude "${REPO_NAME}/**/*.key" \
  --exclude "${REPO_NAME}/.venv/*" \
  --exclude "${REPO_NAME}/__pycache__/*" \
  --exclude "${REPO_NAME}/**/__pycache__/*" \
  --exclude "${REPO_NAME}/*.pyc" \
  --exclude "${REPO_NAME}/**/*.pyc" \
  --exclude "${REPO_NAME}/*.pyo" \
  --exclude "${REPO_NAME}/**/*.pyo" \
  --exclude "${REPO_NAME}/data/*.sqlite" \
  --exclude "${REPO_NAME}/logs/*" \
  --exclude "${REPO_NAME}/state/*" \
  --exclude "${REPO_NAME}/.pytest_cache/*" \
  --exclude "${REPO_NAME}/**/.pytest_cache/*" \
  2>/dev/null

SIZE=$(du -sh "$OUT" | cut -f1)
echo ""
echo "Release created: $OUT ($SIZE)"

# Safety check — verify no .env leaked in
if unzip -l "$OUT" | grep -q "\.env$\|\.env\."; then
  echo ""
  echo "ERROR: .env file detected in zip! Aborting release."
  rm "$OUT"
  exit 1
fi

echo "Security check: no .env files found in zip."
echo "Done."
