#!/usr/bin/env bash
# ============================================================
# START.sh — canonical startup script for the trading bot
#
# Workflow:
#   1. Locate project root
#   2. Verify virtual environment
#   3. Run preflight (exits on failure)
#   4. Launch correct entrypoint for configured mode
#   5. Log startup metadata
#
# The bot will NEVER start if preflight fails.
# LIVE mode is intentionally hard to activate.
#
# Usage:
#   ./START.sh             # reads EXECUTION_MODE from .env
#   ./START.sh --mode PAPER
#   ./START.sh --mode SHADOW
#   ./START.sh --smoke     # run smoke test before starting
# ============================================================

set -euo pipefail

# ── Locate project root ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
cd "$PROJECT_ROOT"

# ── Parse args ────────────────────────────────────────────────────────────────
MODE_OVERRIDE=""
SMOKE_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode=*)  MODE_OVERRIDE="${1#--mode=}"; shift ;;
    --mode)
      [[ $# -ge 2 ]] || { echo "--mode requires a value"; exit 1; }
      MODE_OVERRIDE="$2"; shift 2 ;;
    --smoke)   SMOKE_FLAG="--smoke"; shift ;;
    --help|-h)
      echo "Usage: ./START.sh [--mode PAPER|SHADOW|BACKTEST|LIVE] [--smoke]"
      exit 0 ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: ./START.sh [--mode PAPER|SHADOW|BACKTEST|LIVE] [--smoke]"
      exit 1 ;;
  esac
done

# ── Load .env early (bash-compatible) ────────────────────────────────────────
if [[ -f ".env" ]]; then
  set -o allexport
  # shellcheck disable=SC1091
  source .env 2>/dev/null || true
  set +o allexport
else
  echo ""
  echo "  WARNING: .env not found."
  echo "  Copy .env.example to .env and configure your keys."
  echo "  Run: cp .env.example .env"
  echo ""
fi

# ── Virtual environment ───────────────────────────────────────────────────────
VENV_DIR="$PROJECT_ROOT/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
  echo ""
  echo "  ======================================"
  echo "  NO-GO"
  echo "  ======================================"
  echo "  Virtual environment not found: $VENV_DIR"
  echo ""
  echo "  Create it with:"
  echo "    python3 -m venv .venv"
  echo "    source .venv/bin/activate"
  echo "    pip install -r requirements.txt"
  echo ""
  exit 1
fi

PYTHON="$VENV_DIR/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
  echo ""
  echo "  ======================================"
  echo "  NO-GO"
  echo "  ======================================"
  echo "  Python not found in .venv: $PYTHON"
  echo "  Recreate the virtual environment:"
  echo "    python3 -m venv .venv"
  echo "    source .venv/bin/activate"
  echo "    pip install -r requirements.txt"
  echo ""
  exit 1
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo ""
echo "  Python: $PYTHON ($PY_VERSION)"

# ── Determine execution mode ──────────────────────────────────────────────────
EXEC_MODE="${MODE_OVERRIDE:-${EXECUTION_MODE:-${TRADING_MODE:-PAPER}}}"

# Normalize to uppercase (portable: tr instead of ${var^^})
EXEC_MODE=$(echo "$EXEC_MODE" | tr '[:lower:]' '[:upper:]')

# Reject unknown / obsolete modes immediately
case "$EXEC_MODE" in
  BACKTEST|PAPER|SHADOW|LIVE) ;;
  BALANCED|CONSERVATIVE|AGGRESSIVE|CLAUDE_HF)
    echo ""
    echo "  ======================================"
    echo "  NO-GO"
    echo "  ======================================"
    echo "  '$EXEC_MODE' is a strategy profile, not an execution mode."
    echo "  Set EXECUTION_MODE to one of: BACKTEST PAPER SHADOW LIVE"
    echo ""
    exit 1 ;;
  *)
    echo ""
    echo "  ======================================"
    echo "  NO-GO"
    echo "  ======================================"
    echo "  Unknown execution mode: '$EXEC_MODE'"
    echo "  Allowed: BACKTEST PAPER SHADOW LIVE"
    echo ""
    exit 1 ;;
esac

echo "  Mode:   $EXEC_MODE"

# ── Extra guard for LIVE mode ─────────────────────────────────────────────────
if [[ "$EXEC_MODE" == "LIVE" ]]; then
  ENABLE_LIVE="${ENABLE_LIVE_TRADING:-false}"
  ENABLE_LIVE_LOWER=$(echo "$ENABLE_LIVE" | tr '[:upper:]' '[:lower:]')
  if [[ "$ENABLE_LIVE_LOWER" != "true" ]]; then
    echo ""
    echo "  ======================================"
    echo "  NO-GO"
    echo "  ======================================"
    echo "  EXECUTION_MODE=LIVE but ENABLE_LIVE_TRADING is not 'true'."
    echo "  Both must be explicitly set to deploy real capital."
    echo ""
    exit 1
  fi
  echo ""
  echo "  WARNING: LIVE mode — real capital will be at risk."
  echo "  You have 5 seconds to abort (Ctrl-C)."
  echo ""
  sleep 5
fi

# ── Run preflight ─────────────────────────────────────────────────────────────
echo ""
echo "  Running preflight checks..."
echo ""

PREFLIGHT_ARGS="--mode $EXEC_MODE $SMOKE_FLAG"

if ! "$PYTHON" scripts/preflight.py $PREFLIGHT_ARGS; then
  echo ""
  echo "  ======================================"
  echo "  NO-GO — preflight failed"
  echo "  ======================================"
  echo "  Fix the issues above before starting."
  echo ""
  exit 1
fi

# ── Select entrypoint ─────────────────────────────────────────────────────────
BOT_LOG="logs/bot.log"
mkdir -p logs

case "$EXEC_MODE" in
  BACKTEST)
    ENTRYPOINT="backtest_v3.py"
    ;;
  PAPER)
    ENTRYPOINT="paper_trade_v3.py"
    ;;
  SHADOW)
    ENTRYPOINT="paper_trade_v3.py --shadow"
    ;;
  LIVE)
    ENTRYPOINT="live_test_v3.py"
    ;;
esac

# Verify entrypoint exists
ENTRYPOINT_FILE="$(echo "$ENTRYPOINT" | awk '{print $1}')"
if [[ ! -f "$PROJECT_ROOT/$ENTRYPOINT_FILE" ]]; then
  echo ""
  echo "  ======================================"
  echo "  NO-GO"
  echo "  ======================================"
  echo "  Entrypoint not found: $ENTRYPOINT_FILE"
  echo ""
  exit 1
fi

# ── Launch ────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "" >> "$BOT_LOG"
echo "[$TIMESTAMP] START.sh: launching mode=$EXEC_MODE entrypoint=$ENTRYPOINT" >> "$BOT_LOG"

echo ""
echo "  Starting: $ENTRYPOINT"
echo "  Log:      tail -f $BOT_LOG"
echo "  Stop:     pkill -f $ENTRYPOINT_FILE"
echo ""

# shellcheck disable=SC2086
nohup "$PYTHON" $ENTRYPOINT >> "$BOT_LOG" 2>&1 </dev/null &
BOT_PID=$!

mkdir -p pids
EXEC_MODE_LOWER=$(echo "$EXEC_MODE" | tr '[:upper:]' '[:lower:]')
# The paper entrypoint owns and locks its PID file. Writing it here would race
# the lock owner and could replace the active PID when a duplicate start fails.
if [[ "$EXEC_MODE" != "PAPER" && "$EXEC_MODE" != "SHADOW" ]]; then
  echo "$BOT_PID" > "pids/bot_${EXEC_MODE_LOWER}.pid"
fi

sleep 2

if kill -0 "$BOT_PID" 2>/dev/null; then
  echo "  Bot started (PID $BOT_PID)"
  echo "  Monitor: tail -f $BOT_LOG"
  echo "  Stop:    kill $BOT_PID"
  echo ""
else
  echo "  Bot failed to start — check log:"
  tail -30 "$BOT_LOG"
  exit 1
fi
