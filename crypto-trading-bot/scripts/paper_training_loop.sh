#!/usr/bin/env bash
# paper_training_loop.sh
# Runs the paper trading bot indefinitely for model validation & training.
# Auto-restarts on crash/exit. Logs to logs/paper_training_console.log.
# Usage: ./scripts/paper_training_loop.sh [--capital N]
#
# To run in the background:
#   nohup ./scripts/paper_training_loop.sh &
# To stop:
#   pkill -f paper_training_loop.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
LOG="$PROJECT_ROOT/logs/paper_training_console.log"
CAPITAL="${1:-10000}"

# Parse --capital flag if provided
while [[ $# -gt 0 ]]; do
  case "$1" in
    --capital) CAPITAL="$2"; shift 2 ;;
    *) shift ;;
  esac
done

mkdir -p "$PROJECT_ROOT/logs" "$PROJECT_ROOT/state" "$PROJECT_ROOT/reports/paper_training"

echo "[paper-loop] Starting continuous paper trading — capital=$CAPITAL" >> "$LOG"
echo "[paper-loop] PID=$$  $(date)" >> "$LOG"

RESTART_COUNT=0
MIN_RESTART_INTERVAL=30   # seconds — prevent tight crash loop

while true; do
  LOOP_START=$(date +%s)
  echo "[paper-loop] === Start run #$((RESTART_COUNT+1)) at $(date) ===" >> "$LOG"

  "$PYTHON" -u "$PROJECT_ROOT/scripts/run_intraday_evidence.py" \
    --capital "$CAPITAL" \
    --asset-class both \
    --trading-mode conservative \
    --confidence-threshold 0.55 \
    --approval-threshold 55 \
    --risk-per-trade-pct 0.02 \
    --max-open-positions 3 \
    --max-position-pct 0.10 \
    --target-closed-trades 99999 \
    --check-every-cycles 1 \
    --flatten-every-cycles 480 \
    --max-cycles 99999 \
    --sleep-seconds 60 \
    --trade-log-path "$PROJECT_ROOT/logs/trade_log_paper_live.csv" \
    --bot-log-path "$PROJECT_ROOT/logs/bot_paper_training.log" \
    --state-path "$PROJECT_ROOT/state/paper_training_state.json" \
    --snapshot-out-dir "$PROJECT_ROOT/reports/paper_training" \
    </dev/null >> "$LOG" 2>&1 || true

  LOOP_END=$(date +%s)
  ELAPSED=$((LOOP_END - LOOP_START))
  RESTART_COUNT=$((RESTART_COUNT + 1))

  echo "[paper-loop] Run ended after ${ELAPSED}s. Restarts so far: $RESTART_COUNT" >> "$LOG"

  # Avoid tight crash loop — wait if we crashed very quickly
  if [[ $ELAPSED -lt $MIN_RESTART_INTERVAL ]]; then
    SLEEP=$((MIN_RESTART_INTERVAL - ELAPSED))
    echo "[paper-loop] Crashed too fast — waiting ${SLEEP}s before restart" >> "$LOG"
    sleep "$SLEEP"
  fi

  echo "[paper-loop] Restarting at $(date)..." >> "$LOG"
done
