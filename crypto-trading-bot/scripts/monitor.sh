#!/bin/bash
# Real-time monitoring script for the trading bot
# Usage: ./scripts/monitor.sh

echo "========================================="
echo "TRADING BOT REAL-TIME MONITOR"
echo "========================================="
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Watch logs in real-time with color highlighting
tail -f logs/bot.log | while read line; do
    if [[ $line == *"ERROR"* ]] || [[ $line == *"Kill Switch"* ]]; then
        echo -e "${RED}$line${NC}"
    elif [[ $line == *"Signal"* ]] || [[ $line == *"Trade"* ]] || [[ $line == *"✓"* ]]; then
        echo -e "${GREEN}$line${NC}"
    elif [[ $line == *"WARNING"* ]] || [[ $line == *"Position Health"* ]]; then
        echo -e "${YELLOW}$line${NC}"
    else
        echo "$line"
    fi
done
