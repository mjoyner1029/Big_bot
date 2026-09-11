#!/bin/bash
# Quick testing menu for Ultimate Bot V3

echo "======================================================================"
echo "ULTIMATE BOT V3 - TESTING MENU"
echo "======================================================================"
echo ""
echo "Choose testing mode:"
echo ""
echo "1. 📊 BACKTEST     - Test on historical data (5-10 min, zero risk)"
echo "2. 📝 PAPER TRADE  - Test with live data (24+ hrs, zero risk)"
echo "3. 💰 LIVE TRADE   - Real trading (REAL MONEY! High risk)"
echo ""
echo "4. 📖 View Testing Guide"
echo "5. 📁 Check Results"
echo "6. 🚪 Exit"
echo ""
echo "======================================================================"
read -p "Enter choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "🔄 Starting backtest..."
        echo ""
        python backtest_v3.py
        echo ""
        echo "✅ Backtest complete!"
        echo "📄 Results saved to: logs/backtest_results.json"
        echo ""
        read -p "Press Enter to continue..."
        ;;
    2)
        echo ""
        echo "⚠️  PAPER TRADING MODE"
        echo ""
        echo "This will run continuously until you stop it (Ctrl+C)"
        echo "Recommended: Run for 24+ hours"
        echo ""
        read -p "Press Enter to start, or Ctrl+C to cancel..."
        echo ""
        python paper_trade_v3.py
        ;;
    3)
        echo ""
        echo "⚠️⚠️⚠️  LIVE TRADING MODE - REAL MONEY ⚠️⚠️⚠️"
        echo ""
        echo "Have you completed backtesting and paper trading?"
        read -p "Type 'YES' to continue: " confirm
        
        if [ "$confirm" != "YES" ]; then
            echo "❌ Cancelled. Run backtest and paper trading first."
            exit 0
        fi
        
        echo ""
        python live_test_v3.py
        ;;
    4)
        echo ""
        echo "Run: python backtest_v3.py          # Backtest"
        echo "Run: python paper_trade_v3.py       # Paper trade"
        echo "Run: python live_test_v3.py         # Live trade"
        echo "Logs: tail -f logs/bot.log"
        ;;
    5)
        echo ""
        echo "======================================================================"
        echo "RESULTS"
        echo "======================================================================"
        echo ""
        
        if [ -f "logs/backtest_results.json" ]; then
            echo "📊 BACKTEST RESULTS:"
            cat logs/backtest_results.json
            echo ""
        else
            echo "📊 No backtest results found"
            echo ""
        fi
        
        if [ -f "logs/paper_trade.log" ]; then
            echo "📝 PAPER TRADING (Last 20 lines):"
            tail -20 logs/paper_trade.log
            echo ""
        else
            echo "📝 No paper trading logs found"
            echo ""
        fi
        
        if [ -f "logs/live_trading.log" ]; then
            echo "💰 LIVE TRADING (Last 20 lines):"
            tail -20 logs/live_trading.log
            echo ""
        else
            echo "💰 No live trading logs found"
            echo ""
        fi
        
        read -p "Press Enter to continue..."
        ;;
    6)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
