#!/usr/bin/env python3
"""
Quick status check for the trading bot
Shows: API keys, safety features, current positions, profit pile, etc.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CONFIG
from trading.profit_pile import get_pile_status, format_profit_summary
import json
import os


def check_api_keys():
    """Check API key configuration"""
    print("\n" + "="*60)
    print("API KEY CONFIGURATION")
    print("="*60)
    
    def key_status(key_name):
        key = CONFIG.get(key_name, "")
        if key and key != "your-key-here":
            return f"✅ SET ({len(key)} chars)"
        return "❌ NOT SET"
    
    print(f"Coinbase API Key:     {key_status('coinbase_api_key')}")
    print(f"Coinbase Secret:      {key_status('coinbase_api_secret')}")
    print(f"Alpaca API Key:       {key_status('alpaca_api_key')}")
    print(f"Alpaca Secret:        {key_status('alpaca_secret_key')}")
    
    anthro_key = os.getenv('ANTHROPIC_API_KEY')
    if anthro_key:
        print(f"Anthropic API Key:    ✅ SET ({len(anthro_key)} chars)")
    else:
        print(f"Anthropic API Key:    ❌ NOT SET")


def check_safety_features():
    """Check safety feature configuration"""
    print("\n" + "="*60)
    print("SAFETY FEATURES")
    print("="*60)
    
    features = [
        ("Paper Trading", CONFIG.get("use_paper_trading", True)),
        ("Kill Switch", CONFIG.get("kill_switch_enabled", True)),
        ("Position Health Monitor", CONFIG.get("position_health_monitor_enabled", True)),
        ("Macro Monitoring", CONFIG.get("macro_monitoring_enabled", True)),
        ("Enforce Market Hours", CONFIG.get("enforce_market_hours", True)),
        ("Order Validation", CONFIG.get("order_validation_enabled", True)),
        ("Rate Limiting", CONFIG.get("rate_limiting_enabled", True)),
    ]
    
    for name, enabled in features:
        status = "✅ ENABLED" if enabled else "❌ DISABLED"
        print(f"{name:<30} {status}")
    
    print(f"\n{'Risk Parameters:':<30}")
    print(f"  Max Total Loss:      {CONFIG.get('max_total_loss_pct', 0.20)*100:.1f}%")
    print(f"  Risk Per Trade:      {CONFIG.get('risk_per_trade_pct', 0.02)*100:.1f}%")
    print(f"  Max Positions:       {CONFIG.get('max_open_positions', 5)}")


def check_kill_switch_status():
    """Check if kill switch is tripped"""
    print("\n" + "="*60)
    print("KILL SWITCH STATUS")
    print("="*60)
    
    state_file = os.path.join(CONFIG.get("state_dir", "state"), "killswitch.json")
    
    try:
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            if state.get("tripped"):
                print("🚨 STATUS: TRIPPED")
                print(f"   Reason: {state.get('reason', 'Unknown')}")
                print(f"   Since: {state.get('tripped_at', 'Unknown')}")
                print(f"   Cooldown Until: {state.get('cooldown_until', 'Unknown')}")
            else:
                print("✅ STATUS: ACTIVE (No kill conditions)")
                print(f"   Last Check: {state.get('last_check', 'Never')}")
        else:
            print("✅ STATUS: ACTIVE (No state file yet - first run)")
    except Exception as e:
        print(f"⚠️  Could not load kill switch state: {e}")


def check_profit_pile():
    """Check profit pile status"""
    print("\n" + "="*60)
    print("PROFIT PILE STATUS")
    print("="*60)
    
    try:
        print(format_profit_summary())
    except Exception as e:
        print(f"⚠️  Could not load profit pile: {e}")


def check_recent_logs():
    """Show last few log entries"""
    print("\n" + "="*60)
    print("RECENT LOG ENTRIES (last 10)")
    print("="*60)
    
    log_file = "logs/bot.log"
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(line.strip())
        except Exception as e:
            print(f"⚠️  Could not read log file: {e}")
    else:
        print("⚠️  Log file not found")


def main():
    print("\n" + "="*60)
    print("TRADING BOT STATUS CHECK")
    print("="*60)
    print(f"Asset Class: {CONFIG.get('asset_class', 'both')}")
    print(f"Trading Mode: {CONFIG.get('trading_mode', 'balanced')}")
    print(f"Capital: ${CONFIG.get('capital', 0):,.2f}")
    
    check_api_keys()
    check_safety_features()
    check_kill_switch_status()
    check_profit_pile()
    check_recent_logs()
    
    print("\n" + "="*60)
    print("Status check complete!")
    print("="*60)


if __name__ == "__main__":
    main()
