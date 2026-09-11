#!/usr/bin/env python3
"""
Backtesting Engine for Ultimate Bot V3
Tests the bot on historical data to validate performance
"""
import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# For backtesting, API key is optional (uses fallback strategies)
if not os.getenv('ANTHROPIC_API_KEY'):
    print("ℹ️  No ANTHROPIC_API_KEY found - using rule-based fallback strategies")
    os.environ['ANTHROPIC_API_KEY'] = 'backtest-mode'

from data.fetcher import fetch_latest_market_data
from ultimate_bot_v3_llm import LLMTradingBot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BacktestEngine:
    """Backtest the bot on historical data."""
    
    def __init__(self, start_capital=6090):
        self.start_capital = start_capital
        self.capital = start_capital
        self.positions = []
        self.closed_trades = []
        
        # Performance tracking
        self.equity_curve = []
        self.daily_pnl = []
        
        logger.info("="*70)
        logger.info("BACKTEST ENGINE INITIALIZED")
        logger.info(f"Starting capital: ${start_capital:,.0f}")
        logger.info("="*70)
    
    def run(self, symbols: List[str], days: int = 7):
        """Run backtest on historical data."""
        logger.info(f"\nBacktesting {len(symbols)} symbols for {days} days...")
        
        # Fetch historical data
        historical_data = {}
        for symbol in symbols:
            logger.info(f"Fetching {symbol} data...")
            df = fetch_latest_market_data(symbol, period=f'{days}d', interval='5m')
            if df is not None:
                historical_data[symbol] = df
                logger.info(f"  Got {len(df)} candles")
        
        if not historical_data:
            logger.error("No historical data fetched!")
            return None
        
        # Get min length
        min_length = min(len(df) for df in historical_data.values())
        logger.info(f"\nBacktesting on {min_length} candles (5-min bars)")
        
        # Initialize bot (will select strategies once)
        bot = LLMTradingBot(capital=self.start_capital)
        
        # Mock regime analysis (use default strategies)
        bot.active_strategies = {
            'mean_reversion': bot.all_strategies['mean_reversion'],
            'breakout': bot.all_strategies['breakout'],
            'momentum': bot.all_strategies['momentum']
        }
        logger.info(f"Using strategies: {list(bot.active_strategies.keys())}")
        
        # Simulate trading on each candle
        for i in range(100, min_length):  # Start at 100 to have history
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{min_length} ({i/min_length*100:.1f}%)")
            
            # Get current data up to this point
            current_time = None
            for symbol in symbols:
                df_slice = historical_data[symbol].iloc[:i+1]
                current_time = df_slice.index[-1]
                
                # Manage existing positions
                self._manage_positions(symbol, df_slice)
                
                # Look for new trades (simplified - skip some bot logic)
                if len(self.positions) < 5:  # Max 5 positions
                    self._check_entry(bot, symbol, df_slice)
            
            # Track equity
            current_equity = self._calculate_equity(historical_data, i)
            self.equity_curve.append({
                'time': current_time,
                'equity': current_equity,
                'open_positions': len(self.positions)
            })
        
        # Close all remaining positions at end
        logger.info("\nClosing all remaining positions...")
        for pos in self.positions[:]:
            symbol = pos['symbol']
            df = historical_data[symbol]
            close_price = float(df['close'].iloc[-1])
            self._close_position(pos, close_price, "backtest_end")
        
        # Calculate final results
        return self._calculate_results()
    
    def _check_entry(self, bot, symbol: str, df: pd.DataFrame):
        """Check if should enter a position (simplified)."""
        try:
            current_price = float(df['close'].iloc[-1])
            
            # Get signals from strategies
            buy_votes = 0
            sell_votes = 0
            
            for strategy in bot.active_strategies.values():
                try:
                    sig = strategy.generate_signal(symbol, {'df': df})
                    if sig and hasattr(sig, 'signal'):
                        sig_type = sig.signal.value if hasattr(sig.signal, 'value') else str(sig.signal)
                        if sig_type == 'BUY':
                            buy_votes += 1
                        elif sig_type == 'SELL':
                            sell_votes += 1
                except:
                    pass
            
            # Need majority
            min_votes = len(bot.active_strategies) // 2 + 1
            
            if buy_votes >= min_votes:
                direction = 'LONG'
            elif sell_votes >= min_votes:
                direction = 'SHORT'
            else:
                return
            
            # Calculate position size (5% of capital)
            size = self.capital * 0.05
            
            # Calculate stops and targets (2 ATR)
            atr = self._calculate_atr(df)
            if direction == 'LONG':
                stop_loss = current_price - (atr * 2)
                take_profit = current_price + (atr * 3)
            else:
                stop_loss = current_price + (atr * 2)
                take_profit = current_price - (atr * 3)
            
            # Open position
            position = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': current_price,
                'entry_time': df.index[-1],
                'size': size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'max_price': current_price
            }
            
            self.positions.append(position)
            logger.debug(f"[{symbol}] OPENED {direction} @ ${current_price:.2f}")
            
        except Exception as e:
            logger.error(f"Entry check error: {e}")
    
    def _manage_positions(self, symbol: str, df: pd.DataFrame):
        """Manage existing positions for this symbol."""
        current_price = float(df['close'].iloc[-1])
        current_time = df.index[-1]
        
        for pos in self.positions[:]:
            if pos['symbol'] != symbol:
                continue
            
            entry_price = pos['entry_price']
            direction = pos['direction']
            
            # Calculate P&L
            if direction == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            # Check stop loss
            if direction == 'LONG' and current_price <= pos['stop_loss']:
                self._close_position(pos, current_price, "stop_loss")
                continue
            elif direction == 'SHORT' and current_price >= pos['stop_loss']:
                self._close_position(pos, current_price, "stop_loss")
                continue
            
            # Check take profit
            if direction == 'LONG' and current_price >= pos['take_profit']:
                self._close_position(pos, current_price, "take_profit")
                continue
            elif direction == 'SHORT' and current_price <= pos['take_profit']:
                self._close_position(pos, current_price, "take_profit")
                continue
            
            # Check time limit (4 hours = 48 candles at 5-min)
            time_diff = (current_time - pos['entry_time']).total_seconds() / 3600
            if time_diff > 4:
                self._close_position(pos, current_price, "time_limit")
                continue
            
            # Update max price
            if direction == 'LONG':
                pos['max_price'] = max(pos['max_price'], current_price)
            else:
                pos['max_price'] = min(pos['max_price'], current_price)
    
    def _close_position(self, pos: Dict, close_price: float, reason: str):
        """Close a position and record the trade."""
        entry_price = pos['entry_price']
        direction = pos['direction']
        
        # Calculate P&L
        if direction == 'LONG':
            pnl_pct = (close_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - close_price) / entry_price
        
        pnl_dollars = pnl_pct * pos['size']
        
        # Update capital
        self.capital += pnl_dollars
        
        # Record trade
        trade = {
            'symbol': pos['symbol'],
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': close_price,
            'entry_time': pos['entry_time'],
            'exit_time': datetime.now(),
            'pnl_pct': pnl_pct,
            'pnl_dollars': pnl_dollars,
            'reason': reason
        }
        
        self.closed_trades.append(trade)
        self.positions.remove(pos)
        
        logger.debug(f"[{pos['symbol']}] CLOSED {direction} @ ${close_price:.2f} | P&L: ${pnl_dollars:+.2f} ({reason})")
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            tr = []
            for i in range(1, min(len(df), 50)):
                tr_val = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
                tr.append(tr_val)
            
            return sum(tr[-period:]) / period if len(tr) >= period else sum(tr) / len(tr)
        except:
            return float(df['close'].iloc[-1]) * 0.02
    
    def _calculate_equity(self, historical_data: Dict, candle_idx: int) -> float:
        """Calculate current equity."""
        equity = self.capital
        
        # Add unrealized P&L from open positions
        for pos in self.positions:
            symbol = pos['symbol']
            if symbol in historical_data:
                df = historical_data[symbol]
                current_price = float(df['close'].iloc[candle_idx])
                
                if pos['direction'] == 'LONG':
                    unrealized_pnl = (current_price - pos['entry_price']) / pos['entry_price'] * pos['size']
                else:
                    unrealized_pnl = (pos['entry_price'] - current_price) / pos['entry_price'] * pos['size']
                
                equity += unrealized_pnl
        
        return equity
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest results."""
        if not self.closed_trades:
            logger.warning("No trades executed!")
            return None
        
        # Calculate metrics
        wins = [t for t in self.closed_trades if t['pnl_dollars'] > 0]
        losses = [t for t in self.closed_trades if t['pnl_dollars'] <= 0]
        
        total_pnl = sum(t['pnl_dollars'] for t in self.closed_trades)
        win_rate = len(wins) / len(self.closed_trades)
        
        avg_win = sum(t['pnl_dollars'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl_dollars'] for t in losses) / len(losses) if losses else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = [t['pnl_pct'] for t in self.closed_trades]
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return)**2 for r in returns) / len(returns)) ** 0.5
        sharpe = (avg_return / std_return * (252 ** 0.5)) if std_return > 0 else 0
        
        # Max drawdown
        peak = self.start_capital
        max_dd = 0
        for point in self.equity_curve:
            if point['equity'] > peak:
                peak = point['equity']
            dd = (peak - point['equity']) / peak
            max_dd = max(max_dd, dd)
        
        results = {
            'total_trades': len(self.closed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'start_capital': self.start_capital,
            'end_capital': self.capital,
            'roi': (self.capital - self.start_capital) / self.start_capital
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print backtest results."""
        if not results:
            return
        
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        
        print(f"\n📊 PERFORMANCE:")
        print(f"   Start capital: ${results['start_capital']:,.0f}")
        print(f"   End capital: ${results['end_capital']:,.0f}")
        print(f"   Total P&L: ${results['total_pnl']:+,.2f}")
        print(f"   ROI: {results['roi']*100:+.2f}%")
        
        print(f"\n📈 TRADES:")
        print(f"   Total trades: {results['total_trades']}")
        print(f"   Wins: {results['wins']} ({results['win_rate']*100:.1f}%)")
        print(f"   Losses: {results['losses']}")
        
        print(f"\n💰 PROFIT/LOSS:")
        print(f"   Avg win: ${results['avg_win']:+.2f}")
        print(f"   Avg loss: ${results['avg_loss']:+.2f}")
        print(f"   Profit factor: {results['profit_factor']:.2f}")
        
        print(f"\n📉 RISK:")
        print(f"   Sharpe ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Max drawdown: {results['max_drawdown']*100:.2f}%")
        
        # Grade
        print(f"\n" + "="*70)
        print("VERDICT")
        print("="*70)
        
        if results['roi'] > 0.30 and results['sharpe_ratio'] > 1.5:
            verdict = "🌟 EXCELLENT - Ready for live trading"
        elif results['roi'] > 0.15 and results['sharpe_ratio'] > 1.0:
            verdict = "✅ GOOD - Proceed with caution"
        elif results['roi'] > 0:
            verdict = "⚠️ MARGINAL - Needs improvement"
        else:
            verdict = "❌ POOR - Do not trade"
        
        print(f"\n{verdict}")
        print(f"\nRecommendation:")
        if results['roi'] > 0.20:
            print("  ✅ Bot performs well on historical data")
            print("  ✅ Proceed to paper trading")
        elif results['roi'] > 0:
            print("  ⚠️ Bot is profitable but marginal")
            print("  ⚠️ Consider adjusting parameters")
        else:
            print("  ❌ Bot loses money on historical data")
            print("  ❌ Do NOT use for live trading")
        
        print("\n" + "="*70)

def main():
    """Run backtest."""
    # Symbols to test
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    # Run backtest
    backtest = BacktestEngine(start_capital=6090)
    results = backtest.run(symbols, days=7)
    
    if results:
        backtest.print_results(results)
        
        # Save results
        import json
        with open('logs/backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("\n✅ Results saved to logs/backtest_results.json")

if __name__ == '__main__':
    main()
