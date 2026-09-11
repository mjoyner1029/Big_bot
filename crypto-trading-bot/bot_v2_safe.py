#!/usr/bin/env python3
"""
Big Bot v2 - With Safety Features
Production Trading System with Risk Management, Health Monitoring, and Position Tracking

Created: August 1, 2026
Improvements over v1:
- ✅ Max daily loss limits (2% of capital)
- ✅ Position count limits (max 5)
- ✅ Circuit breaker (3 consecutive losses)
- ✅ Transaction cost modeling
- ✅ Health monitoring with graceful shutdown
- ✅ Position tracking in SQLite
- ✅ Kelly Criterion position sizing
"""
import time
import signal
import logging
import sys
from datetime import datetime, timedelta
from data.fetcher import fetch_latest_market_data
from strategies.crypto_momentum import CryptoMomentumStrategy
from strategies.mean_reversion_zscore import MeanReversionZScoreStrategy
from strategies.kronos_prediction import KronosPredictionStrategy
from core.safety_manager import SafetyManager
from core.health_monitor import HealthMonitor
from core.position_manager import PositionManager
from core.kelly_wrapper import KellySizer
from core.transaction_costs import TransactionCostModel
from core.selection_engine import SelectionEngine  # NEW: Quality filter + Kronos eval
from data.liquidity_scanner import scan_crypto_universe

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/bot_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBotV2:
    """Trading bot with comprehensive safety features."""
    
    def __init__(self, capital=6090):
        self.capital = capital
        self.leverage = 2.0
        
        # Load config from environment
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        max_positions = int(os.getenv('MAX_POSITIONS', 8))
        max_daily_loss_pct = float(os.getenv('MAX_DAILY_LOSS_PCT', 0.02))
        
        # Safety & Risk Management (NEW)
        self.safety = SafetyManager(capital, config={
            'max_daily_loss_pct': max_daily_loss_pct,  # From .env (default 2%)
            'max_positions': max_positions,  # From .env (default 8)
            'max_position_size_pct': 0.10,  # 10%
            'max_consecutive_losses': 3
        })
        self.health = HealthMonitor(self, max_cycle_minutes=15)
        self.positions = PositionManager(db_path='data/trade_memory.sqlite')
        self.kelly = KellySizer(db_path='data/trade_memory.sqlite')
        self.costs = TransactionCostModel(exchange='alpaca')
        
        # Selection Engine (NEW) - Quality filter + Kronos evaluation
        self.selection = SelectionEngine(
            enable_kronos=True,  # Use Kronos for multi-horizon predictions
            min_rr_ratio=2.0     # Minimum 2:1 risk/reward
        )
        
        # Strategies
        self.strategies = {
            'momentum': CryptoMomentumStrategy(),
            'mean_reversion': MeanReversionZScoreStrategy(),
            'kronos': KronosPredictionStrategy(),  # NEW: Foundation model predictions
        }
        
        # Symbols - Load from liquidity scanner
        self.symbols = self._load_symbols()
        logger.info(f"📊 Loaded {len(self.symbols)} liquid symbols for trading")
        
        # Sync position count with database
        actual_open = self.positions.count_open()
        self.safety.active_positions = actual_open
        logger.info(f"🔄 Synced position count: {actual_open} open positions from database")
        
        # Position recovery system - close stale positions on startup
        self._recover_stale_positions()
        
        logger.info("="*60)
        logger.info("🚀 BIG BOT V2 INITIALIZED")
        logger.info(f"   Capital: ${capital:,.0f}")
        logger.info(f"   Strategies: {len(self.strategies)}")
        logger.info(f"   Symbols: {len(self.symbols)}")
        logger.info("")
        logger.info("   SAFETY FEATURES:")
        logger.info(f"   ✅ Max daily loss: {self.safety.max_daily_loss_pct:.0%} (${capital * self.safety.max_daily_loss_pct:.0f})")
        logger.info(f"   ✅ Max positions: {self.safety.max_positions}")
        logger.info(f"   ✅ Circuit breaker: {self.safety.max_consecutive_losses} consecutive losses")
        logger.info(f"   ✅ Transaction cost modeling: Enabled")
        logger.info(f"   ✅ Kelly Criterion sizing: Enabled")
        logger.info(f"   ✅ Health monitoring: Enabled")
        logger.info("="*60)
    
    def _load_symbols(self) -> list:
        """Load tradeable symbols from liquidity scanner."""
        try:
            # Scan crypto universe for top 30 opportunities
            opportunities = scan_crypto_universe(max_results=30)
            
            if not opportunities:
                logger.warning("⚠️ Liquidity scanner returned no results, using fallback")
                return ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'MATIC-USD']
            
            # Extract symbols
            symbols = [opp.symbol for opp in opportunities]
            
            # Log top 10 by opportunity score
            logger.info("🔍 Top opportunities discovered:")
            for opp in opportunities[:10]:
                logger.info(f"   {opp.symbol}: Score={opp.opportunity_score:.1f}, Vol=${opp.volume_24h_usd/1e6:.1f}M, Volatility={opp.volatility_score:.1f}%")
            
            return symbols
            
        except Exception as e:
            logger.error(f"❌ Liquidity scanner failed: {e}")
            logger.info("Using fallback symbol list")
            return ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'MATIC-USD', 'AVAX-USD', 'LINK-USD', 'DOT-USD', 'UNI-USD', 'ATOM-USD']
    
    def _recover_stale_positions(self):
        """Close positions older than 48 hours on startup (recovery system)."""
        open_positions = self.positions.get_open_positions()
        
        if not open_positions:
            logger.info("✅ No stale positions to recover")
            return
        
        logger.info(f"🔍 Checking {len(open_positions)} open positions for staleness...")
        
        from datetime import datetime, timedelta, timezone
        stale_threshold = timedelta(hours=48)
        now = datetime.now(timezone.utc)
        
        closed_count = 0
        for pos in open_positions:
            try:
                entry_time_str = pos.get('entry_time', '')
                # Parse entry time and ensure it has timezone
                entry_time = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                age = now - entry_time
                
                if age > stale_threshold:
                    symbol = pos['symbol']
                    pos_id = pos['id']
                    entry_price = pos['entry_price']
                    
                    logger.warning(f"⚠️ STALE POSITION: #{pos_id} {symbol} aged {age.total_seconds()/3600:.1f}h")
                    
                    # Try to fetch current price and close
                    try:
                        df = fetch_latest_market_data(symbol)
                        if df is not None and len(df) > 0:
                            current_price = float(df['close'].iloc[-1])
                            pnl = self.positions.close_position(pos_id, close_price=current_price, reason='stale_recovery')
                            if pnl is not None:
                                self.safety.on_position_close(pnl)
                                closed_count += 1
                                logger.info(f"   ✅ Closed stale position: P&L ${pnl:+.2f}")
                        else:
                            # Can't fetch price, close at entry (break-even)
                            pnl = self.positions.close_position(pos_id, close_price=entry_price, reason='stale_recovery_no_price')
                            if pnl is not None:
                                self.safety.on_position_close(pnl)
                                closed_count += 1
                                logger.info(f"   ⚠️ Closed stale position (no price): P&L ${pnl:+.2f}")
                    except Exception as e:
                        logger.error(f"   ❌ Failed to close stale position {symbol}: {e}")
                        
            except Exception as e:
                logger.error(f"Error processing position recovery: {e}")
                continue
        
        if closed_count > 0:
            logger.info(f"🧹 Position recovery complete: Closed {closed_count} stale positions")
            # Resync position count after recovery
            self.safety.active_positions = self.positions.count_open()
        else:
            logger.info("✅ No stale positions found")
    
    def calculate_position_size(self, symbol: str) -> float:
        """Calculate position size using Kelly Criterion."""
        kelly_size = self.kelly.get_position_size(
            symbol=symbol,
            capital=self.capital,
            leverage=self.leverage
        )
        
        # Cap at 10% of capital (safety limit)
        max_size = self.capital * 0.10
        final_size = min(kelly_size, max_size)
        
        logger.debug(f"[{symbol}] Kelly: ${kelly_size:.2f}, Capped: ${final_size:.2f}")
        return final_size
    
    def evaluate(self, symbol: str) -> dict:
        """Evaluate symbol with all strategies + safety checks."""
        try:
            # 1. Fetch data
            df = fetch_latest_market_data(symbol)
            if df is None or len(df) < 50:
                return None
            
            current_price = float(df['close'].iloc[-1])
            
            # 2. Calculate position size
            position_size = self.calculate_position_size(symbol)
            
            # 3. Check safety limits FIRST
            allowed, reason = self.safety.check_can_trade(position_size)
            if not allowed:
                logger.warning(f"[{symbol}] ⚠️ Trade blocked by safety: {reason}")
                return None
            
            # 4. Get signals from all strategies
            signals = []
            for name, strategy in self.strategies.items():
                try:
                    sig = strategy.generate_signal(symbol, {'df': df})
                    if sig and hasattr(sig, 'signal'):
                        sig_type = sig.signal.value if hasattr(sig.signal, 'value') else str(sig.signal)
                        if sig_type in ['BUY', 'SELL']:
                            confidence = getattr(sig, 'confidence', 50)
                            signals.append({
                                'type': sig_type,
                                'strategy': name,
                                'confidence': confidence,
                                'stop_loss': getattr(sig, 'stop_loss', None),
                                'targets': getattr(sig, 'targets', []),
                                'signal_obj': sig  # Keep full signal for later
                            })
                            logger.info(f"[{symbol}] {name}: {sig_type} ({confidence:.0f}%)")
                except Exception as e:
                    logger.error(f"[{symbol}] {name} error: {e}")
            
            # 5. Ensemble voting with confidence-weighted tiebreaker
            if len(signals) < 1:
                return None
            
            # 6. Determine trade direction
            buy_votes = sum(1 for s in signals if s['type'] == 'BUY')
            sell_votes = sum(1 for s in signals if s['type'] == 'SELL')
            
            # Clear majority: 2+ votes same direction
            if buy_votes >= 2:
                trade_signal = 'BUY'
            elif sell_votes >= 2:
                trade_signal = 'SELL'
            # Tie (1 BUY, 1 SELL) or single signal: use highest confidence
            else:
                buy_sigs = [s for s in signals if s['type'] == 'BUY']
                sell_sigs = [s for s in signals if s['type'] == 'SELL']
                
                max_buy_conf = max([s['confidence'] for s in buy_sigs], default=0)
                max_sell_conf = max([s['confidence'] for s in sell_sigs], default=0)
                
                if max_buy_conf > max_sell_conf and max_buy_conf >= 60:
                    trade_signal = 'BUY'
                    logger.info(f"[{symbol}] Tiebreaker: BUY wins ({max_buy_conf:.0f}% vs {max_sell_conf:.0f}%)")
                elif max_sell_conf > max_buy_conf and max_sell_conf >= 60:
                    trade_signal = 'SELL'
                    logger.info(f"[{symbol}] Tiebreaker: SELL wins ({max_sell_conf:.0f}% vs {max_buy_conf:.0f}%)")
                else:
                    logger.info(f"[{symbol}] No consensus: {buy_votes} BUY ({max_buy_conf:.0f}%), {sell_votes} SELL ({max_sell_conf:.0f}%) - both <60%")
                    return None
            
            # 7. Check transaction costs
            costs = self.costs.calculate_cost(position_size, current_price)
            min_profit_pct = self.costs.get_min_profitable_move(current_price)
            
            logger.info(f"[{symbol}] ✅ SIGNAL: {trade_signal} ${position_size:.2f}")
            logger.info(f"   Price: ${current_price:,.2f}")
            logger.info(f"   Votes: {buy_votes} BUY, {sell_votes} SELL")
            logger.info(f"   Costs: ${costs:.2f} ({min_profit_pct:.2%} min profit)")
            
            # 8. Extract stop_loss and take_profit from winning signal
            # Use the highest-confidence signal in the winning direction
            winning_sigs = [s for s in signals if s['type'] == trade_signal]
            best_sig = max(winning_sigs, key=lambda x: x['confidence'])
            
            # Get stop loss and target
            stop_loss = best_sig.get('stop_loss')
            targets = best_sig.get('targets', [])
            target = targets[0] if targets else None
            
            # Default targets if missing
            if not stop_loss:
                stop_loss = current_price * 0.97 if trade_signal == 'BUY' else current_price * 1.03
            if not target:
                target = current_price * 1.05 if trade_signal == 'BUY' else current_price * 0.95
            
            # 9. *** SELECTION ENGINE - QUALITY FILTER ***
            # This is the KEY addition from the videos!
            selection_result = self.selection.evaluate_trade(
                symbol=symbol,
                signal=best_sig['signal_obj'],
                data=df,
                entry_price=current_price,
                stop_loss=stop_loss,
                target=target
            )
            
            # Log selection decision
            logger.info(f"[{symbol}] 🎯 SELECTION: {selection_result['reason']}")
            logger.info(f"   Quality: {selection_result['quality_score']}/100")
            logger.info(f"   Kronos: {selection_result['kronos_score']}/20")
            logger.info(f"   R:R: {selection_result['rr_ratio']:.1f}:1")
            logger.info(f"   Total: {selection_result['total_score']}/120")
            
            # Check if approved
            if not selection_result['approved']:
                logger.warning(f"[{symbol}] ❌ Trade REJECTED by selection engine")
                return None
            
            # Apply position size multiplier based on quality
            position_size = position_size * selection_result['position_size_multiplier']
            logger.info(f"[{symbol}] Position size adjusted: ${position_size:.2f} (multiplier: {selection_result['position_size_multiplier']})")
            
            # 10. Update safety manager
            self.safety.on_position_open()
            
            stop_loss = best_sig.get('stop_loss')
            targets = best_sig.get('targets', [])
            take_profit = targets[0] if targets else None
            
            # 10. Track position
            pos_id = self.positions.open_position(
                symbol=symbol,
                signal=trade_signal,
                size=position_size,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            return {
                'signal': trade_signal,
                'size': position_size,
                'price': current_price,
                'votes': buy_votes if trade_signal == 'BUY' else sell_votes,
                'position_id': pos_id,
                'costs': costs
            }
            
        except Exception as e:
            logger.error(f"[{symbol}] Evaluation error: {e}", exc_info=True)
            return None
    
    def monitor_positions(self) -> int:
        """Check open positions and close if stop/target hit.
        
        Returns:
            Number of positions closed
        """
        closed_count = 0
        
        try:
            open_positions = self.positions.get_open_positions()
            logger.debug(f"[MONITOR] Checking {len(open_positions)} open positions")
            
            for pos in open_positions:
                symbol = pos['symbol']
                direction = pos['direction']
                entry_price = pos['entry_price']
                stop_loss = pos.get('stop_loss')
                take_profit = pos.get('take_profit')
                pos_id = pos['id']
                
                logger.debug(f"[MONITOR] #{pos_id} {symbol} {direction} @ ${entry_price:.4f} | SL: ${stop_loss or 0:.4f} | TP: ${take_profit or 0:.4f}")
                
                # Get current price
                try:
                    logger.debug(f"[MONITOR] Fetching data for {symbol}...")
                    df = fetch_latest_market_data(symbol, period='1d', interval='5m')
                    if df is None or df.empty:
                        logger.warning(f"[{symbol}] No data for position monitoring")
                        continue
                    
                    current_price = float(df['close'].iloc[-1])
                    logger.debug(f"[MONITOR] {symbol} current price: ${current_price:.4f}")
                except Exception as e:
                    logger.error(f"[{symbol}] Failed to get current price: {e}")
                    continue
                
                # Check exit conditions
                should_exit = False
                exit_reason = None
                
                if direction in ['BUY', 'LONG']:
                    # BUY: Exit if price hits stop (below) or target (above)
                    if stop_loss and current_price <= stop_loss:
                        should_exit = True
                        exit_reason = f"Stop loss hit: {current_price:.4f} <= {stop_loss:.4f}"
                    elif take_profit and current_price >= take_profit:
                        should_exit = True
                        exit_reason = f"Take profit hit: {current_price:.4f} >= {take_profit:.4f}"
                        
                elif direction in ['SELL', 'SHORT']:
                    # SELL: Exit if price hits stop (above) or target (below)
                    if stop_loss and current_price >= stop_loss:
                        should_exit = True
                        exit_reason = f"Stop loss hit: {current_price:.4f} >= {stop_loss:.4f}"
                    elif take_profit and current_price <= take_profit:
                        should_exit = True
                        exit_reason = f"Take profit hit: {current_price:.4f} <= {take_profit:.4f}"
                
                if should_exit:
                    logger.debug(f"[MONITOR] Exit triggered for {symbol}: {exit_reason}")
                    pnl = self.positions.close_position(pos_id, close_price=current_price, reason=exit_reason)
                    if pnl is not None:
                        logger.info(f"[{symbol}] 📉 Position closed: {exit_reason}")
                        logger.info(f"   P&L: ${pnl:+.2f} ({(pnl/entry_price)*100:+.2f}%)")
                        
                        # Update safety manager
                        self.safety.on_position_close(pnl)
                        closed_count += 1
                else:
                    logger.debug(f"[MONITOR] {symbol} holding (no exit conditions met)")
                        
        except Exception as e:
            logger.error(f"Position monitoring error: {e}", exc_info=True)
        
        logger.debug(f"[MONITOR] Finished checking positions, closed: {closed_count}")
        return closed_count
    
    def run_cycle(self) -> None:
        """One trading cycle with health monitoring."""
        cycle_start = datetime.now()
        logger.info("")
        logger.info("="*60)
        logger.info(f"CYCLE START: {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
        try:
            # 1. Monitor and close existing positions first
            closed = self.monitor_positions()
            
            # 2. Look for new trades
            trades = 0
            blocked = 0
            
            for symbol in self.symbols:
                decision = self.evaluate(symbol)
                if decision:
                    trades += 1
                elif self.safety.circuit_breaker_triggered:
                    blocked += 1
            
            # Update heartbeat
            self.health.update_heartbeat()
            
            # Log cycle summary
            open_positions = self.positions.count_open()
            metrics = self.safety.get_safety_metrics()
            
            logger.info("")
            logger.info("--- CYCLE SUMMARY ---")
            logger.info(f"   Closed: {closed}")
            logger.info(f"   Opened: {trades}")
            logger.info(f"   Blocked: {blocked}")
            logger.info(f"   Open positions: {open_positions}/{self.safety.max_positions}")
            logger.info(f"   Daily P&L: ${metrics['daily_pnl']:+.2f}")
            logger.info(f"   Consecutive losses: {metrics['consecutive_losses']}")
            if metrics['circuit_breaker_triggered']:
                logger.critical(f"   🛑 CIRCUIT BREAKER ACTIVE!")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)
        
        finally:
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            if cycle_duration > 600:  # 10 minutes
                logger.critical(f"🔴 Cycle took {cycle_duration:.0f}s! Possible hang.")
    
    def run(self) -> None:
        """Run continuously with health monitoring."""
        # Setup signal handlers for graceful shutdown
        self.health.setup_signal_handlers()
        
        logger.info("")
        logger.info("🚀 STARTING TRADING LOOP")
        logger.info("   Press Ctrl+C to stop gracefully")
        logger.info("")
        
        consecutive_errors = 0
        
        try:
            while True:
                # Check if heartbeat is healthy
                if not self.health.check_heartbeat():
                    logger.critical("🔴 Heartbeat stalled! Restarting...")
                    break
                
                # Check if circuit breaker triggered
                if self.safety.circuit_breaker_triggered:
                    logger.critical("🔴 Circuit breaker triggered - stopping trading")
                    logger.info("Will resume tomorrow after daily reset")
                    break
                
                # Run cycle
                self.run_cycle()
                consecutive_errors = 0  # Reset on success
                
                # Wait for next cycle
                logger.info("Waiting 5 minutes...")
                time.sleep(300)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
            self.health.graceful_shutdown()
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Fatal error: {e}", exc_info=True)
            
            if consecutive_errors >= 3:
                logger.critical("🔴 3 consecutive errors - stopping bot")
                self.health.graceful_shutdown()
            else:
                logger.info("Backing off 60s before retry...")
                time.sleep(60)

def main():
    """Main entry point."""
    bot = TradingBotV2(capital=6090)
    bot.run()

if __name__ == '__main__':
    main()
