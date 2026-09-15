#!/usr/bin/env python3
"""
Ultimate Bot V3 - LLM-Orchestrated Adaptive Trading

WORKFLOW:
1. LLM analyzes regime for the day
2. Market is analyzed (technicals, sentiment, GEX)
3. Kronos evaluates specific trade setups
4. Bot makes trades based on LLM+Kronos consensus
5. LLM analyzes trade results and adapts strategies
6. End of day: Bot learns from the day's performance

This is the PROPER architecture you described.
"""
import os
import sys
import time
import signal
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# NOTE: credentials are validated at STARTUP for the selected mode via
# config.capabilities — never at import time. Importing this module must not
# terminate the process (see check_startup_requirements in run()).
RECOMMENDED_ENV_VARS = ['ANTHROPIC_API_KEY', 'ALPACA_API_KEY', 'ALPACA_API_SECRET']
_missing_vars = [var for var in RECOMMENDED_ENV_VARS if not os.getenv(var)]
if _missing_vars:
    print(f"ℹ️  Missing optional environment variables: {_missing_vars} "
          f"— some capabilities will be unavailable (validated at startup)")

# Core systems
from data.fetcher import fetch_latest_market_data
from config.config import CONFIG
from core.ohlcv import normalize_ohlcv
from core.llm_orchestrator import LLMOrchestrator
from core.safety_manager import SafetyManager
from core.health_monitor import HealthMonitor
from core.position_manager import PositionManager
from core.kelly_wrapper import KellySizer
from core.transaction_costs import TransactionCostModel

# Analysis modules
from strategies.gamma_exposure import GammaExposureAnalyzer
try:
    from core.regime_detector import RegimeDetector
    REGIME_AVAILABLE = True
except (ImportError, AttributeError) as e:
    REGIME_AVAILABLE = False
    logger.debug(f"Regime detector not available: {e}")

# Kronos (optional)
try:
    from core.kronos_predictor import BigBotKronosPredictor, is_kronos_available
    KRONOS_AVAILABLE = is_kronos_available()
except (ImportError, AttributeError) as e:
    KRONOS_AVAILABLE = False
    logger.debug(f"Kronos not available: {e}")

# All available strategies (LLM will select which to use)
from strategies.crypto_momentum import CryptoMomentumStrategy
from strategies.mean_reversion_zscore import MeanReversionZScoreStrategy
from strategies.breakout import BreakoutStrategy
from strategies.vwap_reversion import VWAPReversionStrategy
from strategies.ema_trend_follow import EMATrendFollowStrategy
from strategies.correlation_lag_strategy import CorrelationLagStrategy
from strategies.statistical_arbitrage import StatisticalArbitrageStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/llm_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLMTradingBot:
    """LLM-orchestrated adaptive trading bot."""
    
    def __init__(self, capital=6090):
        self.capital = capital
        self.leverage = 2.0
        self.symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'XRP-USD']
        
        # Core systems
        self.safety = SafetyManager(capital, config={
            'max_daily_loss_pct': 0.02,
            'max_positions': 5,
            'max_position_size_pct': 0.10,
            'max_consecutive_losses': 3
        })
        self.health = HealthMonitor(self, max_cycle_minutes=15)
        self.positions = PositionManager(db_path='data/trade_memory.sqlite')
        self.kelly = KellySizer(db_path='data/trade_memory.sqlite')
        self.costs = TransactionCostModel(exchange='alpaca')

        # Broker — the ONLY path for order execution
        from core.broker import get_broker
        self.broker = get_broker(capital=capital)

        # Experiment Engine — LLM proposes, operator promotes
        from core.experiment_engine import ExperimentEngine
        self.experiments = ExperimentEngine(db_path='data/trade_memory.sqlite')

        # Universe Scanner — replaces hardcoded symbol list
        from core.universe_scanner import UniverseScanner, ScannerConfig
        self.scanner = UniverseScanner(config=ScannerConfig(
            min_volume_usd_24h=5_000_000,
            min_atr_pct=float(os.environ.get('SCANNER_MIN_ATR_PCT', '0.0005')),
            top_n=10,
        ))
        
        # Intelligence layer
        self.llm = LLMOrchestrator(
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        self.gex = GammaExposureAnalyzer()
        self.regime = RegimeDetector() if REGIME_AVAILABLE else None
        
        # Kronos (optional)
        if KRONOS_AVAILABLE:
            self.kronos = BigBotKronosPredictor(model_size='small')
            logger.info("✅ Kronos predictions enabled")
        else:
            self.kronos = None
            logger.warning("⚠️ Kronos not available")
        
        # ALL available strategies (LLM will pick which to use)
        self.all_strategies = {
            'mean_reversion': MeanReversionZScoreStrategy(),
            'correlation_lag': CorrelationLagStrategy(),
            'breakout': BreakoutStrategy(),
            'vwap_reversion': VWAPReversionStrategy(),
            'stat_arb': StatisticalArbitrageStrategy(),
            'momentum': CryptoMomentumStrategy(),
            'ema_trend_follow': EMATrendFollowStrategy(),
        }
        
        # Active strategies (selected by LLM)
        self.active_strategies = {}
        self.daily_decision = None

        # Adaptation state — modified by _apply_adaptations()
        self._position_size_scale: float = 1.0
        self._skip_regimes: List[str] = []

        # ── Phase 6: Register production params with ExperimentEngine ──────────
        # Claude can only propose experiments against these registered keys.
        # Unknown keys will be rejected at proposal time.
        self.experiments.register_production_params({
            # Kelly / position sizing
            "position_size_scale":        self._position_size_scale,
            "kelly_fraction":             0.5,
            "min_kelly_sample":           10,
            "default_position_pct":       0.05,
            "max_position_pct":           0.15,
            # Strategy selection
            "active_strategy_names":      list(self.all_strategies.keys()),
            "min_signal_confidence":      60.0,
            "weighted_vote_threshold":    0.60,
            # Exit management
            "stop_loss_atr_multiplier":   2.0,
            "take_profit_atr_multiplier": 3.0,
            "trailing_stop_pct":          0.015,
            "partial_profit_threshold":   0.02,
            "max_hold_hours":             4.0,
            # Universe scanner
            "scanner_top_n":              10,
            "scanner_min_volume_usd":     5_000_000,
            "scanner_min_atr_pct":        self.scanner.config.min_atr_pct,
            "scanner_max_atr_pct":        0.08,
        })

        # ── Infrastructure modules (Phases 7-13) ─────────────────────────────
        from core.feature_store import FeatureStore
        from core.meta_model import MetaModel
        from core.portfolio_manager import PortfolioManager
        from core.opportunity_ranker import OpportunityRanker
        from core.trade_memory import TradeMemory

        self.feature_store = FeatureStore(db_path='data/feature_store.sqlite')
        self.meta_model    = MetaModel(model_dir='models/meta_model')
        self.portfolio_mgr = PortfolioManager(capital=capital)
        self.ranker        = OpportunityRanker()
        self.trade_memory  = TradeMemory(db_path='data/trade_memory.sqlite')

        # ── Alpha library / correlation / data-quality gates ──────────────────
        from core.alpha_library import AlphaLibrary
        from core.data_quality import DataQualityMonitor
        from core.strategy_correlation import StrategyCorrelationTracker

        self.alpha_library = AlphaLibrary(db_path='data/trade_memory.sqlite')
        self.data_quality_monitor = DataQualityMonitor()
        self.correlation_tracker = StrategyCorrelationTracker(
            db_path='data/trade_memory.sqlite')
        from core.risk_budgets import DrawdownBudgetManager
        self.drawdown_budgets = DrawdownBudgetManager(
            db_path='data/trade_memory.sqlite', capital=capital)

        # ── Alpha-to-signal pipeline (opportunity-first decision path) ────────
        from core.alpha_signal_engine import AlphaSignalEngine
        from core.ev_model import EconomicEVModel
        from core.feature_registry import FeatureRegistry
        from core.portfolio_allocator import AllocatorConfig, PortfolioAllocator
        from core.trade_attribution import LearningLoop, TradeAttributionStore

        self.feature_registry = FeatureRegistry()
        self.alpha_signal_engine = AlphaSignalEngine(
            self.alpha_library, self.feature_registry,
            data_quality_monitor=self.data_quality_monitor)
        self.ev_model = EconomicEVModel(db_path='data/trade_memory.sqlite')
        self.portfolio_allocator = PortfolioAllocator(
            AllocatorConfig(capital=capital))
        self.trade_attribution = TradeAttributionStore(
            db_path='data/trade_memory.sqlite')
        self.learning_loop = LearningLoop(
            db_path='data/trade_memory.sqlite', alpha_library=self.alpha_library)
        # 'shadow' (default): alpha pipeline runs + logs, legacy decides.
        # 'primary': alpha pipeline decides, legacy is fallback only.
        # 'off': legacy only.
        self.alpha_pipeline_mode = os.environ.get(
            'ALPHA_PIPELINE_MODE', 'shadow').lower()
        # Strategies explicitly approved for LIVE capital (safety default: none)
        self.live_approved_strategies = set(
            s.strip() for s in os.environ.get('LIVE_APPROVED_STRATEGIES', '').split(',')
            if s.strip()
        )

        # ── Autonomous system layer ────────────────────────────────────────────
        from core.champion_challenger import ChampionChallenger
        from core.feature_importance import FeatureImportanceEngine
        from core.model_trainer import ModelTrainer
        from core.strategy_health import StrategyHealthMonitor
        from core.drift_detector import DriftDetector
        from core.order_manager import OrderManager
        from core.execution_quality import ExecutionQualityEngine
        from core.trade_explainability import TradeExplainabilityStore
        from core.research_engine import ResearchEngine
        from core.experiment_worker import ExperimentWorker
        from core.daily_review import DailyReviewer
        from core.weekly_review import WeeklyReviewer
        from dashboard.performance_dashboard import PerformanceDashboard

        self.champion_challenger = ChampionChallenger(
            model_dir='models',
            db_path='data/trade_memory.sqlite',
        )
        self.feature_importance  = FeatureImportanceEngine()
        self.model_trainer       = ModelTrainer(
            feature_store=self.feature_store,
            champion_challenger=self.champion_challenger,
            feature_importance_engine=self.feature_importance,
            db_path='data/trade_memory.sqlite',
        )
        self.strategy_health     = StrategyHealthMonitor(
            db_path='data/trade_memory.sqlite',
            capital=capital,
        )
        self.drift_detector      = DriftDetector(
            feature_store=self.feature_store,
            db_path='data/trade_memory.sqlite',
        )
        self.order_manager       = OrderManager(db_path='data/trade_memory.sqlite')
        self.execution_quality   = ExecutionQualityEngine(db_path='data/trade_memory.sqlite')
        self.trade_explainability = TradeExplainabilityStore(db_path='data/trade_memory.sqlite')
        self.research_engine     = ResearchEngine(db_path='data/trade_memory.sqlite')
        self.experiment_worker   = ExperimentWorker(
            experiment_engine=self.experiments,
            db_path='data/trade_memory.sqlite',
        )
        self.daily_reviewer      = DailyReviewer(
            trade_memory=self.trade_memory,
            strategy_health=self.strategy_health,
            execution_quality=self.execution_quality,
            research_engine=self.research_engine,
            drift_detector=self.drift_detector,
            feature_importance=self.feature_importance,
            position_manager=self.positions,
            db_path='data/trade_memory.sqlite',
        )
        self.weekly_reviewer     = WeeklyReviewer(
            trade_memory=self.trade_memory,
            strategy_health=self.strategy_health,
            champion_challenger=self.champion_challenger,
            research_engine=self.research_engine,
            experiment_engine=self.experiments,
            drift_detector=self.drift_detector,
            feature_importance=self.feature_importance,
            daily_reviewer=self.daily_reviewer,
            db_path='data/trade_memory.sqlite',
        )
        self.dashboard           = PerformanceDashboard(
            position_manager=self.positions,
            strategy_health=self.strategy_health,
            experiment_engine=self.experiments,
            champion_challenger=self.champion_challenger,
            feature_importance=self.feature_importance,
            execution_quality=self.execution_quality,
            research_engine=self.research_engine,
            drift_detector=self.drift_detector,
            trade_memory=self.trade_memory,
            capital=capital,
        )

        # Start experiment worker background thread
        self.experiment_worker.start()

        logger.info("="*60)
        logger.info("🤖 LLM TRADING BOT V3 INITIALIZED")
        logger.info(f"   Capital: ${capital:,.0f}")
        logger.info(f"   Available strategies: {len(self.all_strategies)}")
        logger.info(f"   Symbols: {len(self.symbols)}")
        logger.info(f"   LLM orchestration: {'✅ Active' if self.llm.api_key else '❌ Missing API key'}")
        logger.info(f"   Kronos predictions: {'✅ Active' if self.kronos else '❌ Disabled'}")
        logger.info("="*60)
    
    def daily_regime_analysis(self) -> Dict:
        """STEP 1: LLM analyzes regime for the day."""
        logger.info("\n" + "="*60)
        logger.info("STEP 1: LLM REGIME ANALYSIS")
        logger.info("="*60)
        
        # Gather market data for all symbols
        market_data = {}
        for symbol in self.symbols:
            df = fetch_latest_market_data(symbol, period='7d', interval='1h')
            if df is not None:
                market_data[symbol] = df
        
        # Get GEX regime
        gex_regime = {}
        for symbol in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
            if symbol in market_data:
                current_price = float(market_data[symbol]['close'].iloc[-1])
                gex_data = self.gex.analyze(symbol, current_price)
                if gex_data and 'error' not in gex_data:
                    gex_regime[symbol] = gex_data
        
        # Get recent performance
        try:
            recent_trades = self.positions.get_recent_trades(days=7) if hasattr(self.positions, 'get_recent_trades') else []
        except:
            recent_trades = []
        performance = self._analyze_performance(recent_trades)
        
        # LLM makes the decision (or use defaults if LLM fails)
        try:
            # Wrap performance in a dict for the LLM orchestrator format
            perf_dict = {'overall': performance} if performance else {}
            
            decision = self.llm.analyze_and_decide(
                market_conditions=self._extract_market_conditions(market_data),
                gex_regime=gex_regime,
                recent_performance=perf_dict,
                failure_patterns=self._get_failure_patterns(),
                available_strategies=list(self.all_strategies.keys()),
                market_data=market_data
            )
        except Exception as e:
            logger.error(f"LLM regime analysis failed: {type(e).__name__}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            logger.warning("Falling back to default strategies")
            # Use default strategies if LLM fails
            decision = {
                'selected_strategies': ['mean_reversion', 'breakout', 'momentum'],
                'confidence': 0.7,
                'reasoning': f'Using default strategies (LLM failed: {type(e).__name__})'
            }
        
        # Validate selected strategies exist
        valid_strategies = {}
        invalid_strategies = []
        
        for name in decision['selected_strategies']:
            if name in self.all_strategies:
                valid_strategies[name] = self.all_strategies[name]
            else:
                invalid_strategies.append(name)
                logger.warning(f"⚠️  LLM selected unknown strategy '{name}' - skipping")
        
        # If no valid strategies, use safe defaults
        if not valid_strategies:
            logger.error("❌ No valid strategies selected! Using fallback: mean_reversion + breakout")
            valid_strategies = {
                'mean_reversion': self.all_strategies['mean_reversion'],
                'breakout': self.all_strategies['breakout']
            }
            decision['selected_strategies'] = ['mean_reversion', 'breakout']
        
        if invalid_strategies:
            logger.warning(f"Available strategies: {list(self.all_strategies.keys())}")
            logger.warning(f"Invalid strategies ignored: {invalid_strategies}")
        
        # Update active strategies with validated list
        self.active_strategies = valid_strategies
        
        self.daily_decision = decision
        
        logger.info(f"\n📊 LLM DECISION:")
        logger.info(f"   Selected strategies: {', '.join(decision['selected_strategies'])}")
        logger.info(f"   Confidence: {decision['confidence']:.1%}")
        logger.info(f"   Reasoning: {decision['reasoning']}")
        
        return decision
    
    def check_market_conditions(self, symbol: str, df) -> tuple[bool, str]:
        """Check if market conditions are safe for trading."""
        try:
            # Data-quality gate: a strategy with bad inputs must abstain
            timeframe = df.attrs.get("timeframe", CONFIG.get("interval", "1h"))
            quality = self.data_quality_monitor.check(df, symbol=symbol,
                                                      timeframe=timeframe,
                                                      require_fresh=True)
            if not quality.passed:
                return False, f"Data quality failed: {quality.summary()}"

            # Bar-range volatility proxy. NOTE: this is intrabar range, NOT
            # bid/ask spread — quote data is unavailable here, so this acts as
            # a liquidity-risk fallback gate only.
            high = float(df['high'].iloc[-1])
            low = float(df['low'].iloc[-1])
            mid = (high + low) / 2
            bar_range_pct = (high - low) / mid if mid > 0 else 1.0

            scanner_config = getattr(getattr(self, 'scanner', None), 'config', None)
            max_bar_range = getattr(scanner_config, 'max_bar_range_pct', 0.05)
            if bar_range_pct > max_bar_range:
                return False, (
                    f"Bar range too wide ({bar_range_pct*100:.2f}% high-low "
                    f"> {max_bar_range*100:.2f}% limit, liquidity-risk proxy)"
                )

            # Check volume (liquidity)
            current_vol = float(df['volume'].iloc[-5:].mean())
            avg_vol = float(df['volume'].iloc[-100:].mean())
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 0

            if vol_ratio < 0.3:  # Less than 30% of average
                return False, f"Volume too low ({vol_ratio*100:.0f}% of avg)"

            return True, "OK"
        except Exception as e:
            return False, f"Error checking conditions: {e}"
    
    def analyze_trade_opportunity(self, symbol: str) -> Optional[Dict]:
        """STEP 2-3: Market analysis + Kronos evaluation."""
        logger.info(f"\n[{symbol}] Analyzing opportunity...")
        
        # Fetch data
        df = fetch_latest_market_data(symbol)
        if df is None or len(df) < 50:
            logger.info(f"[{symbol}] NO TRADE — insufficient market history")
            return None
        try:
            df = normalize_ohlcv(df)
        except ValueError as exc:
            logger.warning(f"[{symbol}] NO TRADE — {exc}")
            return None

        current_price = float(df['close'].iloc[-1])
        
        # Check market conditions FIRST
        conditions_ok, condition_reason = self.check_market_conditions(symbol, df)
        if not conditions_ok:
            logger.info(f"[{symbol}] NO TRADE — {condition_reason}")
            return None
        
        # STEP 2: Market analysis
        market_quality = self._analyze_market_quality(symbol, df)
        if market_quality['score'] < 0.60:
            logger.info(f"[{symbol}] Market quality too low: {market_quality['score']:.1%}")
            return None
        
        # STEP 3: Kronos evaluation (if available)
        kronos_signal = None
        if self.kronos:
            kronos_result = self.kronos.get_directional_signal(df, lookback=300, pred_len=12)
            if 'error' not in kronos_result and kronos_result['signal'] != 'NEUTRAL':
                kronos_signal = kronos_result
                logger.info(f"[{symbol}] Kronos: {kronos_signal['signal']} ({kronos_signal['confidence']:.1%})")
        
        # Get signals from active strategies (selected by LLM)
        strategy_signals = []
        strategy_rejections = []
        for name, strategy in self.active_strategies.items():
            try:
                sig = strategy.generate_signal(symbol, {'df': df})
                if sig and hasattr(sig, 'signal'):
                    sig_type = sig.signal.value if hasattr(sig.signal, 'value') else str(sig.signal)
                    if sig_type in ['BUY', 'SELL']:
                        confidence = getattr(sig, 'confidence', 50)
                        strategy_signals.append({
                            'type': sig_type,
                            'strategy': name,
                            'confidence': confidence,
                            'stop_loss': getattr(sig, 'stop_loss', None),
                            'targets': list(getattr(sig, 'targets', []) or []),
                        })
                        logger.info(f"[{symbol}] {name}: {sig_type} ({confidence:.0f}%)")
                    else:
                        reason = getattr(sig, 'reason', None) or sig_type
                        strategy_rejections.append(f"{name}: {reason}")
            except Exception as e:
                logger.error(f"[{symbol}] {name} error: {e}")

        if not strategy_signals:
            logger.info(f"[{symbol}] NO TRADE — no BUY/SELL signal from "
                        f"{len(self.active_strategies)} active strategies; "
                        f"reasons: {' | '.join(strategy_rejections) or 'none returned'}")
            return None

        # ── Weighted voting — strategies weighted by historical expectancy ─────
        trade_signal = self._weighted_vote(strategy_signals, symbol)
        if trade_signal is None:
            logger.info(f"[{symbol}] NO TRADE — strategy vote did not agree")
            return None
        
        # Kronos validation (if available and aligned)
        if kronos_signal:
            kronos_agrees = (
                (trade_signal == 'BUY' and kronos_signal['signal'] == 'LONG') or
                (trade_signal == 'SELL' and kronos_signal['signal'] == 'SHORT')
            )
            if not kronos_agrees:
                logger.warning(f"[{symbol}] Kronos disagrees with strategies - skipping")
                return None
            else:
                logger.info(f"[{symbol}] ✅ Kronos confirms {trade_signal}")
        
        # Safety checks
        position_size = self.kelly.get_position_size(symbol, self.capital, self.leverage)
        position_size = min(position_size, self.capital * 0.10)
        
        allowed, reason = self.safety.check_can_trade(position_size)
        if not allowed:
            logger.warning(f"[{symbol}] ⚠️ Trade blocked: {reason}")
            return None
        
        # Calculate costs
        costs = self.costs.calculate_cost(position_size, current_price)

        # ── Meta-model gate: opportunity-quality layer (P(net return > 0)) ────
        # Only applied when trained; it filters candidates, never generates them.
        meta_confidence = None
        aligned = [s for s in strategy_signals if s['type'] == trade_signal]
        best_conf = max((s['confidence'] for s in aligned), default=50.0)
        if getattr(self.meta_model, '_is_trained', False):
            atr_val = self._calculate_atr(df) if df is not None else current_price * 0.02
            high = float(df['high'].iloc[-1]); low = float(df['low'].iloc[-1])
            mid = (high + low) / 2
            meta_features = {
                'price': current_price,
                'volume_24h': float(df['volume'].iloc[-24:].sum()) if len(df) >= 24 else 0.0,
                # No quote data here: spread is UNKNOWN (0.0 sentinel), the
                # intrabar range proxy is carried separately
                'spread_pct': 0.0,
                'atr_pct': atr_val / current_price if current_price > 0 else 0.0,
                'adx': 0.0,
                'rsi_14': 50.0,
                'kronos_confidence': (kronos_signal or {}).get('confidence', 0.0),
                'llm_confidence': (self.daily_decision or {}).get('confidence', 0.0),
                'opportunity_score': market_quality['score'],
                'signal_confidence': best_conf,
                'trend': 'unknown', 'volatility_regime': 'unknown',
                'market_regime': 'unknown', 'kronos_signal': (kronos_signal or {}).get('signal', 'NONE'),
            }
            meta_pred = self.meta_model.predict(meta_features)
            meta_confidence = meta_pred.confidence
            if meta_pred.decision == 'NO_TRADE':
                logger.info(
                    f"[{symbol}] ❌ Meta-model rejected candidate: {meta_pred.explanation}"
                )
                return None
            logger.info(f"[{symbol}] Meta-model approved (P={meta_pred.confidence:.1%})")

        # Exit plan: prefer the aligned strategy's own exits; ATR fallback later
        best_signal = max(aligned, key=lambda s: s['confidence'], default=None)

        logger.info(f"[{symbol}] ✅ TRADE SIGNAL: {trade_signal}")
        logger.info(f"   Strategies aligned: {[s['strategy'] for s in aligned]}")
        logger.info(f"   Market quality: {market_quality['score']:.1%}")
        logger.info(f"   Position size: ${position_size:.2f}")
        logger.info(f"   Costs: ${costs:.2f}")

        atr_val = self._calculate_atr(df) if df is not None else current_price * 0.02
        high = float(df['high'].iloc[-1]); low = float(df['low'].iloc[-1])
        mid = (high + low) / 2
        expected_gross = best_conf / 100.0 * (atr_val / current_price) if current_price > 0 else 0.0
        expected_net = expected_gross - (costs / position_size if position_size > 0 else 0.0)

        return {
            'signal': trade_signal,
            'size': position_size,
            'price': current_price,
            'votes': len(aligned),
            'market_quality': market_quality,
            'kronos_signal': kronos_signal,
            'strategy_signals': strategy_signals,
            'stop_loss': (best_signal or {}).get('stop_loss'),
            'take_profit': ((best_signal or {}).get('targets') or [None])[0],
            # Ranker candidate fields
            'symbol': symbol,
            'asset_class': 'crypto',
            'signal_confidence': best_conf,
            'kronos_confidence': (kronos_signal or {}).get('confidence', 0.0),
            'momentum_score': market_quality['score'],
            'volume_usd_24h': float(df['volume'].iloc[-24:].sum()) * current_price if len(df) >= 24 else 0.0,
            'spread_pct': None,   # no quote data — never fabricated from bar range
            'bar_range_pct': (high - low) / mid if mid > 0 else 0.0,
            'atr_pct': atr_val / current_price if current_price > 0 else 0.0,
            'expected_net_return': expected_net,
            'meta_model_score': meta_confidence,
            'strategies': [s['strategy'] for s in aligned],
        }
    
    def run_weekly_research_campaign(self, max_instruments: Optional[int] = None
                                     ) -> Optional[dict]:
        """Weekly autonomous research: full discovery loop (external
        intelligence → universal discovery → hypothesis generation →
        validation → PAPER alphas). Fail-soft — research never blocks trading.
        Scale is config-driven (campaign batching handles large universes)."""
        if max_instruments is None:
            # Bound scheduled work; callers can opt into a larger universe
            # after wiring providers with matching history coverage.
            max_instruments = int(os.environ.get(
                'ALPHA_RESEARCH_WEEKLY_MAX', '50'))
        try:
            from core.instruments import InstrumentUniverse
            from core.research_orchestrator import AutonomousResearchOrchestrator
            from data.external_intelligence import (
                ExternalSourceRegistry,
                RawEventStore,
            )
            from data.external_sources import USASpendingSource
            from data.research_interfaces import default_interface_sources

            universe = InstrumentUniverse()
            requested_classes = {
                value.strip().lower() for value in os.environ.get(
                    'ALPHA_RESEARCH_ASSET_CLASSES', 'crypto').split(',')
                if value.strip()
            }
            instruments = []
            if 'crypto' in requested_classes:
                instruments.extend(universe.crypto())
            if 'etf' in requested_classes or 'etfs' in requested_classes:
                instruments.extend(universe.etfs())
            if 'stock' in requested_classes or 'stocks' in requested_classes:
                instruments.extend(universe.stocks())
            # 0 or negative = NO cap: the campaign batches the full eligible
            # universe internally (no hidden truncation — spec §41-42)
            if max_instruments and max_instruments > 0:
                instruments = instruments[:max_instruments]
            logger.info(
                "Weekly research universe: %d instruments (%s)",
                len(instruments), ",".join(sorted(requested_classes)))
            from data.history import fetch_research_history
            data = {}
            for inst in instruments:
                df = fetch_research_history(inst.symbol, period="2y", interval="1d")
                if df is not None and len(df) >= 200:
                    df = normalize_ohlcv(df)
                    data[inst.symbol] = df
            if not data:
                logger.info("Weekly research: no usable history — skipped")
                return None

            registry = ExternalSourceRegistry(
                event_store=RawEventStore("data/trade_memory.sqlite"))
            registry.register(USASpendingSource())
            for src in default_interface_sources():
                registry.register(src)
            orchestrator = AutonomousResearchOrchestrator(
                db_path="data/trade_memory.sqlite", source_registry=registry)
            report = orchestrator.run_campaign(data)
            logger.info(
                "Weekly research campaign: "
                f"{report.get('hypotheses_generated_universal', 0)} universal specs, "
                f"breadth={report.get('search_breadth_total')}, "
                f"new PAPER alphas={len(report.get('new_paper_alphas') or [])}")
            return report
        except Exception as e:
            logger.warning(f"Weekly research campaign failed soft: {e}")
            return None

    def run_alpha_pipeline(self, symbols: List[str]) -> List:
        """RESEARCH → ALPHA LIBRARY → CURRENT MATCH → EV → ALLOCATION.

        Symbols are ALPHA-DRIVEN: each eligible alpha's declared universe
        (e.g. 'sector:semiconductor') is resolved and scanned, with the
        generic scanner list only filling the '*' wildcard.
        """
        from core.portfolio_allocator import alpha_return_correlations
        from core.transaction_costs import execution_feasibility_score

        # Union of scanner symbols and every eligible alpha's resolved universe
        try:
            alpha_symbols = self.alpha_signal_engine.required_symbols(
                scanner_symbols=symbols)
        except Exception as e:
            logger.warning(f"Alpha universe resolution failed: {e}")
            alpha_symbols = []
        fetch_list = list(dict.fromkeys(list(symbols) + alpha_symbols))
        if len(fetch_list) > len(symbols):
            logger.info(
                f"Alpha universes expanded scan: {len(symbols)} scanner symbols "
                f"→ {len(fetch_list)} total (alpha-driven)"
            )

        market_data = {}
        for symbol in fetch_list:
            df = fetch_latest_market_data(symbol)
            if df is not None and len(df) >= 50:
                df = normalize_ohlcv(df)
                df.attrs["timeframe"] = CONFIG.get("interval", "1h")
                market_data[symbol] = df
        if not market_data:
            return []

        # External context feeds (earnings / funding / OI / basis) — best-effort
        context = {}
        try:
            from data.providers import MarketContextBuilder
            context = MarketContextBuilder().build(list(market_data.keys()))
        except Exception as e:
            logger.debug(f"Context feeds unavailable: {e}")

        regime = 'unknown'
        try:
            if self.regime:
                regime = getattr(self.regime, 'current_regime', 'unknown') or 'unknown'
        except Exception:
            pass

        live_mode = os.environ.get("TRADING_MODE", "PAPER").upper() == "LIVE"
        candidates = self.alpha_signal_engine.generate_candidates(
            market_data, regime=regime, context=context, live_mode=live_mode)

        # Economic EV per candidate — GROSS: the OpportunityEnricher applies
        # optimized execution costs so ranking uses true net-execution EV
        for c in candidates:
            price = c.features.get('price') or 0.0
            notional = self.capital * 0.05
            cost = self.costs.estimate_cost_v2(
                asset=c.symbol, asset_class=c.asset_class, side=c.direction,
                quantity=(notional / price) if price else 0.0, price=price,
                market_state={'adv_usd': c.features.get('dollar_volume_24h'),
                              'volatility_pct': c.features.get('atr_pct') or 0.0},
            )
            est = self.ev_model.estimate(c.alpha_id, expected_costs=0.0)
            if est is None:
                # Cold start: strongly validated alphas fall back to their
                # haircut OOS prior until forward observations accumulate
                alpha_record = self.alpha_library.get(c.alpha_id)
                if alpha_record:
                    est = self.ev_model.estimate_with_prior(
                        alpha_record, expected_costs=0.0)
            if est is not None:
                c.expected_net_return = est.expected_net_return
                c.probability_positive = est.probability_positive
                c.expected_upside = est.expected_win
                c.expected_downside = est.expected_loss
                c.expected_shortfall = est.expected_adverse_excursion
                c.ev_lower_bound = est.ev_lower_bound
                c.conservative_ev = est.ev_lower_bound
                c.meta_model_score = est.probability_positive
            c.execution_feasibility = execution_feasibility_score(
                notional_usd=notional,
                adv_usd=c.features.get('dollar_volume_24h'),
                bar_range_pct=c.features.get('bar_range_pct'),
                estimated_cost_bps=cost['total_cost_bps'],
                expected_return_bps=(c.expected_net_return or 0.0) * 10_000,
            )

        # Enrichment: meta-alpha → survival → crowding → half-life/urgency →
        # execution method/cost → borrow → capacity → dollar alpha (spec wiring)
        try:
            from core.opportunity_enrichment import (
                EnrichmentConfig, OpportunityEnricher,
            )
            if not hasattr(self, 'opportunity_enricher'):
                self.opportunity_enricher = OpportunityEnricher(
                    EnrichmentConfig(meta_alpha_mode=os.environ.get(
                        'META_ALPHA_MODE', 'shadow')))
            self.opportunity_enricher.enrich_all(
                candidates, regime=regime, capital=self.capital,
                base_position_frac=self.portfolio_allocator.config.base_position_frac)
            alpha_return_samples = self.opportunity_enricher.return_samples
            for item in self.opportunity_enricher.re_research_queue:
                logger.warning(f"Re-research flagged: {item}")
            self.opportunity_enricher.re_research_queue.clear()
        except Exception as e:
            logger.warning(f"Opportunity enrichment failed soft: {e}")
            alpha_return_samples = {}

        try:
            # Production ordinary + downside/stress matrices (cached, spec §26)
            from core.portfolio_paths import ProductionCorrelationService
            if not hasattr(self, '_correlation_service'):
                self._correlation_service = ProductionCorrelationService()
            self._correlation_service.populate_allocator(self.portfolio_allocator)
        except Exception as e:
            from core.trade_history import TradeMemorySchemaError
            if isinstance(e, TradeMemorySchemaError):
                # Broken accounting ≠ zero risk: block new exposure this cycle
                self._db_health = 'CRITICAL'
                logger.critical(
                    f"CRITICAL_ACCOUNTING_FAILURE: risk history unavailable "
                    f"({e}) — no new positions until resolved")
                return []
            try:
                self.portfolio_allocator.alpha_correlations = alpha_return_correlations()
            except Exception:
                pass
        alpha_matrix = {}
        try:
            alpha_matrix = self._correlation_service.daily_alpha_returns()
        except Exception as e:
            from core.trade_history import TradeMemorySchemaError
            if isinstance(e, TradeMemorySchemaError):
                self._db_health = 'CRITICAL'
                logger.critical(f"CRITICAL_ACCOUNTING_FAILURE: {e}")
                return []
        decisions = self.portfolio_allocator.allocate(
            candidates, open_positions=self.positions.get_open_positions(),
            alpha_return_samples=alpha_return_samples,
            alpha_return_matrix=alpha_matrix or None)

        n_accepted = sum(1 for d in decisions if d.accepted)
        logger.info(
            f"Alpha pipeline [{self.alpha_pipeline_mode}]: "
            f"{len(candidates)} candidates → {n_accepted} accepted; "
            f"rejections: {self.alpha_signal_engine.rejection_summary()}"
        )
        # Capital-allocation + rejected-opportunity report (spec §47-48)
        try:
            from core.opportunity_enrichment import OpportunityEnricher as _OE
            for row in _OE.allocation_report(decisions):
                logger.info(f"ALLOCATION: {row}")
        except Exception:
            pass
        # Persist every pipeline decision (audit trail + maturity evidence,
        # regardless of mode)
        self._record_shadow_comparison(decisions)
        return decisions

    def _legacy_fallback_allowed(self) -> bool:
        """Legacy fallback retires only on EVIDENCE-BASED maturity: resolved
        outcomes, EV calibration and forward expectancy — never elapsed time
        alone. After maturity, a pipeline NO TRADE is final."""
        from core.pipeline_maturity import (
            MaturityThresholds, PipelineMaturityEvaluator,
        )
        thresholds = MaturityThresholds(
            min_days=int(os.environ.get('ALPHA_MATURITY_MIN_DAYS', '14')),
            min_decisions=int(os.environ.get('ALPHA_MATURITY_MIN_DECISIONS', '100')),
            min_resolved_outcomes=int(
                os.environ.get('ALPHA_MATURITY_MIN_RESOLVED', '50')),
        )
        assessment = PipelineMaturityEvaluator(
            'data/trade_memory.sqlite', thresholds).assess()
        if not assessment.legacy_fallback_allowed:
            logger.info(assessment.report())
        return assessment.legacy_fallback_allowed

    def _record_shadow_comparison(self, decisions: List) -> None:
        """Persist alpha-pipeline decisions for legacy-vs-alpha comparison."""
        import sqlite3 as _sqlite3
        try:
            with _sqlite3.connect('data/trade_memory.sqlite') as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS alpha_pipeline_shadow ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, candidate_id TEXT, "
                    "alpha_id TEXT, symbol TEXT, direction TEXT, accepted INTEGER, "
                    "allocation_usd REAL, expected_net_return REAL, ev_lower_bound REAL, "
                    "reason_codes TEXT, recorded_at TEXT)"
                )
                for d in decisions:
                    c = d.candidate
                    conn.execute(
                        "INSERT INTO alpha_pipeline_shadow (candidate_id, alpha_id, "
                        "symbol, direction, accepted, allocation_usd, "
                        "expected_net_return, ev_lower_bound, reason_codes, recorded_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?,datetime('now'))",
                        (c.candidate_id, c.alpha_id, c.symbol, c.direction,
                         int(d.accepted), d.allocation_usd, c.expected_net_return,
                         c.ev_lower_bound, ",".join(d.reason_codes)),
                    )
                conn.commit()
        except Exception as e:
            logger.warning(f"Shadow comparison record failed: {e}")

    def _paper_evidence(self):
        """One canonical PaperEvidenceTracker instance (spec §2)."""
        if not hasattr(self, 'paper_evidence_tracker'):
            try:
                from core.paper_evidence import PaperEvidenceTracker
                self.paper_evidence_tracker = PaperEvidenceTracker(
                    'data/trade_memory.sqlite')
            except Exception as e:
                # Core accounting unavailable = CRITICAL, not a warning
                logger.critical(f"CRITICAL_ACCOUNTING_FAILURE: "
                                f"PaperEvidenceTracker unavailable: {e}")
                self.paper_evidence_tracker = None
        return self.paper_evidence_tracker

    _TIMEFRAME_SECONDS = {'1m': 60, '5m': 300, '15m': 900, '30m': 1800,
                          '1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800}

    def _bar_interval_seconds(self, candidate) -> float:
        """Real bar interval for half-life unit conversion — from the
        candidate/alpha timeframe, defaulting to the campaign's daily bars."""
        tf = ((candidate.features or {}).get('timeframe')
              or (candidate.entry_plan or {}).get('timeframe'))
        if not tf:
            try:
                record = self.alpha_library.get(candidate.alpha_id) or {}
                tf = (record.get('extra') or {}).get('data_timeframe') \
                    if isinstance(record.get('extra'), dict) else None
            except Exception:
                tf = None
        return float(LLMTradingBot._TIMEFRAME_SECONDS.get(
            str(tf or '1d'), 86_400))

    def _position_filled_notional(self, pos_id: int) -> Optional[float]:
        try:
            import sqlite3 as _sq
            with _sq.connect(self.positions.db_path) as conn:
                row = conn.execute("SELECT size FROM positions WHERE id=?",
                                   (pos_id,)).fetchone()
            return float(row[0]) if row else None
        except Exception:
            return None

    def _alpha_decision_to_trade(self, decision) -> Dict:
        """Convert an accepted allocator decision into the execute_trade format."""
        c = decision.candidate
        signal = 'BUY' if c.direction == 'long' else 'SELL'
        strategy_name = c.alpha_id.split(':')[0]
        return {
            'signal': signal,
            'size': decision.allocation_usd,
            'price': c.features.get('price') or c.entry_plan.get('reference_price'),
            'votes': 1,
            'market_quality': {'score': c.final_opportunity_score or 0.0},
            'kronos_signal': None,
            'strategy_signals': [{'type': signal, 'strategy': strategy_name,
                                  'confidence': (c.probability_positive or 0.5) * 100,
                                  'stop_loss': c.exit_plan.get('stop_price'),
                                  'targets': [c.exit_plan.get('target_price')]}],
            'strategies': [strategy_name],
            'stop_loss': c.exit_plan.get('stop_price'),
            'take_profit': c.exit_plan.get('target_price'),
            'symbol': c.symbol,
            'alpha_id': c.alpha_id,
            'candidate_id': c.candidate_id,
            'execution_decision': self._build_execution_decision(decision),
        }

    def _build_execution_decision(self, decision):
        """Carry the optimizer's order method to the broker (spec §2-4)."""
        try:
            from core.execution_decision import build_execution_decision
            c = decision.candidate
            price = c.features.get('price') or c.entry_plan.get('reference_price')
            if not price:
                return None
            return build_execution_decision(
                c, allocation_usd=decision.allocation_usd, price=float(price))
        except Exception as e:
            logger.warning(f"ExecutionDecision build failed: {e}")
            return None

    def _get_strategy_weights(self) -> Dict[str, float]:
        """
        Evidence-aware strategy weights.

        Replaces the old `expectancy × win_rate × sqrt(n)` heuristic with the
        LOWER confidence bound of net expectancy (autocorrelation-adjusted),
        scaled by edge health and penalized for correlation with other active
        strategies. Uncertain / small-sample / redundant strategies rank lower
        even when their point estimate looks good.
        """
        from core.strategy_correlation import evidence_aware_strategy_score

        weights: Dict[str, float] = {}
        try:
            corr_matrix = self.correlation_tracker.correlation_matrix()
        except Exception:
            corr_matrix = {}
        active_names = list(self.active_strategies)

        for name in active_names:
            pnls = self._recent_strategy_pnls(name)
            if len(pnls) < 5:
                weights[name] = 0.5   # no evidence → neutral exploration weight
                continue
            health = 1.0
            try:
                if self.strategy_health.is_paused(name):
                    health = 0.0
            except Exception:
                pass
            corr_penalty = self.correlation_tracker.correlation_penalty(
                name, active_names, corr_matrix)
            dd = 0.0
            cum, peak = 0.0, 0.0
            for p in pnls:
                cum += p
                peak = max(peak, cum)
                dd = max(dd, peak - cum)
            score = evidence_aware_strategy_score(
                pnls,
                edge_health=health,
                correlation_penalty=corr_penalty,
                max_drawdown_frac=dd / max(self.capital, 1.0),
            )
            weights[name] = max(score, 0.1)   # floor preserves exploration

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        logger.debug(f"Strategy weights (evidence-aware): {weights}")
        return weights

    def _recent_strategy_pnls(self, strategy: str, days: int = 30) -> List[float]:
        """Net P&L stream for one strategy from trade memory (most recent first)."""
        import sqlite3 as _sqlite3
        try:
            with _sqlite3.connect('data/trade_memory.sqlite') as conn:
                rows = conn.execute(
                    "SELECT net_pnl FROM trade_memory WHERE strategy=? "
                    "AND exit_time >= datetime('now', ?) ORDER BY exit_time ASC",
                    (strategy, f'-{days} days'),
                ).fetchall()
            return [float(r[0]) for r in rows if r[0] is not None]
        except Exception:
            return []

    def _weighted_vote(
        self,
        strategy_signals: List[Dict],
        symbol: str,
    ) -> Optional[str]:
        """
        Weighted voting across strategy signals.

        Each strategy's vote is weighted by its historical expectancy.
        The winning side must accumulate > 60% of total weight to trigger a trade.

        Returns 'BUY', 'SELL', or None (no consensus).
        """
        weights = self._get_strategy_weights()
        buy_weight  = 0.0
        sell_weight = 0.0
        total_weight = 0.0

        for sig in strategy_signals:
            strat  = sig["strategy"]
            w      = weights.get(strat, 1.0 / max(len(self.active_strategies), 1))
            # Multiply by signal confidence (0-100 → 0-1)
            conf   = sig.get("confidence", 50) / 100.0
            vote   = w * conf
            total_weight += vote
            if sig["type"] == "BUY":
                buy_weight += vote
            else:
                sell_weight += vote

        if total_weight == 0:
            return None

        buy_pct  = buy_weight  / total_weight
        sell_pct = sell_weight / total_weight
        THRESHOLD = 0.60   # need 60% of weighted votes to act

        if buy_pct >= THRESHOLD:
            logger.info(
                f"[{symbol}] Weighted vote: BUY {buy_pct:.0%} "
                f"(SELL {sell_pct:.0%}, threshold {THRESHOLD:.0%})"
            )
            return "BUY"
        elif sell_pct >= THRESHOLD:
            logger.info(
                f"[{symbol}] Weighted vote: SELL {sell_pct:.0%} "
                f"(BUY {buy_pct:.0%}, threshold {THRESHOLD:.0%})"
            )
            return "SELL"
        else:
            logger.info(
                f"[{symbol}] No weighted consensus: BUY {buy_pct:.0%} "
                f"SELL {sell_pct:.0%} (need {THRESHOLD:.0%})"
            )
            return None

    def _check_live_authorization(self, strategies: List[str]) -> tuple[bool, str]:
        """Hard gate: in LIVE mode, every contributing strategy must be
        explicitly approved (env LIVE_APPROVED_STRATEGIES or an alpha in a
        LIVE_* lifecycle state). DISCOVERED/VALIDATING/REJECTED/PAPER
        strategies can never send live orders. Paper/shadow modes pass."""
        mode = os.environ.get("TRADING_MODE", "PAPER").upper()
        if mode != "LIVE":
            return True, f"non-live mode ({mode})"
        if not strategies:
            return False, "no attributable strategy for live order"
        for name in strategies:
            if name in self.live_approved_strategies:
                continue
            try:
                if self.alpha_library.is_live_approved(name):
                    continue
            except Exception:
                pass
            return False, (
                f"strategy '{name}' is not approved for live trading "
                "(needs LIVE_APPROVED_STRATEGIES or alpha LIVE_* state)"
            )
        return True, "all strategies live-approved"

    @staticmethod
    def _exit_levels_sane(signal: str, entry: float, stop: Optional[float],
                          target: Optional[float]) -> bool:
        """Validate strategy-provided exits are on the correct side of entry."""
        if stop is None or target is None or entry <= 0:
            return False
        if signal == 'BUY':
            return stop < entry < target
        return target < entry < stop

    def _family_exposure_exceeded(self, decision: Dict,
                                  max_family_frac: float = 0.40) -> bool:
        """True when adding this trade would concentrate >40% of capital in
        one strategy family (TREND / MOMENTUM / MEAN_REVERSION / ...)."""
        from core.strategy_correlation import family_of
        try:
            open_positions = self.positions.get_open_positions()
            exposure = self.correlation_tracker.family_exposure(open_positions)
            families = {family_of(s) for s in decision.get('strategies', [])} or {'MOMENTUM'}
            new_size = float(decision.get('size', 0.0))
            for fam in families:
                if exposure.get(fam, 0.0) + new_size > self.capital * max_family_frac:
                    return True
            return False
        except Exception as e:
            logger.warning(f"Family exposure check failed (allowing): {e}")
            return False

    def execute_trade(self, symbol: str, decision: Dict) -> Optional[int]:
        """STEP 4: Execute trade through broker → confirm fill → open local record.

        Execution path (canonical):
          1. SafetyManager.check_can_trade() — hard risk gate
          2. Broker.submit_and_wait()        — sends order, waits for fill
          3. On FILLED: PositionManager.open_position(broker_order_id=...)
          4. SafetyManager.on_position_open()

        A position is NEVER opened locally before the broker confirms a fill.
        """
        logger.info(f"\n[{symbol}] Executing {decision['signal']} trade...")

        # ── FAIL-CLOSED GATE: broken core accounting blocks NEW exposure ──────
        if getattr(self, '_db_health', 'HEALTHY') != 'HEALTHY':
            logger.critical(f"[{symbol}] 🛑 BLOCKED: database health "
                            f"{getattr(self, '_db_health', '?')} — "
                            "position monitoring/closing only")
            return None

        entry_price = decision['price']

        # ── LIVE-SAFETY ASSERTION ────────────────────────────────────────────
        # Unvalidated strategies may NEVER route real capital. In LIVE mode
        # every contributing strategy must be explicitly approved.
        allowed, live_reason = self._check_live_authorization(
            [s['strategy'] for s in decision.get('strategy_signals', [])
             if s['type'] == decision['signal']]
        )
        if not allowed:
            logger.critical(f"[{symbol}] 🛑 LIVE ORDER BLOCKED: {live_reason}")
            return None

        # Calculate ATR for dynamic stops
        df = fetch_latest_market_data(symbol)
        atr = self._calculate_atr(df) if df is not None else entry_price * 0.02

        # Exit plan: strategy-specific exits when provided; generic ATR fallback
        stop_loss = decision.get('stop_loss')
        take_profit = decision.get('take_profit')
        exit_source = 'strategy'
        if stop_loss is None or take_profit is None or not self._exit_levels_sane(
                decision['signal'], entry_price, stop_loss, take_profit):
            exit_source = 'atr_fallback'
            if decision['signal'] == 'BUY':
                stop_loss   = entry_price - (atr * 2.0)
                take_profit = entry_price + (atr * 3.0)
            else:
                stop_loss   = entry_price + (atr * 2.0)
                take_profit = entry_price - (atr * 3.0)

        logger.info(
            f"[{symbol}] Entry: ${entry_price:.4f}  SL: ${stop_loss:.4f}  "
            f"TP: ${take_profit:.4f}  (exits: {exit_source})"
        )

        # Dollar size → coin quantity for the order
        size_dollars = decision['size'] * self._position_size_scale

        # Strategy/family/asset drawdown budgets scale or veto the allocation
        lead_strategy = (decision.get('strategies') or ['unknown'])[0]
        budget_mult = self.drawdown_budgets.allocation_multiplier(lead_strategy, symbol)
        if budget_mult <= 0:
            logger.warning(f"[{symbol}] ⚠️ Blocked: drawdown budget exhausted for {lead_strategy}")
            return None
        size_dollars *= budget_mult

        quantity     = size_dollars / entry_price if entry_price > 0 else 0
        if quantity <= 0:
            logger.warning(f"[{symbol}] Invalid quantity {quantity:.6f} — skipping")
            return None

        # ── STEP 1: Safety gate ────────────────────────────────────────────────
        can_trade, reason = self.safety.check_can_trade(position_size=size_dollars)
        if not can_trade:
            logger.warning(f"[{symbol}] Safety gate blocked: {reason}")
            return None

        # ── STEP 2: Submit to broker and wait for fill ─────────────────────────
        # ExecutionDecision path: the optimizer's method reaches the broker;
        # expiration, capability fallback, and a final EV recheck run first.
        exec_decision = decision.get('execution_decision')
        order_type, limit_price, timeout = "MARKET", None, 30.0
        exec_store = None
        exec_mode = os.environ.get(
            'EXECUTION_OPTIMIZER_MODE',
            'shadow' if os.environ.get('TRADING_MODE', 'PAPER').upper() == 'LIVE'
            else 'active')
        if exec_decision is not None:
            try:
                from core.execution_decision import (
                    BrokerCapabilities,
                    ExecutionDecisionStore,
                    pre_submission_recheck,
                    resolve_for_broker,
                )
                exec_decision = resolve_for_broker(
                    exec_decision, BrokerCapabilities.detect(self.broker))
                if exec_decision.resolution.startswith("NO_TRADE"):
                    logger.warning(
                        f"[{symbol}] Execution blocked: {exec_decision.resolution} "
                        f"{exec_decision.detail.get('why', '')}")
                    ExecutionDecisionStore().record(exec_decision)
                    return None
                current_spread = decision.get('current_spread_pct')
                if current_spread is not None:
                    exec_decision = pre_submission_recheck(
                        exec_decision,
                        current_spread_pct=float(current_spread),
                        reference_spread_pct=exec_decision.expected_spread_bps
                        * 2 / 10_000)
                    if exec_decision.resolution.startswith("NO_TRADE"):
                        logger.warning(
                            f"[{symbol}] Pre-submission recheck cancelled order: "
                            f"{exec_decision.detail.get('why', '')}")
                        ExecutionDecisionStore().record(exec_decision)
                        return None
                if exec_mode == 'active':
                    order_type = exec_decision.order_type
                    limit_price = exec_decision.limit_price
                    timeout = min(exec_decision.max_execution_delay_seconds, 120.0)
                else:
                    logger.info(
                        f"[{symbol}] ExecutionOptimizer SHADOW: would use "
                        f"{exec_decision.order_type} @ {exec_decision.limit_price}")
                exec_store = ExecutionDecisionStore()
            except Exception as e:
                logger.warning(f"ExecutionDecision handling failed soft: {e}")
                exec_decision = None

        logger.info(f"[{symbol}] Submitting {decision['signal']} {quantity:.6f} "
                    f"@ ${entry_price:.4f} ({order_type}) to broker...")
        fill = self.broker.submit_and_wait(
            symbol=symbol,
            side=decision['signal'],          # "BUY" or "SELL"
            quantity=quantity,
            current_price=entry_price,
            order_type=order_type,
            limit_price=limit_price,
            timeout_seconds=timeout,
        )

        if not fill.status.value in ('FILLED', 'PARTIAL'):
            logger.warning(
                f"[{symbol}] Order NOT filled — status={fill.status.value} "
                f"reason={fill.reject_reason}"
            )
            return None

        if fill.partial:
            logger.warning(f"[{symbol}] Partial fill: {fill.fill_quantity:.6f} / {quantity:.6f}")
            # Chase the remainder only if EV still clears costs (spec §10)
            if exec_decision is not None:
                from core.execution_decision import remaining_ev_sufficient
                spread_now = decision.get('current_spread_pct') or \
                    exec_decision.expected_spread_bps * 2 / 10_000
                if not remaining_ev_sufficient(
                        exec_decision, filled_qty=fill.fill_quantity,
                        requested_qty=quantity,
                        current_spread_pct=float(spread_now) * 1.0,
                        reference_spread_pct=exec_decision.expected_spread_bps
                        * 2 / 10_000):
                    try:
                        self.broker.cancel_order(fill.order_id)
                        logger.info(f"[{symbol}] Cancelled unfilled remainder — "
                                    "EV no longer sufficient")
                    except Exception:
                        pass

        # Realized execution attribution: spread/slippage/impact/fees SEPARATE
        if exec_decision is not None and exec_store is not None:
            try:
                from core.execution_decision import realized_execution_attribution
                realized = realized_execution_attribution(
                    exec_decision, fill_price=fill.fill_price,
                    mid_at_decision=entry_price, fees_usd=fill.fees,
                    notional_usd=fill.notional)
                exec_store.record(exec_decision, realized=realized)
            except Exception as e:
                logger.debug(f"Execution attribution failed: {e}")

        # ── STEP 3: Open local position record using confirmed fill data ────────
        pos_id = self.positions.open_position(
            symbol=symbol,
            signal=decision['signal'],
            size=fill.notional,                # actual dollar value filled
            entry_price=fill.fill_price,       # actual fill price (incl. slippage)
            entry_fill_price=fill.fill_price,
            entry_fees=fill.fees,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategies_used=decision.get('strategy_signals', []),
            kronos_confidence=(decision.get('kronos_signal') or {}).get('confidence', 0),
            broker_order_id=fill.order_id,
        )

        # ── STEP 4: Update safety manager ──────────────────────────────────────
        self.safety.on_position_open()

        # ── STEP 5: Place broker-native bracket orders (crash protection) ───────
        bracket = self.broker.place_bracket_orders(
            symbol=symbol,
            quantity=fill.fill_quantity,
            stop_price=stop_loss,
            take_profit_price=take_profit,
        )
        if bracket.get('sl_order_id'):
            logger.info(
                f"[{symbol}] Bracket orders placed: "
                f"SL={bracket['sl_order_id']} TP={bracket['tp_order_id']}"
            )
        else:
            logger.info(f"[{symbol}] Broker-native brackets not supported — software stops active")

        logger.info(
            f"[{symbol}] ✅ Position #{pos_id} opened | "
            f"filled ${fill.notional:.2f} @ ${fill.fill_price:.4f} "
            f"fees=${fill.fees:.4f}"
        )
        return pos_id
    
    def manage_open_positions(self):
        """Manage all open positions - check stops, targets, trailing, etc."""
        open_positions = self.positions.get_open_positions()
        
        if not open_positions:
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"MANAGING {len(open_positions)} OPEN POSITIONS")
        logger.info(f"{'='*60}")
        
        for pos in open_positions:
            try:
                symbol = pos['symbol']
                
                # Get current price
                df = fetch_latest_market_data(symbol)
                if df is None:
                    continue
                
                current_price = float(df['close'].iloc[-1])
                entry_price = pos['entry_price']
                direction = pos['direction']
                
                # Calculate P&L
                if direction == 'LONG':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:  # SHORT
                    pnl_pct = (entry_price - current_price) / entry_price
                
                pnl_dollars = pnl_pct * pos['size']

                # Persist excursion extremes — MFE/MAE must exist at close time
                try:
                    mfe = max(float(pos.get('mfe_pct') or 0.0), pnl_pct * 100)
                    mae = min(float(pos.get('mae_pct') or 0.0), pnl_pct * 100)
                    if mfe != (pos.get('mfe_pct') or 0.0) or \
                            mae != (pos.get('mae_pct') or 0.0):
                        self.positions.update_position(pos['id'],
                                                       mfe_pct=mfe, mae_pct=mae)
                except Exception as _ex:
                    logger.debug(f"[{symbol}] excursion update failed: {_ex}")

                logger.info(f"\n[{symbol}] {direction} @ ${entry_price:.2f} → ${current_price:.2f}")
                logger.info(f"   P&L: {pnl_pct*100:+.2f}% (${pnl_dollars:+.2f})")

                # ── Check broker-native bracket triggers first ─────────────────
                # (PaperBroker simulates this; AlpacaBroker orders fire at broker)
                if hasattr(self.broker, 'check_bracket_triggers'):
                    broker_trigger = self.broker.check_bracket_triggers(symbol, current_price)
                    if broker_trigger == 'stop_loss':
                        logger.warning(f"[{symbol}] 🛑 BROKER STOP LOSS TRIGGERED!")
                        self._close_position(pos, current_price, "broker_stop_loss")
                        continue
                    elif broker_trigger == 'take_profit':
                        logger.info(f"[{symbol}] 🎯 BROKER TAKE PROFIT TRIGGERED!")
                        self._close_position(pos, current_price, "broker_take_profit")
                        continue

                # Check stop loss
                stop_loss = pos.get('stop_loss')
                if stop_loss and self._should_stop_out(current_price, stop_loss, direction):
                    logger.warning(f"[{symbol}] 🛑 STOP LOSS HIT!")
                    self._close_position(pos, current_price, "stop_loss")
                    continue
                
                # Check take profit
                take_profit = pos.get('take_profit')
                if take_profit and self._should_take_profit(current_price, take_profit, direction):
                    logger.info(f"[{symbol}] 🎯 TAKE PROFIT HIT!")
                    self._close_position(pos, current_price, "take_profit")
                    continue
                
                # Check trailing stop
                if self._should_trail_stop(pos, current_price, pnl_pct):
                    logger.info(f"[{symbol}] 📉 TRAILING STOP HIT!")
                    self._close_position(pos, current_price, "trailing_stop")
                    continue
                
                # Check time limit (4 hours for day trading)
                time_in_trade = (datetime.now() - pos['entry_time']).total_seconds() / 3600
                if time_in_trade > 4:
                    logger.info(f"[{symbol}] ⏰ TIME LIMIT ({time_in_trade:.1f}h)")
                    self._close_position(pos, current_price, "time_limit")
                    continue
                
                # Check signal reversal
                if self._check_signal_reversal(symbol, direction):
                    logger.warning(f"[{symbol}] 🔄 SIGNAL REVERSAL!")
                    self._close_position(pos, current_price, "signal_reversal")
                    continue
                
                # Partial profit taking (scale out at +2%)
                if pnl_pct > 0.02 and not pos.get('partial_closed', False):
                    logger.info(f"[{symbol}] 💰 PARTIAL PROFIT @ +{pnl_pct*100:.1f}%")
                    self._take_partial_profit(pos, current_price)
                    continue
                
                # Update trailing stop if in profit
                if pnl_pct > 0.015:  # >1.5% profit
                    self._update_trailing_stop(pos, current_price, pnl_pct)
                
            except Exception as e:
                logger.error(f"Error managing position {pos.get('symbol', 'unknown')}: {e}")
    
    def _calculate_atr(self, df, period=14) -> float:
        """Calculate Average True Range for dynamic stops."""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            tr = []
            for i in range(1, len(df)):
                tr_val = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1])
                )
                tr.append(tr_val)
            
            atr = sum(tr[-period:]) / period if len(tr) >= period else sum(tr) / len(tr)
            return atr
        except (KeyError, IndexError, ValueError) as e:
            logger.warning(f"ATR calculation failed: {e}, using 2% fallback")
            # Fallback: 2% of price
            return float(df['close'].iloc[-1]) * 0.02
    
    def _should_stop_out(self, current_price: float, stop_loss: float, direction: str) -> bool:
        """Check if stop loss is hit."""
        if direction == 'LONG':
            return current_price <= stop_loss
        else:  # SHORT
            return current_price >= stop_loss
    
    def _should_take_profit(self, current_price: float, take_profit: float, direction: str) -> bool:
        """Check if take profit is hit."""
        if direction == 'LONG':
            return current_price >= take_profit
        else:  # SHORT
            return current_price <= take_profit
    
    def _should_trail_stop(self, pos: Dict, current_price: float, pnl_pct: float) -> bool:
        """Check if trailing stop is hit."""
        # Only trail if in profit >2%
        if pnl_pct < 0.02:
            return False
        
        # Get max price reached
        max_price = pos.get('max_price', pos['entry_price'])
        
        # Trail by 1.5% from max
        trail_pct = 0.015
        
        if pos['direction'] == 'LONG':
            trail_price = max_price * (1 - trail_pct)
            return current_price <= trail_price
        else:  # SHORT
            trail_price = max_price * (1 + trail_pct)
            return current_price >= trail_price
    
    def _check_signal_reversal(self, symbol: str, current_direction: str) -> bool:
        """Check if strategies now say opposite direction."""
        try:
            df = fetch_latest_market_data(symbol)
            if df is None:
                return False
            
            # Get current signals from active strategies
            buy_votes = 0
            sell_votes = 0
            
            for strategy in self.active_strategies.values():
                sig = strategy.generate_signal(symbol, {'df': df})
                if sig and hasattr(sig, 'signal'):
                    sig_type = sig.signal.value if hasattr(sig.signal, 'value') else str(sig.signal)
                    if sig_type == 'BUY':
                        buy_votes += 1
                    elif sig_type == 'SELL':
                        sell_votes += 1
            
            # Check if majority reversed
            if current_direction == 'LONG' and sell_votes > buy_votes:
                return True
            elif current_direction == 'SHORT' and buy_votes > sell_votes:
                return True
            
            return False
        except Exception as e:
            logger.warning(f"Signal reversal check failed: {e}")
            return False
    
    def _take_partial_profit(self, pos: Dict, current_price: float, pnl_dollars: float = 0):
        """
        Take 50% profit through the canonical partial-close lifecycle:

            1. Broker.close_position(quantity=50%) — submit exit order, wait for fill
            2. PositionManager.partial_close()     — record actual fill prices/fees
            3. SafetyManager.on_partial_close()    — record P&L (no position count change)
            4. Move stop to breakeven
        """
        symbol = pos['symbol']
        pos_id = pos['id']
        fraction = 0.5

        logger.info(f"[{symbol}] Partial profit — closing {fraction:.0%} of #{pos_id}")

        # Estimate quantity for 50% close
        entry_fill   = pos.get('entry_fill_price') or pos['entry_price']
        size_dollars = pos['size']
        full_qty     = size_dollars / entry_fill if entry_fill > 0 else 0
        half_qty     = full_qty * fraction

        # ── Step 1: Broker partial exit ────────────────────────────────────────
        fill = self.broker.close_position(symbol, current_price, quantity=half_qty)

        if fill.status.value not in ('FILLED', 'PARTIAL'):
            logger.error(
                f"[{symbol}] Partial exit NOT filled — status={fill.status.value} "
                f"reason={fill.reject_reason}. Skipping partial close."
            )
            return

        # ── Step 2: Record in DB ───────────────────────────────────────────────
        net_pnl = self.positions.partial_close(
            pos_id,
            fraction=fraction,
            close_price=current_price,
            exit_fill_price=fill.fill_price,
            exit_fees=fill.fees,
            broker_order_id=fill.order_id,
        )

        # ── Step 3: Update safety (does NOT decrement active_positions) ─────────
        if net_pnl is not None:
            self.safety.on_partial_close(net_pnl)

        # ── Step 4: Move stop to breakeven ─────────────────────────────────────
        self.positions.update_position(pos_id, stop_loss=pos['entry_price'])

        logger.info(
            f"[{symbol}] Partial close #{pos_id}: {fraction:.0%} @ ${fill.fill_price:.4f} "
            f"net_pnl=${net_pnl:+.2f} | stop → breakeven"
        )
    
    def _update_trailing_stop(self, pos: Dict, current_price: float, pnl_pct: float):
        """Update trailing stop to lock in profits."""
        # Update max price
        max_price = pos.get('max_price', pos['entry_price'])
        
        if pos['direction'] == 'LONG':
            if current_price > max_price:
                self.positions.update_position(pos['id'], max_price=current_price)
                logger.debug(f"   New max: ${current_price:.2f} (trail: {pnl_pct*100:.1f}%)")
        else:  # SHORT
            if current_price < max_price:
                self.positions.update_position(pos['id'], max_price=current_price)
                logger.debug(f"   New max: ${current_price:.2f} (trail: {pnl_pct*100:.1f}%)")
    
    def _close_position(self, pos: Dict, current_price: float, reason: str, pnl_dollars: float = 0):
        """
        Close a position through the canonical exit lifecycle:

            1. Broker.close_position()           — submit exit order, wait for fill
            2. PositionManager.close_position()  — record actual fill prices/fees
            3. SafetyManager.on_position_closed() — update risk state

        The local DB is NEVER updated until the broker confirms the fill.
        """
        symbol = pos['symbol']
        pos_id = pos['id']

        logger.info(f"[{symbol}] Closing position #{pos_id} — reason: {reason}")

        # ── Step 1: Execute exit at broker ─────────────────────────────────────
        fill = self.broker.close_position(symbol, current_price)

        if fill.status.value not in ('FILLED', 'PARTIAL'):
            logger.error(
                f"[{symbol}] Exit order NOT filled — status={fill.status.value} "
                f"reason={fill.reject_reason}. Position still OPEN locally."
            )
            return   # Do NOT update local state if broker rejected the close

        actual_exit = fill.fill_price
        actual_fees = fill.fees

        # ── Step 2: Record actual fill in DB ───────────────────────────────────
        net_pnl = self.positions.close_position(
            pos_id,
            close_price=current_price,
            exit_fill_price=actual_exit,
            exit_fees=actual_fees,
            reason=reason,
            broker_order_id=fill.order_id,
        )

        if net_pnl is None:
            logger.error(f"[{symbol}] PositionManager.close_position returned None — check DB")
            return

        # ── Step 3: Update risk state ──────────────────────────────────────────
        self.safety.on_position_closed(net_pnl)

        # ── Step 4: Write to permanent trade memory ────────────────────────────
        full_pos = None
        try:
            import sqlite3
            with sqlite3.connect(self.positions.db_path) as conn:
                conn.row_factory = sqlite3.Row
                full_pos = conn.execute(
                    "SELECT * FROM positions WHERE id=?", (pos_id,)
                ).fetchone()
            if full_pos:
                self.trade_memory.record(dict(full_pos))
        except Exception as e:
            # Cannot persist a closed trade → accounting no longer trustworthy:
            # block NEW exposure until resolved (monitoring/closing continues)
            self._db_health = 'DEGRADED_READ_ONLY'
            logger.critical(f"[{symbol}] CRITICAL_ACCOUNTING_FAILURE: "
                            f"TradeMemory write failed ({e}) — "
                            "new exposure disabled")

        # ── Step 5: Resolve paper evidence on the canonical close (spec §9-11) ─
        evidence = self._paper_evidence()
        if evidence is not None:
            try:
                from core.trade_history import compute_trade_net_return
                size = float(full_pos['size']) if full_pos else None
                ret = compute_trade_net_return(net_pnl, size)
                if ret is not None and full_pos:
                    p = dict(full_pos)
                    holding = p.get('holding_hours')
                    if holding is None and p.get('entry_time') and p.get('exit_time'):
                        try:
                            t0 = datetime.fromisoformat(str(p['entry_time']))
                            t1 = datetime.fromisoformat(str(p['exit_time']))
                            holding = (t1 - t0).total_seconds() / 3600.0
                        except (ValueError, TypeError):
                            holding = None
                    fee_bps = ((p.get('fees') or 0.0) / size * 10_000
                               if size else None)
                    slip_bps = 0.0
                    if p.get('entry_price') and p.get('entry_fill_price'):
                        slip_bps += abs(p['entry_fill_price'] - p['entry_price']) \
                            / p['entry_price'] * 10_000
                    if p.get('exit_price') and p.get('exit_fill_price'):
                        slip_bps += abs(p['exit_fill_price'] - p['exit_price']) \
                            / p['exit_price'] * 10_000
                    evidence.resolve_by_position(
                        pos_id, realized_net_return=ret,
                        realized_net_pnl=net_pnl,
                        realized_execution_cost_bps=(fee_bps or 0.0) + slip_bps,
                        realized_holding_hours=holding,
                        realized_mfe_pct=p.get('mfe_pct'),
                        realized_mae_pct=p.get('mae_pct'),
                        realized_fee_bps=fee_bps,
                        realized_slippage_bps=slip_bps,
                        close_reason=reason)
            except Exception as _pe:
                logger.warning(f"[{symbol}] Paper evidence resolve failed: {_pe}")

        logger.info(
            f"[{symbol}] ✅ CLOSED #{pos_id} @ ${actual_exit:.4f} "
            f"net_pnl=${net_pnl:+.2f} reason={reason}"
        )
    
    def check_portfolio_risk(self) -> tuple[bool, str]:
        """Check portfolio-level risk before opening new position."""
        open_positions = self.positions.get_open_positions()
        
        # Check position count
        if len(open_positions) >= self.safety.max_positions:
            return False, f"At max positions ({self.safety.max_positions})"
        
        # Calculate total portfolio heat (risk across all positions)
        total_risk = 0
        for pos in open_positions:
            # Risk = distance to stop loss
            entry = pos['entry_price']
            stop = pos.get('stop_loss', entry * 0.98)
            risk_per_position = abs(entry - stop) / entry * pos['size']
            total_risk += risk_per_position
        
        portfolio_heat_pct = total_risk / self.capital
        
        if portfolio_heat_pct > 0.10:  # Max 10% total portfolio risk
            return False, f"Portfolio heat too high ({portfolio_heat_pct*100:.1f}%)"
        
        # Check correlation (avoid too many crypto positions)
        crypto_positions = sum(1 for p in open_positions if 'USD' in p['symbol'])
        if crypto_positions >= 4:  # Max 4 crypto positions (they're correlated)
            return False, f"Too many correlated crypto positions ({crypto_positions})"
        
        return True, "OK"
    
    def llm_trade_analysis(self):
        """STEP 5: LLM analyzes trades and suggests adaptations."""
        logger.info("\n" + "="*60)
        logger.info("STEP 5: LLM TRADE ANALYSIS")
        logger.info("="*60)
        
        # Get trades since last analysis
        try:
            recent_trades = self.positions.get_recent_trades(hours=4) if hasattr(self.positions, 'get_recent_trades') else []
        except:
            recent_trades = []
        
        if not recent_trades:
            logger.info("No recent trades to analyze")
            return
        
        # Analyze what worked and what didn't
        analysis = self.llm.analyze_trade_performance(
            trades=recent_trades,
            current_strategies=list(self.active_strategies.keys()),
            market_data=self._get_current_market_snapshot()
        )
        
        logger.info(f"\n🤖 LLM ANALYSIS:")
        logger.info(f"   Recommendations: {analysis.get('recommendations', 'None')}")
        
        # Apply adaptations if suggested
        if analysis.get('should_adapt', False):
            logger.info(f"   Adapting strategies: {analysis.get('adaptations', {})}")
            self._apply_adaptations(analysis['adaptations'])
    
    def end_of_day_learning(self):
        """STEP 6: Bot learns from the day."""
        logger.info("\n" + "="*60)
        logger.info("STEP 6: END OF DAY LEARNING")
        logger.info("="*60)

        # Measure realized signal decay for resolved paper evidence (daily job)
        evidence = self._paper_evidence()
        if evidence is not None:
            try:
                decay = evidence.measure_realized_decay(fetch_latest_market_data)
                if decay['candidates']:
                    logger.info(f"Signal-decay measurement: {decay}")
            except Exception as _de:
                logger.warning(f"Decay measurement failed soft: {_de}")

        # Get all today's trades
        try:
            today_trades = self.positions.get_recent_trades(days=1) if hasattr(self.positions, 'get_recent_trades') else []
        except:
            today_trades = []
        
        # LLM performs deep analysis
        learning = self.llm.end_of_day_analysis(
            trades=today_trades,
            daily_decision=self.daily_decision,
            strategies_used=list(self.active_strategies.keys())
        )
        
        logger.info(f"\n📚 LEARNING SUMMARY:")
        logger.info(f"   What worked: {learning.get('successes', [])}")
        logger.info(f"   What failed: {learning.get('failures', [])}")
        logger.info(f"   Insights: {learning.get('insights', '')}")
        logger.info(f"   Tomorrow's plan: {learning.get('tomorrow_plan', '')}")
        
        # Save learnings
        self._save_daily_learnings(learning)

        # ── Autonomous system: daily review ───────────────────────────────────
        try:
            self.daily_reviewer.llm = self.llm
            self.daily_reviewer.run()
        except Exception as _e:
            logger.warning(f"DailyReviewer error: {_e}")

        # ── Drift detection ───────────────────────────────────────────────────
        try:
            drift_report = self.drift_detector.run()
            if drift_report.retrain_required:
                logger.warning("DriftDetector: retrain triggered")
                self.model_trainer.retrain(trigger='drift')
            elif drift_report.drift_detected:
                logger.info(f"DriftDetector: mild drift — alloc factor {drift_report.allocation_factor:.0%}")
        except Exception as _e:
            logger.warning(f"DriftDetector error: {_e}")

        # ── Scheduled retraining (daily trigger) ─────────────────────────────
        try:
            self.model_trainer.retrain(trigger='scheduled')
        except Exception as _e:
            logger.warning(f"ModelTrainer error: {_e}")

        # ── Research hypotheses ───────────────────────────────────────────────
        try:
            health_summary = self.strategy_health.summary()
            self.research_engine.analyze(
                trade_memory=self.trade_memory,
                strategy_health=health_summary,
                execution_quality=self.execution_quality.get_alerts(),
                drift_detector=self.drift_detector,
                meta_model=self.meta_model,
                llm=None,   # don't consume extra tokens in daily run
            )
        except Exception as _e:
            logger.warning(f"ResearchEngine error: {_e}")
    
    def _analyze_market_quality(self, symbol: str, df) -> Dict:
        """Analyze if market conditions are good for trading."""
        # Volume, volatility, trend strength
        close = df['close'].values
        volume = df['volume'].values
        
        vol_ratio = volume[-5:].mean() / volume[-20:].mean() if volume[-20:].mean() > 0 else 1.0
        returns = (close[1:] / close[:-1]) - 1
        volatility = returns[-20:].std()
        
        # Hourly crypto return volatility is normally well below 1%. The old
        # 1%-4% band made ordinary liquid markets score 0.3-0.5 forever.
        score = 0.5
        if vol_ratio > 1.2:
            score += 0.2
        elif vol_ratio >= 0.8:
            score += 0.1
        elif vol_ratio < 0.3:
            score -= 0.2
        if 0.001 <= volatility < 0.04:
            score += 0.15

        return {'score': min(1.0, max(0.0, score)),
                'vol_ratio': vol_ratio, 'volatility': volatility}
    
    def _extract_market_conditions(self, market_data: Dict) -> Dict:
        """Extract key market metrics for LLM."""
        conditions = {}
        for symbol, df in market_data.items():
            close = df['close'].values
            conditions[symbol] = {
                'price': float(close[-1]),
                'change_24h': (close[-1] / close[-24] - 1) if len(close) >= 24 else 0,
                'volatility': ((close[1:] / close[:-1]) - 1).std(),
            }
        return conditions
    
    def _analyze_performance(self, trades: List) -> Dict:
        """Analyze recent trading performance."""
        if not trades or len(trades) == 0:
            return {'win_rate': 0.0, 'avg_pnl': 0.0, 'total_trades': 0}
        
        wins = sum(1 for t in trades if isinstance(t, dict) and t.get('pnl', 0) > 0)
        total = len(trades)
        
        return {
            'win_rate': wins / total if total > 0 else 0.0,
            'avg_pnl': sum(t.get('pnl', 0) for t in trades if isinstance(t, dict)) / total if total > 0 else 0.0,
            'total_trades': total
        }
    
    def _get_failure_patterns(self) -> List:
        """Detect recurring patterns in losing trades from recent history.

        Analyses losing trades over the last 7 days across:
            • Strategy loss rates
            • Regime clustering
            • Time-of-day clustering (UTC hour buckets)
            • Technical conditions at entry: ADX, RSI, Volume Ratio, ATR,
              VWAP Distance, Spread, Volatility, Kronos Confidence, Claude Confidence
            • Trade lifecycle: MFE, MAE, Holding Time, Close Reason

        Returns a list of pattern strings (capped at 15) for the LLM prompt.
        Uses simple statistical clustering (> 2 SD from mean loss rate) to
        identify significant groupings rather than naive counts.
        """
        import math
        from collections import Counter, defaultdict

        try:
            recent = self.positions.get_recent_trades(days=7)
        except Exception:
            return []

        if not recent:
            return []

        losers  = [t for t in recent if isinstance(t, dict) and (t.get('net_pnl') or t.get('pnl', 0)) < 0]
        winners = [t for t in recent if isinstance(t, dict) and (t.get('net_pnl') or t.get('pnl', 0)) >= 0]
        total   = len(recent)

        if not losers:
            return [f"No losing trades in last 7 days ({total} total trades)"]

        overall_loss_rate = len(losers) / total
        patterns: List[str] = []

        # ── Helper: is this group's loss rate significantly elevated? ─────────
        def is_significant(group_losers: int, group_total: int) -> bool:
            if group_total < 3:
                return False
            group_rate = group_losers / group_total
            # Binomial standard error of overall rate
            se = math.sqrt(overall_loss_rate * (1 - overall_loss_rate) / group_total)
            return se > 0 and (group_rate - overall_loss_rate) / se > 1.5

        # ── 1. Strategy-level failure rate ────────────────────────────────────
        strat_data: Dict[str, Dict] = defaultdict(lambda: {"wins": 0, "losses": 0})
        for t in recent:
            s = t.get('strategy', 'unknown')
            if (t.get('net_pnl') or t.get('pnl', 0)) < 0:
                strat_data[s]["losses"] += 1
            else:
                strat_data[s]["wins"] += 1

        for strat, data in strat_data.items():
            g_total = data["wins"] + data["losses"]
            if is_significant(data["losses"], g_total):
                loss_rate = data["losses"] / g_total
                patterns.append(
                    f"Strategy '{strat}': {loss_rate:.0%} loss rate "
                    f"({data['losses']}/{g_total} trades, last 7d) — "
                    f"significantly above overall {overall_loss_rate:.0%}"
                )

        # ── 2. Regime clustering ──────────────────────────────────────────────
        regime_data: Dict[str, Dict] = defaultdict(lambda: {"wins": 0, "losses": 0})
        for t in recent:
            r = t.get('regime', 'unknown')
            if (t.get('net_pnl') or t.get('pnl', 0)) < 0:
                regime_data[r]["losses"] += 1
            else:
                regime_data[r]["wins"] += 1

        for regime, data in regime_data.items():
            g_total = data["wins"] + data["losses"]
            if is_significant(data["losses"], g_total):
                loss_rate = data["losses"] / g_total
                patterns.append(
                    f"Regime '{regime}': {loss_rate:.0%} loss rate "
                    f"({data['losses']}/{g_total} trades) — statistically elevated"
                )

        # ── 3. Time-of-day clustering (UTC hour) ──────────────────────────────
        hour_data: Dict[int, Dict] = defaultdict(lambda: {"wins": 0, "losses": 0})
        for t in recent:
            ts = t.get('exit_time') or t.get('entry_time', '')
            try:
                hour = datetime.fromisoformat(str(ts).replace('Z', '+00:00')).hour
                key = "losses" if (t.get('net_pnl') or t.get('pnl', 0)) < 0 else "wins"
                hour_data[hour][key] += 1
            except Exception:
                pass

        for hour, data in hour_data.items():
            g_total = data["wins"] + data["losses"]
            if is_significant(data["losses"], g_total):
                loss_rate = data["losses"] / g_total
                patterns.append(
                    f"UTC hour {hour:02d}:00 — {loss_rate:.0%} loss rate "
                    f"({data['losses']}/{g_total} trades)"
                )

        # ── 4. Technical conditions at entry ──────────────────────────────────
        # Extract stored technical metadata from closed trade records
        loser_kronos  = [t.get('kronos_confidence', 0) or 0 for t in losers]
        loser_llm_raw = [t.get('confidence', 0) or 0 for t in losers]
        winner_kronos = [t.get('kronos_confidence', 0) or 0 for t in winners]
        winner_llm    = [t.get('confidence', 0) or 0 for t in winners]

        def mean(lst): return sum(lst) / len(lst) if lst else 0

        if loser_kronos and winner_kronos:
            avg_loser_k  = mean(loser_kronos)
            avg_winner_k = mean(winner_kronos)
            if avg_loser_k < avg_winner_k * 0.8:
                patterns.append(
                    f"Losers had lower Kronos confidence ({avg_loser_k:.2f}) "
                    f"vs winners ({avg_winner_k:.2f}) — consider higher threshold"
                )

        if loser_llm_raw and winner_llm:
            avg_loser_llm  = mean(loser_llm_raw)
            avg_winner_llm = mean(winner_llm)
            if avg_loser_llm < avg_winner_llm * 0.8:
                patterns.append(
                    f"Losers had lower Claude confidence ({avg_loser_llm:.2f}) "
                    f"vs winners ({avg_winner_llm:.2f})"
                )

        # ── 5. Holding time analysis ──────────────────────────────────────────
        loser_durations = []
        for t in losers:
            try:
                entry = datetime.fromisoformat(str(t.get('entry_time', '')).replace('Z', '+00:00'))
                exit_ = datetime.fromisoformat(str(t.get('exit_time',  '')).replace('Z', '+00:00'))
                loser_durations.append((exit_ - entry).total_seconds() / 60)  # minutes
            except Exception:
                pass

        if loser_durations:
            avg_dur = mean(loser_durations)
            if avg_dur < 5:
                patterns.append(
                    f"Losers held an average of only {avg_dur:.1f} min — "
                    f"potential stop-hunt / spread issue"
                )
            elif avg_dur > 240:
                patterns.append(
                    f"Losers held an average of {avg_dur/60:.1f}h — "
                    f"consider tighter time-limit exits"
                )

        # ── 6. Close reason analysis ──────────────────────────────────────────
        reason_counts = Counter(t.get('close_reason', 'unknown') for t in losers)
        for reason, count in reason_counts.most_common(3):
            if count >= 3:
                patterns.append(
                    f"{count} losses closed via '{reason}' "
                    f"({count/len(losers):.0%} of all losers)"
                )

        # ── 7. MAE / MFE (if stored) ─────────────────────────────────────────
        # Future: store max_adverse_excursion and max_favorable_excursion in DB
        # and include here once PositionManager tracks them during position updates.

        return patterns[:15]  # cap to avoid prompt bloat
    
    def _get_current_market_snapshot(self) -> Dict:
        """Get current market snapshot for LLM analysis."""
        snapshot = {}
        for symbol in self.symbols[:3]:  # BTC, ETH, SOL
            df = fetch_latest_market_data(symbol, period='1d', interval='5m')
            if df is not None:
                snapshot[symbol] = df
        return snapshot
    
    def _apply_adaptations(self, adaptations: Dict):
        """Route LLM-suggested adaptations through the ExperimentEngine.

        Claude may PROPOSE parameter changes.
        Claude may NEVER directly modify production parameters.

        All proposals are stored as PROPOSED experiments that must pass:
            backtest → out-of-sample validation → paper trading → operator promotion

        Immutable safety params (risk limits, leverage, circuit breakers) are
        blocked by ExperimentEngine.propose() and will raise ValueError.
        """
        if not adaptations or not isinstance(adaptations, dict):
            return

        # Extract experiment description from the adaptations dict
        description = adaptations.pop("description", "LLM-proposed adaptation")
        proposed_by = "claude"

        # Filter to only parameter-change keys (not meta-keys)
        param_changes = {
            k: v for k, v in adaptations.items()
            if k not in ("should_adapt", "recommendations", "analysis")
        }

        if not param_changes:
            logger.info("LLM adaptation: no parameter changes proposed")
            return

        try:
            exp = self.experiments.propose(
                description=description,
                params=param_changes,
                proposed_by=proposed_by,
            )
            logger.info(
                f"LLM proposed experiment {exp.id[:8]}: {description} "
                f"— params: {list(param_changes.keys())} "
                f"(must pass backtest+OOS before any production effect)"
            )
        except (ValueError, PermissionError) as e:
            logger.warning(f"LLM adaptation BLOCKED by ExperimentEngine: {e}")
    
    def _save_daily_learnings(self, learning: Dict):
        """Save daily learnings to file."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        filename = f'logs/learnings_{date_str}.json'
        
        import json
        with open(filename, 'w') as f:
            json.dump(learning, f, indent=2)
        
        logger.info(f"✅ Learnings saved to {filename}")
    
    def run_trading_cycle(self):
        """Run one complete trading cycle with position management."""
        logger.info("\n" + "="*60)
        logger.info(f"TRADING CYCLE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)

        # Periodic core-accounting health check (throttled, spec §33):
        # runtime degradation must not wait for the next restart to be noticed
        now_ts = time.time()
        if now_ts - getattr(self, '_last_core_health_check', 0.0) > 1800:
            self._last_core_health_check = now_ts
            try:
                from core.trade_history import DatabaseSchemaHealthCheck
                report = DatabaseSchemaHealthCheck(
                    'data/trade_memory.sqlite').run()
                if not report['ok']:
                    self._db_health = 'CRITICAL'
                    logger.critical(f"CRITICAL_ACCOUNTING_FAILURE: periodic "
                                    f"health check failed: {report['critical']}")
                elif getattr(self, '_db_health', 'HEALTHY') == 'CRITICAL':
                    self._db_health = 'HEALTHY'   # recovered
                    logger.warning("Core database health recovered → HEALTHY")
            except Exception as e:
                self._db_health = 'CRITICAL'
                logger.critical(f"CRITICAL_ACCOUNTING_FAILURE: health check "
                                f"unrunnable: {e}")

        # STEP 1: Manage existing positions FIRST (critical!)
        self.manage_open_positions()
        
        # STEP 2: Check portfolio risk before opening new positions
        can_trade, risk_reason = self.check_portfolio_risk()
        
        if not can_trade:
            logger.info(f"\n⚠️ Not opening new positions: {risk_reason}")
            return
        
        # STEP 3: Alpha pipeline — validated edges matched to current markets
        trades_made = 0

        # Dynamic universe — scan and rank all assets, use top candidates
        try:
            candidate_symbols = self.scanner.get_symbols(top_n=10)
            logger.info(f"Universe scan: {len(candidate_symbols)} candidates — {candidate_symbols[:5]}...")
        except Exception as e:
            logger.warning(f"Universe scan failed, using fallback list: {e}")
            candidate_symbols = self.symbols  # fallback to hardcoded list

        alpha_decisions = []
        if self.alpha_pipeline_mode != 'off':
            try:
                alpha_decisions = self.run_alpha_pipeline(candidate_symbols)
            except Exception as e:
                logger.error(f"Alpha pipeline error (falling back to legacy): {e}",
                             exc_info=True)

        accepted = [d for d in alpha_decisions if d.accepted]
        if self.alpha_pipeline_mode == 'primary' and accepted:
            for d in accepted:
                decision = self._alpha_decision_to_trade(d)
                # Prediction recorded BEFORE the outcome exists (spec §3, §15).
                # FAIL CLOSED: exposure is never opened without its prediction
                # record — unrecorded trades corrupt every calibration.
                evidence = self._paper_evidence()
                if evidence is None:
                    logger.critical(
                        "CRITICAL_ACCOUNTING_FAILURE: evidence tracker down — "
                        "skipping new position (fail closed)")
                    continue
                try:
                    c = d.candidate
                    family = c.alpha_id.split('_')[0].upper()
                    evidence.record_prediction(
                        c, family=family,
                        regime=getattr(self.regime, 'current_regime',
                                       'unknown') or 'unknown',
                        bar_interval_seconds=self._bar_interval_seconds(c))
                except Exception as _pe:
                    logger.critical(f"CRITICAL_ACCOUNTING_FAILURE: prediction "
                                    f"record failed ({_pe}) — trade skipped")
                    continue
                pos_id = self.execute_trade(d.candidate.symbol, decision)
                if evidence is not None:
                    try:
                        if pos_id is not None:
                            filled = self._position_filled_notional(pos_id)
                            evidence.attach_position(
                                d.candidate.candidate_id, pos_id,
                                filled_notional_usd=filled)
                        else:
                            evidence.mark_not_executed(
                                d.candidate.candidate_id, "EXECUTION_REJECTED")
                    except Exception as _pe:
                        logger.warning(f"Paper evidence link failed: {_pe}")
                if pos_id is not None:
                    trades_made += 1
                    self.trade_attribution.record(
                        d.candidate.alpha_id,
                        alpha_version=d.candidate.alpha_version,
                        position_id=pos_id,
                        candidate_id=d.candidate.candidate_id,
                        signal_id=d.candidate.candidate_id,
                        meta_model_version=self.ev_model.version,
                        feature_version=self.feature_registry.version,
                        execution_model_version='cost_v2',
                    )
                can_trade, _ = self.check_portfolio_risk()
                if not can_trade:
                    break
        elif self.alpha_pipeline_mode == 'primary':
            logger.info("Alpha pipeline: NO TRADE — no opportunity beats cash")

        # Legacy strategy-vote path (phased retirement, spec §65-66):
        #   shadow    — legacy decides, alpha pipeline records
        #   primary   — legacy is fallback ONLY until the pipeline has earned
        #               enough shadow/forward evidence; once mature,
        #               NO TRADE means NO TRADE
        #   exclusive — legacy never runs
        if self.alpha_pipeline_mode == 'exclusive':
            run_legacy = False
        elif self.alpha_pipeline_mode == 'primary':
            run_legacy = trades_made == 0 and self._legacy_fallback_allowed()
        else:
            run_legacy = True
        # Collect ALL candidates first — rank across strategies/symbols instead
        # of executing greedily per symbol (portfolio-level selection).
        open_positions = self.positions.get_open_positions()
        held = {p['symbol'] for p in open_positions}
        candidates = []
        if run_legacy:
            for symbol in candidate_symbols:
                if symbol in held:
                    logger.debug(f"[{symbol}] Already have position - skipping")
                    continue
                decision = self.analyze_trade_opportunity(symbol)
                if decision:
                    candidates.append(decision)

        if candidates:
            drawdown_pct = 0.0
            try:
                m = self.safety.get_safety_metrics()
                peak = max(self.safety.peak_equity, 1.0)
                drawdown_pct = max(0.0, (peak - self.safety.current_equity) / peak)
            except Exception:
                pass
            regime = 'unknown'
            try:
                if self.regime:
                    regime = getattr(self.regime, 'current_regime', 'unknown') or 'unknown'
            except Exception:
                pass
            ranked = self.ranker.top_trades(
                candidates, regime=regime, drawdown_pct=drawdown_pct)
            ranked_symbols = {r.symbol for r in ranked}
            decisions_by_symbol = {c['symbol']: c for c in candidates}

            for r in ranked:
                decision = decisions_by_symbol.get(r.symbol)
                if decision is None:
                    continue
                # Strategy-family exposure limit: don't stack the same factor
                if self._family_exposure_exceeded(decision):
                    logger.info(
                        f"[{r.symbol}] Skipped: strategy-family exposure limit reached"
                    )
                    continue
                pos_id = self.execute_trade(r.symbol, decision)
                if pos_id is not None:
                    trades_made += 1

                # Re-check portfolio risk after each trade
                can_trade, _ = self.check_portfolio_risk()
                if not can_trade:
                    logger.info(f"⚠️ Portfolio risk limit reached - stopping new trades")
                    break

            for c in candidates:
                if c['symbol'] not in ranked_symbols:
                    logger.info(
                        f"[{c['symbol']}] NO TRADE: below opportunity-ranker threshold "
                        f"(competing with cash)"
                    )
        
        # Summary
        open_count = len(self.positions.get_open_positions())
        metrics = self.safety.get_safety_metrics()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"CYCLE SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"   New trades: {trades_made}")
        logger.info(f"   Open positions: {open_count}/{self.safety.max_positions}")
        logger.info(f"   Daily P&L: ${metrics['daily_pnl']:+.2f}")
        logger.info(f"   Consecutive losses: {metrics['consecutive_losses']}")
        if metrics.get('circuit_breaker'):
            logger.critical(f"   🛑 CIRCUIT BREAKER ACTIVE!")
        logger.info(f"{'='*60}")
        
        # LLM analysis every 4 hours
        if datetime.now().hour % 4 == 0:
            self.llm_trade_analysis()
    
    def _initialize_core_accounting(
            self, db_path: str = 'data/trade_memory.sqlite') -> None:
        """Startup initialization of core accounting/risk infrastructure.

        FAIL CLOSED: any core database/schema/migration/reconciliation error
        sets health CRITICAL and raises — the bot never starts trading on
        records it cannot trust. Optional intelligence sources are NOT
        initialized here and remain fail-soft elsewhere.
        """
        import sqlite3 as _sq

        from core.trade_history import (
            CoreAccountingInitializationError,
            DatabaseSchemaHealthCheck,
            TradeMemorySchemaError,
            data_integrity_scan,
            migrate_legacy_position_size,
        )
        try:
            # 1. Required migrations
            migration = migrate_legacy_position_size(db_path)
            if migration.get('migrated') or migration.get('flagged'):
                logger.warning(f"Legacy schema migration: {migration}")

            # 2. Required schema health
            schema_report = DatabaseSchemaHealthCheck(db_path).run()
            if not schema_report['ok']:
                raise CoreAccountingInitializationError(
                    f"DATABASE SCHEMA UNHEALTHY: {schema_report['critical']}")

            # 3. Data integrity (flags only — corruption is surfaced loudly)
            integrity = data_integrity_scan(db_path)
            if integrity['flagged']:
                logger.critical(
                    f"DATA INTEGRITY: {integrity['flagged']} flagged trade rows "
                    "— review before trusting risk estimates")

            # 4. Paper evidence store + reconciliation (required in
            #    PAPER_EVIDENCE_MODE — constructor/reconcile failures are core)
            from config.config import PAPER_EVIDENCE_MODE
            from core.paper_evidence import PaperEvidenceTracker
            try:
                self.paper_evidence_tracker = PaperEvidenceTracker(db_path)
                recon = self.paper_evidence_tracker.reconcile()
                if recon['resolved'] or recon['flags']:
                    logger.warning(f"Paper evidence reconciliation: {recon}")
            except (TradeMemorySchemaError, _sq.DatabaseError):
                raise
            except Exception as e:
                if PAPER_EVIDENCE_MODE:
                    raise CoreAccountingInitializationError(
                        f"paper evidence initialization failed: {e}") from e
                logger.warning(f"Paper evidence unavailable (optional): {e}")
                self.paper_evidence_tracker = None
        except (TradeMemorySchemaError, _sq.DatabaseError,
                CoreAccountingInitializationError) as exc:
            self._db_health = 'CRITICAL'
            logger.critical(
                "CRITICAL_ACCOUNTING_FAILURE: core accounting initialization "
                f"failed | component=startup db={db_path} "
                f"error={type(exc).__name__}: {exc} | health=CRITICAL — "
                "new exposure disabled")
            raise RuntimeError(
                "Core accounting initialization failed; new exposure disabled"
            ) from exc

        self._db_health = 'HEALTHY'

    def _startup_reconciliation(self) -> None:
        """Compare broker state vs local DB on startup.

        If discrepancies exist, trading is HALTED and a full report is logged.
        The operator must resolve discrepancies manually before the bot restarts.
        This prevents the bot from opening duplicate positions or ignoring
        existing broker positions that aren't tracked locally.
        """
        logger.info("\n" + "="*60)
        logger.info("STARTUP: BROKER RECONCILIATION")
        logger.info("="*60)

        try:
            broker_account = self.broker.get_account()
            logger.info(
                f"Broker account: equity=${broker_account.equity:,.2f} "
                f"cash=${broker_account.cash:,.2f}"
            )
        except Exception as e:
            logger.error(f"Cannot reach broker: {e}")
            raise RuntimeError(f"Startup reconciliation failed — broker unreachable: {e}")

        local_positions = self.positions.get_open_positions()
        result = self.broker.reconcile(local_positions)

        logger.info(f"Local DB open positions: {len(local_positions)}")
        logger.info(f"Broker open positions:   {len(result['broker_positions'])}")

        if result['ok']:
            logger.info("✅ Reconciliation PASSED — state is consistent")
        else:
            logger.critical("🛑 RECONCILIATION FAILED — discrepancies found:")
            for d in result['discrepancies']:
                logger.critical(f"   • {d}")
            logger.critical("Required actions:")
            for a in result.get('action_required', []):
                logger.critical(f"   → {a}")
            logger.critical(
                "Bot will NOT trade until reconciliation is resolved. "
                "Fix discrepancies and restart."
            )
            raise RuntimeError(
                f"Startup reconciliation failed with {len(result['discrepancies'])} discrepancies. "
                "See logs for details."
            )

        # Sync safety manager position count from broker truth
        self.safety.active_positions = len(result['broker_positions'])
        logger.info(f"Safety manager synced: {self.safety.active_positions} active positions")

    def run(self):
        """Main bot loop."""
        self.health.setup_signal_handlers()

        logger.info("\n🚀 STARTING LLM TRADING BOT")

        # Startup-time capability validation (mode-specific, never at import)
        from config.capabilities import capability_matrix, check_startup_requirements
        mode = os.environ.get("TRADING_MODE", "PAPER").upper()
        errors = check_startup_requirements(mode)
        if errors:
            for e in errors:
                logger.critical(f"Startup requirement failed: {e}")
            raise RuntimeError(f"Cannot start in {mode} mode: {errors}")
        for name, cap in capability_matrix().items():
            logger.info(f"Capability {name}: {cap.status}"
                        + (f" ({cap.reason})" if cap.reason else ""))

        # ── STARTUP: core accounting must initialize or the bot fails closed ──
        self._initialize_core_accounting()

        # ── STARTUP: Broker reconciliation (must pass before trading) ─────────
        self._startup_reconciliation()

        # Daily regime analysis at start
        self.daily_regime_analysis()
        
        last_daily_analysis = datetime.now().date()
        last_weekly_review  = datetime.now().isocalendar()[1]  # ISO week number
        
        try:
            while True:
                # Check if new day - run regime analysis
                if datetime.now().date() > last_daily_analysis:
                    self.end_of_day_learning()
                    self.daily_regime_analysis()
                    last_daily_analysis = datetime.now().date()

                # Check if new week — run weekly review
                current_week = datetime.now().isocalendar()[1]
                if current_week != last_weekly_review:
                    try:
                        self.weekly_reviewer.llm = self.llm
                        self.weekly_reviewer.run()
                    except Exception as _we:
                        logger.warning(f"WeeklyReviewer error: {_we}")
                    try:
                        self.run_weekly_research_campaign()
                    except Exception as _re:
                        logger.warning(f"Weekly research campaign error: {_re}")
                    last_weekly_review = current_week
                
                # Health check
                if not self.health.check_heartbeat():
                    logger.critical("🔴 Heartbeat stalled!")
                    break
                
                # Circuit breaker check
                if self.safety.circuit_breaker_triggered:
                    logger.critical("🔴 Circuit breaker - waiting until midnight")
                    time.sleep(3600)
                    continue
                
                # Run trading cycle
                self.run_trading_cycle()
                
                # Wait 5 minutes
                logger.info("\n💤 Waiting 5 minutes...\n")
                time.sleep(300)
                
        except KeyboardInterrupt:
            logger.info("\n⚠️ Shutdown requested")
            self.end_of_day_learning()
            self.experiment_worker.stop()
            self.health.graceful_shutdown()
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
            self.experiment_worker.stop()
            self.health.graceful_shutdown()

def main():
    """Main entry point."""
    bot = LLMTradingBot(capital=6090)
    bot.run()

if __name__ == '__main__':
    main()
