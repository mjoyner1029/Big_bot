"""Autonomous Mode Extensions for Main Trading Loop.

Integrates:
  • Autonomous agent decision-making
  • World events analysis
  • Continuous learning from outcomes
  • Self-reflection and adaptation
  • Cost tracking and velocity optimization

Import and use these functions in main.py to enable full autonomy.
"""
import logging
from typing import Dict, Any, Optional
import pandas as pd

from models.autonomous_agent import get_autonomous_agent
from models.world_events_analyzer import get_world_events_analyzer
from strategies.velocity_optimizer import get_velocity_optimizer
from strategies.cost_tracker import get_cost_tracker
from config.config import CONFIG


# Global state for autonomous mode
_world_events_cache: Optional[Dict[str, Any]] = None
_last_world_events_cycle: int = 0
_last_reflection_cycle: int = 0


def autonomous_pre_cycle_checks(cycle_count: int) -> Dict[str, Any]:
    """Run autonomous checks before each trading cycle.
    
    Returns:
        Dict with:
          - should_pause: bool
          - world_events: dict (current analysis)
          - adaptive_params: dict (parameter adjustments)
    """
    global _world_events_cache, _last_world_events_cycle
    
    result = {
        "should_pause": False,
        "pause_reason": "",
        "world_events": None,
        "adaptive_params": {},
    }
    
    # ── World Events Analysis ─────────────────────────────────────
    if CONFIG.get("enable_world_events_analysis", True):
        refresh_interval = CONFIG.get("world_events_analysis_interval", 4)
        
        # Refresh world events periodically (only if LLM is enabled)
        if CONFIG.get("use_llm", True) and (cycle_count - _last_world_events_cycle >= refresh_interval or 
            _world_events_cache is None):
            
            logging.info("[Autonomous] Refreshing world events analysis...")
            analyzer = get_world_events_analyzer()
            _world_events_cache = analyzer.analyze_market_environment()
            _last_world_events_cycle = cycle_count
            
            # Check if we should pause trading
            should_pause, reason = analyzer.should_pause_trading(_world_events_cache)
            if should_pause:
                result["should_pause"] = True
                result["pause_reason"] = reason
                logging.warning(f"[Autonomous] TRADING PAUSED: {reason}")
                return result
            
            # Get adaptive parameters
            result["adaptive_params"] = analyzer.get_adaptive_parameters(_world_events_cache)
            
            if result["adaptive_params"]:
                logging.info(
                    f"[Autonomous] World events adaptive params: "
                    f"size_mult={result['adaptive_params'].get('position_size_mult', 1.0):.2f}, "
                    f"conf_mult={result['adaptive_params'].get('confidence_mult', 1.0):.2f}"
                )
        
        result["world_events"] = _world_events_cache
    
    return result


def autonomous_enhance_signal(
    signal: Dict[str, Any],
    market_data: pd.DataFrame,
    world_events: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Enhance a trading signal with autonomous decision-making.
    
    Args:
        signal: Base signal from strategy engine
        market_data: Market data DataFrame for the symbol
        world_events: Current world events analysis
    
    Returns:
        Enhanced decision from autonomous agent
    """
    if not CONFIG.get("enable_autonomous_learning", True):
        # Autonomous mode disabled — return original
        return {
            "should_execute": True,
            "adjusted_confidence": signal.get("confidence", 0.5),
            "reasoning": "Autonomous mode disabled",
            "learned_adjustments": {},
        }
    
    # Get the autonomous agent
    agent = get_autonomous_agent()
    
    # Build market context
    market_context = {
        "symbol": signal["symbol"],
        "regime": signal.get("_regime", "unknown"),
        "indicators": {},
    }
    
    # Add latest indicator values
    if market_data is not None and not market_data.empty:
        latest = market_data.iloc[-1]
        market_context["indicators"] = {
            "rsi": latest.get("rsi"),
            "macd_hist": latest.get("macd_hist"),
            "bb_pctb": latest.get("bb_pctb"),
            "adx": latest.get("adx"),
            "volume": latest.get("Volume"),
        }
    
    # Get symbol-specific news analysis
    news_analysis = None
    if world_events:
        analyzer = get_world_events_analyzer()
        news_analysis = analyzer.get_symbol_specific_analysis(
            signal["symbol"],
            world_events
        )
    
    # Get autonomous decision
    decision = agent.make_autonomous_decision(
        signal,
        market_context,
        news_analysis
    )
    
    return decision


def autonomous_post_cycle_tasks(cycle_count: int, portfolio) -> None:
    """Perform autonomous tasks after trading cycle.
    
    Args:
        cycle_count: Current cycle number
        portfolio: Portfolio manager instance
    """
    global _last_reflection_cycle
    
    # ── Periodic Self-Reflection ─────────────────────────────────
    reflection_interval = CONFIG.get("autonomous_reflection_interval", 12)
    
    if cycle_count - _last_reflection_cycle >= reflection_interval:
        if CONFIG.get("enable_autonomous_learning", True):
            logging.info("[Autonomous] Performing self-reflection and adaptation...")
            agent = get_autonomous_agent()
            adaptations = agent.self_reflect_and_adapt()
            
            # Log learning summary
            summary = agent.get_learning_summary()
            logging.info(
                f"[Autonomous] Learning Summary: "
                f"{summary['total_executed']} trades, "
                f"{summary['overall_win_rate']:.1%} win rate, "
                f"{summary['symbols_learned']} symbols learned"
            )
            
            _last_reflection_cycle = cycle_count
    
    # ── Velocity Status ───────────────────────────────────────────
    if cycle_count % 6 == 0:  # Every ~30 min in balanced mode
        velocity_opt = get_velocity_optimizer()
        velocity_opt.log_status()
    
    # ── Cost Tracking ─────────────────────────────────────────────
    if cycle_count % 12 == 0:  # Every hour
        cost_tracker = get_cost_tracker()
        
        # Get total P&L from portfolio
        try:
            summary = portfolio.summary(current_prices={})
            total_pnl = summary.get("total_realised_pnl", 0)
            cost_tracker.log_cost_summary(total_pnl)
        except Exception as e:
            logging.warning(f"[Autonomous] Cost summary failed: {e}")


def get_autonomous_status() -> Dict[str, Any]:
    """Get comprehensive autonomous system status.
    
    Returns:
        Status dict with metrics from all autonomous components
    """
    status = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "mode": CONFIG.get("trading_mode", "balanced"),
        "autonomous_learning_enabled": CONFIG.get("enable_autonomous_learning", True),
        "world_events_enabled": CONFIG.get("enable_world_events_analysis", True),
    }
    
    # Autonomous agent status
    if CONFIG.get("enable_autonomous_learning", True):
        try:
            agent = get_autonomous_agent()
            status["learning"] = agent.get_learning_summary()
        except Exception as e:
            status["learning"] = {"error": str(e)}
    
    # Velocity optimizer status
    try:
        velocity_opt = get_velocity_optimizer()
        status["velocity"] = velocity_opt.get_velocity_metrics()
    except Exception as e:
        status["velocity"] = {"error": str(e)}
    
    # Cost tracker status
    try:
        cost_tracker = get_cost_tracker()
        status["costs"] = cost_tracker.get_cost_breakdown()
    except Exception as e:
        status["costs"] = {"error": str(e)}
    
    # World events status
    if CONFIG.get("enable_world_events_analysis", True) and _world_events_cache:
        status["world_events"] = {
            "last_updated": _world_events_cache.get("timestamp", "unknown"),
            "overall_sentiment": _world_events_cache.get("overall_sentiment", 0),
            "magnitude": _world_events_cache.get("magnitude", 0),
            "summary": _world_events_cache.get("summary", ""),
        }
    
    return status
