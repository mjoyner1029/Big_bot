"""Autonomous Trading Agent — fully self-directed decision making.

Inspired by Claude's 48hr autonomous experiment: no human intervention,
continuous learning from outcomes, comprehensive market analysis.

This agent:
  • Makes all trading decisions autonomously
  • Learns from every trade outcome (win/loss patterns)
  • Analyzes news, world events, and market sentiment
  • Adapts strategy based on performance feedback
  • Self-diagnoses issues and adjusts parameters

The agent maintains a "trading journal" to track decision rationale
and outcome, enabling continuous improvement.
"""
import json
import logging
import os
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from config.config import CONFIG, is_crypto, get_mode_config
from models.llm_analyst import _call_claude


class AutonomousAgent:
    """Fully autonomous trading decision maker with self-learning capability.
    
    Similar to how Claude operated for 48 hours without human intervention,
    this agent makes all decisions independently and learns from outcomes.
    """
    
    def __init__(self, journal_path: Optional[str] = None):
        self.journal_path = journal_path or "logs/autonomous_journal.json"
        
        # Decision history for learning
        self.decision_history: deque = deque(maxlen=500)
        self.outcome_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # Performance tracking
        self.win_rate_by_confidence: Dict[str, Tuple[int, int]] = {}  # confidence_bucket: (wins, total)
        self.win_rate_by_symbol: Dict[str, Tuple[int, int]] = {}
        self.win_rate_by_market_condition: Dict[str, Tuple[int, int]] = {}
        
        # Adaptive parameters (learned from experience)
        self.confidence_adjustment: float = 0.0
        self.symbol_preferences: Dict[str, float] = {}  # symbol: confidence_modifier
        self.news_impact_weights: Dict[str, float] = {}
        
        # Session state
        self.decisions_today = 0
        self.autonomous_cycles = 0
        
        self._load_journal()
        logging.info(
            f"[Autonomous] Agent initialized. "
            f"{len(self.decision_history)} historical decisions loaded."
        )
    
    def _load_journal(self) -> None:
        """Load historical decisions and learned patterns."""
        if os.path.exists(self.journal_path):
            try:
                with open(self.journal_path, "r") as f:
                    data = json.load(f)
                self.outcome_patterns = defaultdict(list, data.get("outcome_patterns", {}))
                self.win_rate_by_confidence = data.get("win_rate_by_confidence", {})
                self.win_rate_by_symbol = data.get("win_rate_by_symbol", {})
                self.win_rate_by_market_condition = data.get("win_rate_by_market_condition", {})
                self.confidence_adjustment = data.get("confidence_adjustment", 0.0)
                self.symbol_preferences = data.get("symbol_preferences", {})
                self.news_impact_weights = data.get("news_impact_weights", {})
                
                # Reconstruct recent history
                history = data.get("recent_decisions", [])
                self.decision_history = deque(history[-500:], maxlen=500)
                
                logging.info(
                    f"[Autonomous] Loaded journal: {len(self.decision_history)} decisions, "
                    f"{len(self.symbol_preferences)} symbol preferences learned"
                )
            except Exception as e:
                logging.warning(f"[Autonomous] Could not load journal: {e}")
    
    def save_journal(self) -> None:
        """Persist learned patterns and decision history."""
        os.makedirs(os.path.dirname(self.journal_path) or ".", exist_ok=True)
        with open(self.journal_path, "w") as f:
            json.dump({
                "outcome_patterns": dict(self.outcome_patterns),
                "win_rate_by_confidence": self.win_rate_by_confidence,
                "win_rate_by_symbol": self.win_rate_by_symbol,
                "win_rate_by_market_condition": self.win_rate_by_market_condition,
                "confidence_adjustment": self.confidence_adjustment,
                "symbol_preferences": self.symbol_preferences,
                "news_impact_weights": self.news_impact_weights,
                "recent_decisions": list(self.decision_history),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)
    
    def make_autonomous_decision(
        self,
        signal: Dict[str, Any],
        market_context: Dict[str, Any],
        news_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a fully autonomous trading decision with self-learning.
        
        Args:
            signal: Base trading signal from strategy engine
            market_context: Current market state, indicators, regime
            news_analysis: News sentiment and world events impact
        
        Returns:
            Enhanced decision with:
              - should_execute: bool
              - adjusted_confidence: float
              - reasoning: str (agent's thought process)
              - learned_adjustments: dict (what the agent learned/applied)
        """
        self.autonomous_cycles += 1
        symbol = signal["symbol"]
        base_confidence = signal.get("confidence", 0.5)
        
        # Start with base signal
        decision = {
            "should_execute": False,
            "adjusted_confidence": base_confidence,
            "reasoning": [],
            "learned_adjustments": {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # ── 1. Apply learned patterns ────────────────────────────────
        learned_adj = self._apply_learned_patterns(symbol, market_context)
        decision["adjusted_confidence"] += learned_adj
        if learned_adj != 0:
            decision["reasoning"].append(
                f"Applied learned adjustment for {symbol}: {learned_adj:+.3f}"
            )
            decision["learned_adjustments"]["pattern_learning"] = learned_adj
        
        # ── 2. News and world events impact ──────────────────────────
        if news_analysis:
            news_adj = self._evaluate_news_impact(news_analysis, symbol)
            decision["adjusted_confidence"] += news_adj
            if abs(news_adj) > 0.01:
                decision["reasoning"].append(
                    f"News/events impact: {news_adj:+.3f} "
                    f"({news_analysis.get('summary', 'N/A')})"
                )
                decision["learned_adjustments"]["news_impact"] = news_adj
        
        # ── 3. Market regime adaptation ──────────────────────────────
        regime = market_context.get("regime", "unknown")
        regime_adj = self._adapt_to_regime(regime, signal)
        decision["adjusted_confidence"] += regime_adj
        if regime_adj != 0:
            decision["reasoning"].append(f"Regime adaptation ({regime}): {regime_adj:+.3f}")
            decision["learned_adjustments"]["regime_adaptation"] = regime_adj
        
        # ── 4. Self-diagnosis and caution ────────────────────────────
        caution_adj = self._apply_caution_based_on_performance()
        decision["adjusted_confidence"] += caution_adj
        if caution_adj != 0:
            decision["reasoning"].append(
                f"Performance-based caution: {caution_adj:+.3f}"
            )
            decision["learned_adjustments"]["performance_caution"] = caution_adj
        
        # ── 5. Final decision gate ───────────────────────────────────
        final_confidence = decision["adjusted_confidence"]
        threshold = get_mode_config("confidence_threshold")
        
        decision["should_execute"] = final_confidence >= threshold
        decision["reasoning"] = " | ".join(decision["reasoning"]) if decision["reasoning"] else "Base signal, no adjustments"
        
        # Record this decision
        self._record_decision(signal, decision, market_context)
        
        if decision["should_execute"]:
            logging.info(
                f"[Autonomous] EXECUTE {symbol}: confidence={final_confidence:.3f} "
                f"(base={base_confidence:.3f}, adj={final_confidence-base_confidence:+.3f})"
            )
            logging.info(f"[Autonomous] Reasoning: {decision['reasoning']}")
        else:
            logging.debug(
                f"[Autonomous] SKIP {symbol}: confidence={final_confidence:.3f} "
                f"below threshold {threshold:.3f}"
            )
        
        return decision
    
    def _apply_learned_patterns(
        self,
        symbol: str,
        market_context: Dict[str, Any],
    ) -> float:
        """Apply patterns learned from past trades."""
        adjustment = 0.0
        
        # Symbol-specific learning
        if symbol in self.symbol_preferences:
            symbol_adj = self.symbol_preferences[symbol]
            adjustment += symbol_adj
        
        # Confidence bucket performance
        # (If we historically do poorly at 0.6-0.7 confidence, reduce it)
        # This will be populated as trades close
        
        return adjustment
    
    def _evaluate_news_impact(
        self,
        news_analysis: Dict[str, Any],
        symbol: str,
    ) -> float:
        """Evaluate how news and world events should impact decision."""
        impact = 0.0
        
        sentiment = news_analysis.get("sentiment_score", 0.0)  # -1 to +1
        magnitude = news_analysis.get("magnitude", 0.5)  # 0 to 1 (how significant)
        
        # Base impact from sentiment
        impact = sentiment * magnitude * 0.05  # Cap at ±5% confidence adjustment
        
        # Learn news impact effectiveness over time
        if "news_sentiment" in self.news_impact_weights:
            learned_weight = self.news_impact_weights["news_sentiment"]
            impact *= learned_weight
        
        return impact
    
    def _adapt_to_regime(
        self,
        regime: str,
        signal: Dict[str, Any],
    ) -> float:
        """Adapt confidence based on market regime and learned patterns."""
        # Check historical performance in this regime
        if regime in self.win_rate_by_market_condition:
            wins, total = self.win_rate_by_market_condition[regime]
            if total >= 10:  # Need enough samples
                win_rate = wins / total
                # If we do well in this regime, boost confidence
                if win_rate > 0.6:
                    return 0.02
                elif win_rate < 0.4:
                    return -0.03
        
        return 0.0
    
    def _apply_caution_based_on_performance(self) -> float:
        """Apply caution if recent performance is poor."""
        if len(self.decision_history) < 20:
            return 0.0
        
        # Check last 20 decisions
        recent = list(self.decision_history)[-20:]
        executed = [d for d in recent if d.get("was_executed")]
        
        if len(executed) < 5:
            return 0.0
        
        # Calculate recent win rate
        wins = sum(1 for d in executed if d.get("outcome") == "win")
        win_rate = wins / len(executed)
        
        # If win rate is below 40%, be more cautious
        if win_rate < 0.4:
            return -0.05
        # If win rate is above 60%, be more confident
        elif win_rate > 0.6:
            return 0.02
        
        return 0.0
    
    def _record_decision(
        self,
        signal: Dict[str, Any],
        decision: Dict[str, Any],
        market_context: Dict[str, Any],
    ) -> None:
        """Record this decision for learning."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": signal["symbol"],
            "side": signal["side"],
            "base_confidence": signal.get("confidence", 0.5),
            "adjusted_confidence": decision["adjusted_confidence"],
            "was_executed": decision["should_execute"],
            "reasoning": decision["reasoning"],
            "regime": market_context.get("regime"),
            "adjustments": decision["learned_adjustments"],
            # Outcome will be added later when trade closes
            "outcome": None,
            "pnl": None,
        }
        
        self.decision_history.append(record)
        
        # Auto-save every 10 decisions
        if len(self.decision_history) % 10 == 0:
            self.save_journal()
    
    def learn_from_trade_outcome(
        self,
        trade_id: str,
        symbol: str,
        pnl: float,
        confidence_used: float,
        market_condition: str,
    ) -> None:
        """Learn from a closed trade outcome.
        
        This is the key learning loop: every trade teaches the agent.
        """
        outcome = "win" if pnl > 0 else "loss"
        
        # Update symbol preferences
        if symbol not in self.symbol_preferences:
            self.symbol_preferences[symbol] = 0.0
        
        # Adjust symbol preference based on outcome
        learning_rate = 0.01
        if outcome == "win":
            self.symbol_preferences[symbol] += learning_rate
        else:
            self.symbol_preferences[symbol] -= learning_rate
        
        # Clamp to reasonable bounds
        self.symbol_preferences[symbol] = max(-0.1, min(0.1, self.symbol_preferences[symbol]))
        
        # Update confidence bucket stats
        conf_bucket = f"{int(confidence_used * 10) / 10:.1f}"
        if conf_bucket not in self.win_rate_by_confidence:
            self.win_rate_by_confidence[conf_bucket] = (0, 0)
        
        wins, total = self.win_rate_by_confidence[conf_bucket]
        if outcome == "win":
            wins += 1
        self.win_rate_by_confidence[conf_bucket] = (wins, total + 1)
        
        # Update symbol stats
        if symbol not in self.win_rate_by_symbol:
            self.win_rate_by_symbol[symbol] = (0, 0)
        wins, total = self.win_rate_by_symbol[symbol]
        if outcome == "win":
            wins += 1
        self.win_rate_by_symbol[symbol] = (wins, total + 1)
        
        # Update market condition stats
        if market_condition not in self.win_rate_by_market_condition:
            self.win_rate_by_market_condition[market_condition] = (0, 0)
        wins, total = self.win_rate_by_market_condition[market_condition]
        if outcome == "win":
            wins += 1
        self.win_rate_by_market_condition[market_condition] = (wins, total + 1)
        
        # Update decision history with outcome
        for decision in reversed(self.decision_history):
            if (decision.get("symbol") == symbol and 
                decision.get("was_executed") and 
                decision.get("outcome") is None):
                decision["outcome"] = outcome
                decision["pnl"] = pnl
                break
        
        logging.info(
            f"[Autonomous] Learned from {outcome}: {symbol} "
            f"(P&L: ${pnl:.2f}, confidence: {confidence_used:.3f}) "
            f"→ preference now {self.symbol_preferences[symbol]:+.3f}"
        )
        
        self.save_journal()
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of what the agent has learned."""
        total_decisions = len(self.decision_history)
        executed = [d for d in self.decision_history if d.get("was_executed")]
        outcomes = [d for d in executed if d.get("outcome") is not None]
        
        wins = sum(1 for d in outcomes if d.get("outcome") == "win")
        
        return {
            "total_decisions": total_decisions,
            "total_executed": len(executed),
            "total_with_outcomes": len(outcomes),
            "overall_win_rate": wins / len(outcomes) if outcomes else 0,
            "symbols_learned": len(self.symbol_preferences),
            "top_performing_symbols": sorted(
                self.symbol_preferences.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "worst_performing_symbols": sorted(
                self.symbol_preferences.items(),
                key=lambda x: x[1]
            )[:5],
            "win_rate_by_confidence": {
                k: f"{v[0]}/{v[1]} ({v[0]/v[1]*100:.1f}%)" if v[1] > 0 else "0/0"
                for k, v in sorted(self.win_rate_by_confidence.items())
            },
            "autonomous_cycles": self.autonomous_cycles,
        }
    
    def self_reflect_and_adapt(self) -> Dict[str, Any]:
        """Periodic self-reflection: analyze performance and adapt strategy.
        
        Should be called every N trading cycles (e.g., every hour).
        """
        summary = self.get_learning_summary()
        
        adaptations = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions_taken": [],
        }
        
        # If overall win rate is low, become more conservative
        win_rate = summary["overall_win_rate"]
        if summary["total_with_outcomes"] >= 20:
            if win_rate < 0.45:
                old_adj = self.confidence_adjustment
                self.confidence_adjustment -= 0.01
                self.confidence_adjustment = max(-0.1, self.confidence_adjustment)
                adaptations["actions_taken"].append(
                    f"Win rate {win_rate:.1%} is low → reduced confidence threshold "
                    f"(adj: {old_adj:.3f} → {self.confidence_adjustment:.3f})"
                )
            elif win_rate > 0.60 and self.confidence_adjustment < 0:
                old_adj = self.confidence_adjustment
                self.confidence_adjustment += 0.01
                self.confidence_adjustment = min(0.1, self.confidence_adjustment)
                adaptations["actions_taken"].append(
                    f"Win rate {win_rate:.1%} is good → increased confidence "
                    f"(adj: {old_adj:.3f} → {self.confidence_adjustment:.3f})"
                )
        
        if adaptations["actions_taken"]:
            logging.info(f"[Autonomous] Self-reflection performed:")
            for action in adaptations["actions_taken"]:
                logging.info(f"[Autonomous]   • {action}")
        else:
            logging.info(
                f"[Autonomous] Self-reflection: {summary['total_with_outcomes']} trades, "
                f"win rate {win_rate:.1%} — no adaptations needed"
            )
        
        self.save_journal()
        return adaptations


# Singleton instance
_autonomous_agent: Optional[AutonomousAgent] = None


def get_autonomous_agent() -> AutonomousAgent:
    """Get or create the global autonomous agent instance."""
    global _autonomous_agent
    if _autonomous_agent is None:
        _autonomous_agent = AutonomousAgent()
    return _autonomous_agent
