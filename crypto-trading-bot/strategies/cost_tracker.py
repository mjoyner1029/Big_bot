"""Cost Tracker — monitors transaction costs and API expenses.

Inspired by Claude's autonomous experiment: +1,322% return while covering API costs.
Critical for high-frequency trading where costs can erode profits.

Tracks:
  • Exchange/broker transaction fees
  • Estimated slippage
  • API call costs (Anthropic Claude, OpenAI, etc.)
  • Cost-to-profit ratio
  • Break-even analysis
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from config.config import CONFIG


class CostTracker:
    """Comprehensive cost tracking for trading operations.
    
    Essential for high-frequency strategies where costs accumulate rapidly.
    The successful Claude agent's profit covered all API expenses — this
    tracker helps ensure we maintain that efficiency.
    """
    
    def __init__(self, state_path: Optional[str] = None):
        self.state_path = state_path or CONFIG.get("cost_log_path", "logs/cost_tracker.json")
        
        # Running totals
        self.total_exchange_fees: float = 0.0
        self.total_slippage_cost: float = 0.0
        self.total_api_costs: float = 0.0
        
        # Detailed logs
        self.transaction_costs: List[Dict] = []
        self.api_call_costs: List[Dict] = []
        
        # Statistics
        self.total_trades = 0
        self.total_api_calls = 0
        self.api_call_breakdown: Dict[str, int] = {}
        
        self._load_state()
        logging.info(f"[CostTracker] Initialized. Total costs: ${self.get_total_costs():.4f}")
    
    def _load_state(self) -> None:
        """Load persisted cost data."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    data = json.load(f)
                self.total_exchange_fees = data.get("total_exchange_fees", 0.0)
                self.total_slippage_cost = data.get("total_slippage_cost", 0.0)
                self.total_api_costs = data.get("total_api_costs", 0.0)
                self.total_trades = data.get("total_trades", 0)
                self.total_api_calls = data.get("total_api_calls", 0)
                self.api_call_breakdown = data.get("api_call_breakdown", {})
                # Keep last 1000 entries in memory
                self.transaction_costs = data.get("transaction_costs", [])[-1000:]
                self.api_call_costs = data.get("api_call_costs", [])[-1000:]
                logging.info(f"[CostTracker] Loaded state: {self.total_trades} trades tracked")
            except Exception as e:
                logging.warning(f"[CostTracker] Could not load state: {e}")
    
    def save_state(self) -> None:
        """Persist cost data."""
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump({
                "total_exchange_fees": self.total_exchange_fees,
                "total_slippage_cost": self.total_slippage_cost,
                "total_api_costs": self.total_api_costs,
                "total_trades": self.total_trades,
                "total_api_calls": self.total_api_calls,
                "api_call_breakdown": self.api_call_breakdown,
                "transaction_costs": self.transaction_costs[-1000:],  # Keep last 1000
                "api_call_costs": self.api_call_costs[-1000:],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)
    
    def record_trade_cost(
        self,
        symbol: str,
        trade_value: float,
        fee_pct: Optional[float] = None,
        slippage_pct: Optional[float] = None,
    ) -> Dict[str, float]:
        """Record costs for a single trade.
        
        Args:
            symbol: Trading symbol
            trade_value: Dollar value of the trade
            fee_pct: Exchange fee percentage (uses config default if None)
            slippage_pct: Slippage percentage (uses config default if None)
        
        Returns:
            Dict with breakdown: {"fee": X, "slippage": Y, "total": Z}
        """
        if not CONFIG.get("track_transaction_costs", True):
            return {"fee": 0, "slippage": 0, "total": 0}
        
        fee_pct = fee_pct if fee_pct is not None else CONFIG.get("exchange_fee_pct", 0.001)
        slippage_pct = slippage_pct if slippage_pct is not None else CONFIG.get("slippage_estimate_pct", 0.0005)
        
        fee = trade_value * fee_pct
        slippage = trade_value * slippage_pct
        total = fee + slippage
        
        self.total_exchange_fees += fee
        self.total_slippage_cost += slippage
        self.total_trades += 1
        
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "trade_value": round(trade_value, 2),
            "fee": round(fee, 4),
            "slippage": round(slippage, 4),
            "total_cost": round(total, 4),
        }
        self.transaction_costs.append(entry)
        
        # Auto-save every 10 trades to prevent data loss
        if self.total_trades % 10 == 0:
            self.save_state()
        
        return {"fee": fee, "slippage": slippage, "total": total}
    
    def record_api_call(
        self,
        service: str,
        call_type: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
    ) -> float:
        """Record cost of an API call (Claude, OpenAI, etc.).
        
        Args:
            service: "anthropic", "openai", etc.
            call_type: "chat", "completion", "embedding", etc.
            tokens_input: Input tokens used
            tokens_output: Output tokens generated
        
        Returns:
            Estimated cost in USD
        """
        if not CONFIG.get("track_api_costs", True):
            return 0.0
        
        # Get pricing from config
        pricing = CONFIG.get("api_cost_per_call", {})
        
        cost = 0.0
        if service == "anthropic":
            input_cost_per_1k = pricing.get("anthropic_input_1k", 0.003)
            output_cost_per_1k = pricing.get("anthropic_output_1k", 0.015)
            cost = (tokens_input / 1000 * input_cost_per_1k) + (tokens_output / 1000 * output_cost_per_1k)
        elif service == "openai":
            # Simplified — assume GPT-4
            cost_per_1k = pricing.get("openai_gpt4_1k", 0.03)
            cost = ((tokens_input + tokens_output) / 1000) * cost_per_1k
        
        self.total_api_costs += cost
        self.total_api_calls += 1
        
        key = f"{service}_{call_type}"
        self.api_call_breakdown[key] = self.api_call_breakdown.get(key, 0) + 1
        
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": service,
            "call_type": call_type,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "cost": round(cost, 6),
        }
        self.api_call_costs.append(entry)
        
        # Auto-save every 50 API calls
        if self.total_api_calls % 50 == 0:
            self.save_state()
        
        return cost
    
    def get_total_costs(self) -> float:
        """Get sum of all costs."""
        return self.total_exchange_fees + self.total_slippage_cost + self.total_api_costs
    
    def get_avg_cost_per_trade(self) -> float:
        """Average transaction cost per trade."""
        if self.total_trades == 0:
            return 0.0
        return (self.total_exchange_fees + self.total_slippage_cost) / self.total_trades
    
    def get_cost_breakdown(self) -> Dict[str, any]:
        """Get detailed cost breakdown."""
        total_costs = self.get_total_costs()
        
        return {
            "total_costs": round(total_costs, 4),
            "exchange_fees": round(self.total_exchange_fees, 4),
            "slippage_costs": round(self.total_slippage_cost, 4),
            "api_costs": round(self.total_api_costs, 4),
            "total_trades": self.total_trades,
            "total_api_calls": self.total_api_calls,
            "avg_cost_per_trade": round(self.get_avg_cost_per_trade(), 4),
            "api_call_breakdown": self.api_call_breakdown,
            "cost_composition_pct": {
                "fees": round((self.total_exchange_fees / total_costs * 100), 1) if total_costs > 0 else 0,
                "slippage": round((self.total_slippage_cost / total_costs * 100), 1) if total_costs > 0 else 0,
                "api": round((self.total_api_costs / total_costs * 100), 1) if total_costs > 0 else 0,
            },
        }
    
    def check_cost_efficiency(self, total_pnl: float) -> Dict[str, any]:
        """Analyze cost efficiency relative to P&L.
        
        Claude's experiment: +1,322% return while covering API costs.
        This checks if we're maintaining similar efficiency.
        
        Args:
            total_pnl: Total profit/loss from trading
        
        Returns:
            Efficiency metrics and warnings
        """
        total_costs = self.get_total_costs()
        
        if total_pnl == 0:
            return {
                "cost_to_pnl_ratio": float('inf') if total_costs > 0 else 0,
                "net_pnl": -total_costs,
                "costs_covered": False,
                "efficiency_status": "No P&L yet",
            }
        
        cost_to_pnl = (total_costs / abs(total_pnl)) * 100 if total_pnl != 0 else 0
        net_pnl = total_pnl - total_costs
        costs_covered = total_pnl > total_costs
        
        # Efficiency benchmarks
        if costs_covered and cost_to_pnl < 5:
            status = "Excellent (Claude-level efficiency)"
        elif costs_covered and cost_to_pnl < 15:
            status = "Good (costs covered with margin)"
        elif costs_covered:
            status = "Acceptable (costs barely covered)"
        else:
            status = "WARNING  Poor (costs exceed profit)"
        
        return {
            "cost_to_pnl_ratio_pct": round(cost_to_pnl, 2),
            "net_pnl": round(net_pnl, 2),
            "costs_covered": costs_covered,
            "efficiency_status": status,
        }
    
    def log_cost_summary(self, total_pnl: Optional[float] = None) -> None:
        """Log comprehensive cost summary."""
        breakdown = self.get_cost_breakdown()
        
        logging.info(
            f"[CostTracker] Total costs: ${breakdown['total_costs']:.4f} "
            f"(Fees: ${breakdown['exchange_fees']:.4f}, "
            f"Slippage: ${breakdown['slippage_costs']:.4f}, "
            f"API: ${breakdown['api_costs']:.4f})"
        )
        logging.info(
            f"[CostTracker] {breakdown['total_trades']} trades, "
            f"{breakdown['total_api_calls']} API calls, "
            f"Avg ${breakdown['avg_cost_per_trade']:.4f}/trade"
        )
        
        if total_pnl is not None:
            efficiency = self.check_cost_efficiency(total_pnl)
            logging.info(
                f"[CostTracker] P&L: ${total_pnl:.2f}, "
                f"Net (after costs): ${efficiency['net_pnl']:.2f}, "
                f"Cost ratio: {efficiency['cost_to_pnl_ratio_pct']:.2f}% — "
                f"{efficiency['efficiency_status']}"
            )


# Singleton instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get or create the global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


def reset_cost_tracker() -> None:
    """Reset the cost tracker (useful for testing)."""
    global _cost_tracker
    _cost_tracker = None
