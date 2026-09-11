"""Calculate real transaction costs."""

class TransactionCostModel:
    """Models exchange fees, slippage, funding."""
    
    EXCHANGE_FEES = {
        'alpaca': 0.0,      # Free for crypto
        'binance': 0.001,   # 0.1%
        'coinbase': 0.006,  # 0.6%
    }
    
    def __init__(self, exchange: str = 'alpaca', slippage_pct: float = 0.0005):
        """
        Args:
            exchange: Exchange name for fee lookup
            slippage_pct: Expected slippage (default 0.05%)
        """
        self.exchange = exchange
        self.fee_pct = self.EXCHANGE_FEES.get(exchange, 0.001)
        self.slippage_pct = slippage_pct
    
    def calculate_cost(self, position_size: float, price: float) -> float:
        """Calculate total round-trip cost in dollars.
        
        Args:
            position_size: Position size in dollars
            price: Entry price
            
        Returns:
            Total cost in dollars (fees + slippage for buy and sell)
        """
        # Round trip (buy + sell)
        total_cost_pct = 2 * (self.fee_pct + self.slippage_pct)
        return total_cost_pct * position_size
    
    def get_min_profitable_move(self, entry_price: float) -> float:
        """Get minimum price movement needed to break even.
        
        Args:
            entry_price: Entry price
            
        Returns:
            Minimum profitable move as percentage (e.g., 0.005 = 0.5%)
        """
        # Need to overcome round-trip costs
        return 2 * (self.fee_pct + self.slippage_pct)
    
    def adjust_profit_target(self, target_price: float, entry_price: float, 
                            position_size: float) -> float:
        """Adjust profit target to account for costs.
        
        Args:
            target_price: Gross profit target
            entry_price: Entry price
            position_size: Position size in dollars
            
        Returns:
            Adjusted target price that accounts for costs
        """
        gross_return = (target_price - entry_price) / entry_price
        cost = self.calculate_cost(position_size, entry_price)
        cost_pct = cost / position_size
        
        net_return = gross_return - cost_pct
        adjusted_target = entry_price * (1 + net_return)
        
        return adjusted_target
    
    def calculate_net_pnl(self, entry_price: float, exit_price: float, 
                         position_size: float) -> float:
        """Calculate net P&L after all costs.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price  
            position_size: Position size in dollars
            
        Returns:
            Net P&L in dollars
        """
        # Gross P&L
        gross_pnl = (exit_price - entry_price) / entry_price * position_size
        
        # Costs
        costs = self.calculate_cost(position_size, entry_price)
        
        # Net P&L
        return gross_pnl - costs

    def estimate_cost(
        self,
        asset: str,
        side: str,
        quantity: float,
        price: float,
        timestamp=None,
        market_state: dict = None,
        order_type: str = "market",
    ) -> dict:
        """Estimate one-way execution cost with a full breakdown.

        Models commission, bid/ask spread, slippage and size-dependent market
        impact. Uses real quote/liquidity data from ``market_state`` when
        available; falls back to conservative assumptions when not.

        market_state keys (all optional):
            bid, ask            — live quotes (true spread)
            adv_usd             — average daily volume in USD
            volatility_pct      — recent bar volatility (e.g. ATR%)
            hour_utc            — hour of day (0-23)

        Returns dict: commission, spread_cost, slippage_cost,
        estimated_market_impact, total_cost, cost_bps.
        """
        market_state = market_state or {}
        notional = abs(quantity) * price
        if notional <= 0:
            return {"commission": 0.0, "spread_cost": 0.0, "slippage_cost": 0.0,
                    "estimated_market_impact": 0.0, "total_cost": 0.0, "cost_bps": 0.0}

        commission = self.fee_pct * notional

        # Spread: real bid/ask when available; conservative default otherwise.
        bid, ask = market_state.get("bid"), market_state.get("ask")
        if bid and ask and ask > bid > 0:
            spread_pct = (ask - bid) / ((ask + bid) / 2)
            spread_source = "QUOTE"
        else:
            spread_pct = 0.0008  # conservative 8 bps fallback (no quote data)
            spread_source = "FALLBACK"
        spread_cost = (spread_pct / 2) * notional  # pay half-spread per side

        # Slippage scales with volatility and time of day (thin sessions cost more)
        vol_mult = 1.0 + 10.0 * max(0.0, float(market_state.get("volatility_pct", 0.0)) - 0.01)
        hour = market_state.get("hour_utc")
        tod_mult = 1.3 if hour is not None and (hour < 6 or hour >= 22) else 1.0
        limit_mult = 0.3 if order_type == "limit" else 1.0
        slippage_cost = self.slippage_pct * notional * vol_mult * tod_mult * limit_mult

        # Market impact: square-root model on participation of ADV
        adv = float(market_state.get("adv_usd", 0.0) or 0.0)
        if adv > 0:
            participation = min(notional / adv, 1.0)
            impact_pct = 0.001 * (participation ** 0.5)  # 10 bps at 1% ADV
        else:
            impact_pct = 0.0005  # conservative fallback without liquidity data
        estimated_market_impact = impact_pct * notional

        total = commission + spread_cost + slippage_cost + estimated_market_impact
        return {
            "commission": commission,
            "spread_cost": spread_cost,
            "slippage_cost": slippage_cost,
            "estimated_market_impact": estimated_market_impact,
            "total_cost": total,
            "cost_bps": (total / notional) * 10_000,
            "spread_source": spread_source,   # QUOTE | FALLBACK — auditable
            "estimated_spread_bps": spread_pct * 10_000,
        }

    def estimate_cost_v2(
        self,
        asset: str,
        asset_class: str,
        side: str,
        quantity: float,
        price: float,
        timestamp=None,
        order_type: str = "market",
        market_state: dict = None,
        maker: bool = False,
        funding_rate_8h: float = None,
        holding_hours: float = 0.0,
    ) -> dict:
        """Cost model V2: full basis-point breakdown incl. funding.

        Returns commission_bps, spread_bps, slippage_bps, impact_bps,
        funding_bps, total_cost_bps (+ dollar totals). Conservative fallbacks
        apply when quote/liquidity data is unavailable.
        """
        base = self.estimate_cost(asset, side, quantity, price,
                                  timestamp=timestamp,
                                  market_state=market_state,
                                  order_type=order_type)
        notional = abs(quantity) * price
        if notional <= 0:
            return {k: 0.0 for k in (
                "commission_bps", "spread_bps", "slippage_bps", "impact_bps",
                "funding_bps", "total_cost_bps", "total_cost")} | {
                "spread_source": "UNAVAILABLE"}

        commission_bps = (base["commission"] / notional) * 10_000
        if maker and asset_class == "crypto":
            commission_bps *= 0.5   # maker rebate approximation

        funding_bps = 0.0
        if asset_class == "crypto" and funding_rate_8h is not None and holding_hours > 0:
            periods = holding_hours / 8.0
            sign = 1.0 if side.lower() in ("buy", "long") else -1.0
            funding_bps = sign * funding_rate_8h * periods * 10_000

        spread_bps = (base["spread_cost"] / notional) * 10_000
        slippage_bps = (base["slippage_cost"] / notional) * 10_000
        impact_bps = (base["estimated_market_impact"] / notional) * 10_000
        total_bps = commission_bps + spread_bps + slippage_bps + impact_bps + max(funding_bps, 0.0)
        return {
            "commission_bps": commission_bps,
            "spread_bps": spread_bps,
            "slippage_bps": slippage_bps,
            "impact_bps": impact_bps,
            "funding_bps": funding_bps,
            "total_cost_bps": total_bps,
            "total_cost": total_bps / 10_000 * notional,
            "spread_source": base.get("spread_source", "UNAVAILABLE"),
        }


def execution_feasibility_score(
    notional_usd: float,
    adv_usd: float = None,
    bar_range_pct: float = None,
    estimated_cost_bps: float = None,
    expected_return_bps: float = None,
    hour_utc: int = None,
    data_fresh: bool = True,
) -> float:
    """0-100 feasibility of executing an opportunity as planned.

    Penalizes: illiquidity, large size vs ADV, high costs relative to expected
    return, erratic pricing (bar range), thin trading hours, stale data.
    """
    score = 100.0
    if not data_fresh:
        score -= 40
    if adv_usd is not None and adv_usd > 0:
        participation = notional_usd / adv_usd
        if participation > 0.05:
            score -= 40
        elif participation > 0.01:
            score -= 20
        elif participation > 0.001:
            score -= 5
    else:
        score -= 15   # unknown liquidity — conservative
    if bar_range_pct is not None and bar_range_pct > 0.02:
        score -= 15
    if estimated_cost_bps is not None and expected_return_bps is not None \
            and expected_return_bps > 0:
        cost_ratio = estimated_cost_bps / expected_return_bps
        if cost_ratio > 0.8:
            score -= 30
        elif cost_ratio > 0.5:
            score -= 15
    if hour_utc is not None and (hour_utc < 6 or hour_utc >= 22):
        score -= 10
    return max(0.0, min(100.0, score))
