"""
Portfolio Manager — capital allocation across all open and candidate positions.

ARCHITECTURE RULE:
    No trade may be sized or approved without consulting the PortfolioManager.
    The PortfolioManager is the single source of truth for:
        - current exposure by asset class
        - current exposure by sector/strategy
        - correlation-adjusted portfolio risk
        - maximum capital available for new positions
        - beta-weighted exposure

Usage:
    pm = PortfolioManager(capital=10_000)
    allocation = pm.allocate(candidate, position_manager)
    if allocation.approved:
        size_dollars = allocation.size_dollars
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Correlation estimates (static; updated periodically) ──────────────────────
# These are approximate pairwise correlations between asset classes.
# Highly-correlated assets reduce the total position size allowed.
_ASSET_CLASS_CORRELATIONS: Dict[Tuple[str, str], float] = {
    ('crypto', 'crypto'):     0.75,   # BTC/ETH/SOL are highly correlated
    ('equity', 'equity'):     0.60,
    ('etf',    'equity'):     0.70,
    ('etf',    'etf'):        0.50,
    ('crypto', 'equity'):     0.20,
    ('equity', 'crypto'):     0.20,
    ('prediction', 'crypto'): 0.05,
    ('prediction', 'equity'): 0.05,
}

# ── Sector beta estimates (vs S&P 500) ────────────────────────────────────────
_SECTOR_BETA: Dict[str, float] = {
    'crypto':      1.8,
    'tech':        1.3,
    'financials':  1.1,
    'healthcare':  0.8,
    'utilities':   0.5,
    'etf_broad':   1.0,
    'prediction':  0.0,   # uncorrelated with market
    'unknown':     1.0,
}


@dataclass
class AllocationResult:
    """Result of PortfolioManager.allocate()."""
    approved:       bool
    size_dollars:   float
    reject_reason:  Optional[str] = None
    # Context
    portfolio_heat:     float = 0.0   # current total risk %
    correlation_penalty: float = 0.0  # reduction applied due to correlation
    max_available:      float = 0.0   # max dollars available for new positions


@dataclass
class PortfolioState:
    """Snapshot of current portfolio risk."""
    total_equity:        float
    cash_available:      float
    positions_open:      int
    exposure_by_class:   Dict[str, float] = field(default_factory=dict)
    correlation_risk:    float = 0.0
    beta_exposure:       float = 0.0
    portfolio_heat_pct:  float = 0.0


class PortfolioManager:
    """
    Manages capital allocation with correlation, beta, and exposure constraints.
    """

    def __init__(
        self,
        capital: float,
        max_total_exposure_pct: float = 0.80,    # max 80% of capital deployed
        max_class_exposure_pct: float = 0.50,    # max 50% in any one asset class
        max_correlation_exposure: float = 0.60,  # max 60% in highly-correlated assets
        max_beta_exposure: float = 2.0,          # max portfolio beta
        cash_reserve_pct: float = 0.10,          # always keep 10% as cash
    ):
        self.capital                  = capital
        self.max_total_exposure_pct   = max_total_exposure_pct
        self.max_class_exposure_pct   = max_class_exposure_pct
        self.max_correlation_exposure = max_correlation_exposure
        self.max_beta_exposure        = max_beta_exposure
        self.cash_reserve_pct         = cash_reserve_pct

        logger.info(
            f"PortfolioManager ready: capital=${capital:,.0f} "
            f"max_exposure={max_total_exposure_pct:.0%} "
            f"cash_reserve={cash_reserve_pct:.0%}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def allocate(
        self,
        symbol: str,
        asset_class: str,
        requested_size: float,
        open_positions: List[Dict],
        current_equity: float = None,
    ) -> AllocationResult:
        """
        Decide how much capital to allocate to a new position.

        Args:
            symbol:          Trading symbol
            asset_class:     'crypto' | 'equity' | 'etf' | 'prediction'
            requested_size:  Dollar size requested by Kelly/strategy
            open_positions:  List of current open position dicts from PositionManager
            current_equity:  Current portfolio equity (defaults to self.capital)

        Returns AllocationResult with approved bool and final size_dollars.
        """
        equity = current_equity or self.capital
        state  = self._compute_state(open_positions, equity)

        # ── Gate 1: Cash reserve ───────────────────────────────────────────────
        cash_floor = equity * self.cash_reserve_pct
        available  = state.cash_available - cash_floor
        if available <= 0:
            return AllocationResult(
                approved=False,
                size_dollars=0.0,
                reject_reason=f"Cash reserve floor: ${cash_floor:,.0f} minimum",
                portfolio_heat=state.portfolio_heat_pct,
                max_available=0.0,
            )

        # ── Gate 2: Total exposure cap ─────────────────────────────────────────
        current_deployed = sum(state.exposure_by_class.values())
        max_deploy = equity * self.max_total_exposure_pct
        if current_deployed >= max_deploy:
            return AllocationResult(
                approved=False,
                size_dollars=0.0,
                reject_reason=(
                    f"Total exposure cap: ${current_deployed:,.0f} deployed "
                    f"(max ${max_deploy:,.0f})"
                ),
                portfolio_heat=state.portfolio_heat_pct,
                max_available=0.0,
            )

        # ── Gate 3: Asset-class exposure cap ──────────────────────────────────
        class_deployed = state.exposure_by_class.get(asset_class, 0.0)
        max_class      = equity * self.max_class_exposure_pct
        if class_deployed >= max_class:
            return AllocationResult(
                approved=False,
                size_dollars=0.0,
                reject_reason=(
                    f"Asset class cap ({asset_class}): ${class_deployed:,.0f} "
                    f"deployed (max ${max_class:,.0f})"
                ),
                portfolio_heat=state.portfolio_heat_pct,
                max_available=max_class - class_deployed,
            )

        # ── Gate 4: Beta cap ──────────────────────────────────────────────────
        new_beta = state.beta_exposure + _SECTOR_BETA.get(asset_class, 1.0) * (
            requested_size / equity
        )
        if new_beta > self.max_beta_exposure:
            return AllocationResult(
                approved=False,
                size_dollars=0.0,
                reject_reason=(
                    f"Beta cap: adding {asset_class} would push beta to "
                    f"{new_beta:.2f} (max {self.max_beta_exposure:.1f})"
                ),
                portfolio_heat=state.portfolio_heat_pct,
                max_available=requested_size * 0.5,
            )

        # ── Correlation penalty ────────────────────────────────────────────────
        corr_penalty = self._correlation_penalty(asset_class, open_positions)
        adjusted_size = requested_size * (1.0 - corr_penalty)

        # ── Final size: min of all constraints ────────────────────────────────
        room_total  = max_deploy - current_deployed
        room_class  = max_class  - class_deployed
        room_cash   = available
        size_dollars = min(adjusted_size, room_total, room_class, room_cash)
        size_dollars = max(size_dollars, 0.0)

        if size_dollars < 10.0:
            return AllocationResult(
                approved=False,
                size_dollars=0.0,
                reject_reason=f"Allocation too small after constraints: ${size_dollars:.2f}",
                portfolio_heat=state.portfolio_heat_pct,
                correlation_penalty=corr_penalty,
                max_available=room_cash,
            )

        logger.info(
            f"PortfolioManager: APPROVED {symbol} ${size_dollars:,.2f} "
            f"(requested ${requested_size:,.2f}, corr_penalty={corr_penalty:.0%}, "
            f"heat={state.portfolio_heat_pct:.1%})"
        )
        return AllocationResult(
            approved=True,
            size_dollars=size_dollars,
            portfolio_heat=state.portfolio_heat_pct,
            correlation_penalty=corr_penalty,
            max_available=min(room_total, room_class, room_cash),
        )

    def get_state(self, open_positions: List[Dict], equity: float = None) -> PortfolioState:
        """Return a snapshot of current portfolio risk state."""
        return self._compute_state(open_positions, equity or self.capital)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _compute_state(self, open_positions: List[Dict], equity: float) -> PortfolioState:
        exposure_by_class: Dict[str, float] = {}
        beta_exposure = 0.0
        total_deployed = 0.0

        for pos in open_positions:
            ac   = pos.get('asset_class', 'crypto')
            size = pos.get('size', 0.0)
            exposure_by_class[ac] = exposure_by_class.get(ac, 0.0) + size
            total_deployed       += size
            beta_exposure        += _SECTOR_BETA.get(ac, 1.0) * (size / equity)

        cash_available   = equity - total_deployed
        heat_pct         = total_deployed / equity if equity > 0 else 0.0

        return PortfolioState(
            total_equity=equity,
            cash_available=cash_available,
            positions_open=len(open_positions),
            exposure_by_class=exposure_by_class,
            beta_exposure=round(beta_exposure, 3),
            portfolio_heat_pct=round(heat_pct, 4),
        )

    def _correlation_penalty(
        self, new_class: str, open_positions: List[Dict]
    ) -> float:
        """
        Estimate correlation penalty for adding a new position.

        If many positions share a high correlation with the new asset class,
        reduce the allocation to maintain diversification.

        Returns a penalty fraction in [0, 0.5].
        """
        if not open_positions:
            return 0.0

        corr_sum = 0.0
        for pos in open_positions:
            ac = pos.get('asset_class', 'crypto')
            key  = tuple(sorted([new_class, ac]))
            corr = _ASSET_CLASS_CORRELATIONS.get(key, 0.30)
            corr_sum += corr

        avg_corr = corr_sum / len(open_positions)
        # Penalty = 0 at corr=0, 0.5 at corr=1.0
        return min(avg_corr * 0.5, 0.5)
