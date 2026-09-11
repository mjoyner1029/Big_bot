"""
Execution Quality Engine — measures broker and order execution performance.

Tracks per-trade and aggregate:
    SLIPPAGE:       expected_price vs actual_fill_price
    SPREAD_COST:    half-spread paid on each fill
    LATENCY:        time from signal to fill
    FILL_QUALITY:   market impact, partial fill rate, rejected rate
    FEE_ANALYSIS:   fee-drag on strategy returns
    BROKER_SCORECARD: aggregate per-broker metrics

Separate from strategy quality — this measures execution, not alpha.
Results feed Research Engine hypotheses about execution cost reduction.
"""
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


@dataclass
class FillRecord:
    """A single fill event for quality analysis."""
    order_id:      str
    symbol:        str
    side:          str
    signal_price:  float     # price at signal time
    fill_price:    float     # actual fill
    quantity:      float
    fee:           float
    latency_ms:    float
    spread_pct:    float
    timestamp:     str = field(default_factory=_utcnow)


@dataclass
class SlippageStats:
    mean_bps:      float   # basis points
    median_bps:    float
    worst_bps:     float
    pct_adverse:   float   # fraction of fills with adverse slippage
    total_cost:    float   # total $ lost to slippage


@dataclass
class ExecutionReport:
    """Aggregate execution quality report."""
    period_days:    int
    fills_analyzed: int
    slippage:       SlippageStats
    avg_fee_pct:    float
    partial_fill_rate: float
    rejection_rate: float
    avg_latency_ms: float
    total_fee_drag: float
    by_broker:      Dict[str, Dict] = field(default_factory=dict)
    alerts:         List[str] = field(default_factory=list)
    computed_at:    str = field(default_factory=_utcnow)


_CREATE_FILLS = """
CREATE TABLE IF NOT EXISTS execution_fills (
    order_id       TEXT PRIMARY KEY,
    symbol         TEXT NOT NULL,
    side           TEXT NOT NULL,
    signal_price   REAL,
    fill_price     REAL NOT NULL,
    quantity       REAL NOT NULL,
    fee            REAL DEFAULT 0,
    latency_ms     REAL DEFAULT 0,
    spread_pct     REAL DEFAULT 0,
    broker         TEXT DEFAULT 'paper',
    timestamp      TEXT NOT NULL
)
"""


class ExecutionQualityEngine:
    """
    Measures and reports on broker execution quality.

    Reads from the orders table and cross-references with positions.
    Records fills in a separate execution_fills table for time-series analysis.
    """

    # Alert thresholds
    MAX_SLIPPAGE_BPS   = 10.0   # alert if mean slippage > 10 bps
    MAX_FEE_PCT        = 0.002  # alert if avg fee > 0.2%
    MAX_REJECTION_RATE = 0.10   # alert if >10% orders rejected
    MAX_PARTIAL_RATE   = 0.20   # alert if >20% fills are partial

    def __init__(self, db_path: str = "data/trade_memory.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_FILLS)
            conn.commit()

    # ── Recording ────────────────────────────────────────────────────────────

    def record_fill(
        self,
        order_id:     str,
        symbol:       str,
        side:         str,
        signal_price: float,
        fill_price:   float,
        quantity:     float,
        fee:          float = 0.0,
        latency_ms:   float = 0.0,
        spread_pct:   float = 0.0,
        broker:       str = 'paper',
    ) -> None:
        """Record an execution fill for quality tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO execution_fills "
                "(order_id,symbol,side,signal_price,fill_price,quantity,"
                "fee,latency_ms,spread_pct,broker,timestamp) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (order_id, symbol, side, signal_price, fill_price, quantity,
                 fee, latency_ms, spread_pct, broker, _utcnow()),
            )
            conn.commit()

    # ── Analysis ──────────────────────────────────────────────────────────────

    def run(self, period_days: int = 7) -> ExecutionReport:
        """Compute execution quality report for the given period."""
        fills  = self._load_fills(period_days)
        orders = self._load_orders(period_days)

        if not fills:
            return ExecutionReport(
                period_days=period_days,
                fills_analyzed=0,
                slippage=SlippageStats(0, 0, 0, 0, 0),
                avg_fee_pct=0.0,
                partial_fill_rate=0.0,
                rejection_rate=0.0,
                avg_latency_ms=0.0,
                total_fee_drag=0.0,
                alerts=["No fills in period"],
            )

        slippage = self._compute_slippage(fills)
        fee_pct  = self._avg_fee_pct(fills)
        fee_drag = self._total_fee_drag(fills)
        partial_rate = self._partial_fill_rate(orders)
        rej_rate     = self._rejection_rate(orders)
        avg_latency  = sum(f['latency_ms'] for f in fills) / len(fills) if fills else 0.0
        by_broker    = self._per_broker_stats(fills)
        alerts       = self._generate_alerts(slippage, fee_pct, rej_rate, partial_rate)

        return ExecutionReport(
            period_days=period_days,
            fills_analyzed=len(fills),
            slippage=slippage,
            avg_fee_pct=fee_pct,
            partial_fill_rate=partial_rate,
            rejection_rate=rej_rate,
            avg_latency_ms=avg_latency,
            total_fee_drag=fee_drag,
            by_broker=by_broker,
            alerts=alerts,
        )

    def summarize(self, period_days: int = 7) -> str:
        """Return a human-readable summary."""
        r = self.run(period_days)
        lines = [
            f"=== Execution Quality ({period_days}d) ===",
            f"Fills analyzed:   {r.fills_analyzed}",
            f"Mean slippage:    {r.slippage.mean_bps:.1f} bps",
            f"Avg fee:          {r.avg_fee_pct:.3%}",
            f"Partial rate:     {r.partial_fill_rate:.1%}",
            f"Rejection rate:   {r.rejection_rate:.1%}",
            f"Avg latency:      {r.avg_latency_ms:.0f}ms",
            f"Total fee drag:   ${r.total_fee_drag:.2f}",
        ]
        if r.alerts:
            lines.append("\nAlerts:")
            for a in r.alerts:
                lines.append(f"  ⚠ {a}")
        if r.by_broker:
            lines.append("\nBy broker:")
            for broker, stats in r.by_broker.items():
                lines.append(f"  {broker}: {stats.get('fills',0)} fills, "
                             f"slip={stats.get('mean_slip_bps',0):.1f}bps")
        return "\n".join(lines)

    def get_alerts(self) -> List[str]:
        r = self.run(7)
        return r.alerts

    # ── Private methods ───────────────────────────────────────────────────────

    def _compute_slippage(self, fills: List[Dict]) -> SlippageStats:
        slippages = []
        for f in fills:
            signal = f.get('signal_price', f['fill_price'])
            if signal and signal > 0:
                if f['side'].upper() == 'BUY':
                    bps = (f['fill_price'] - signal) / signal * 10000
                else:
                    bps = (signal - f['fill_price']) / signal * 10000
                slippages.append(bps)

        if not slippages:
            return SlippageStats(0, 0, 0, 0, 0)

        slippages.sort()
        n = len(slippages)
        mean   = sum(slippages) / n
        median = slippages[n // 2]
        worst  = max(slippages)
        adverse = sum(1 for s in slippages if s > 0) / n
        total_cost = sum(
            abs(s / 10000 * f['fill_price'] * f['quantity'])
            for s, f in zip(slippages, fills)
        )

        return SlippageStats(
            mean_bps=round(mean, 2),
            median_bps=round(median, 2),
            worst_bps=round(worst, 2),
            pct_adverse=round(adverse, 3),
            total_cost=round(total_cost, 2),
        )

    @staticmethod
    def _avg_fee_pct(fills: List[Dict]) -> float:
        if not fills:
            return 0.0
        pcts = [
            f['fee'] / (f['fill_price'] * f['quantity'])
            for f in fills
            if f['fill_price'] and f['quantity'] and f['fill_price'] * f['quantity'] > 0
        ]
        return sum(pcts) / len(pcts) if pcts else 0.0

    @staticmethod
    def _total_fee_drag(fills: List[Dict]) -> float:
        return sum(f.get('fee', 0.0) for f in fills)

    @staticmethod
    def _partial_fill_rate(orders: List[Dict]) -> float:
        if not orders:
            return 0.0
        partials = sum(1 for o in orders if o['status'] == 'PARTIALLY_FILLED')
        return partials / len(orders)

    @staticmethod
    def _rejection_rate(orders: List[Dict]) -> float:
        if not orders:
            return 0.0
        rejected = sum(1 for o in orders if o['status'] == 'REJECTED')
        return rejected / len(orders)

    def _per_broker_stats(self, fills: List[Dict]) -> Dict[str, Dict]:
        by_broker: Dict[str, List] = {}
        for f in fills:
            broker = f.get('broker', 'unknown')
            by_broker.setdefault(broker, []).append(f)

        result = {}
        for broker, broker_fills in by_broker.items():
            slippage = self._compute_slippage(broker_fills)
            result[broker] = {
                'fills':          len(broker_fills),
                'mean_slip_bps':  slippage.mean_bps,
                'avg_fee_pct':    self._avg_fee_pct(broker_fills),
                'total_fee_drag': self._total_fee_drag(broker_fills),
            }
        return result

    def _generate_alerts(
        self,
        slippage:     SlippageStats,
        avg_fee_pct:  float,
        rej_rate:     float,
        partial_rate: float,
    ) -> List[str]:
        alerts = []
        if slippage.mean_bps > self.MAX_SLIPPAGE_BPS:
            alerts.append(f"High mean slippage: {slippage.mean_bps:.1f}bps (max {self.MAX_SLIPPAGE_BPS})")
        if avg_fee_pct > self.MAX_FEE_PCT:
            alerts.append(f"High avg fee: {avg_fee_pct:.3%} (max {self.MAX_FEE_PCT:.1%})")
        if rej_rate > self.MAX_REJECTION_RATE:
            alerts.append(f"High rejection rate: {rej_rate:.1%} (max {self.MAX_REJECTION_RATE:.0%})")
        if partial_rate > self.MAX_PARTIAL_RATE:
            alerts.append(f"High partial fill rate: {partial_rate:.1%} (max {self.MAX_PARTIAL_RATE:.0%})")
        return alerts

    # ── DB queries ────────────────────────────────────────────────────────────

    def _load_fills(self, period_days: int) -> List[Dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=period_days)).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM execution_fills WHERE timestamp>=?", (cutoff,)
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"ExecutionQualityEngine: fills load error: {e}")
            return []

    def _load_orders(self, period_days: int) -> List[Dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=period_days)).isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM orders WHERE created_at>=?", (cutoff,)
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"ExecutionQualityEngine: orders load error: {e}")
            return []
