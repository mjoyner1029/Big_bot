"""
Broker abstraction layer — the ONLY path for all order execution.

Architecture rule:
    No position may be opened locally until an order is acknowledged by
    the broker and a fill is confirmed. The correct execution path is:

        Strategy signal
          → SafetyManager.check_can_trade()
          → Broker.submit_order()          ← order sent to market/simulator
          → Broker.wait_for_fill()         ← blocking until FILLED/REJECTED
          → PositionManager.open_position(broker_order_id=...)  ← local record

Trading modes (set TRADING_MODE env var):
    BACKTEST — no-op / historical simulator
    PAPER    — PaperBroker (simulated fills, no network calls)  ← default
    LIVE     — AlpacaBroker (real orders)

LIVE requires BOTH:
    TRADING_MODE=LIVE
    ENABLE_LIVE_TRADING=true
Neither alone is sufficient.
"""
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Enums & value objects
# ─────────────────────────────────────────────────────────────────────────────

class TradingMode(str, Enum):
    BACKTEST = "BACKTEST"
    PAPER    = "PAPER"
    LIVE     = "LIVE"


class OrderStatus(str, Enum):
    PENDING   = "PENDING"
    FILLED    = "FILLED"
    PARTIAL   = "PARTIAL"
    REJECTED  = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class Order:
    symbol:         str
    side:           str             # "BUY" | "SELL"
    quantity:       float           # amount in base asset (e.g. BTC)
    order_type:     str   = "MARKET"
    limit_price:    Optional[float] = None
    order_id:       Optional[str]   = None
    status:         OrderStatus     = OrderStatus.PENDING
    fill_price:     Optional[float] = None  # market price at submission
    fill_quantity:  float = 0.0
    fees:           float = 0.0
    submitted_at:   datetime = field(default_factory=_utcnow)
    filled_at:      Optional[datetime] = None
    reject_reason:  Optional[str] = None

    @property
    def is_filled(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.PARTIAL)

    @property
    def notional(self) -> float:
        """Dollar value of filled portion."""
        return (self.fill_price or 0.0) * self.fill_quantity


@dataclass
class FillResult:
    """Everything the bot needs after a fill to open a local position."""
    order_id:       str
    symbol:         str
    side:           str
    fill_price:     float
    fill_quantity:  float       # actual coins/shares filled
    notional:       float       # dollar value
    fees:           float
    status:         OrderStatus
    reject_reason:  Optional[str] = None
    partial:        bool = False


@dataclass
class BrokerPosition:
    symbol:         str
    side:           str
    quantity:       float
    avg_price:      float
    market_value:   float
    unrealized_pnl: float


@dataclass
class AccountState:
    cash:           float
    equity:         float
    buying_power:   float
    positions:      List[BrokerPosition] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract interface
# ─────────────────────────────────────────────────────────────────────────────

class Broker(ABC):
    """All trade execution must go through this interface.

    The key contract is:
        result = broker.submit_and_wait(order, current_price)

    which atomically submits and waits for the fill, returning a FillResult.
    The bot must use the FillResult to open a local position record.
    """

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order. Returns immediately with PENDING/REJECTED status."""

    @abstractmethod
    def wait_for_fill(
        self,
        order_id: str,
        timeout_seconds: float = 30.0,
        poll_interval: float = 0.5,
    ) -> Order:
        """Block until order is FILLED, CANCELLED, or REJECTED (or timeout)."""

    def submit_and_wait(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None,
        timeout_seconds: float = 30.0,
    ) -> FillResult:
        """
        High-level entry point for the bot's execution path.

        Creates an order, submits it, waits for fill, and returns
        a FillResult the bot uses to open a local position record.

        On rejection or timeout, returns a FillResult with
        status=REJECTED so the bot can log and move on.
        """
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=current_price,
            order_type=order_type,
            limit_price=limit_price,
        )
        order = self.submit_order(order)

        if order.status == OrderStatus.REJECTED:
            return FillResult(
                order_id=order.order_id or "rejected",
                symbol=symbol,
                side=side,
                fill_price=current_price,
                fill_quantity=0.0,
                notional=0.0,
                fees=0.0,
                status=OrderStatus.REJECTED,
                reject_reason=order.reject_reason,
            )

        # Wait for fill
        order = self.wait_for_fill(order.order_id, timeout_seconds=timeout_seconds)

        partial = order.status == OrderStatus.PARTIAL
        return FillResult(
            order_id=order.order_id,
            symbol=symbol,
            side=side,
            fill_price=order.fill_price or current_price,
            fill_quantity=order.fill_quantity,
            notional=order.notional,
            fees=order.fees,
            status=order.status,
            partial=partial,
        )

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if successfully cancelled."""

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """Fetch current state of an order by ID."""

    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """All pending/partial orders."""

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Current broker position for a symbol, or None."""

    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]:
        """All currently open broker positions."""

    @abstractmethod
    def get_account(self) -> AccountState:
        """Current account equity, cash, and buying power."""

    @abstractmethod
    def close_position(
        self,
        symbol: str,
        current_price: float,
        quantity: Optional[float] = None,
    ) -> FillResult:
        """Close all (or a portion) of the broker position for a symbol."""

    def place_bracket_orders(
        self,
        symbol: str,
        quantity: float,
        stop_price: float,
        take_profit_price: float,
    ) -> Dict[str, Optional[str]]:
        """
        Place native broker-side stop-loss and take-profit orders.

        Returns dict with keys:
            sl_order_id  — stop-loss order ID (None if unsupported)
            tp_order_id  — take-profit order ID (None if unsupported)

        If the broker implementation doesn't support bracket orders, returns
        {'sl_order_id': None, 'tp_order_id': None} and the execution manager
        must enforce stops in-process.
        """
        # Default: not supported — subclasses override
        logger.debug(
            f"place_bracket_orders not implemented for {type(self).__name__} — "
            f"using software fallback stops"
        )
        return {"sl_order_id": None, "tp_order_id": None}

    @abstractmethod
    def reconcile(self, local_positions: List[Dict]) -> Dict:
        """
        Compare local DB positions with broker state.

        Returns dict:
            ok              bool    — True when no discrepancies
            discrepancies   list    — human-readable mismatch descriptions
            broker_positions list   — current broker positions
            local_positions list    — passed-in local positions
            action_required list    — suggested corrective actions
        """


# ─────────────────────────────────────────────────────────────────────────────
# PaperBroker — deterministic paper trading, no network calls
# ─────────────────────────────────────────────────────────────────────────────

class PaperBroker(Broker):
    """Simulated broker for PAPER and BACKTEST modes.

    Simulates:
        • Immediate market fills (PAPER is always filled synchronously)
        • Configurable slippage (default 0.05%)
        • Configurable taker fee (default 0.10%)
        • Rejection on insufficient buying power
        • Partial fills are not simulated (always full fill or reject)
    """

    DEFAULT_SLIPPAGE_PCT = 0.0005   # 0.05 % adverse slippage
    DEFAULT_FEE_PCT      = 0.001    # 0.10 % taker fee

    def __init__(self, starting_cash: float = 10_000.0, config: Optional[Dict] = None):
        cfg = config or {}
        self.cash         = starting_cash
        self.slippage_pct = cfg.get("slippage_pct", self.DEFAULT_SLIPPAGE_PCT)
        self.fee_pct      = cfg.get("fee_pct",      self.DEFAULT_FEE_PCT)
        self._positions: Dict[str, BrokerPosition] = {}
        self._orders:    Dict[str, Order]          = {}
        self._seq = 0
        logger.info(
            f"PaperBroker ready | cash=${starting_cash:,.2f} "
            f"slippage={self.slippage_pct:.3%} fee={self.fee_pct:.3%}"
        )

    # ── Submission ────────────────────────────────────────────────────────────

    def submit_order(self, order: Order) -> Order:
        """Submit and immediately fill (paper trading fills synchronously)."""
        self._seq += 1
        order.order_id = f"PAPER-{self._seq:06d}"

        if order.fill_price is None:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "fill_price (current market price) must be set"
            self._orders[order.order_id] = order
            logger.warning(f"PaperBroker: REJECTED {order.order_id} — no fill_price")
            return order

        # Apply adverse slippage
        slip = order.fill_price * self.slippage_pct
        fill_price = order.fill_price + slip if order.side == "BUY" else order.fill_price - slip

        notional = fill_price * order.quantity
        fees     = notional * self.fee_pct

        # Buying power check
        if order.side == "BUY" and (notional + fees) > self.cash:
            order.status = OrderStatus.REJECTED
            order.reject_reason = (
                f"Insufficient buying power: need ${notional+fees:.2f}, "
                f"available ${self.cash:.2f}"
            )
            self._orders[order.order_id] = order
            logger.warning(f"PaperBroker: REJECTED {order.order_id} — {order.reject_reason}")
            return order

        # Execute fill
        order.fill_price    = fill_price
        order.fill_quantity = order.quantity
        order.fees          = fees
        order.status        = OrderStatus.FILLED
        order.filled_at     = _utcnow()

        self._apply_fill(order, fill_price, fees)
        self._orders[order.order_id] = order

        logger.info(
            f"PaperBroker FILL {order.order_id}: {order.side} {order.quantity:.6f} "
            f"{order.symbol} @ ${fill_price:.4f} fees=${fees:.4f}"
        )
        return order

    def wait_for_fill(self, order_id: str, timeout_seconds: float = 30.0,
                      poll_interval: float = 0.5) -> Order:
        """Paper orders are filled synchronously in submit_order — no waiting needed."""
        order = self._orders.get(order_id)
        if order is None:
            return Order(symbol="", side="", quantity=0,
                         status=OrderStatus.REJECTED, reject_reason=f"Unknown order {order_id}")
        return order

    # ── Position update helpers ───────────────────────────────────────────────

    def _apply_fill(self, order: Order, fill_price: float, fees: float) -> None:
        if order.side == "BUY":
            self.cash -= (fill_price * order.quantity + fees)
            pos = self._positions.get(order.symbol)
            if pos:
                total = pos.quantity + order.quantity
                pos.avg_price = (pos.avg_price * pos.quantity + fill_price * order.quantity) / total
                pos.quantity  = total
            else:
                self._positions[order.symbol] = BrokerPosition(
                    symbol=order.symbol, side="LONG",
                    quantity=order.quantity, avg_price=fill_price,
                    market_value=fill_price * order.quantity,
                    unrealized_pnl=0.0,
                )
        else:  # SELL
            self.cash += (fill_price * order.quantity - fees)
            pos = self._positions.get(order.symbol)
            if pos:
                pos.quantity = round(pos.quantity - order.quantity, 10)
                if pos.quantity <= 1e-10:
                    del self._positions[order.symbol]

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            logger.info(f"PaperBroker: cancelled {order_id}")
            return True
        return False

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.status == OrderStatus.PENDING]

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        return self._positions.get(symbol)

    def get_positions(self) -> List[BrokerPosition]:
        return list(self._positions.values())

    def get_account(self) -> AccountState:
        market_value = sum(p.quantity * p.avg_price for p in self._positions.values())
        return AccountState(
            cash=self.cash,
            equity=self.cash + market_value,
            buying_power=self.cash,
            positions=list(self._positions.values()),
        )

    def close_position(
        self,
        symbol: str,
        current_price: float,
        quantity: Optional[float] = None,
    ) -> FillResult:
        pos = self._positions.get(symbol)
        if not pos:
            return FillResult(
                order_id="", symbol=symbol, side="SELL",
                fill_price=current_price, fill_quantity=0.0, notional=0.0,
                fees=0.0, status=OrderStatus.REJECTED,
                reject_reason=f"No broker position for {symbol}",
            )
        qty = quantity if quantity else pos.quantity
        order = Order(symbol=symbol, side="SELL", quantity=qty, fill_price=current_price)
        order = self.submit_order(order)
        return FillResult(
            order_id=order.order_id,
            symbol=symbol,
            side="SELL",
            fill_price=order.fill_price or current_price,
            fill_quantity=order.fill_quantity,
            notional=order.notional,
            fees=order.fees,
            status=order.status,
        )

    def place_bracket_orders(
        self,
        symbol: str,
        quantity: float,
        stop_price: float,
        take_profit_price: float,
    ) -> Dict[str, Optional[str]]:
        """
        Paper-simulated bracket orders.

        Stores virtual SL and TP levels; the PaperBroker's
        `check_bracket_triggers()` method is called each tick to simulate
        broker-side stop protection (crucial: if the Python process crashes,
        real positions would be unprotected, but in paper mode this is fine).
        """
        self._seq += 1
        sl_id = f"PAPER-SL-{self._seq:06d}"
        self._seq += 1
        tp_id = f"PAPER-TP-{self._seq:06d}"
        self._bracket_orders = getattr(self, '_bracket_orders', {})
        self._bracket_orders[symbol] = {
            'sl_price':    stop_price,
            'tp_price':    take_profit_price,
            'quantity':    quantity,
            'sl_order_id': sl_id,
            'tp_order_id': tp_id,
        }
        logger.info(
            f"PaperBroker bracket: {symbol} SL=${stop_price:.4f} "
            f"TP=${take_profit_price:.4f} qty={quantity:.6f}"
        )
        return {"sl_order_id": sl_id, "tp_order_id": tp_id}

    def check_bracket_triggers(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check if current price has triggered a virtual bracket order.

        Returns 'stop_loss', 'take_profit', or None.
        Call this each tick for each open position that has bracket orders.
        """
        brackets = getattr(self, '_bracket_orders', {})
        b = brackets.get(symbol)
        if not b:
            return None
        pos = self._positions.get(symbol)
        if not pos:
            return None
        if pos.side == "LONG":
            if current_price <= b['sl_price']:
                logger.warning(f"PaperBroker: SL triggered for {symbol} @ ${current_price:.4f}")
                brackets.pop(symbol, None)
                return 'stop_loss'
            if current_price >= b['tp_price']:
                logger.info(f"PaperBroker: TP triggered for {symbol} @ ${current_price:.4f}")
                brackets.pop(symbol, None)
                return 'take_profit'
        return None

    def reconcile(self, local_positions: List[Dict]) -> Dict:
        local_syms  = {p["symbol"] for p in local_positions}
        broker_syms = set(self._positions.keys())
        discrepancies = []
        actions = []

        for sym in local_syms - broker_syms:
            discrepancies.append(f"Local has '{sym}' OPEN but broker has no position")
            actions.append(f"Investigate '{sym}' — may need manual close")

        for sym in broker_syms - local_syms:
            discrepancies.append(f"Broker has '{sym}' position but local DB shows none")
            actions.append(f"Create local record for '{sym}' or close at broker")

        # Quantity check for shared symbols
        for sym in local_syms & broker_syms:
            local_size = next((p.get("size", 0) for p in local_positions if p["symbol"] == sym), 0)
            broker_qty = self._positions[sym].quantity
            if abs(local_size - broker_qty) > 0.0001:
                discrepancies.append(
                    f"Size mismatch for '{sym}': local=${local_size:.4f} broker={broker_qty:.6f}"
                )

        ok = len(discrepancies) == 0
        if not ok:
            logger.warning(f"Reconciliation found {len(discrepancies)} discrepancies")
        return {
            "ok":               ok,
            "discrepancies":    discrepancies,
            "broker_positions": list(self._positions.values()),
            "local_positions":  local_positions,
            "action_required":  actions,
        }


# ─────────────────────────────────────────────────────────────────────────────
# AlpacaBroker — real money via Alpaca REST API
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaBroker(Broker):
    """Live trading via Alpaca Markets API.

    HARD GATE: requires BOTH env vars:
        TRADING_MODE=LIVE
        ENABLE_LIVE_TRADING=true
    """

    def __init__(self):
        mode   = os.environ.get("TRADING_MODE", "PAPER").upper()
        enable = os.environ.get("ENABLE_LIVE_TRADING", "false").lower()

        if mode != "LIVE" or enable != "true":
            raise RuntimeError(
                "AlpacaBroker requires TRADING_MODE=LIVE and ENABLE_LIVE_TRADING=true. "
                "Use PaperBroker for PAPER/BACKTEST modes."
            )

        try:
            import alpaca_trade_api as tradeapi
        except ImportError:
            raise ImportError("alpaca-trade-api not installed. Run: pip install alpaca-trade-api")

        key    = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_API_SECRET")
        base   = os.environ.get("ALPACA_BASE_URL", "https://api.alpaca.markets")

        if not key or not secret:
            raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET must be set for LIVE trading")

        self._api = tradeapi.REST(key, secret, base, api_version="v2")
        logger.warning("⚠️  AlpacaBroker LIVE — real orders will be placed")

    # ── Submission ────────────────────────────────────────────────────────────

    def submit_order(self, order: Order) -> Order:
        try:
            resp = self._api.submit_order(
                symbol=order.symbol,
                qty=str(order.quantity),
                side=order.side.lower(),
                type=order.order_type.lower(),
                time_in_force="gtc",
                limit_price=str(order.limit_price) if order.limit_price else None,
            )
            order.order_id = resp.id
            order.status   = OrderStatus.PENDING
            logger.info(f"AlpacaBroker submitted: {order.order_id} {order.side} {order.quantity} {order.symbol}")
        except Exception as e:
            order.status       = OrderStatus.REJECTED
            order.reject_reason = str(e)
            logger.error(f"AlpacaBroker submit failed: {e}")
        return order

    def wait_for_fill(self, order_id: str, timeout_seconds: float = 30.0,
                      poll_interval: float = 1.0) -> Order:
        """Poll Alpaca until filled, cancelled, or timeout."""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                resp = self._api.get_order(order_id)
                if resp.status == "filled":
                    return Order(
                        symbol=resp.symbol, side=resp.side.upper(),
                        quantity=float(resp.qty),
                        order_id=resp.id,
                        status=OrderStatus.FILLED,
                        fill_price=float(resp.filled_avg_price),
                        fill_quantity=float(resp.filled_qty),
                        filled_at=_utcnow(),
                    )
                if resp.status in ("cancelled", "rejected", "expired"):
                    order = Order(
                        symbol=resp.symbol, side=resp.side.upper(),
                        quantity=float(resp.qty),
                        order_id=resp.id,
                        status=OrderStatus.CANCELLED if resp.status == "cancelled" else OrderStatus.REJECTED,
                    )
                    order.reject_reason = f"Alpaca status: {resp.status}"
                    return order
                # Partial fill
                if float(resp.filled_qty) > 0:
                    logger.info(f"AlpacaBroker: partial fill {resp.filled_qty}/{resp.qty}")
            except Exception as e:
                logger.warning(f"AlpacaBroker poll error: {e}")
            time.sleep(poll_interval)

        # Timeout — cancel the order
        logger.warning(f"AlpacaBroker: fill timeout for {order_id}, cancelling")
        self.cancel_order(order_id)
        order = Order(symbol="", side="", quantity=0,
                      status=OrderStatus.CANCELLED, order_id=order_id)
        order.reject_reason = "Fill timeout — order cancelled"
        return order

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._api.cancel_order(order_id)
            return True
        except Exception as e:
            logger.error(f"AlpacaBroker cancel failed: {e}")
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        try:
            r = self._api.get_order(order_id)
            return Order(
                symbol=r.symbol, side=r.side.upper(), quantity=float(r.qty),
                order_id=r.id,
                status=OrderStatus.FILLED if r.status == "filled" else OrderStatus.PENDING,
                fill_price=float(r.filled_avg_price) if r.filled_avg_price else None,
                fill_quantity=float(r.filled_qty),
            )
        except Exception as e:
            logger.error(f"AlpacaBroker get_order failed: {e}")
            return None

    def get_open_orders(self) -> List[Order]:
        try:
            return [
                Order(symbol=o.symbol, side=o.side.upper(), quantity=float(o.qty),
                      order_id=o.id)
                for o in self._api.list_orders(status="open")
            ]
        except Exception as e:
            logger.error(f"AlpacaBroker get_open_orders failed: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        try:
            p = self._api.get_position(symbol)
            return BrokerPosition(
                symbol=symbol, side=p.side.upper(),
                quantity=float(p.qty), avg_price=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pnl=float(p.unrealized_pl),
            )
        except Exception:
            return None

    def get_positions(self) -> List[BrokerPosition]:
        try:
            return [
                BrokerPosition(
                    symbol=p.symbol, side=p.side.upper(),
                    quantity=float(p.qty), avg_price=float(p.avg_entry_price),
                    market_value=float(p.market_value),
                    unrealized_pnl=float(p.unrealized_pl),
                )
                for p in self._api.list_positions()
            ]
        except Exception as e:
            logger.error(f"AlpacaBroker get_positions failed: {e}")
            return []

    def get_account(self) -> AccountState:
        try:
            a = self._api.get_account()
            return AccountState(
                cash=float(a.cash), equity=float(a.equity),
                buying_power=float(a.buying_power),
                positions=self.get_positions(),
            )
        except Exception as e:
            logger.error(f"AlpacaBroker get_account failed: {e}")
            return AccountState(cash=0, equity=0, buying_power=0)

    def close_position(
        self,
        symbol: str,
        current_price: float,
        quantity: Optional[float] = None,
    ) -> FillResult:
        try:
            resp = self._api.close_position(symbol)
            return FillResult(
                order_id=resp.id, symbol=symbol, side="SELL",
                fill_price=current_price, fill_quantity=float(resp.qty),
                notional=current_price * float(resp.qty), fees=0.0,
                status=OrderStatus.PENDING,
            )
        except Exception as e:
            logger.error(f"AlpacaBroker close_position failed for {symbol}: {e}")
            return FillResult(
                order_id="", symbol=symbol, side="SELL",
                fill_price=current_price, fill_quantity=0.0, notional=0.0,
                fees=0.0, status=OrderStatus.REJECTED, reject_reason=str(e),
            )

    def reconcile(self, local_positions: List[Dict]) -> Dict:
        broker_positions = self.get_positions()
        broker_syms  = {p.symbol for p in broker_positions}
        local_syms   = {p["symbol"] for p in local_positions}
        discrepancies = []
        actions = []

        for sym in local_syms - broker_syms:
            discrepancies.append(f"Local has '{sym}' OPEN but Alpaca has no position")
            actions.append(f"Close local record for '{sym}'")

        for sym in broker_syms - local_syms:
            discrepancies.append(f"Alpaca has '{sym}' position but local DB is empty")
            actions.append(f"Recreate local DB record for '{sym}' or close at Alpaca")

        ok = len(discrepancies) == 0
        if not ok:
            logger.warning(f"Reconciliation: {len(discrepancies)} discrepancies")
        return {
            "ok":               ok,
            "discrepancies":    discrepancies,
            "broker_positions": broker_positions,
            "local_positions":  local_positions,
            "action_required":  actions,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_broker(capital: float = 10_000.0) -> Broker:
    """Return the appropriate broker for the current TRADING_MODE.

    PAPER (default) and BACKTEST → PaperBroker
    LIVE (requires ENABLE_LIVE_TRADING=true) → AlpacaBroker
    """
    mode = os.environ.get("TRADING_MODE", "PAPER").upper()
    if mode == "LIVE":
        logger.warning("TRADING_MODE=LIVE — instantiating AlpacaBroker")
        return AlpacaBroker()
    logger.info(f"TRADING_MODE={mode} — using PaperBroker")
    return PaperBroker(starting_cash=capital)
