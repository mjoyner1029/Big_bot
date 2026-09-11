"""
Order Management System — tracks every order from submission to settlement.

Provides a persistent audit trail for all orders placed through the broker.
The OMS reconciles local state with broker state on every cycle.

Order lifecycle:
    SUBMITTED → ACCEPTED → [PARTIALLY_FILLED →] FILLED
    SUBMITTED → REJECTED
    ACCEPTED  → CANCELLED
    FILLED    → SETTLED (after broker confirmation)

    For bracket orders:
    BRACKET_PENDING → STOP_TRIGGERED | TP_TRIGGERED | MANUAL_CLOSE

    Stale detection:
    Orders not updated within STALE_THRESHOLD_MINUTES → STALE → auto-cancel
"""
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()

STALE_THRESHOLD_MINUTES = 30


class OrderStatus(str, Enum):
    SUBMITTED        = "SUBMITTED"
    ACCEPTED         = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED           = "FILLED"
    CANCELLED        = "CANCELLED"
    REJECTED         = "REJECTED"
    STALE            = "STALE"
    SETTLED          = "SETTLED"
    BRACKET_PENDING  = "BRACKET_PENDING"
    STOP_TRIGGERED   = "STOP_TRIGGERED"
    TP_TRIGGERED     = "TP_TRIGGERED"


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET   = "MARKET"
    LIMIT    = "LIMIT"
    STOP     = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class ManagedOrder:
    order_id:        str
    broker_order_id: Optional[str]
    position_id:     Optional[str]
    symbol:          str
    side:            OrderSide
    order_type:      OrderType
    quantity:        float
    price:           Optional[float]       # limit price
    stop_price:      Optional[float]       # stop price
    fill_price:      Optional[float] = None  # actual fill
    filled_qty:      float = 0.0
    status:          OrderStatus = OrderStatus.SUBMITTED
    reject_reason:   Optional[str] = None
    order_class:     str = "entry"         # entry | stop | tp | partial_close
    created_at:      str = field(default_factory=_utcnow)
    updated_at:      str = field(default_factory=_utcnow)
    settled_at:      Optional[str] = None
    notes:           str = ''


_CREATE_ORDERS = """
CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    broker_order_id TEXT,
    position_id     TEXT,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    order_type      TEXT NOT NULL,
    quantity        REAL NOT NULL,
    price           REAL,
    stop_price      REAL,
    fill_price      REAL,
    filled_qty      REAL DEFAULT 0,
    status          TEXT DEFAULT 'SUBMITTED',
    reject_reason   TEXT,
    order_class     TEXT DEFAULT 'entry',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    settled_at      TEXT,
    notes           TEXT DEFAULT ''
)
"""


class OrderManager:
    """
    Canonical order lifecycle management.

    All broker submissions go through record_submission().
    All fills/updates go through update_status().
    Reconcile() compares local and broker state.
    """

    def __init__(self, db_path: str = "data/trade_memory.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_ORDERS)
            conn.commit()

    # ── Submission ────────────────────────────────────────────────────────────

    def record_submission(
        self,
        symbol:          str,
        side:            str,
        order_type:      str,
        quantity:        float,
        price:           Optional[float] = None,
        stop_price:      Optional[float] = None,
        position_id:     Optional[str] = None,
        broker_order_id: Optional[str] = None,
        order_class:     str = "entry",
        notes:           str = '',
    ) -> str:
        """Record a new order submission. Returns order_id."""
        order_id = str(uuid.uuid4())
        order = ManagedOrder(
            order_id=order_id,
            broker_order_id=broker_order_id,
            position_id=position_id,
            symbol=symbol,
            side=OrderSide(side.upper()),
            order_type=OrderType(order_type.upper()),
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            order_class=order_class,
            notes=notes,
        )
        self._save(order)
        logger.debug(f"OMS: submitted {order_class} order {order_id[:8]} {side} {quantity} {symbol}")
        return order_id

    def link_broker_id(self, order_id: str, broker_order_id: str) -> None:
        """Link the broker's order ID to our internal order ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE orders SET broker_order_id=?, updated_at=? WHERE order_id=?",
                (broker_order_id, _utcnow(), order_id),
            )
            conn.commit()

    # ── Status updates ────────────────────────────────────────────────────────

    def update_status(
        self,
        order_id:   str,
        status:     str,
        fill_price: Optional[float] = None,
        filled_qty: Optional[float] = None,
        reject_reason: Optional[str] = None,
    ) -> None:
        now = _utcnow()
        settled_at = now if status in (OrderStatus.SETTLED, OrderStatus.FILLED) else None

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE orders SET status=?, fill_price=COALESCE(?,fill_price), "
                "filled_qty=COALESCE(?,filled_qty), reject_reason=COALESCE(?,reject_reason), "
                "settled_at=COALESCE(?,settled_at), updated_at=? WHERE order_id=?",
                (status, fill_price, filled_qty, reject_reason, settled_at, now, order_id),
            )
            conn.commit()
        logger.debug(f"OMS: {order_id[:8]} → {status} fill={fill_price}")

    def record_fill(
        self,
        order_id:   str,
        fill_price: float,
        filled_qty: float,
    ) -> None:
        self.update_status(order_id, OrderStatus.FILLED, fill_price=fill_price, filled_qty=filled_qty)

    def record_rejection(self, order_id: str, reason: str) -> None:
        self.update_status(order_id, OrderStatus.REJECTED, reject_reason=reason)

    def record_cancellation(self, order_id: str) -> None:
        self.update_status(order_id, OrderStatus.CANCELLED)

    # ── Reconciliation ────────────────────────────────────────────────────────

    def reconcile(self, broker) -> List[str]:
        """
        Compare open orders against broker state.

        Returns list of discrepancy messages.
        """
        discrepancies = []
        open_orders = self.get_open_orders()

        for order in open_orders:
            if not order.broker_order_id:
                # Order was never confirmed by broker
                age = self._age_minutes(order.created_at)
                if age > STALE_THRESHOLD_MINUTES:
                    logger.warning(f"OMS: orphaned order {order.order_id[:8]} ({age:.0f}m old) — marking STALE")
                    self.update_status(order.order_id, OrderStatus.STALE)
                    discrepancies.append(
                        f"Orphaned order {order.order_id[:8]} {order.symbol} {order.order_class}"
                    )
            else:
                # Query broker for current status
                try:
                    broker_status = broker.get_order_status(order.broker_order_id)
                    if broker_status and broker_status != order.status:
                        self.update_status(order.order_id, broker_status)
                        discrepancies.append(
                            f"Order {order.order_id[:8]} status: local={order.status} broker={broker_status}"
                        )
                except Exception as e:
                    logger.debug(f"OMS: broker status check failed for {order.order_id[:8]}: {e}")

        return discrepancies

    def detect_stale(self) -> List[ManagedOrder]:
        """Return orders that are stale (open > STALE_THRESHOLD_MINUTES)."""
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=STALE_THRESHOLD_MINUTES)
        ).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM orders WHERE status IN ('SUBMITTED','ACCEPTED') AND created_at<?",
                (cutoff,),
            ).fetchall()
        stale = [self._row_to_order(r) for r in rows]
        for o in stale:
            self.update_status(o.order_id, OrderStatus.STALE)
        return stale

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_open_orders(self) -> List[ManagedOrder]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM orders WHERE status IN "
                "('SUBMITTED','ACCEPTED','PARTIALLY_FILLED','BRACKET_PENDING')"
            ).fetchall()
        return [self._row_to_order(r) for r in rows]

    def get_orders_for_position(self, position_id: str) -> List[ManagedOrder]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM orders WHERE position_id=? ORDER BY created_at",
                (position_id,),
            ).fetchall()
        return [self._row_to_order(r) for r in rows]

    def get_recent_fills(self, hours: int = 24) -> List[ManagedOrder]:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM orders WHERE status='FILLED' AND updated_at>=?",
                (cutoff,),
            ).fetchall()
        return [self._row_to_order(r) for r in rows]

    def stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*), "
                "SUM(CASE WHEN status='FILLED' THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN status='REJECTED' THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN status='CANCELLED' THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN status='STALE' THEN 1 ELSE 0 END) "
                "FROM orders"
            ).fetchone()
        return {
            'total':     row[0],
            'filled':    row[1],
            'rejected':  row[2],
            'cancelled': row[3],
            'stale':     row[4],
            'fill_rate': row[1] / row[0] if row[0] else 0.0,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, o: ManagedOrder) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO orders "
                "(order_id,broker_order_id,position_id,symbol,side,order_type,quantity,"
                "price,stop_price,fill_price,filled_qty,status,reject_reason,order_class,"
                "created_at,updated_at,settled_at,notes) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (o.order_id, o.broker_order_id, o.position_id, o.symbol, o.side.value,
                 o.order_type.value, o.quantity, o.price, o.stop_price, o.fill_price,
                 o.filled_qty, o.status.value, o.reject_reason, o.order_class,
                 o.created_at, o.updated_at, o.settled_at, o.notes),
            )
            conn.commit()

    @staticmethod
    def _row_to_order(row) -> ManagedOrder:
        r = dict(row)
        return ManagedOrder(
            order_id=r['order_id'],
            broker_order_id=r.get('broker_order_id'),
            position_id=r.get('position_id'),
            symbol=r['symbol'],
            side=OrderSide(r['side']),
            order_type=OrderType(r['order_type']),
            quantity=r['quantity'],
            price=r.get('price'),
            stop_price=r.get('stop_price'),
            fill_price=r.get('fill_price'),
            filled_qty=r.get('filled_qty', 0.0),
            status=OrderStatus(r['status']),
            reject_reason=r.get('reject_reason'),
            order_class=r.get('order_class', 'entry'),
            created_at=r['created_at'],
            updated_at=r['updated_at'],
            settled_at=r.get('settled_at'),
            notes=r.get('notes', ''),
        )

    @staticmethod
    def _age_minutes(created_at: str) -> float:
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            return (datetime.now(timezone.utc) - dt).total_seconds() / 60
        except Exception:
            return 0.0
