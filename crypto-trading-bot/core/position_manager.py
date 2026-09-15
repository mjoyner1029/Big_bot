"""
Position lifecycle management.

Canonical API used by all bot versions. Every caller (v2, v3) must use these
signatures. Do not call SQLite directly from strategy or bot code.
"""
import sqlite3
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict

_utcnow = lambda: datetime.now(timezone.utc)

logger = logging.getLogger(__name__)

_CREATE_POSITIONS = """
CREATE TABLE IF NOT EXISTS positions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol              TEXT    NOT NULL,
    asset_class         TEXT    DEFAULT 'crypto',
    strategy            TEXT,
    strategy_version    TEXT,
    strategies_used     TEXT,
    signal              TEXT    NOT NULL,
    side                TEXT,
    direction           TEXT,
    size                REAL    NOT NULL,
    entry_price         REAL    NOT NULL,
    entry_fill_price    REAL,
    entry_time          TIMESTAMP NOT NULL,
    stop_loss           REAL,
    take_profit         REAL,
    max_price           REAL,
    exit_price          REAL,
    exit_fill_price     REAL,
    exit_time           TIMESTAMP,
    pnl                 REAL,
    fees                REAL    DEFAULT 0.0,
    entry_fees          REAL    DEFAULT 0.0,
    exit_fees           REAL    DEFAULT 0.0,
    net_pnl             REAL,
    regime              TEXT,
    confidence          REAL    DEFAULT 0.0,
    kronos_confidence   REAL    DEFAULT 0.0,
    llm_decision        TEXT,
    close_reason        TEXT,
    broker_order_id     TEXT,
    broker_exit_order_id TEXT,
    broker_position_id  TEXT,
    status              TEXT    DEFAULT 'OPEN',
    partial_closed      INTEGER DEFAULT 0,
    mfe_pct             REAL    DEFAULT 0.0,
    mae_pct             REAL    DEFAULT 0.0,
    features_json       TEXT,
    ml_prediction       TEXT,
    holding_hours       REAL
)
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_pos_status   ON positions(status)",
    "CREATE INDEX IF NOT EXISTS idx_pos_symbol   ON positions(symbol)",
    "CREATE INDEX IF NOT EXISTS idx_pos_strategy ON positions(strategy)",
    "CREATE INDEX IF NOT EXISTS idx_pos_entry    ON positions(entry_time)",
]


class PositionManager:
    """Tracks the full lifecycle of every position/trade.

    Architectural boundary: this is the ONLY place that reads/writes the
    positions table. All P&L arithmetic is done here for consistency.
    """

    def __init__(self, db_path: str = 'data/trade_memory.sqlite'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_POSITIONS)
            self._migrate(conn)  # Run migration BEFORE creating indexes
            for idx in _INDEXES:
                conn.execute(idx)
            conn.commit()
        logger.info(f"PositionManager ready: {self.db_path} ({self.count_open()} open)")

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add columns that may be missing in older DBs."""
        cursor = conn.execute("PRAGMA table_info(positions)")
        existing = {row[1] for row in cursor.fetchall()}
        additions = {
            'signal':                "TEXT",
            'asset_class':           "TEXT DEFAULT 'crypto'",
            'strategy':              "TEXT",
            'strategy_version':      "TEXT",
            'strategies_used':       "TEXT",
            'direction':             "TEXT",
            'stop_loss':             "REAL",
            'take_profit':           "REAL",
            'max_price':             "REAL",
            'close_reason':          "TEXT",
            'fees':                  "REAL DEFAULT 0.0",
            'entry_fees':            "REAL DEFAULT 0.0",
            'exit_fees':             "REAL DEFAULT 0.0",
            'net_pnl':               "REAL",
            'regime':                "TEXT",
            'confidence':            "REAL DEFAULT 0.0",
            'kronos_confidence':     "REAL DEFAULT 0.0",
            'llm_decision':          "TEXT",
            'broker_order_id':       "TEXT",
            'broker_exit_order_id':  "TEXT",
            'broker_position_id':    "TEXT",
            'partial_closed':        "INTEGER DEFAULT 0",
            'side':                  "TEXT",
            'entry_fill_price':      "REAL",
            'exit_fill_price':       "REAL",
            'mfe_pct':               "REAL DEFAULT 0.0",
            'mae_pct':               "REAL DEFAULT 0.0",
            'features_json':         "TEXT",
            'ml_prediction':         "TEXT",
            'holding_hours':         "REAL",
        }
        for col, defn in additions.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE positions ADD COLUMN {col} {defn}")

    # ── Write API ─────────────────────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        signal: str,
        size: float,
        entry_price: float,
        stop_loss: float = None,
        take_profit: float = None,
        direction: str = None,
        strategy: str = None,
        strategy_version: str = None,
        strategies_used: list = None,
        kronos_confidence: float = 0.0,
        confidence: float = 0.0,
        regime: str = None,
        llm_decision: str = None,
        asset_class: str = 'crypto',
        broker_order_id: str = None,
        entry_fees: float = 0.0,
        entry_fill_price: float = None,
        features_json: str = None,
        ml_prediction: str = None,
    ) -> int:
        """Open a new position. Returns position ID."""
        if direction is None:
            direction = 'LONG' if signal in ('BUY', 'LONG') else 'SHORT'
        strats_json = json.dumps(strategies_used) if strategies_used else None
        actual_fill = entry_fill_price if entry_fill_price is not None else entry_price

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO positions
                   (symbol, asset_class, strategy, strategy_version, strategies_used,
                    signal, side, direction, size, entry_price, entry_fill_price, entry_time,
                    stop_loss, take_profit, max_price, regime, confidence,
                    kronos_confidence, llm_decision, broker_order_id, entry_fees,
                    features_json, ml_prediction, status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'OPEN')""",
                (symbol, asset_class, strategy, strategy_version, strats_json,
                 signal, signal, direction, size, entry_price, actual_fill, _utcnow(),
                 stop_loss, take_profit, entry_price, regime, confidence,
                 kronos_confidence, llm_decision, broker_order_id, entry_fees,
                 features_json, ml_prediction),
            )
            conn.commit()
            pos_id = cur.lastrowid

        logger.info(
            f"OPEN #{pos_id} {direction} {symbol} @ ${entry_price:.4f} "
            f"size=${size:.2f} strategy={strategy}"
        )
        return pos_id

    def update_position(self, position_id: int, **kwargs) -> bool:
        """Update one or more fields on an open position."""
        allowed = {
            'stop_loss', 'take_profit', 'max_price', 'size',
            'partial_closed', 'broker_position_id', 'regime', 'confidence',
            'mfe_pct', 'mae_pct', 'ml_prediction',
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        set_clause = ', '.join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [position_id]
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                f"UPDATE positions SET {set_clause} WHERE id=? AND status='OPEN'",
                values,
            )
            conn.commit()
        return cur.rowcount > 0

    def close_position(
        self,
        position_id: int,
        close_price: float,
        pnl: float = None,
        reason: str = 'manual',
        fees: float = 0.0,
        broker_order_id: str = None,
        exit_fill_price: float = None,
        exit_fees: float = None,
    ) -> Optional[float]:
        """
        Close a position by ID. Returns realized net P&L or None.

        PnL is computed from actual broker fill prices when provided:
            pnl = (exit_fill_price - entry_fill_price) / entry_fill_price * size
        """
        row = self._get_by_id(position_id)
        if not row:
            logger.warning(f"close_position: ID {position_id} not found / not open")
            return None

        # Use actual fill prices when available
        actual_exit  = exit_fill_price if exit_fill_price is not None else close_price
        actual_entry = row.get('entry_fill_price') or row['entry_price']
        actual_fees  = exit_fees if exit_fees is not None else fees
        size         = row['size']

        if pnl is None:
            if row['direction'] == 'LONG':
                pnl = (actual_exit - actual_entry) / actual_entry * size
            else:
                pnl = (actual_entry - actual_exit) / actual_entry * size

        total_fees = actual_fees + (row.get('entry_fees') or 0.0)
        net_pnl    = pnl - actual_fees   # entry fees already subtracted at open

        entry_dt = row.get('entry_time')
        if isinstance(entry_dt, str):
            try:
                entry_dt = datetime.fromisoformat(entry_dt)
            except ValueError:
                entry_dt = None
        holding_hours = None
        if entry_dt:
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=timezone.utc)
            holding_hours = (_utcnow() - entry_dt).total_seconds() / 3600

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE positions
                   SET exit_price=?, exit_fill_price=?, exit_time=?,
                       pnl=?, exit_fees=?, fees=?, net_pnl=?,
                       close_reason=?, holding_hours=?, status='CLOSED',
                       broker_exit_order_id=COALESCE(?,broker_exit_order_id)
                   WHERE id=?""",
                (close_price, actual_exit, _utcnow(),
                 pnl, actual_fees, total_fees, net_pnl,
                 reason, holding_hours, broker_order_id, position_id),
            )
            conn.commit()

        logger.info(
            f"CLOSE #{position_id} {row['symbol']} @ ${actual_exit:.4f} "
            f"pnl=${pnl:+.2f} net=${net_pnl:+.2f} reason={reason}"
        )
        return net_pnl

    def partial_close(
        self,
        position_id: int,
        fraction: float,
        close_price: float,
        fees: float = 0.0,
        exit_fill_price: float = None,
        exit_fees: float = None,
        broker_order_id: str = None,
    ) -> Optional[float]:
        """
        Close a fraction of a position (e.g. 0.5 = 50%). Returns realized P&L.

        Uses actual broker fill price when provided.
        Reduces local size; does NOT mark position CLOSED.
        """
        row = self._get_by_id(position_id)
        if not row:
            return None

        actual_exit  = exit_fill_price if exit_fill_price is not None else close_price
        actual_entry = row.get('entry_fill_price') or row['entry_price']
        actual_fees  = exit_fees if exit_fees is not None else fees

        close_size = row['size'] * fraction
        remaining  = row['size'] * (1.0 - fraction)

        if row['direction'] == 'LONG':
            pnl = (actual_exit - actual_entry) / actual_entry * close_size
        else:
            pnl = (actual_entry - actual_exit) / actual_entry * close_size
        net_pnl = pnl - actual_fees

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE positions SET size=?, partial_closed=1 WHERE id=?",
                (remaining, position_id),
            )
            conn.commit()

        logger.info(
            f"PARTIAL_CLOSE #{position_id} {fraction:.0%} @ ${actual_exit:.4f} "
            f"pnl=${pnl:+.2f} net=${net_pnl:+.2f} remaining=${remaining:.2f}"
        )
        return net_pnl

    # ── Read API ──────────────────────────────────────────────────────────────

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Most recent open position for symbol."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """SELECT * FROM positions WHERE symbol=? AND status='OPEN'
                   ORDER BY entry_time DESC LIMIT 1""",
                (symbol,),
            ).fetchone()
        return dict(row) if row else None

    def get_open_positions(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM positions WHERE status='OPEN' ORDER BY entry_time DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def count_open(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM positions WHERE status='OPEN'"
            ).fetchone()[0]

    def get_recent_trades(self, days: int = 7, hours: int = None) -> List[Dict]:
        """Closed trades within the last N days (or hours)."""
        if hours is not None:
            cutoff = _utcnow() - timedelta(hours=hours)
        else:
            cutoff = _utcnow() - timedelta(days=days)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT * FROM positions WHERE status='CLOSED' AND exit_time>=?
                   ORDER BY exit_time DESC""",
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_strategy_stats(self, strategy: str = None, days: int = 30) -> Dict:
        """Win rate and expectancy, optionally filtered to one strategy."""
        cutoff = _utcnow() - timedelta(days=days)
        with sqlite3.connect(self.db_path) as conn:
            if strategy:
                rows = conn.execute(
                    """SELECT net_pnl FROM positions
                       WHERE status='CLOSED' AND strategy=? AND exit_time>=?""",
                    (strategy, cutoff),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT net_pnl FROM positions WHERE status='CLOSED' AND exit_time>=?",
                    (cutoff,),
                ).fetchall()

        pnls = [r[0] for r in rows if r[0] is not None]
        if not pnls:
            return {'trades': 0, 'win_rate': 0.0, 'expectancy': 0.0,
                    'avg_win': 0.0, 'avg_loss': 0.0, 'total_pnl': 0.0}
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        return {
            'trades': len(pnls),
            'win_rate': len(wins) / len(pnls),
            'expectancy': sum(pnls) / len(pnls),
            'avg_win': sum(wins) / len(wins) if wins else 0.0,
            'avg_loss': sum(losses) / len(losses) if losses else 0.0,
            'total_pnl': sum(pnls),
        }

    def _get_by_id(self, position_id: int) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM positions WHERE id=? AND status='OPEN'",
                (position_id,),
            ).fetchone()
        return dict(row) if row else None
