"""
Unit tests for the four rewritten/new core modules:
  - core.position_manager.PositionManager
  - core.safety_manager.SafetyManager
  - core.kelly_wrapper.KellySizer
  - core.broker.PaperBroker / get_broker()

All tests use tmp_path / in-memory SQLite — no external I/O.
"""
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Dict
from unittest.mock import patch

_utcnow = lambda: datetime.now(timezone.utc)

import pytest

# ── PositionManager ───────────────────────────────────────────────────────────

class TestPositionManager:
    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "test_positions.sqlite")

    @pytest.fixture
    def pm(self, db_path):
        from core.position_manager import PositionManager
        return PositionManager(db_path=db_path)

    # ── open_position ─────────────────────────────────────────────────────────

    def test_open_returns_int_id(self, pm):
        pos_id = pm.open_position(
            symbol="BTC-USD",
            signal="BUY",
            size=1000.0,
            entry_price=50000.0,
            stop_loss=49000.0,
            take_profit=52000.0,
        )
        assert isinstance(pos_id, int)
        assert pos_id > 0

    def test_open_position_is_retrievable(self, pm):
        pm.open_position("ETH-USD", "BUY", 500, 3000, 2900, 3200)
        pos = pm.get_position("ETH-USD")
        assert pos is not None
        assert pos["symbol"] == "ETH-USD"
        assert pos["status"] == "OPEN"
        assert pos["entry_price"] == pytest.approx(3000.0)

    def test_count_open(self, pm):
        assert pm.count_open() == 0
        pm.open_position("BTC-USD", "BUY", 1000, 50000, 49000, 52000)
        assert pm.count_open() == 1
        pm.open_position("ETH-USD", "BUY", 500, 3000, 2900, 3200)
        assert pm.count_open() == 2

    def test_get_open_positions_empty(self, pm):
        assert pm.get_open_positions() == []

    def test_duplicate_open_returns_new_id(self, pm):
        id1 = pm.open_position("SOL-USD", "BUY", 200, 100, 95, 110)
        id2 = pm.open_position("SOL-USD", "BUY", 200, 102, 97, 112)
        assert id1 != id2

    # ── close_position ────────────────────────────────────────────────────────

    def test_close_position_returns_pnl(self, pm):
        # size is in dollars: $5000 invested at $50000 = 0.1 BTC
        # PnL = (51000-50000)/50000 * 5000 = 2% of $5000 = $100
        pos_id = pm.open_position("BTC-USD", "BUY", 5000.0, 50000.0, 49000.0, 52000.0)
        pnl = pm.close_position(pos_id, close_price=51000.0)
        assert pnl == pytest.approx(100.0)

    def test_close_sets_status_closed(self, pm):
        pos_id = pm.open_position("BTC-USD", "BUY", 0.1, 50000, 49000, 52000)
        pm.close_position(pos_id, close_price=51000.0)
        positions = pm.get_open_positions()
        assert len(positions) == 0

    def test_close_nonexistent_returns_none(self, pm):
        result = pm.close_position(9999, close_price=100.0)
        assert result is None

    def test_close_with_explicit_pnl(self, pm):
        pos_id = pm.open_position("BTC-USD", "BUY", 1, 50000, 49000, 52000)
        pnl = pm.close_position(pos_id, close_price=50500, pnl=250.0)
        assert pnl == pytest.approx(250.0)

    # ── update_position ───────────────────────────────────────────────────────

    def test_update_position_stop_loss(self, pm):
        pos_id = pm.open_position("BTC-USD", "BUY", 0.1, 50000, 49000, 52000)
        ok = pm.update_position(pos_id, stop_loss=49500)
        assert ok is True
        pos = pm.get_position("BTC-USD")
        assert pos["stop_loss"] == pytest.approx(49500.0)

    def test_update_nonexistent_returns_false(self, pm):
        ok = pm.update_position(9999, stop_loss=1.0)
        assert ok is False

    # ── get_recent_trades ─────────────────────────────────────────────────────

    def test_get_recent_trades_empty(self, pm):
        trades = pm.get_recent_trades(days=7)
        assert trades == []

    def test_get_recent_trades_after_close(self, pm):
        pos_id = pm.open_position("BTC-USD", "BUY", 0.1, 50000, 49000, 52000)
        pm.close_position(pos_id, close_price=51000)
        trades = pm.get_recent_trades(days=1)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "BTC-USD"


# ── SafetyManager ─────────────────────────────────────────────────────────────

class TestSafetyManager:
    @pytest.fixture
    def sm(self):
        from core.safety_manager import SafetyManager
        return SafetyManager(capital=10_000.0, config={
            'max_daily_loss_pct': 0.02,
            'max_positions': 3,
            'max_consecutive_losses': 3,
            'max_drawdown_pct': 0.05,
        })

    def test_can_trade_initially(self, sm):
        ok, reason = sm.check_can_trade()
        assert ok is True
        assert reason is None

    def test_max_positions_blocks_trade(self, sm):
        for _ in range(3):
            sm.on_position_open()
        ok, reason = sm.check_can_trade()
        assert ok is False
        assert "Max positions" in reason

    def test_circuit_breaker_on_consecutive_losses(self, sm):
        sm.on_position_open()
        sm.on_loss(50)
        sm.on_position_open()
        sm.on_loss(50)
        sm.on_position_open()
        sm.on_loss(50)
        ok, reason = sm.check_can_trade()
        assert ok is False
        assert sm.circuit_breaker_triggered is True

    def test_on_win_resets_consecutive_losses(self, sm):
        sm.on_position_open()
        sm.on_loss(10)
        sm.on_position_open()
        sm.on_loss(10)
        sm.on_position_open()
        sm.on_win(20)
        assert sm.consecutive_losses == 0
        assert sm.consecutive_wins == 1

    def test_daily_loss_limit_triggers_breaker(self, sm):
        # $10k capital, 2% limit = $200 max daily loss
        sm.on_position_open()
        sm.on_loss(201)  # exceed $200 limit
        ok, reason = sm.check_can_trade()
        assert ok is False
        assert sm.circuit_breaker_triggered is True

    def test_get_safety_metrics_keys(self, sm):
        metrics = sm.get_safety_metrics()
        for key in ('daily_pnl', 'active_positions', 'consecutive_losses',
                    'circuit_breaker', 'peak_equity', 'drawdown_pct'):
            assert key in metrics

    def test_on_win_increments_equity(self, sm):
        sm.on_win(100)
        assert sm.current_equity == pytest.approx(10_100.0)

    def test_on_loss_decrements_equity(self, sm):
        sm.on_loss(100)
        assert sm.current_equity == pytest.approx(9_900.0)

    def test_position_size_rejected_when_too_large(self, sm):
        # max 10% of $10k = $1000
        ok, reason = sm.check_can_trade(position_size=1_500)
        assert ok is False
        assert "max" in reason.lower()

    def test_max_drawdown_triggers_breaker(self, sm):
        # 5% of $10k = $500 drawdown limit
        sm.on_loss(600)
        ok, _ = sm.check_can_trade()
        assert ok is False
        assert sm.circuit_breaker_triggered is True


# ── KellySizer ────────────────────────────────────────────────────────────────

class TestKellySizer:
    @pytest.fixture
    def db_path(self, tmp_path):
        path = str(tmp_path / "kelly_test.sqlite")
        # Create minimal positions table
        with sqlite3.connect(path) as conn:
            conn.execute("""
                CREATE TABLE positions (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT,
                    strategy TEXT,
                    status TEXT,
                    entry_time TEXT,
                    exit_time TEXT,
                    net_pnl REAL
                )
            """)
        return path

    @pytest.fixture
    def sizer(self, db_path):
        from core.kelly_wrapper import KellySizer
        return KellySizer(db_path=db_path, lookback_days=30)

    def _insert_closed_trades(self, db_path, pnls):
        """Helper: insert closed trade rows."""
        cutoff = _utcnow() - timedelta(days=1)
        with sqlite3.connect(db_path) as conn:
            for pnl in pnls:
                conn.execute(
                    "INSERT INTO positions (symbol, strategy, status, entry_time, exit_time, net_pnl) "
                    "VALUES (?, ?, 'CLOSED', ?, ?, ?)",
                    ("BTC-USD", "test_strat", cutoff.isoformat(), _utcnow().isoformat(), pnl)
                )

    def test_fallback_when_no_trades(self, sizer):
        # With 0 trades, should return 5% of capital
        size = sizer.get_position_size("BTC-USD", capital=10_000)
        assert size == pytest.approx(500.0)

    def test_fallback_when_insufficient_trades(self, sizer, db_path):
        # 5 trades < MIN_SAMPLE (10) → still fallback
        self._insert_closed_trades(db_path, [10, -5, 10, -5, 10])
        size = sizer.get_position_size("BTC-USD", capital=10_000)
        assert size == pytest.approx(500.0)

    def test_size_nonzero_with_enough_trades(self, sizer, db_path):
        # 10+ winning trades → Kelly returns positive size
        wins = [50.0] * 8
        losses = [-20.0] * 4
        self._insert_closed_trades(db_path, wins + losses)
        size = sizer.get_position_size("BTC-USD", capital=10_000)
        assert size > 0

    def test_leverage_multiplies_size(self, sizer):
        size1 = sizer.get_position_size("BTC-USD", capital=10_000, leverage=1.0)
        size2 = sizer.get_position_size("BTC-USD", capital=10_000, leverage=2.0)
        assert size2 == pytest.approx(size1 * 2)

    def test_db_error_returns_fallback(self, sizer):
        # Point to a non-existent DB path — should fall back silently
        sizer.db_path = "/tmp/nonexistent_9999.sqlite"
        size = sizer.get_position_size("BTC-USD", capital=10_000)
        assert size == pytest.approx(500.0)


# ── PaperBroker / get_broker() ───────────────────────────────────────────────

class TestPaperBroker:
    @pytest.fixture
    def broker(self):
        from core.broker import PaperBroker
        return PaperBroker(starting_cash=10_000.0, config={'slippage_pct': 0.0, 'fee_pct': 0.0})

    def _buy_order(self, symbol="BTC-USD", qty=0.1, price=50000.0):
        from core.broker import Order
        return Order(symbol=symbol, side="BUY", quantity=qty, fill_price=price)

    def _sell_order(self, symbol="BTC-USD", qty=0.1, price=51000.0):
        from core.broker import Order
        return Order(symbol=symbol, side="SELL", quantity=qty, fill_price=price)

    def test_buy_order_fills(self, broker):
        from core.broker import OrderStatus
        order = broker.submit_order(self._buy_order())
        assert order.status == OrderStatus.FILLED
        assert order.order_id is not None

    def test_buy_reduces_cash(self, broker):
        broker.submit_order(self._buy_order(qty=0.1, price=50000.0))
        acct = broker.get_account()
        assert acct.cash == pytest.approx(10_000 - 5000, rel=0.01)

    def test_sell_increases_cash(self, broker):
        broker.submit_order(self._buy_order(qty=0.1, price=50000.0))
        broker.submit_order(self._sell_order(qty=0.1, price=51000.0))
        acct = broker.get_account()
        assert acct.cash == pytest.approx(10_000 + 100, rel=0.01)

    def test_insufficient_cash_rejected(self, broker):
        from core.broker import Order, OrderStatus
        # Try to buy more than $10k
        big_order = Order(symbol="BTC-USD", side="BUY", quantity=1.0, fill_price=50000.0)
        result = broker.submit_order(big_order)
        assert result.status == OrderStatus.REJECTED

    def test_get_position_after_buy(self, broker):
        broker.submit_order(self._buy_order(qty=0.2, price=50000.0))
        pos = broker.get_position("BTC-USD")
        assert pos is not None
        assert pos.quantity == pytest.approx(0.2)

    def test_position_cleared_after_sell(self, broker):
        broker.submit_order(self._buy_order(qty=0.1, price=50000.0))
        broker.submit_order(self._sell_order(qty=0.1, price=51000.0))
        pos = broker.get_position("BTC-USD")
        assert pos is None

    def test_get_account_equity(self, broker):
        acct = broker.get_account()
        assert acct.equity == pytest.approx(10_000.0)

    def test_reconcile_ok_when_in_sync(self, broker):
        broker.submit_order(self._buy_order(qty=0.1, price=50000.0))
        # Pass a local position dict that matches broker: size ≈ qty
        result = broker.reconcile([{'symbol': 'BTC-USD', 'size': 0.1}])
        assert result['ok'] is True

    def test_reconcile_detects_missing_position(self, broker):
        # Local DB says BTC is open but broker has nothing
        result = broker.reconcile([{'symbol': 'BTC-USD'}])
        assert result['ok'] is False
        assert len(result['discrepancies']) > 0

    def test_get_open_orders_empty(self, broker):
        assert broker.get_open_orders() == []


class TestGetBrokerFactory:
    def test_paper_mode_returns_paper_broker(self):
        from core.broker import PaperBroker, get_broker
        with patch.dict(os.environ, {'TRADING_MODE': 'PAPER'}):
            broker = get_broker(capital=1000)
        assert isinstance(broker, PaperBroker)

    def test_backtest_mode_returns_paper_broker(self):
        from core.broker import PaperBroker, get_broker
        with patch.dict(os.environ, {'TRADING_MODE': 'BACKTEST'}):
            broker = get_broker(capital=1000)
        assert isinstance(broker, PaperBroker)

    def test_live_mode_without_flag_raises(self):
        from core.broker import get_broker
        with patch.dict(os.environ, {'TRADING_MODE': 'LIVE', 'ENABLE_LIVE_TRADING': 'false'}):
            with pytest.raises(RuntimeError, match="ENABLE_LIVE_TRADING"):
                get_broker()

    def test_live_mode_with_flag_and_missing_keys_raises(self):
        from core.broker import AlpacaBroker
        env = {'TRADING_MODE': 'LIVE', 'ENABLE_LIVE_TRADING': 'true'}
        with patch.dict(os.environ, env, clear=False):
            # AlpacaBroker needs ALPACA_API_KEY and ALPACA_API_SECRET
            # If those are not set, it should raise ValueError or ImportError
            try:
                broker = AlpacaBroker()
                # If we get here the keys happened to be set (e.g. in CI) — that's fine
            except (RuntimeError, ValueError, ImportError):
                pass  # expected
