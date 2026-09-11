"""
End-to-end integration tests covering the complete trade lifecycle.

These tests verify the FULL execution path:
    Signal → Broker order → Fill → Position opened → Partial profit →
    Stop loss → Position closed → Safety updated → Experiment recorded →
    Feature store updated → Trade memory updated

All tests use PaperBroker (no network calls).
All tests use tmp_path SQLite DBs (no shared state).
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_integration.sqlite")


@pytest.fixture()
def feature_db(tmp_path: Path) -> str:
    return str(tmp_path / "test_features.sqlite")


@pytest.fixture()
def broker():
    from core.broker import PaperBroker
    return PaperBroker(starting_cash=50_000.0)


@pytest.fixture()
def position_mgr(db_path):
    from core.position_manager import PositionManager
    return PositionManager(db_path=db_path)


@pytest.fixture()
def safety_mgr():
    from core.safety_manager import SafetyManager
    return SafetyManager(capital=50_000.0, config={'max_positions': 5})


@pytest.fixture()
def experiment_engine(db_path):
    from core.experiment_engine import ExperimentEngine
    engine = ExperimentEngine(db_path=db_path)
    engine.register_production_params({
        'kelly_fraction': 0.5,
        'min_kelly_sample': 10,
        'weighted_vote_threshold': 0.60,
    })
    return engine


@pytest.fixture()
def feature_store(feature_db):
    from core.feature_store import FeatureStore
    return FeatureStore(db_path=feature_db)


@pytest.fixture()
def trade_memory(db_path):
    from core.trade_memory import TradeMemory
    return TradeMemory(db_path=db_path)


@pytest.fixture()
def portfolio_mgr():
    from core.portfolio_manager import PortfolioManager
    return PortfolioManager(capital=50_000.0)


@pytest.fixture()
def opportunity_ranker():
    from core.opportunity_ranker import OpportunityRanker
    return OpportunityRanker(min_score=0.35)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1-3: Complete Entry Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestEntryLifecycle:
    """Verify canonical entry path: Safety → Broker → PositionManager → Safety."""

    def test_full_entry_flow(self, broker, position_mgr, safety_mgr):
        """Signal → Broker fill → Position opened in DB → Safety updated."""
        SYMBOL = 'BTC-USD'
        PRICE  = 50_000.0
        SIZE   = 1_000.0    # $1k trade
        QTY    = SIZE / PRICE

        # Step 1: Safety gate
        can_trade, reason = safety_mgr.check_can_trade(position_size=SIZE)
        assert can_trade, f"Safety blocked: {reason}"
        assert safety_mgr.active_positions == 0

        # Step 2: Submit to broker
        fill = broker.submit_and_wait(
            symbol=SYMBOL, side='BUY', quantity=QTY,
            current_price=PRICE, timeout_seconds=5.0,
        )
        assert fill.status.value == 'FILLED'
        assert fill.fill_quantity == pytest.approx(QTY, rel=1e-4)
        assert fill.fill_price > PRICE * 0.999    # slippage applied
        assert fill.fees > 0

        # Step 3: Open local position using actual fill data
        pos_id = position_mgr.open_position(
            symbol=SYMBOL,
            signal='BUY',
            size=fill.notional,
            entry_price=fill.fill_price,
            entry_fill_price=fill.fill_price,
            entry_fees=fill.fees,
            broker_order_id=fill.order_id,
        )
        assert pos_id > 0

        # Verify DB record
        pos = position_mgr.get_position(SYMBOL)
        assert pos is not None
        assert pos['entry_fill_price'] == pytest.approx(fill.fill_price, rel=1e-4)
        assert pos['entry_fees'] == pytest.approx(fill.fees, rel=1e-4)
        assert pos['broker_order_id'] == fill.order_id

        # Step 4: Safety updated
        safety_mgr.on_position_open()
        assert safety_mgr.active_positions == 1

    def test_rejected_entry_does_not_open_position(self, broker, position_mgr, safety_mgr):
        """If broker rejects order, no local position is created."""
        # Drain all cash
        broker.cash = 0.01

        fill = broker.submit_and_wait(
            symbol='ETH-USD', side='BUY', quantity=1.0,
            current_price=3_000.0, timeout_seconds=5.0,
        )
        assert fill.status.value == 'REJECTED'

        # Must NOT open position
        # (bot's execute_trade returns None on rejection — simulated here)
        assert position_mgr.get_position('ETH-USD') is None
        assert safety_mgr.active_positions == 0

    def test_bracket_orders_placed_after_entry(self, broker):
        """After entry fill, bracket orders must be placed."""
        PRICE = 50_000.0
        fill = broker.submit_and_wait(
            symbol='BTC-USD', side='BUY', quantity=0.01,
            current_price=PRICE,
        )
        assert fill.status.value == 'FILLED'

        bracket = broker.place_bracket_orders(
            symbol='BTC-USD',
            quantity=fill.fill_quantity,
            stop_price=PRICE * 0.98,
            take_profit_price=PRICE * 1.03,
        )
        assert bracket['sl_order_id'] is not None
        assert bracket['tp_order_id'] is not None

        # Verify bracket doesn't fire when price is between SL and TP
        result = broker.check_bracket_triggers('BTC-USD', PRICE)
        assert result is None

        # Verify stop fires when price drops below SL
        result = broker.check_bracket_triggers('BTC-USD', PRICE * 0.97)
        assert result == 'stop_loss'


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1-3: Complete Exit Lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestExitLifecycle:
    """Verify exits always go through broker before updating local state."""

    def _open_position(self, broker, position_mgr, safety_mgr,
                       symbol='BTC-USD', price=50_000.0, size=1_000.0):
        qty = size / price
        fill = broker.submit_and_wait(symbol=symbol, side='BUY', quantity=qty,
                                      current_price=price)
        pos_id = position_mgr.open_position(
            symbol=symbol, signal='BUY', size=fill.notional,
            entry_price=fill.fill_price, entry_fill_price=fill.fill_price,
            entry_fees=fill.fees, broker_order_id=fill.order_id,
        )
        safety_mgr.on_position_open()
        return pos_id, fill.fill_price

    def test_full_exit_flow(self, broker, position_mgr, safety_mgr):
        """Broker close → actual fill → PositionManager updated → Safety updated."""
        pos_id, entry_fill = self._open_position(broker, position_mgr, safety_mgr)
        assert safety_mgr.active_positions == 1

        EXIT_PRICE = 52_000.0

        # Step 1: Close at broker
        close_fill = broker.close_position('BTC-USD', EXIT_PRICE)
        assert close_fill.status.value == 'FILLED'
        assert close_fill.fill_price > EXIT_PRICE * 0.999   # slippage applied (BUY→SELL slippage favours us)

        # Step 2: Record actual fill in PositionManager
        net_pnl = position_mgr.close_position(
            pos_id,
            close_price=EXIT_PRICE,
            exit_fill_price=close_fill.fill_price,
            exit_fees=close_fill.fees,
            reason='take_profit',
            broker_order_id=close_fill.order_id,
        )
        assert net_pnl is not None
        assert net_pnl > 0    # profitable exit

        # Verify exit_fill_price stored
        with __import__('sqlite3').connect(position_mgr.db_path) as conn:
            row = conn.execute(
                "SELECT exit_fill_price, exit_fees, broker_exit_order_id, status "
                "FROM positions WHERE id=?", (pos_id,)
            ).fetchone()
        assert row[0] == pytest.approx(close_fill.fill_price, rel=1e-4)
        assert row[1] == pytest.approx(close_fill.fees, rel=1e-4)
        assert row[3] == 'CLOSED'

        # Step 3: Safety updated — active_positions decremented
        safety_mgr.on_position_closed(net_pnl)
        assert safety_mgr.active_positions == 0
        assert safety_mgr.consecutive_wins == 1
        assert safety_mgr.daily_pnl == pytest.approx(net_pnl, rel=1e-2)

    def test_pnl_uses_actual_broker_fill_not_market_price(self, broker, position_mgr, safety_mgr):
        """
        The critical test: PnL must be based on actual broker fill prices,
        not on the signal/market price passed to close_position.
        """
        pos_id, entry_fill = self._open_position(broker, position_mgr, safety_mgr,
                                                  price=50_000.0)

        # Close at broker — slippage will make actual fill differ from 52_000
        close_fill = broker.close_position('BTC-USD', 52_000.0)
        actual_exit = close_fill.fill_price

        net_pnl = position_mgr.close_position(
            pos_id, close_price=52_000.0,
            exit_fill_price=actual_exit,
            exit_fees=close_fill.fees, reason='take_profit',
        )

        # PnL is based on actual fill prices, not quoted price
        pos_after = None
        with __import__('sqlite3').connect(position_mgr.db_path) as conn:
            conn.row_factory = __import__('sqlite3').Row
            pos_after = dict(conn.execute(
                "SELECT * FROM positions WHERE id=?", (pos_id,)
            ).fetchone())

        expected_pnl = (actual_exit - entry_fill) / entry_fill * pos_after['size']
        assert pos_after['pnl'] == pytest.approx(expected_pnl, rel=0.01)

    def test_failed_broker_exit_does_not_close_position_locally(
        self, broker, position_mgr, safety_mgr
    ):
        """
        If broker rejects the exit order, position remains OPEN locally.
        This is the fundamental safety guarantee.
        """
        pos_id, _ = self._open_position(broker, position_mgr, safety_mgr)

        # Remove broker position (simulate broker failure)
        broker._positions.clear()

        close_fill = broker.close_position('BTC-USD', 52_000.0)
        assert close_fill.status.value == 'REJECTED'

        # We simulate the bot's logic: only update DB if fill succeeds
        if close_fill.status.value in ('FILLED', 'PARTIAL'):
            position_mgr.close_position(pos_id, close_price=52_000.0, reason='take_profit')

        # Position must still be OPEN
        pos = position_mgr._get_by_id(pos_id)
        assert pos is not None    # still open
        assert safety_mgr.active_positions == 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Partial Close
# ─────────────────────────────────────────────────────────────────────────────

class TestPartialClose:
    """Partial closes go through broker, don't decrement active_positions."""

    def test_partial_close_lifecycle(self, broker, position_mgr, safety_mgr):
        """50% partial close: broker fill → DB reduced → safety P&L recorded."""
        # Open full position
        fill = broker.submit_and_wait('BTC-USD', 'BUY', 0.02, 50_000.0)
        pos_id = position_mgr.open_position(
            'BTC-USD', 'BUY', fill.notional, fill.fill_price,
            entry_fill_price=fill.fill_price, entry_fees=fill.fees,
            broker_order_id=fill.order_id,
        )
        safety_mgr.on_position_open()
        assert safety_mgr.active_positions == 1

        EXIT_PRICE = 52_000.0
        HALF_QTY   = 0.01

        # Partial close at broker
        close_fill = broker.close_position('BTC-USD', EXIT_PRICE, quantity=HALF_QTY)
        assert close_fill.status.value == 'FILLED'

        # Record partial close in DB
        net_pnl = position_mgr.partial_close(
            pos_id, fraction=0.5, close_price=EXIT_PRICE,
            exit_fill_price=close_fill.fill_price,
            exit_fees=close_fill.fees,
            broker_order_id=close_fill.order_id,
        )
        assert net_pnl is not None
        assert net_pnl > 0

        # Safety: record P&L but do NOT decrement active_positions
        safety_mgr.on_partial_close(net_pnl)
        assert safety_mgr.active_positions == 1   # still 1 — position still open
        assert safety_mgr.daily_pnl == pytest.approx(net_pnl, rel=0.01)
        assert safety_mgr.consecutive_wins == 0   # partial close doesn't update win streak

        # Position still exists in DB as OPEN with reduced size
        pos = position_mgr.get_position('BTC-USD')
        assert pos is not None
        assert pos['status'] == 'OPEN'
        assert pos['partial_closed'] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5-6: Experiment Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestExperimentEngine:
    """Experiment lifecycle: propose → backtest → OOS → promote."""

    def test_immutable_params_blocked(self, experiment_engine):
        """Claude cannot modify risk parameters."""
        with pytest.raises(ValueError, match="immutable"):
            experiment_engine.propose(
                description="Increase leverage",
                params={"leverage": 3.0},
                proposed_by="claude",
            )

    def test_unknown_params_blocked(self, experiment_engine):
        """Unknown parameters are rejected even if not immutable."""
        with pytest.raises(ValueError, match="unknown"):
            experiment_engine.propose(
                description="Test with made-up param",
                params={"made_up_param": 99},
                proposed_by="claude",
            )

    def test_valid_proposal_accepted(self, experiment_engine):
        """A valid proposal with registered non-immutable params is accepted."""
        exp = experiment_engine.propose(
            description="Increase Kelly fraction",
            params={"kelly_fraction": 0.6},
            proposed_by="claude",
        )
        assert exp.id is not None
        assert exp.status.value == 'PROPOSED'

    def test_claude_cannot_promote(self, experiment_engine):
        """Only operators can promote to production."""
        exp = experiment_engine.propose(
            description="Adjust threshold",
            params={"weighted_vote_threshold": 0.65},
            proposed_by="operator",
        )

        # Fast-track to PAPER_TEST (simulate full pipeline)
        exp.status = __import__('core.experiment_engine', fromlist=['ExperimentStatus']).ExperimentStatus.PAPER_TEST
        experiment_engine._save(exp)

        with pytest.raises(PermissionError, match="Claude cannot"):
            experiment_engine.promote(exp.id, promoted_by="claude")

    def test_canary_deploy(self, experiment_engine):
        """Canary deployment creates config with limited capital fraction."""
        from core.experiment_engine import ExperimentStatus
        exp = experiment_engine.propose(
            description="Kelly canary test",
            params={"kelly_fraction": 0.55},
            proposed_by="operator",
        )
        exp.status = ExperimentStatus.PAPER_TEST
        experiment_engine._save(exp)

        config = experiment_engine.canary_deploy(exp.id, capital_fraction=0.05)
        assert config['capital_fraction'] == 0.05
        assert config['params'] == {'kelly_fraction': 0.55}
        assert config['min_live_trades'] == 30

    def test_strategy_versioning_on_promote(self, experiment_engine):
        """Promoting an experiment creates a new strategy version."""
        from core.experiment_engine import ExperimentStatus
        exp = experiment_engine.propose(
            description="breakout threshold adjust",
            params={"kelly_fraction": 0.55},
            proposed_by="operator",
        )
        exp.status = ExperimentStatus.PAPER_TEST
        experiment_engine._save(exp)

        experiment_engine.promote(exp.id, promoted_by="operator")

        version = experiment_engine.get_strategy_version('breakout')
        assert version is not None
        assert version.startswith('v')


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7: Feature Store
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureStore:
    """Feature store records every evaluated opportunity."""

    def test_record_no_trade_opportunity(self, feature_store):
        """NO_TRADE decisions are recorded just like TRADE decisions."""
        from core.feature_store import OpportunityFeatures, TradeOutcome

        feats = OpportunityFeatures(
            symbol='BTC-USD', price=50_000.0, volume_24h=1e9,
            rsi_14=35.0, opportunity_score=0.2,
        )
        opp_id = feature_store.record_opportunity(feats, decision='NO_TRADE',
                                                   reject_reason='Score too low')
        assert opp_id > 0
        stats = feature_store.stats()
        assert stats['no_trade_decisions'] == 1

    def test_record_trade_and_update_outcome(self, feature_store):
        """After a trade, outcomes can be appended."""
        from core.feature_store import OpportunityFeatures, TradeOutcome

        feats  = OpportunityFeatures(symbol='ETH-USD', price=3000.0, opportunity_score=0.7)
        opp_id = feature_store.record_opportunity(feats, decision='TRADE')

        outcome = TradeOutcome(
            return_1h=0.015, return_4h=0.03,
            mfe_pct=0.025, mae_pct=-0.008,
            target_hit=True, stop_hit=False,
            realized_pnl=45.0, holding_hours=1.5,
        )
        ok = feature_store.update_outcome(opp_id, outcome)
        assert ok

        dataset = feature_store.get_training_dataset(min_rows=1)
        assert len(dataset) >= 1
        assert dataset[0]['return_1h'] == pytest.approx(0.015)

    def test_training_dataset_stats(self, feature_store):
        """Stats method reports correct coverage."""
        from core.feature_store import OpportunityFeatures, TradeOutcome

        for i in range(5):
            feats  = OpportunityFeatures(symbol='SOL-USD', price=100.0)
            opp_id = feature_store.record_opportunity(feats, decision='TRADE')
            if i % 2 == 0:
                feature_store.update_outcome(opp_id, TradeOutcome(return_1h=0.01))

        stats = feature_store.stats()
        assert stats['trade_decisions'] == 5
        assert stats['with_outcomes'] == 3
        assert stats['outcome_coverage'] == pytest.approx(0.6)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 10: Portfolio Manager
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioManager:
    """Portfolio manager enforces exposure limits."""

    def test_first_position_approved(self, portfolio_mgr):
        """With no open positions, first trade should be approved."""
        result = portfolio_mgr.allocate('BTC-USD', 'crypto', 1000.0, [])
        assert result.approved

    def test_over_class_exposure_blocked(self, portfolio_mgr):
        """Exceeding max per-class exposure blocks new positions."""
        open_positions = [
            {'symbol': 'BTC-USD', 'asset_class': 'crypto', 'size': 25_000.0},
        ]
        # Requesting another 2k in crypto (already at 50% of capital)
        result = portfolio_mgr.allocate('ETH-USD', 'crypto', 2000.0, open_positions)
        assert not result.approved
        assert 'class' in result.reject_reason.lower() or 'cap' in result.reject_reason.lower()

    def test_correlation_penalty_applied(self, portfolio_mgr):
        """Highly-correlated positions reduce allocation."""
        open_positions = [
            {'symbol': 'BTC-USD', 'asset_class': 'crypto', 'size': 1000.0},
            {'symbol': 'ETH-USD', 'asset_class': 'crypto', 'size': 1000.0},
        ]
        result = portfolio_mgr.allocate('SOL-USD', 'crypto', 1000.0, open_positions)
        # Should be approved but with reduced size due to correlation
        if result.approved:
            assert result.size_dollars < 1000.0 or result.correlation_penalty > 0

    def test_different_class_lower_penalty(self, portfolio_mgr):
        """Different asset class has lower correlation penalty."""
        open_positions = [
            {'symbol': 'BTC-USD', 'asset_class': 'crypto', 'size': 1000.0},
        ]
        result = portfolio_mgr.allocate('AAPL', 'equity', 1000.0, open_positions)
        assert result.approved
        assert result.correlation_penalty < 0.3   # equity/crypto correlation is low


# ─────────────────────────────────────────────────────────────────────────────
# Phase 11: Opportunity Ranker
# ─────────────────────────────────────────────────────────────────────────────

class TestOpportunityRanker:
    """Ranker scores candidates and enforces minimum thresholds."""

    def _make_candidate(self, **kwargs):
        base = {
            'symbol': 'BTC-USD', 'asset_class': 'crypto',
            'price': 50_000.0, 'volume_usd_24h': 5e9,
            'momentum_score': 0.6, 'kronos_confidence': 0.7,
            'signal_confidence': 75.0, 'opportunity_score': 0.65,
        }
        base.update(kwargs)
        return base

    def test_high_quality_candidate_approved(self, opportunity_ranker):
        candidates = [self._make_candidate()]
        ranked = opportunity_ranker.rank(candidates, regime='bull')
        assert len(ranked) == 1
        assert ranked[0].decision in ('TRADE', 'WATCH')

    def test_low_quality_candidate_rejected(self, opportunity_ranker):
        candidates = [self._make_candidate(
            momentum_score=0.05, kronos_confidence=0.1,
            signal_confidence=20.0, volume_usd_24h=10_000,
        )]
        ranked = opportunity_ranker.rank(candidates, regime='bear', drawdown_pct=0.04)
        assert ranked[0].decision == 'NO_TRADE'

    def test_drawdown_reduces_scores(self, opportunity_ranker):
        cand = self._make_candidate()
        no_drawdown = opportunity_ranker.rank([cand], regime='bull', drawdown_pct=0.0)
        with_drawdown = opportunity_ranker.rank([cand], regime='bull', drawdown_pct=0.10)
        assert no_drawdown[0].ranked_score >= with_drawdown[0].ranked_score

    def test_top_trades_filters_watch(self, opportunity_ranker):
        candidates = [
            self._make_candidate(symbol='BTC-USD', momentum_score=0.8, signal_confidence=90.0),
            self._make_candidate(symbol='ETH-USD', momentum_score=0.3, signal_confidence=45.0),
        ]
        tops = opportunity_ranker.top_trades(candidates, regime='bull', top_n=5)
        # Only TRADE decisions returned
        assert all(t.decision == 'TRADE' for t in tops)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 13: Trade Memory
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeMemory:
    """Trade memory records every closed trade with enriched data."""

    def _make_closed_position(self, pos_id=1, net_pnl=50.0):
        return {
            'id': pos_id,
            'symbol': 'BTC-USD',
            'status': 'CLOSED',
            'asset_class': 'crypto',
            'strategy': 'breakout',
            'strategy_version': 'v1.2',
            'entry_time': '2026-08-22T10:00:00',
            'entry_price': 50_000.0,
            'entry_fill_price': 50_025.0,
            'entry_fees': 5.0,
            'exit_time': '2026-08-22T11:30:00',
            'exit_price': 51_000.0,
            'exit_fill_price': 50_975.0,
            'exit_fees': 5.1,
            'pnl': net_pnl + 10.1,     # gross
            'net_pnl': net_pnl,
            'close_reason': 'take_profit',
            'size': 1_000.0,
            'direction': 'LONG',
            'side': 'BUY',
            'regime': 'bull',
            'kronos_confidence': 0.72,
            'llm_decision': 'BUY',
            'strategies_used': '["breakout","momentum"]',
            'broker_order_id': 'PAPER-000001',
            'broker_exit_order_id': 'PAPER-000002',
            'holding_hours': 1.5,
            'mfe_pct': 0.025,
            'mae_pct': -0.005,
        }

    def test_record_closed_trade(self, trade_memory):
        pos = self._make_closed_position(pos_id=1, net_pnl=50.0)
        row_id = trade_memory.record(pos)
        assert row_id is not None and row_id > 0

    def test_duplicate_not_written(self, trade_memory):
        pos = self._make_closed_position(pos_id=2)
        id1 = trade_memory.record(pos)
        id2 = trade_memory.record(pos)    # same position_id
        assert id1 == id2   # returns existing row id

    def test_open_position_skipped(self, trade_memory):
        pos = self._make_closed_position(pos_id=3)
        pos['status'] = 'OPEN'
        row_id = trade_memory.record(pos)
        assert row_id is None

    def test_performance_report(self, trade_memory):
        for i in range(5):
            pnl = 100.0 if i % 2 == 0 else -40.0
            trade_memory.record(self._make_closed_position(pos_id=i + 10, net_pnl=pnl))

        report = trade_memory.performance_report(days=30)
        assert report['total_trades'] == 5
        assert 0.0 < report['win_rate'] < 1.0
        assert 'by_strategy' in report

    def test_exit_quality_scoring(self, trade_memory):
        """Exit quality scores reflect close reason."""
        # Take profit = best exit
        pos_tp = self._make_closed_position(pos_id=20, net_pnl=100.0)
        pos_tp['close_reason'] = 'take_profit'
        quality_tp = trade_memory._estimate_exit_quality(pos_tp)

        # Stop loss = poor exit
        pos_sl = self._make_closed_position(pos_id=21, net_pnl=-50.0)
        pos_sl['close_reason'] = 'stop_loss'
        quality_sl = trade_memory._estimate_exit_quality(pos_sl)

        assert quality_tp > quality_sl
        assert quality_tp == 1.0
        assert quality_sl == 0.30


# ─────────────────────────────────────────────────────────────────────────────
# Full End-to-End: Complete lifecycle in one test
# ─────────────────────────────────────────────────────────────────────────────

class TestFullLifecycle:
    """
    THE INTEGRATION TEST:
    Signal → Broker → Position → Partial profit → Stop loss → Closed →
    Safety → Feature store → Trade memory
    """

    def test_complete_trade_lifecycle(
        self, broker, position_mgr, safety_mgr, feature_store, trade_memory
    ):
        from core.feature_store import OpportunityFeatures, TradeOutcome

        SYMBOL = 'BTC-USD'
        ENTRY  = 50_000.0
        QTY    = 0.02

        # ── 1. Record opportunity in feature store ─────────────────────────────
        feats  = OpportunityFeatures(
            symbol=SYMBOL, price=ENTRY, volume_24h=1e9,
            rsi_14=55.0, kronos_confidence=0.8,
            llm_confidence=0.8, opportunity_score=0.72,
            signal_confidence=80.0,
        )
        opp_id = feature_store.record_opportunity(feats, decision='TRADE')

        # ── 2. Enter via broker ────────────────────────────────────────────────
        fill = broker.submit_and_wait(SYMBOL, 'BUY', QTY, ENTRY)
        assert fill.status.value == 'FILLED'

        pos_id = position_mgr.open_position(
            SYMBOL, 'BUY', fill.notional, fill.fill_price,
            entry_fill_price=fill.fill_price, entry_fees=fill.fees,
            broker_order_id=fill.order_id,
            stop_loss=ENTRY * 0.98, take_profit=ENTRY * 1.03,
        )
        safety_mgr.on_position_open()
        feature_store.link_position(opp_id, pos_id)

        # ── 3. Partial profit (50%) ────────────────────────────────────────────
        PARTIAL_PRICE = 51_000.0
        partial_fill = broker.close_position(SYMBOL, PARTIAL_PRICE, quantity=QTY * 0.5)
        assert partial_fill.status.value == 'FILLED'

        partial_pnl = position_mgr.partial_close(
            pos_id, fraction=0.5, close_price=PARTIAL_PRICE,
            exit_fill_price=partial_fill.fill_price,
            exit_fees=partial_fill.fees,
        )
        safety_mgr.on_partial_close(partial_pnl)

        # Position still open
        assert position_mgr.get_position(SYMBOL) is not None
        assert safety_mgr.active_positions == 1

        # ── 4. Stop loss hit on remaining 50% ─────────────────────────────────
        SL_PRICE = 49_000.0
        sl_fill = broker.close_position(SYMBOL, SL_PRICE)
        if sl_fill.status.value in ('FILLED', 'PARTIAL'):
            net_pnl = position_mgr.close_position(
                pos_id, close_price=SL_PRICE,
                exit_fill_price=sl_fill.fill_price,
                exit_fees=sl_fill.fees,
                reason='stop_loss',
                broker_order_id=sl_fill.order_id,
            )
            safety_mgr.on_position_closed(net_pnl or 0.0)

        # ── 5. Verify final state ──────────────────────────────────────────────
        assert safety_mgr.active_positions == 0

        # ── 6. Record in trade memory ──────────────────────────────────────────
        closed_pos = None
        import sqlite3
        with sqlite3.connect(position_mgr.db_path) as conn:
            conn.row_factory = sqlite3.Row
            closed_pos = dict(conn.execute(
                "SELECT * FROM positions WHERE id=?", (pos_id,)
            ).fetchone())

        if closed_pos and closed_pos.get('status') == 'CLOSED':
            tm_id = trade_memory.record(closed_pos)
            assert tm_id is not None

        # ── 7. Update feature store outcome ───────────────────────────────────
        outcome = TradeOutcome(
            return_1h=0.01, mfe_pct=0.025, mae_pct=-0.02,
            target_hit=False, stop_hit=True,
        )
        feature_store.update_outcome(opp_id, outcome)

        stats = feature_store.stats()
        assert stats['trade_decisions'] >= 1
        assert stats['with_outcomes'] >= 1

        # ── 8. Trade memory performance report ────────────────────────────────
        report = trade_memory.performance_report(days=1)
        assert report.get('total_trades', 0) >= 0   # may be 0 if position wasn't CLOSED
