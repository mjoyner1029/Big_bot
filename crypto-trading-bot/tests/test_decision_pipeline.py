"""Tests for the decision pipeline & risk architecture:

- Alpha library lifecycle enforcement (research-only edges cannot go live)
- Live-safety assertions (unvalidated strategies can never send live orders)
- Edge health states incl. decaying edge -> DEGRADED/RETIRED (Case H)
- Drawdown budgets (strategy/family/asset)
- Execution cost model breakdown
- Evidence-aware strategy scoring & correlation penalties
"""
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from core.alpha_library import (
    AlphaLibrary,
    AlphaState,
    LIVE_ELIGIBLE_STATES,
    SIGNAL_ELIGIBLE_STATES,
)
from core.risk_budgets import DrawdownBudgetManager
from core.strategy_correlation import (
    StrategyCorrelationTracker,
    evidence_aware_strategy_score,
    family_of,
)
from core.strategy_health import EdgeState, StrategyHealthMonitor
from core.transaction_costs import TransactionCostModel


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / f"test_{uuid.uuid4().hex}.sqlite")


# ── Alpha library lifecycle ───────────────────────────────────────────────────


class TestAlphaLibraryLifecycle:
    def test_new_alpha_is_not_signal_eligible(self, db):
        lib = AlphaLibrary(db)
        lib.register("a1", "test alpha", strategy_id="momentum")
        assert lib.state_of("a1") == AlphaState.DISCOVERED
        assert lib.eligible_alphas() == []
        assert not lib.is_live_approved("a1")

    def test_cannot_jump_from_discovered_to_live(self, db):
        lib = AlphaLibrary(db)
        lib.register("a1", "test alpha")
        assert not lib.transition("a1", AlphaState.LIVE_SCALED)
        assert lib.state_of("a1") == AlphaState.DISCOVERED

    def test_full_lifecycle_path(self, db):
        lib = AlphaLibrary(db)
        lib.register("a1", "test alpha", market="crypto")
        for state in (AlphaState.VALIDATING, AlphaState.PAPER,
                      AlphaState.LIVE_ELIGIBLE, AlphaState.LIVE_LIMITED):
            assert lib.transition("a1", state, reason="test"), state
        assert lib.is_live_approved("a1")
        assert any(a["alpha_id"] == "a1" for a in lib.eligible_alphas(live_only=True))

    def test_rejected_alpha_not_eligible(self, db):
        lib = AlphaLibrary(db)
        lib.register("a1", "test alpha")
        lib.transition("a1", AlphaState.VALIDATING)
        lib.transition("a1", AlphaState.REJECTED, reason="failed OOS")
        assert lib.eligible_alphas() == []

    def test_retirement_is_permanent(self, db):
        lib = AlphaLibrary(db)
        lib.register("a1", "test alpha")
        lib.transition("a1", AlphaState.VALIDATING)
        lib.transition("a1", AlphaState.RETIRED, reason="edge decayed")
        assert not lib.transition("a1", AlphaState.PAPER)
        assert not lib.transition("a1", AlphaState.LIVE_ELIGIBLE)
        assert any(g["alpha_id"] == "a1" for g in lib.graveyard())

    def test_regime_filter(self, db):
        lib = AlphaLibrary(db)
        lib.register("a1", "bull only", market="crypto",
                     valid_regimes=["bullish"])
        lib.transition("a1", AlphaState.VALIDATING)
        lib.transition("a1", AlphaState.PAPER)
        assert lib.eligible_alphas(regime="bullish")
        assert lib.eligible_alphas(regime="bearish") == []

    def test_paper_state_never_live_approved(self, db):
        lib = AlphaLibrary(db)
        lib.register("a1", "paper alpha")
        lib.transition("a1", AlphaState.VALIDATING)
        lib.transition("a1", AlphaState.PAPER)
        assert AlphaState.PAPER in SIGNAL_ELIGIBLE_STATES
        assert AlphaState.PAPER not in LIVE_ELIGIBLE_STATES
        assert not lib.is_live_approved("a1")


# ── Edge health: decaying edge (adversarial CASE H) ───────────────────────────


def _insert_closed_trades(db_path, strategy, pnls, start=None):
    start = start or (datetime.now(timezone.utc) - timedelta(days=5))
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS positions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, strategy TEXT, "
            "status TEXT, net_pnl REAL, holding_hours REAL, mfe_pct REAL, "
            "mae_pct REAL, exit_time TEXT)"
        )
        for i, p in enumerate(pnls):
            exit_time = (start + timedelta(hours=i)).isoformat()
            conn.execute(
                "INSERT INTO positions (symbol, strategy, status, net_pnl, "
                "holding_hours, mfe_pct, mae_pct, exit_time) "
                "VALUES (?,?,?,?,?,?,?,?)",
                ("TEST-USD", strategy, "CLOSED", p, 1.0, 0.01, -0.01, exit_time),
            )
        conn.commit()


class TestCaseHDecayingEdge:
    def test_decaying_edge_degrades_then_retires(self, db):
        monitor = StrategyHealthMonitor(db_path=db, capital=10_000.0)
        # Decayed edge: consistent losses, plenty of evidence
        _insert_closed_trades(db, "decayed_edge", [-12.0] * 40)

        states = []
        for _ in range(3):
            monitor.run()
            states.append(monitor.state_of("decayed_edge"))

        # Escalation: never healthy, ends in permanent retirement
        assert states[-1] == EdgeState.RETIRED
        assert monitor.is_paused("decayed_edge")
        # Retired strategies cannot be unpaused
        monitor.unpause("decayed_edge")
        assert monitor.is_paused("decayed_edge")

    def test_healthy_edge_stays_healthy(self, db):
        monitor = StrategyHealthMonitor(db_path=db, capital=10_000.0)
        _insert_closed_trades(db, "good_edge", [8.0, -4.0] * 20)
        monitor.run()
        assert monitor.state_of("good_edge") == EdgeState.HEALTHY
        assert not monitor.is_paused("good_edge")

    def test_states_persist_across_instances(self, db):
        monitor = StrategyHealthMonitor(db_path=db, capital=10_000.0)
        monitor.retire("dead_strategy", "test retirement")
        fresh = StrategyHealthMonitor(db_path=db, capital=10_000.0)
        assert fresh.is_paused("dead_strategy")
        assert fresh.state_of("dead_strategy") == EdgeState.RETIRED


# ── Drawdown budgets ──────────────────────────────────────────────────────────


def _insert_trade_memory(db_path, strategy, symbol, pnls):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS trade_memory ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, strategy TEXT, "
            "net_pnl REAL, exit_time TEXT)"
        )
        start = datetime.now(timezone.utc) - timedelta(days=2)
        for i, p in enumerate(pnls):
            conn.execute(
                "INSERT INTO trade_memory (symbol, strategy, net_pnl, exit_time) "
                "VALUES (?,?,?,?)",
                (symbol, strategy, p, (start + timedelta(hours=i)).isoformat()),
            )
        conn.commit()


class TestDrawdownBudgets:
    def test_fresh_strategy_full_allocation(self, db):
        mgr = DrawdownBudgetManager(db_path=db, capital=10_000.0)
        assert mgr.allocation_multiplier("momentum", "BTC-USD") == 1.0

    def test_exhausted_strategy_budget_pauses(self, db):
        # strategy budget = 3% of 10k = $300; drawdown of $400 exceeds it
        _insert_trade_memory(db, "bleeding", "BTC-USD", [-40.0] * 10)
        mgr = DrawdownBudgetManager(db_path=db, capital=10_000.0)
        assert mgr.strategy_status("bleeding").action == "pause"
        assert mgr.allocation_multiplier("bleeding", "ETH-USD") == 0.0

    def test_partial_budget_reduces(self, db):
        # $240 drawdown = 80% of $300 budget -> reduce
        _insert_trade_memory(db, "pressured", "SOL-USD", [-24.0] * 10)
        mgr = DrawdownBudgetManager(db_path=db, capital=10_000.0)
        assert mgr.strategy_status("pressured").action == "reduce"
        assert mgr.allocation_multiplier("pressured", "ADA-USD") == 0.5


# ── Execution cost model ──────────────────────────────────────────────────────


class TestExecutionCostModel:
    def test_estimate_cost_breakdown(self):
        model = TransactionCostModel("binance")
        est = model.estimate_cost(
            asset="BTC-USD", side="buy", quantity=0.01, price=50_000.0,
            market_state={"bid": 49_990.0, "ask": 50_010.0, "adv_usd": 1e9,
                          "volatility_pct": 0.02, "hour_utc": 14},
        )
        for key in ("commission", "spread_cost", "slippage_cost",
                    "estimated_market_impact", "total_cost", "cost_bps"):
            assert key in est
        assert est["total_cost"] > 0
        assert est["total_cost"] == pytest.approx(
            est["commission"] + est["spread_cost"] + est["slippage_cost"]
            + est["estimated_market_impact"])
        # true bid/ask spread used: 20/50000 = 4bps -> half-spread 2bps of notional
        assert est["spread_cost"] == pytest.approx(0.0002 * 500.0, rel=0.01)

    def test_conservative_fallback_without_quotes(self):
        model = TransactionCostModel("binance")
        est = model.estimate_cost("X", "buy", 1.0, 100.0)
        assert est["spread_cost"] > 0       # conservative default, not zero
        assert est["estimated_market_impact"] > 0

    def test_size_dependent_impact(self):
        model = TransactionCostModel("binance")
        small = model.estimate_cost("X", "buy", 1.0, 100.0,
                                    market_state={"adv_usd": 1e6})
        large = model.estimate_cost("X", "buy", 1000.0, 100.0,
                                    market_state={"adv_usd": 1e6})
        assert (large["estimated_market_impact"] / 100_000
                > small["estimated_market_impact"] / 100)  # impact % grows with size


# ── Evidence-aware scoring & correlation ─────────────────────────────────────


class TestEvidenceAwareScoring:
    def test_uncertain_edge_ranks_below_confident_edge(self):
        # Wide CI straddling zero (mean +0.40) vs tight CI positive (mean +0.28)
        import random
        rng = random.Random(1)
        uncertain = [rng.gauss(0.40, 6.0) for _ in range(30)]
        confident = [rng.gauss(0.28, 0.5) for _ in range(30)]
        s_uncertain = evidence_aware_strategy_score(uncertain)
        s_confident = evidence_aware_strategy_score(confident)
        assert s_confident > s_uncertain

    def test_no_evidence_scores_zero(self):
        assert evidence_aware_strategy_score([1.0, 2.0]) == 0.0  # tiny sample
        assert evidence_aware_strategy_score([-1.0] * 50) == 0.0  # negative edge

    def test_family_mapping(self):
        assert family_of("breakout") == family_of("ema_trend_follow") == "TREND"
        assert family_of("mean_reversion") == "MEAN_REVERSION"

    def test_correlation_penalty_same_family_fallback(self, db):
        tracker = StrategyCorrelationTracker(db_path=db)
        # No P&L data -> family fallback: same family penalized more
        same = tracker.correlation_penalty("breakout", ["ema_trend_follow"], {})
        diff = tracker.correlation_penalty("breakout", ["mean_reversion"], {})
        assert same < diff

    def test_realized_correlation_used_when_available(self, db):
        tracker = StrategyCorrelationTracker(db_path=db)
        matrix = {("a", "b"): 0.95, ("b", "a"): 0.95}
        redundant = tracker.correlation_penalty("a", ["b"], matrix)
        independent = tracker.correlation_penalty("a", ["c"], {("a", "c"): 0.0})
        assert redundant < independent


# ── Live-safety authorization (execution layer) ──────────────────────────────


class _BotStub:
    """Minimal stub exposing the live-authorization logic from the bot."""

    def __init__(self, db_path):
        self.alpha_library = AlphaLibrary(db_path)
        self.live_approved_strategies = set()


class TestLiveSafetyAuthorization:
    def _auth(self, db_path, strategies, mode, approved=(), alpha_states=()):
        from ultimate_bot_v3_llm import LLMTradingBot
        stub = _BotStub(db_path)
        stub.live_approved_strategies = set(approved)
        for alpha_id, states in alpha_states:
            stub.alpha_library.register(alpha_id, alpha_id)
            for s in states:
                stub.alpha_library.transition(alpha_id, s)
        old_mode = os.environ.get("TRADING_MODE")
        os.environ["TRADING_MODE"] = mode
        try:
            return LLMTradingBot._check_live_authorization(stub, strategies)
        finally:
            if old_mode is None:
                os.environ.pop("TRADING_MODE", None)
            else:
                os.environ["TRADING_MODE"] = old_mode

    def test_paper_mode_always_allowed(self, db):
        ok, _ = self._auth(db, ["anything"], "PAPER")
        assert ok

    def test_live_blocks_unapproved_strategy(self, db):
        ok, reason = self._auth(db, ["new_discovery"], "LIVE")
        assert not ok
        assert "not approved" in reason

    def test_live_blocks_paper_stage_alpha(self, db):
        ok, _ = self._auth(
            db, ["paper_alpha"], "LIVE",
            alpha_states=[("paper_alpha",
                           [AlphaState.VALIDATING, AlphaState.PAPER])],
        )
        assert not ok

    def test_live_allows_env_approved(self, db):
        ok, _ = self._auth(db, ["momentum"], "LIVE", approved=["momentum"])
        assert ok

    def test_live_allows_live_eligible_alpha(self, db):
        ok, _ = self._auth(
            db, ["validated_alpha"], "LIVE",
            alpha_states=[("validated_alpha",
                           [AlphaState.VALIDATING, AlphaState.PAPER,
                            AlphaState.LIVE_ELIGIBLE])],
        )
        assert ok

    def test_live_blocks_when_no_strategy_attributed(self, db):
        ok, _ = self._auth(db, [], "LIVE")
        assert not ok
