"""Tests: alpha-driven universes, broad instrument universe, data providers,
cold-start EV, and legacy-fallback retirement."""
import json
import sqlite3
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from core.alpha_library import AlphaLibrary, AlphaState
from core.alpha_signal_engine import AlphaSignalEngine
from core.ev_model import EconomicEVModel
from core.instruments import InstrumentUniverse, UniverseResolver

LIVE_PATH = [AlphaState.VALIDATING, AlphaState.PAPER, AlphaState.LIVE_ELIGIBLE]


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / f"u_{uuid.uuid4().hex}.sqlite")


def make_df(n=250, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n)))
    idx = pd.date_range("2026-06-01", periods=n, freq="1D")
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": rng.uniform(9e5, 1.1e6, n)},
                        index=idx)


# ── Universe resolution: alphas drive scanning ────────────────────────────────


class TestUniverseResolver:
    def test_sector_token_resolves_to_semiconductor_universe(self):
        resolver = UniverseResolver()
        symbols = resolver.resolve(["sector:semiconductor"])
        assert {"NVDA", "AMD", "MU", "AVGO"} <= set(symbols)
        assert "JPM" not in symbols

    def test_class_and_etf_tokens(self):
        resolver = UniverseResolver()
        crypto = resolver.resolve(["class:crypto"])
        assert "BTC-USD" in crypto
        sector_etfs = resolver.resolve(["etf:sector"])
        assert "SMH" in sector_etfs and "SPY" not in sector_etfs

    def test_wildcard_uses_scanner_symbols(self):
        resolver = UniverseResolver()
        assert resolver.resolve(["*"], scanner_symbols=["X-USD"]) == ["X-USD"]

    def test_explicit_symbols_and_dedupe_and_bound(self):
        resolver = UniverseResolver(max_symbols_per_alpha=3)
        out = resolver.resolve(["NVDA", "nvda", "sector:semiconductor"])
        assert out[0] == "NVDA" and len(out) == 3

    def test_alpha_universe_drives_required_symbols(self, db):
        lib = AlphaLibrary(db)
        lib.register("semi_alpha", "semi overnight", universe=["sector:semiconductor"],
                     direction="long", asset_class="stock")
        for s in LIVE_PATH:
            lib.transition("semi_alpha", s)
        engine = AlphaSignalEngine(lib)
        required = engine.required_symbols(scanner_symbols=["BTC-USD"])
        assert "MU" in required and "NVDA" in required
        assert "BTC-USD" not in required  # no wildcard declared

    def test_candidates_generated_across_resolved_universe(self, db):
        lib = AlphaLibrary(db)
        lib.register("semi_alpha", "semi", universe=["sector:semiconductor"],
                     direction="long", asset_class="stock", edge_health_score=0.9)
        for s in LIVE_PATH:
            lib.transition("semi_alpha", s)
        engine = AlphaSignalEngine(lib)
        market_data = {"NVDA": make_df(seed=1), "MU": make_df(seed=2)}
        cands = engine.generate_candidates(market_data, regime="bullish")
        assert {c.symbol for c in cands} == {"NVDA", "MU"}


# ── Broad universe + file loading ─────────────────────────────────────────────


class TestBroadUniverse:
    def test_universe_is_broad(self):
        uni = InstrumentUniverse()
        assert len(uni.stocks()) >= 250
        assert len(uni.etfs()) >= 50
        assert len(uni.sectors()) >= 12

    def test_sector_metadata_present(self):
        uni = InstrumentUniverse()
        nvda = uni.get("NVDA")
        assert nvda.sector == "semiconductor" and nvda.sector_etf == "SMH"

    def test_universe_file_loading_with_delisted(self, tmp_path):
        f = tmp_path / "universe.csv"
        f.write_text(
            "symbol,asset_class,sector,sector_etf,active,listed_to\n"
            "NEWCO,stock,technology,XLK,true,\n"
            "DEADCO,stock,energy,XLE,false,2024-06-30\n"
        )
        uni = InstrumentUniverse()
        assert uni.load_universe_file(str(f)) == 2
        actives = {i.symbol for i in uni.stocks()}
        assert "NEWCO" in actives and "DEADCO" not in actives
        # historical membership retained for research
        assert "DEADCO" in {i.symbol for i in uni.stocks(include_delisted=True)}


# ── Data providers (no fabrication, fail-soft) ────────────────────────────────


class TestDataProviders:
    def test_unavailable_feeds_return_nothing(self):
        from data.providers import (
            AnalystChangesProvider, CorporateActionsProvider,
            LiquidationsProvider, SECFilingsProvider,
        )
        for provider in (SECFilingsProvider(), CorporateActionsProvider(),
                         AnalystChangesProvider(), LiquidationsProvider()):
            assert not provider.available()

    def test_crypto_symbols_have_no_earnings(self):
        from data.providers import YFinanceEarningsProvider
        assert YFinanceEarningsProvider().days_to_next_earnings("BTC-USD") is None

    def test_derivatives_symbol_mapping(self):
        from data.providers import BinanceDerivativesProvider
        assert BinanceDerivativesProvider._perp_symbol("BTC-USD") == "BTCUSDT"
        assert BinanceDerivativesProvider._perp_symbol("AAPL") is None

    def test_context_builder_never_fabricates(self):
        from data.providers import (
            BinanceDerivativesProvider, EarningsProvider, MarketContextBuilder,
        )

        class OfflineDeriv(BinanceDerivativesProvider):
            def available(self):
                return False

        builder = MarketContextBuilder(earnings=EarningsProvider(),
                                       derivatives=OfflineDeriv())
        ctx = builder.build(["BTC-USD", "NVDA"])
        # No feeds available -> no fabricated values
        assert "funding_rate" not in ctx["BTC-USD"]
        assert "earnings_days_away" not in ctx["NVDA"]

    def test_context_flows_into_conditions(self, db):
        lib = AlphaLibrary(db)
        lib.register("funding_alpha", "extreme funding", universe=["BTC-USD"],
                     direction="short", asset_class="crypto",
                     entry_conditions=json.dumps(
                         [{"feature": "funding_rate", "op": ">", "value": 0.0005}]))
        for s in LIVE_PATH:
            lib.transition("funding_alpha", s)
        engine = AlphaSignalEngine(lib)
        data = {"BTC-USD": make_df()}
        # Without funding context: abstain
        assert engine.generate_candidates(data, regime="bullish") == []
        # With provider-supplied context: candidate fires
        cands = engine.generate_candidates(
            data, regime="bullish",
            context={"BTC-USD": {"funding_rate": 0.001}})
        assert len(cands) == 1 and cands[0].direction == "short"


# ── Cold-start EV ─────────────────────────────────────────────────────────────


def _alpha_with_oos(db, alpha_id, mean_net_return=0.008, se_return=0.002,
                    win_rate=0.58, trades=30):
    lib = AlphaLibrary(db)
    lib.register(alpha_id, alpha_id, universe=["TEST-USD"], direction="long")
    lib.update_evidence(alpha_id, oos_metrics={
        "mean_net_return": mean_net_return,
        "standard_error_return": se_return,
        "win_rate": win_rate, "trades": trades,
    })
    for s in LIVE_PATH:
        lib.transition(alpha_id, s)
    return lib


def _add_forward_obs(db, alpha_id, returns_pct):
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS trade_memory (id INTEGER PRIMARY KEY "
            "AUTOINCREMENT, entry_time TEXT, exit_time TEXT, net_return_pct REAL, "
            "gross_pnl REAL, total_fees REAL, size_dollars REAL, mae_pct REAL, "
            "mfe_pct REAL, holding_hours REAL, close_reason TEXT)"
        )
        from core.trade_attribution import TradeAttributionStore
        store = TradeAttributionStore(db)
        for i, r in enumerate(returns_pct):
            cur = conn.execute(
                "INSERT INTO trade_memory (entry_time, exit_time, net_return_pct, "
                "size_dollars, holding_hours) VALUES (?,?,?,?,?)",
                (f"2026-08-{(i % 28) + 1:02d}T00:00:00", f"2026-08-{(i % 28) + 1:02d}T04:00:00",
                 r, 1000.0, 4.0),
            )
            conn.commit()
            store.record(alpha_id, trade_memory_id=cur.lastrowid)


class TestColdStartEV:
    def test_validated_paper_alpha_gets_conservative_prior(self, db):
        lib = _alpha_with_oos(db, "cold_alpha", mean_net_return=0.008,
                              se_return=0.002)
        model = EconomicEVModel(db)
        est = model.estimate_with_prior(lib.get("cold_alpha"))
        assert est is not None and est.source == "oos_prior"
        # haircut: 50% of 0.8% = 0.4% — conservative, not raw OOS
        assert est.expected_net_return == pytest.approx(0.004, abs=1e-6)
        assert est.probability_positive < 0.58   # shrunk toward 0.5
        # UNIT CONSISTENCY: uncertainty must be fractional-return scale,
        # so the lower bound stays in a sane band (not 0.004 - 1.645*2.0)
        assert -0.02 < est.ev_lower_bound < est.expected_net_return

    def test_weak_oos_gives_no_prior(self, db):
        lib = _alpha_with_oos(db, "weak", mean_net_return=-0.002)
        assert EconomicEVModel(db).estimate_with_prior(lib.get("weak")) is None
        lib2 = _alpha_with_oos(db, "tiny_sample", trades=3)
        assert EconomicEVModel(db).estimate_with_prior(lib2.get("tiny_sample")) is None

    def test_forward_evidence_progressively_replaces_prior(self, db):
        lib = _alpha_with_oos(db, "blend_alpha", mean_net_return=0.008,
                              se_return=0.002)
        model = EconomicEVModel(db, min_samples=8)
        cold = model.estimate_with_prior(lib.get("blend_alpha"))
        assert cold.source == "oos_prior"

        # add forward observations with a DIFFERENT realized expectancy (2%)
        _add_forward_obs(db, "blend_alpha", [2.0, 2.1, 1.9, 2.0, 2.2, 1.8, 2.0,
                                             2.1, 1.9, 2.0])
        blended = model.estimate_with_prior(lib.get("blend_alpha"))
        assert blended.source == "blended"
        # pulled from 0.4% prior toward 2% forward mean
        assert 0.004 < blended.expected_net_return < 0.02
        assert blended.forward_weight is not None and 0 < blended.forward_weight < 1

    def test_prior_fully_retired_with_large_forward_sample(self, db):
        lib = _alpha_with_oos(db, "mature_alpha", mean_net_return=0.008,
                              se_return=0.002)
        model = EconomicEVModel(db, min_samples=8)
        _add_forward_obs(db, "mature_alpha",
                         list(np.random.default_rng(3).normal(1.0, 0.4, 70)))
        est = model.estimate_with_prior(lib.get("mature_alpha"), prior_strength=20)
        assert est.forward_weight > 0.75   # forward evidence dominates


# ── Legacy fallback retirement ────────────────────────────────────────────────


class TestLegacyRetirement:
    def test_maturity_requires_performance_evidence(self):
        """Days+decisions alone must NOT mature the pipeline (evidence-based)."""
        from core.pipeline_maturity import (
            MaturityEvidence, MaturityState, PipelineMaturityEvaluator,
        )
        ev = MaturityEvidence(days_observed=30, decisions=500,
                              resolved_outcomes=0)
        assessment = PipelineMaturityEvaluator(":memory:").assess(ev)
        assert assessment.state != MaturityState.MATURE
        assert assessment.legacy_fallback_allowed

    def test_exclusive_mode_exists(self):
        """'exclusive' mode short-circuits legacy entirely (code inspection)."""
        import inspect
        import ultimate_bot_v3_llm as m
        src = inspect.getsource(m.LLMTradingBot.run_trading_cycle)
        assert "exclusive" in src and "run_legacy = False" in src
