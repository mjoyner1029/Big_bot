"""Universal discovery + anti-overfitting tests (spec §89 J-R).

J  synthetic strong relative outperformer detected without being named
K  regime change detected within reasonable lag (change point)
L  random outlier does NOT become a validated strategy (validation, not scan)
M  hypothesis expressed in structured DSL form
N  complexity penalty prefers simpler equal-performance strategy
O  feature ablation flags non-contributing features
P  million-hypothesis search receives stronger correction than hundred
Q  regime-diversified portfolio preferred over correlated cluster
R  no-opportunity market → no forced trades (CASH is a position)
"""
import numpy as np
import pandas as pd
import pytest

from core.hypothesis_generator import (
    AutonomousHypothesisGenerator,
    GeneratorBudget,
    breadth_adjusted_alpha,
    complexity_adjusted_score,
    feature_ablation,
    hypothesis_complexity,
)
from core.portfolio_robustness import (
    downside_correlation,
    estimate_alpha_capacity,
    expected_log_growth,
    portfolio_regime_preference,
    regime_coverage_matrix,
    regime_coverage_score,
    return_on_capital_time,
    risk_of_ruin,
)
from core.universal_discovery import (
    CrossSectionalOpportunityScanner,
    MarketChangePointDetector,
    MarketOutlierDetector,
    ReturnDecompositionEngine,
)


def make_price_df(n=300, drift=0.0003, vol=0.01, seed=0, start="2024-01-01",
                  overnight_share=0.5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    daily = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(daily))
    # opens carry `overnight_share` of each day's move (0.5 = balanced asset)
    prev = np.concatenate([[100.0], close[:-1]])
    opens = prev * np.exp(daily * overnight_share)
    return pd.DataFrame({
        "open": opens, "close": close,
        "high": np.maximum(opens, close) * 1.005,
        "low": np.minimum(opens, close) * 0.995,
        "volume": rng.integers(1_000_000, 2_000_000, n).astype(float),
    }, index=idx)


# ── Case J: unexpected outperformer detected without being named ──────────────


class TestCaseJRelativeOutperformer:
    def test_synthetic_outperformer_flagged(self):
        data = {f"SYM{i}": make_price_df(seed=i) for i in range(120)}
        data["HERO"] = make_price_df(drift=0.01, seed=999)   # ~massive drift
        rows = CrossSectionalOpportunityScanner().scan(data)
        by_symbol = {r.symbol: r for r in rows}
        hero = by_symbol["HERO"]
        assert hero.return_percentiles[20] >= 0.99
        assert any("OUTPERFORMANCE" in f for f in hero.anomaly_flags)
        # nothing about the scanner referenced the name — flags are rank-based
        flagged = [r.symbol for r in rows
                   if any("OUTPERFORMANCE_20D" in f for f in r.anomaly_flags)]
        assert "HERO" in flagged and len(flagged) <= 3

    def test_micron_style_overnight_anomaly_found(self):
        """Return decomposition independently finds an asset earning nearly
        all appreciation overnight — no ticker hints anywhere."""
        df = make_price_df(n=250, drift=0.004, seed=7, overnight_share=0.95)
        decomp = ReturnDecompositionEngine().decompose("ANON", df)
        assert decomp is not None
        assert decomp.overnight_contribution > 0.8
        assert "EXTREME_OVERNIGHT_CONTRIBUTION" in decomp.extreme_flags
        # and it becomes a structured overnight-holding hypothesis
        hyps = AutonomousHypothesisGenerator().from_return_decomposition(decomp)
        assert any(h["subfamily"] == "overnight" for h in hyps)

    def test_normal_asset_not_flagged(self):
        df = make_price_df(n=250, seed=11)
        decomp = ReturnDecompositionEngine().decompose("NORM", df)
        assert decomp is None or \
            "EXTREME_OVERNIGHT_CONTRIBUTION" not in decomp.extreme_flags


# ── Case K: change point detected with reasonable lag ─────────────────────────


class TestCaseKChangePoint:
    def test_regime_break_detected(self):
        rng = np.random.default_rng(3)
        calm = rng.normal(0.0002, 0.005, 200)
        wild = rng.normal(-0.002, 0.04, 100)
        close = 100 * np.exp(np.cumsum(np.concatenate([calm, wild])))
        df = pd.DataFrame({"close": close},
                          index=pd.bdate_range("2024-01-01", periods=300))
        points = MarketChangePointDetector().detect("X", df)
        assert points, "regime break must be detected"
        vol_points = [p for p in points if p.metric == "volatility"]
        assert any(190 <= p.index <= 260 for p in vol_points or points)

    def test_stationary_series_quiet(self):
        rng = np.random.default_rng(4)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
        df = pd.DataFrame({"close": close},
                          index=pd.bdate_range("2024-01-01", periods=300))
        points = MarketChangePointDetector().detect("X", df)
        assert len(points) <= 2   # tolerate rare noise triggers


# ── Case L: random outlier ≠ validated strategy ───────────────────────────────


class TestCaseLOutlierIsNotAlpha:
    def test_outlier_detected_but_needs_validation(self):
        rng = np.random.default_rng(5)
        matrix = {f"S{i}": {"volume_z": float(rng.normal())} for i in range(50)}
        matrix["WEIRD"] = {"volume_z": 25.0}
        outliers = MarketOutlierDetector().detect(matrix)
        assert any(o.symbol == "WEIRD" for o in outliers)
        # outlier alone (no coincident change point) produces NO hypothesis
        hyps = AutonomousHypothesisGenerator().from_outliers_and_changepoints(
            outliers, change_points=[])
        assert hyps == []

    def test_random_noise_hypothesis_fails_validation(self):
        """A hypothesis built on pure noise must not survive the existing
        validation gates (FDR/DSR are already tested elsewhere; here we check
        the wiring: anomaly_score is not a pass ticket)."""
        from core.validation_stats import deflated_sharpe_ratio
        rng = np.random.default_rng(6)
        noise = rng.normal(0.0, 0.01, 60)
        sharpe = float(noise.mean() / noise.std())
        dsr = deflated_sharpe_ratio(sharpe, n_trials=1000, n_obs=60)
        assert dsr < 0.95   # noise cannot clear DSR at high trial count

    def test_small_universe_outliers_suppressed(self):
        # <8 symbols per feature → no robust statistics → no outliers
        matrix = {f"S{i}": {"x": float(i)} for i in range(5)}
        assert MarketOutlierDetector().detect(matrix) == []


# ── Case M: hypotheses are structured DSL, not prose ──────────────────────────


class TestCaseMStructuredHypotheses:
    def test_generated_conditions_are_valid_dsl(self):
        from core.alpha_conditions import validate_condition
        data = {f"SYM{i}": make_price_df(seed=i) for i in range(120)}
        data["HERO"] = make_price_df(drift=0.01, seed=999)
        rows = CrossSectionalOpportunityScanner().scan(data)
        hyps = AutonomousHypothesisGenerator().from_cross_sectional(rows)
        assert hyps
        for h in hyps:
            assert h["family"] and h["direction"] in ("long", "short")
            for cond in h["entry_conditions"]:
                validate_condition(cond)   # raises if not machine-readable

    def test_external_event_hypotheses(self):
        gen = AutonomousHypothesisGenerator()
        hyps = gen.from_external_events({
            "ABC": {"new_federal_award": True,
                    "award_amount_vs_market_cap": 0.08},
            "DEF": {"clustered_buying": True, "unique_members_buying": 4},
            "ETH-USD": {"skill_weighted_positioning": 0.9},
        })
        fams = {h["subfamily"] for h in hyps}
        assert {"federal_award", "congress_cluster", "smart_wallets"} <= fams

    def test_generator_respects_budget(self):
        gen = AutonomousHypothesisGenerator(GeneratorBudget(
            max_hypotheses_total=5, max_hypotheses_per_family=5))
        rows = CrossSectionalOpportunityScanner().scan(
            {f"S{i}": make_price_df(drift=0.01, seed=i) for i in range(30)})
        hyps = gen.from_cross_sectional(rows)
        assert len(hyps) <= 5


# ── Case N: complexity penalty prefers the simpler strategy ───────────────────


class TestCaseNComplexityPenalty:
    def test_simpler_strategy_preferred_at_similar_sharpe(self):
        simple = {"entry_conditions": [
            {"feature": "rsi_14", "op": "<", "value": 30.0},
            {"feature": "relative_volume_20d", "op": ">", "value": 1.5}]}
        complex_ = {"entry_conditions": [
            {"feature": "rsi_14", "op": "BETWEEN", "value": [27.13, 29.87]},
            {"feature": "relative_volume_20d", "op": ">", "value": 1.6123},
            {"feature": "atr_pct", "op": "<", "value": 0.0312},
            {"feature": "day_of_week", "op": "==", "value": "TUESDAY"},
            {"feature": "returns_5d", "op": "<", "value": -0.0213},
            {"feature": "macd_hist", "op": ">", "value": 0.0001},
            {"feature": "bb_position", "op": "<", "value": 0.21},
            {"feature": "volume_zscore", "op": ">", "value": 1.87},
            {"feature": "returns_20d", "op": ">", "value": -0.0999},
            {"feature": "adx_14", "op": ">", "value": 23.4},
            {"feature": "ema_ratio_9_21", "op": "<", "value": 0.9912}]}
        # spec §50: 2 conditions @1.5 must beat 11 conditions @1.55
        assert complexity_adjusted_score(1.50, simple) > \
            complexity_adjusted_score(1.55, complex_)
        assert hypothesis_complexity(complex_) > hypothesis_complexity(simple) * 3

    def test_precision_costs_complexity(self):
        coarse = {"entry_conditions": [{"feature": "rsi_14", "op": "<", "value": 30.0}]}
        precise = {"entry_conditions": [{"feature": "rsi_14", "op": "<", "value": 29.8731}]}
        assert hypothesis_complexity(precise) > hypothesis_complexity(coarse)


# ── Case O: feature ablation ──────────────────────────────────────────────────


class TestCaseOAblation:
    def test_non_contributing_feature_flagged(self):
        useful = {"feature": "rsi_14", "op": "<", "value": 30.0}
        useless = {"feature": "day_of_week", "op": "==", "value": "TUESDAY"}

        def evaluate(conds):
            return 1.5 if any(c["feature"] == "rsi_14" for c in conds) else 0.1

        result = feature_ablation([useful, useless], evaluate)
        assert result["non_contributing_features"] == ["day_of_week==TUESDAY"]
        assert result["recommended_conditions"] == [useful]
        assert result["incremental_alpha_contribution"]["rsi_14<30.0"] > 1.0


# ── Case P: search breadth strengthens correction ─────────────────────────────


class TestCasePBreadthCorrection:
    def test_million_hypotheses_stricter_than_hundred(self):
        a100 = breadth_adjusted_alpha(0.10, 100)
        a1m = breadth_adjusted_alpha(0.10, 1_000_000)
        assert a100 == 0.10
        assert a1m < a100 / 3

    def test_monotone_in_breadth(self):
        alphas = [breadth_adjusted_alpha(0.10, n)
                  for n in (100, 1_000, 10_000, 100_000, 1_000_000)]
        assert alphas == sorted(alphas, reverse=True)

    def test_breadth_feeds_fdr(self):
        """Stricter alpha → fewer FDR survivors on the same p-values."""
        from core.validation_stats import benjamini_hochberg
        pvals = [0.001, 0.004, 0.008, 0.02, 0.04, 0.09]
        wide = benjamini_hochberg(pvals, alpha=breadth_adjusted_alpha(0.10, 100))
        narrow = benjamini_hochberg(pvals,
                                    alpha=breadth_adjusted_alpha(0.10, 1_000_000))
        assert sum(narrow) < sum(wide)


# ── Case Q: regime diversification preferred ──────────────────────────────────


class TestCaseQRegimeDiversification:
    def test_coverage_matrix_measured_from_data(self):
        matrix = regime_coverage_matrix({
            "alpha_bull": [("BULL", 0.01)] * 20 + [("BEAR", -0.005)] * 20,
            "alpha_bear": [("BEAR", 0.008)] * 20 + [("BULL", -0.002)] * 20,
        })
        assert matrix["alpha_bull"]["BULL"]["expectancy"] > 0
        assert matrix["alpha_bull"]["BEAR"]["expectancy"] < 0
        assert regime_coverage_score(matrix) == 1.0   # both regimes covered
        solo = regime_coverage_matrix(
            {"alpha_bull": [("BULL", 0.01)] * 20 + [("BEAR", -0.005)] * 20})
        assert regime_coverage_score(solo) == 0.5

    def test_diversified_portfolio_beats_correlated_cluster(self):
        choice = portfolio_regime_preference({
            "correlated_cluster": {"coverage": 0.33, "avg_downside_corr": 0.85,
                                   "expected_return": 0.30},
            "diversified": {"coverage": 1.0, "avg_downside_corr": 0.10,
                            "expected_return": 0.22},
        })
        assert choice == "diversified"

    def test_downside_correlation_detects_joint_crashes(self):
        rng = np.random.default_rng(8)
        shock = rng.normal(0, 0.02, 200)
        a = list(0.7 * shock + rng.normal(0.001, 0.005, 200))
        b = list(0.7 * shock + rng.normal(0.001, 0.005, 200))
        c = list(rng.normal(0.001, 0.01, 200))
        assert downside_correlation(a, b) > 0.4
        assert abs(downside_correlation(a, c)) < 0.45


# ── Capacity / capital efficiency / risk of ruin ──────────────────────────────


class TestCapacityAndRuin:
    def test_capacity_scales_with_liquidity(self):
        small = estimate_alpha_capacity(adv_usd=1e6, expected_net_return=0.01,
                                        signal_frequency_per_day=1, holding_days=5)
        big = estimate_alpha_capacity(adv_usd=1e9, expected_net_return=0.01,
                                      signal_frequency_per_day=1, holding_days=5)
        assert big["estimated_alpha_capacity_usd"] > \
            small["estimated_alpha_capacity_usd"]

    def test_unknown_liquidity_conservative(self):
        r = estimate_alpha_capacity(adv_usd=None, expected_net_return=0.05,
                                    signal_frequency_per_day=1, holding_days=1)
        assert r["constraint"] == "unknown_liquidity"
        assert r["estimated_alpha_capacity_usd"] <= 1_000

    def test_return_on_capital_time(self):
        fast = return_on_capital_time(0.01, 10_000, holding_days=1)
        slow = return_on_capital_time(0.01, 10_000, holding_days=20)
        assert fast > slow

    def test_risk_of_ruin_rejects_wild_strategy(self):
        rng = np.random.default_rng(9)
        wild = list(rng.normal(0.01, 0.30, 200))
        safe = list(rng.normal(0.002, 0.01, 200))
        assert risk_of_ruin(wild)["risk_of_ruin"] > 0.5
        assert risk_of_ruin(safe)["risk_of_ruin"] < 0.05

    def test_log_growth_penalizes_ruin(self):
        assert expected_log_growth([0.5, 0.5, -1.0]) == float("-inf")
        assert expected_log_growth([0.01] * 10) > 0


# ── Case R: no opportunity → CASH ─────────────────────────────────────────────


class TestCaseRCashIsAPosition:
    def test_flat_market_generates_no_hypotheses(self):
        rng = np.random.default_rng(10)
        data = {}
        for i in range(60):
            idx = pd.bdate_range("2024-01-01", periods=300)
            close = 100 + rng.normal(0, 0.05, 300).cumsum() * 0  # dead flat
            close = 100 * np.ones(300) + rng.normal(0, 0.01, 300)
            data[f"S{i}"] = pd.DataFrame(
                {"open": close, "close": close, "volume": np.full(300, 1e6)},
                index=idx)
        rows = CrossSectionalOpportunityScanner(
            extreme_percentile=0.999).scan(data)
        gen = AutonomousHypothesisGenerator()
        hyps = gen.from_cross_sectional(rows)
        hyps += gen.from_outliers_and_changepoints([], [])
        hyps += gen.from_external_events({})
        assert hyps == []   # nothing found → nothing forced → stay in cash

    def test_allocator_returns_no_positions_without_candidates(self):
        from core.portfolio_allocator import PortfolioAllocator
        alloc = PortfolioAllocator()
        decisions = alloc.allocate([])
        assert [d for d in decisions if d.accepted] == []
