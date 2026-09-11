from research.opportunity_lifecycle import (
    EdgeMonitor,
    PaperTradingExperiment,
    StrategyLifecycleManager,
    StrategyGraveyard,
    OpportunityBoard,
)


def test_paper_experiment_tracks_expected_vs_actual():
    exp = PaperTradingExperiment(
        strategy_id="strategy-a",
        start_date="2026-01-01",
        expected_performance=0.12,
        actual_performance=0.09,
        expected_win_rate=0.60,
        actual_win_rate=0.58,
        expected_drawdown=0.12,
        actual_drawdown=0.15,
        sample_count=40,
    )
    assert exp.strategy_id == "strategy-a"
    assert exp.actual_performance < exp.expected_performance
    assert exp.status in {"PAPER_TRADING", "DEGRADED"}


def test_edge_monitor_classifies_health():
    monitor = EdgeMonitor()
    health = monitor.evaluate_edge(
        expected_performance=0.10,
        realized_performance=0.09,
        rolling_sharpe=1.2,
        rolling_win_rate=0.54,
        rolling_profit_factor=1.7,
        rolling_drawdown=0.12,
        slippage=0.002,
        signal_frequency=0.9,
    )
    assert health["edge_health_score"] >= 0
    assert health["state"] in {"HEALTHY", "WATCH"}


def test_strategy_lifecycle_rejects_live_promotion_without_gate():
    mgr = StrategyLifecycleManager(auto_live_promotion=False)
    result = mgr.evaluate_promotion(
        strategy_id="strategy-b",
        paper_trades=10,
        sharpe=0.4,
        drawdown=0.22,
        sample_size=10,
    )
    assert result["eligible_for_live"] is False
    assert "gates" in result["reason"].lower()


def test_graveyard_prevents_duplicate_rejection():
    graveyard = StrategyGraveyard()
    graveyard.register("duplicate-signal", "look-ahead leak", {"score": 0.5})
    assert graveyard.is_known("duplicate-signal") is True
    assert graveyard.lookup("duplicate-signal")["reason"] == "look-ahead leak"


def test_opportunity_board_ranks_candidates():
    board = OpportunityBoard()
    board.push({"opportunity": "alpha-1", "score": 82.0, "status": "PAPER"})
    board.push({"opportunity": "alpha-2", "score": 88.0, "status": "PAPER"})
    ranked = board.rank()
    assert ranked[0]["opportunity"] == "alpha-2"
    assert ranked[0]["score"] >= ranked[1]["score"]
