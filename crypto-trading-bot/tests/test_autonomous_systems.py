"""
Tests for the autonomous system layer (Phases 3-17).

Covers:
    ChampionChallenger — register, evaluate, promote, version history
    ModelTrainer       — should_retrain logic
    StrategyHealthMonitor — compute, alerts, pausing
    DriftDetector      — PSI computation, report
    OrderManager       — submission, fill, reconciliation
    ExecutionQualityEngine — record, compute slippage
    TradeExplainabilityStore — record, retrieve, format
    ResearchEngine     — analyze, get_open_hypotheses
    DailyReviewer      — run, store, retrieve
    WeeklyReviewer     — run, store, retrieve
    PerformanceDashboard — render all sections
"""
import os
import sqlite3
import tempfile
import unittest
from dataclasses import dataclass
from typing import Dict, List
from unittest.mock import MagicMock, patch


# ─── helpers ────────────────────────────────────────────────────────────────────

def _tmp_db():
    fd, path = tempfile.mkstemp(suffix='.sqlite')
    os.close(fd)
    return path


# ─── ChampionChallenger ──────────────────────────────────────────────────────────

class TestChampionChallenger(unittest.TestCase):

    def setUp(self):
        self.db = _tmp_db()
        from core.champion_challenger import ChampionChallenger
        self.cc = ChampionChallenger(model_dir=tempfile.mkdtemp(), db_path=self.db)

    def tearDown(self):
        os.unlink(self.db)

    def test_no_champion_initially(self):
        self.assertIsNone(self.cc.get_champion())

    def test_register_challenger_creates_record(self):
        mock_model = MagicMock()
        mock_model.__reduce__ = MagicMock(return_value=(MagicMock, ()))
        version = self.cc.register_challenger(mock_model, training_rows=200,
                                              metrics={'cv_auc': 0.65})
        self.assertIsNotNone(version)
        self.assertEqual(version.training_rows, 200)

    def test_no_champion_promotes_directly(self):
        mock_model = MagicMock()
        version = self.cc.register_challenger(mock_model, training_rows=100,
                                              metrics={'cv_auc': 0.60})
        # When no champion exists, evaluate_challenger promotes immediately
        result = self.cc.evaluate_challenger(version, oos_metrics={'cv_auc': 0.60})
        self.assertTrue(result)
        # Champion now exists
        self.assertIsNotNone(self.cc.get_champion())

    def test_insufficient_improvement_rejects(self):
        # Promote a champion with known OOS metrics directly
        mock_model = MagicMock()
        version = self.cc.register_challenger(mock_model, training_rows=100,
                                              metrics={'cv_auc': 0.70})
        version.oos_metrics = {'cv_auc': 0.70}
        self.cc.promote_challenger(version)  # champion OOS AUC = 0.70
        # Challenger with only 0.01 improvement — below 0.02 threshold
        v2 = self.cc.register_challenger(mock_model, training_rows=150,
                                          metrics={'cv_auc': 0.71})
        accepted = self.cc.evaluate_challenger(v2, oos_metrics={'cv_auc': 0.71})
        self.assertFalse(accepted)  # 0.01 < MIN_IMPROVEMENT_AUC (0.02)

    def test_sufficient_improvement_accepted(self):
        mock_model = MagicMock()
        v1 = self.cc.register_challenger(mock_model, training_rows=100,
                                          metrics={'cv_auc': 0.60})
        self.cc.evaluate_challenger(v1, oos_metrics={'cv_auc': 0.60})
        v2 = self.cc.register_challenger(mock_model, training_rows=150,
                                          metrics={'cv_auc': 0.63})
        accepted = self.cc.evaluate_challenger(v2, oos_metrics={'cv_auc': 0.63})
        self.assertTrue(accepted)

    def test_version_history_returns_list(self):
        history = self.cc.get_version_history()
        self.assertIsInstance(history, list)

    def test_should_promote_insufficient_paper_trades(self):
        mock_model = MagicMock()
        # Need a champion so should_promote doesn't return True immediately
        v_champ = self.cc.register_challenger(mock_model, training_rows=100,
                                               metrics={'cv_auc': 0.60})
        v_champ.oos_metrics = {'cv_auc': 0.60}
        self.cc.promote_challenger(v_champ)
        # Now create challenger with insufficient paper trades
        v = self.cc.register_challenger(mock_model, training_rows=150,
                                         metrics={'cv_auc': 0.65})
        v.paper_metrics = {'paper_trades': 5, 'win_rate': 0.60}
        should, reason = self.cc.should_promote(v)
        self.assertFalse(should)
        self.assertIn('Insufficient', reason)

    def test_champion_summary_structure(self):
        summary = self.cc.get_champion_summary()
        self.assertIn('champion', summary)
        self.assertIn('challenger', summary)
        self.assertIn('history', summary)


# ─── ModelTrainer ────────────────────────────────────────────────────────────────

class TestModelTrainer(unittest.TestCase):

    def setUp(self):
        self.db = _tmp_db()

    def tearDown(self):
        os.unlink(self.db)

    def test_should_retrain_force(self):
        from core.model_trainer import ModelTrainer
        trainer = ModelTrainer(db_path=self.db)
        should, reason = trainer.should_retrain(force=True)
        self.assertTrue(should)

    def test_should_retrain_drift(self):
        from core.model_trainer import ModelTrainer
        trainer = ModelTrainer(db_path=self.db)
        should, reason = trainer.should_retrain(trigger='drift')
        self.assertTrue(should)

    def test_should_not_retrain_without_data(self):
        from core.model_trainer import ModelTrainer
        mock_fs = MagicMock()
        mock_fs.stats.return_value = {'with_outcomes': 10}
        trainer = ModelTrainer(feature_store=mock_fs, db_path=self.db)
        should, reason = trainer.should_retrain()
        self.assertFalse(should)

    def test_should_retrain_with_enough_data(self):
        from core.model_trainer import ModelTrainer
        mock_fs = MagicMock()
        mock_fs.stats.return_value = {'with_outcomes': 200}
        trainer = ModelTrainer(feature_store=mock_fs, db_path=self.db)
        # 200 total, last_trained=0, new=200 ≥ 50
        should, reason = trainer.should_retrain()
        self.assertTrue(should)


# ─── StrategyHealthMonitor ───────────────────────────────────────────────────────

class TestStrategyHealthMonitor(unittest.TestCase):

    def setUp(self):
        self.db = _tmp_db()
        # Populate positions table with test data
        with sqlite3.connect(self.db) as conn:
            conn.execute("""
                CREATE TABLE positions (
                    id TEXT PRIMARY KEY, strategy TEXT, status TEXT,
                    net_pnl REAL, holding_hours REAL, mfe_pct REAL, mae_pct REAL,
                    exit_time TEXT, market_regime TEXT
                )
            """)
            # 20 closed trades for 'test_strat', alternating win/loss
            from datetime import datetime, timezone
            for i in range(20):
                pnl = 10.0 if i % 2 == 0 else -5.0
                conn.execute("INSERT INTO positions VALUES (?,?,?,?,?,?,?,?,?)",
                             (str(i), 'test_strat', 'CLOSED', pnl, 2.0, 0.01, -0.005,
                              datetime.now(timezone.utc).isoformat(), 'trending'))
            conn.commit()

    def tearDown(self):
        os.unlink(self.db)

    def test_compute_health_healthy_strategy(self):
        from core.strategy_health import StrategyHealthMonitor
        monitor = StrategyHealthMonitor(db_path=self.db, window_days=30)
        health_map = monitor.run()
        self.assertIn('test_strat', health_map)
        h = health_map['test_strat']
        # 10/20 wins = 50% > 42% threshold
        self.assertAlmostEqual(h.win_rate, 0.5, places=1)
        self.assertTrue(h.healthy)

    def test_is_paused_after_degradation(self):
        from core.strategy_health import StrategyHealthMonitor
        monitor = StrategyHealthMonitor(db_path=self.db, window_days=30)
        # Force all losses
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE positions SET net_pnl=-10 WHERE strategy='test_strat'")
        monitor.run()
        self.assertTrue(monitor.is_paused('test_strat'))

    def test_get_alerts_returns_list(self):
        from core.strategy_health import StrategyHealthMonitor
        monitor = StrategyHealthMonitor(db_path=self.db, window_days=30)
        monitor.run()
        alerts = monitor.get_alerts()
        self.assertIsInstance(alerts, list)


# ─── DriftDetector ───────────────────────────────────────────────────────────────

class TestDriftDetector(unittest.TestCase):

    def setUp(self):
        self.db = _tmp_db()

    def tearDown(self):
        os.unlink(self.db)

    def test_no_data_returns_no_drift(self):
        from core.drift_detector import DriftDetector
        dd = DriftDetector(db_path=self.db)
        report = dd.run()
        self.assertFalse(report.drift_detected)
        self.assertEqual(report.allocation_factor, 1.0)

    def test_psi_same_distribution_zero(self):
        from core.drift_detector import DriftDetector
        dd = DriftDetector(db_path=self.db)
        ref = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
        psi = dd._psi(ref, ref)
        self.assertAlmostEqual(psi, 0.0, places=1)

    def test_psi_different_distribution_high(self):
        from core.drift_detector import DriftDetector
        dd = DriftDetector(db_path=self.db)
        ref = [1.0, 1.1, 1.2, 1.3, 1.4] * 10
        cur = [10.0, 11.0, 12.0, 13.0, 14.0] * 10
        psi = dd._psi(ref, cur)
        self.assertGreater(psi, 0.25)

    def test_allocation_factor_default(self):
        from core.drift_detector import DriftDetector
        dd = DriftDetector(db_path=self.db)
        self.assertEqual(dd.get_allocation_factor(), 1.0)


# ─── OrderManager ────────────────────────────────────────────────────────────────

class TestOrderManager(unittest.TestCase):

    def setUp(self):
        self.db = _tmp_db()
        from core.order_manager import OrderManager
        self.om = OrderManager(db_path=self.db)

    def tearDown(self):
        os.unlink(self.db)

    def test_record_submission_returns_id(self):
        order_id = self.om.record_submission('BTC-USD', 'BUY', 'MARKET', 0.01)
        self.assertIsInstance(order_id, str)
        self.assertEqual(len(order_id), 36)

    def test_record_fill_updates_status(self):
        order_id = self.om.record_submission('ETH-USD', 'BUY', 'MARKET', 0.1)
        self.om.record_fill(order_id, fill_price=3000.0, filled_qty=0.1)
        with sqlite3.connect(self.db) as conn:
            row = conn.execute("SELECT status FROM orders WHERE order_id=?",
                               (order_id,)).fetchone()
        self.assertEqual(row[0], 'FILLED')

    def test_record_rejection(self):
        order_id = self.om.record_submission('SOL-USD', 'SELL', 'MARKET', 1.0)
        self.om.record_rejection(order_id, 'Insufficient margin')
        with sqlite3.connect(self.db) as conn:
            row = conn.execute("SELECT status, reject_reason FROM orders WHERE order_id=?",
                               (order_id,)).fetchone()
        self.assertEqual(row[0], 'REJECTED')
        self.assertIn('margin', row[1])

    def test_get_open_orders_filters_correctly(self):
        id1 = self.om.record_submission('BTC-USD', 'BUY', 'MARKET', 0.01)
        id2 = self.om.record_submission('ETH-USD', 'BUY', 'MARKET', 0.1)
        self.om.record_fill(id1, 50000.0, 0.01)
        open_orders = self.om.get_open_orders()
        ids = [o.order_id for o in open_orders]
        self.assertNotIn(id1, ids)
        self.assertIn(id2, ids)

    def test_stats_counts_correctly(self):
        id1 = self.om.record_submission('BTC-USD', 'BUY', 'MARKET', 0.01)
        id2 = self.om.record_submission('ETH-USD', 'BUY', 'MARKET', 0.1)
        self.om.record_fill(id1, 50000.0, 0.01)
        self.om.record_rejection(id2, 'Test')
        stats = self.om.stats()
        self.assertEqual(stats['total'], 2)
        self.assertEqual(stats['filled'], 1)
        self.assertEqual(stats['rejected'], 1)

    def test_detect_stale_marks_old_orders(self):
        from datetime import datetime, timedelta, timezone
        order_id = self.om.record_submission('BTC-USD', 'BUY', 'MARKET', 0.01)
        # Backdate the order to 2 hours ago
        old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE orders SET created_at=? WHERE order_id=?",
                         (old_time, order_id))
        stale = self.om.detect_stale()
        stale_ids = [o.order_id for o in stale]
        self.assertIn(order_id, stale_ids)


# ─── ExecutionQualityEngine ──────────────────────────────────────────────────────

class TestExecutionQualityEngine(unittest.TestCase):

    def setUp(self):
        self.db = _tmp_db()
        from core.execution_quality import ExecutionQualityEngine
        self.eq = ExecutionQualityEngine(db_path=self.db)

    def tearDown(self):
        os.unlink(self.db)

    def test_empty_period_returns_zero_fills(self):
        report = self.eq.run(period_days=7)
        self.assertEqual(report.fills_analyzed, 0)

    def test_record_fill_then_analyze(self):
        import uuid
        self.eq.record_fill(
            order_id=str(uuid.uuid4()),
            symbol='BTC-USD', side='BUY',
            signal_price=50000.0, fill_price=50050.0,
            quantity=0.1, fee=5.0, latency_ms=120.0, spread_pct=0.001,
        )
        report = self.eq.run(period_days=7)
        self.assertEqual(report.fills_analyzed, 1)
        self.assertGreater(report.slippage.mean_bps, 0)

    def test_adverse_slippage_detected(self):
        import uuid
        # BUY at 50100 when signal was 50000 = 2bps adverse slippage
        self.eq.record_fill(
            order_id=str(uuid.uuid4()),
            symbol='BTC-USD', side='BUY',
            signal_price=50000.0, fill_price=50100.0,
            quantity=0.1, fee=1.0,
        )
        report = self.eq.run(period_days=7)
        self.assertGreater(report.slippage.pct_adverse, 0.0)

    def test_summarize_returns_string(self):
        summary = self.eq.summarize(period_days=7)
        self.assertIsInstance(summary, str)
        self.assertIn('Execution Quality', summary)


# ─── TradeExplainabilityStore ────────────────────────────────────────────────────

class TestTradeExplainabilityStore(unittest.TestCase):

    def setUp(self):
        self.db = _tmp_db()
        from core.trade_explainability import TradeExplainabilityStore
        self.store = TradeExplainabilityStore(db_path=self.db)

    def tearDown(self):
        os.unlink(self.db)

    def test_record_and_retrieve(self):
        from core.trade_explainability import TradeExplanation
        expl = TradeExplanation(
            position_id='pos-123',
            symbol='BTC-USD', strategy='momentum',
            why_chosen='High momentum signal',
            alternatives_rejected=[{'symbol': 'ETH-USD', 'score': 0.4, 'reason': 'WATCH'}],
            expected_value=50.0, opportunity_score=0.72,
            portfolio_reason='Approved $500', risk_reason='Safety passed',
            claude_explanation='BTC showed strong breakout',
            metamodel_decision='TRADE', metamodel_confidence=0.80, metamodel_ev=0.05,
            kronos_confidence=0.75, strategy_votes={'momentum': 'BUY'},
            regime='trending', ranker_rank=1, total_candidates=5,
        )
        self.store.record(expl)
        retrieved = self.store.get('pos-123')
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.symbol, 'BTC-USD')
        self.assertEqual(retrieved.metamodel_confidence, 0.80)

    def test_deduplication(self):
        from core.trade_explainability import TradeExplanation
        expl = TradeExplanation(
            position_id='pos-dup',
            symbol='ETH-USD', strategy='breakout',
            why_chosen='First record', alternatives_rejected=[],
            expected_value=10.0, opportunity_score=0.5,
            portfolio_reason='', risk_reason='',
            claude_explanation='', metamodel_decision='TRADE',
            metamodel_confidence=0.6, metamodel_ev=0.0,
            kronos_confidence=0.0, strategy_votes={},
            regime='trending', ranker_rank=2, total_candidates=3,
        )
        self.store.record(expl)
        expl.why_chosen = 'Updated record'
        self.store.record(expl)
        # Should only have one record
        records = self.store.get_recent(limit=10)
        ids = [r.position_id for r in records]
        self.assertEqual(ids.count('pos-dup'), 1)

    def test_format_explanation_found(self):
        from core.trade_explainability import TradeExplanation
        expl = TradeExplanation(
            position_id='pos-format',
            symbol='SOL-USD', strategy='mean_reversion',
            why_chosen='Mean reversion signal', alternatives_rejected=[],
            expected_value=25.0, opportunity_score=0.65,
            portfolio_reason='Approved', risk_reason='OK',
            claude_explanation='SOL oversold', metamodel_decision='TRADE',
            metamodel_confidence=0.70, metamodel_ev=0.03,
            kronos_confidence=0.0, strategy_votes={'mean_reversion': 'BUY'},
            regime='ranging', ranker_rank=1, total_candidates=4,
        )
        self.store.record(expl)
        fmt = self.store.format_explanation('pos-format')
        self.assertIn('SOL-USD', fmt)
        self.assertIn('mean_reversion', fmt)

    def test_format_not_found(self):
        result = self.store.format_explanation('nonexistent')
        self.assertIn('No explanation found', result)

    def test_record_from_context(self):
        mock_pred = MagicMock()
        mock_pred.decision      = 'TRADE'
        mock_pred.confidence    = 0.75
        mock_pred.expected_value = 0.04
        self.store.record_from_context(
            position_id='pos-ctx',
            symbol='BTC-USD', strategy='breakout',
            meta_prediction=mock_pred,
            regime='trending',
        )
        retrieved = self.store.get('pos-ctx')
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.metamodel_decision, 'TRADE')


# ─── DailyReviewer ──────────────────────────────────────────────────────────────

class TestDailyReviewer(unittest.TestCase):

    def setUp(self):
        self.db = _tmp_db()
        with sqlite3.connect(self.db) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS positions
                (id TEXT, strategy TEXT, status TEXT, net_pnl REAL,
                 exit_time TEXT, market_regime TEXT, holding_hours REAL,
                 mfe_pct REAL, mae_pct REAL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS daily_reviews
                (date TEXT PRIMARY KEY, pnl_summary TEXT, strategy_health TEXT,
                 execution_quality TEXT, research_output TEXT, drift_report TEXT,
                 regime_summary TEXT, feature_importance TEXT, recommendations TEXT,
                 claude_narrative TEXT, generated_at TEXT)""")

    def tearDown(self):
        os.unlink(self.db)

    def test_run_stores_review(self):
        from core.daily_review import DailyReviewer
        reviewer = DailyReviewer(db_path=self.db)
        review = reviewer.run(date='2025-01-01')
        self.assertEqual(review.date, '2025-01-01')
        retrieved = reviewer.get_review('2025-01-01')
        self.assertIsNotNone(retrieved)

    def test_run_is_idempotent(self):
        from core.daily_review import DailyReviewer
        reviewer = DailyReviewer(db_path=self.db)
        reviewer.run(date='2025-01-02')
        reviewer.run(date='2025-01-02')  # second run should update, not error
        retrieved = reviewer.get_review('2025-01-02')
        self.assertIsNotNone(retrieved)

    def test_recommendations_generated(self):
        from core.daily_review import DailyReviewer
        reviewer = DailyReviewer(db_path=self.db)
        review = reviewer.run(date='2025-01-03')
        self.assertIsInstance(review.recommendations, list)
        self.assertGreater(len(review.recommendations), 0)

    def test_format_review_output(self):
        from core.daily_review import DailyReviewer
        reviewer = DailyReviewer(db_path=self.db)
        review = reviewer.run(date='2025-01-04')
        formatted = reviewer.format_review(review)
        self.assertIn('2025-01-04', formatted)


# ─── WeeklyReviewer ─────────────────────────────────────────────────────────────

class TestWeeklyReviewer(unittest.TestCase):

    def setUp(self):
        self.db = _tmp_db()
        with sqlite3.connect(self.db) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS positions
                (id TEXT, strategy TEXT, status TEXT, net_pnl REAL,
                 exit_time TEXT, market_regime TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS experiments
                (id TEXT, description TEXT, status TEXT, created_at TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS weekly_reviews
                (week_start TEXT PRIMARY KEY, week_end TEXT, performance TEXT,
                 strategy_breakdown TEXT, model_changes TEXT, research_completed TEXT,
                 experiments_promoted TEXT, key_metrics TEXT, recommendations TEXT,
                 claude_narrative TEXT, generated_at TEXT)""")

    def tearDown(self):
        os.unlink(self.db)

    def test_run_stores_review(self):
        from core.weekly_review import WeeklyReviewer
        reviewer = WeeklyReviewer(db_path=self.db)
        review = reviewer.run(week_start='2025-01-06')
        self.assertEqual(review.week_start, '2025-01-06')
        retrieved = reviewer.get_review('2025-01-06')
        self.assertIsNotNone(retrieved)

    def test_recommendations_generated(self):
        from core.weekly_review import WeeklyReviewer
        reviewer = WeeklyReviewer(db_path=self.db)
        review = reviewer.run(week_start='2025-01-13')
        self.assertIsInstance(review.recommendations, list)
        self.assertGreater(len(review.recommendations), 0)

    def test_format_review_includes_week(self):
        from core.weekly_review import WeeklyReviewer
        reviewer = WeeklyReviewer(db_path=self.db)
        review = reviewer.run(week_start='2025-01-20')
        formatted = reviewer.format_review(review)
        self.assertIn('2025-01-20', formatted)


# ─── PerformanceDashboard ────────────────────────────────────────────────────────

class TestPerformanceDashboard(unittest.TestCase):

    def test_render_with_no_components(self):
        from dashboard.performance_dashboard import PerformanceDashboard
        db = PerformanceDashboard(capital=10000.0)
        output = db.run()
        self.assertIsInstance(output, str)
        self.assertIn('DASHBOARD', output)

    def test_render_specific_sections(self):
        from dashboard.performance_dashboard import PerformanceDashboard
        db = PerformanceDashboard(capital=10000.0)
        output = db.run(sections=['header', 'portfolio'])
        self.assertIn('PORTFOLIO', output)
        self.assertNotIn('EXECUTION QUALITY', output)

    def test_section_error_does_not_crash(self):
        from dashboard.performance_dashboard import PerformanceDashboard
        mock_sh = MagicMock()
        mock_sh.summary.side_effect = RuntimeError("DB error")
        db = PerformanceDashboard(strategy_health=mock_sh)
        output = db.run(sections=['strategies'])
        # Should contain the section name even if error occurred
        self.assertIn('STRATEGY', output)


if __name__ == '__main__':
    unittest.main()
