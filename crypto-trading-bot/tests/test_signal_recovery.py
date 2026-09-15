from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from core.data_quality import DataQualityMonitor
from core.llm_orchestrator import LLMOrchestrator
from core.ohlcv import normalize_ohlcv
from core.paper_maintenance import PaperMaintenance
from core.universe_scanner import ScannerConfig
from data.fetcher import _add_ohlcv_alias_columns
from data.history import verify_history_coverage


def candles():
    prices = pd.Series([100 + i * .01 for i in range(60)]).to_numpy()
    return pd.DataFrame(dict(open=prices, high=prices + .1, low=prices - .1,
                             close=prices, volume=1000),
                        index=pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=60, freq='1h'))


def test_real_fetcher_aliases_pass_hourly_gate():
    from ultimate_bot_v3_llm import LLMTradingBot
    bot = SimpleNamespace(data_quality_monitor=DataQualityMonitor())
    df = _add_ohlcv_alias_columns(candles())
    assert LLMTradingBot.check_market_conditions(bot, 'TEST', df) == (True, 'OK')
    assert len(normalize_ohlcv(df).columns) == 5


def test_normal_hourly_crypto_range_is_not_rejected():
    from ultimate_bot_v3_llm import LLMTradingBot
    df = candles()
    df['high'] = df['close'] * 1.0085
    df['low'] = df['close'] * 0.9915
    df.attrs['timeframe'] = '1h'
    bot = SimpleNamespace(
        data_quality_monitor=DataQualityMonitor(),
        scanner=SimpleNamespace(config=ScannerConfig()),
    )
    assert LLMTradingBot.check_market_conditions(bot, 'TEST', df) == (True, 'OK')


def test_scanner_default_matches_five_minute_volatility(monkeypatch):
    import core.universe_scanner as scanner_module
    import data.fetcher as fetcher
    df = candles()
    df.index = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=60, freq='5min')
    df['volume'] = 100_000
    monkeypatch.setattr(scanner_module, 'DEFAULT_CRYPTO_UNIVERSE', ['TEST-USD'])
    monkeypatch.setattr(fetcher, 'fetch_latest_market_data', lambda *a, **k: df)
    scanner = scanner_module.UniverseScanner()
    assert scanner.get_symbols() == ['TEST-USD']


def test_conflicting_aliases_fail_closed():
    df = _add_ohlcv_alias_columns(candles())
    df['Close'] += 1
    result = DataQualityMonitor().check(df, timeframe='1h')
    assert not result.passed
    assert 'Conflicting OHLCV aliases' in result.issues[0]


def test_rejection_is_visible_at_info(caplog):
    from ultimate_bot_v3_llm import LLMTradingBot
    bot = SimpleNamespace(check_market_conditions=lambda *args: (False, 'test rejection'))
    import ultimate_bot_v3_llm as module
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(module, 'fetch_latest_market_data', lambda *a, **k: candles())
        with caplog.at_level('INFO'):
            assert LLMTradingBot.analyze_trade_opportunity(bot, 'TEST') is None
    assert 'NO TRADE — test rejection' in caplog.text


def test_scanner_rejections_are_visible(caplog, monkeypatch):
    import core.universe_scanner as scanner_module
    import data.fetcher as fetcher
    monkeypatch.setattr(scanner_module, 'DEFAULT_CRYPTO_UNIVERSE', ['TEST'])
    monkeypatch.setattr(fetcher, 'fetch_latest_market_data', lambda *a, **k: None)
    scanner = scanner_module.UniverseScanner()
    with caplog.at_level('INFO'):
        assert scanner.scan() == []
    assert scanner.last_rejections == {'TEST': 'INSUFFICIENT_HISTORY'}
    assert 'INSUFFICIENT_HISTORY' in caplog.text


def test_research_coverage_rejects_short_and_stale_history():
    now = datetime(2026, 9, 15, tzinfo=timezone.utc)
    short = pd.DataFrame(index=pd.date_range(end=now, periods=350, freq='1D'))
    full = pd.DataFrame(index=pd.date_range(end=now, periods=731, freq='1D'))
    assert not verify_history_coverage(short, now=now)[0]
    assert verify_history_coverage(full, now=now)[0]
    assert not verify_history_coverage(full.iloc[:-10], now=now)[0]


def test_paginated_coinbase_history(monkeypatch):
    import data.fetcher as fetcher
    calls = []
    def get(url, params, timeout):
        calls.append(params)
        idx = pd.date_range(params['start'], params['end'], freq='1D')
        rows = [[int(t.timestamp()), 99, 101, 100, 100, 10] for t in idx]
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: rows)
    monkeypatch.setattr(fetcher.requests, 'get', get)
    monkeypatch.setattr(fetcher.time, 'sleep', lambda _: None)
    df = fetcher._fetch_coinbase_candles('BTC-USD', '1d', period='2y')
    assert len(calls) == 3
    assert df.index.is_unique
    assert verify_history_coverage(df)[0]


def test_scheduler_runs_initially_and_weekly_survives_restart(tmp_path):
    path = tmp_path / 'schedule.json'
    now = datetime(2026, 9, 14)
    bot = Mock()
    bot.run_weekly_research_campaign.return_value = {'new_paper_alphas': []}
    PaperMaintenance(path, now, background=False).tick(bot, now)
    PaperMaintenance(path, now, background=False).tick(bot, now)
    assert bot.run_weekly_research_campaign.call_count == 1
    PaperMaintenance(path, now, background=False).tick(bot, datetime(2026, 9, 21))
    assert bot.run_weekly_research_campaign.call_count == 2
    bot.end_of_day_learning.assert_called_once()


def test_failed_research_retries_next_day(tmp_path):
    bot = Mock()
    bot.run_weekly_research_campaign.return_value = None
    now = datetime(2026, 9, 14)
    scheduler = PaperMaintenance(tmp_path / 'schedule.json', now, background=False)
    scheduler.tick(bot, now)
    scheduler.tick(bot, now)
    assert bot.run_weekly_research_campaign.call_count == 1
    scheduler.tick(bot, datetime(2026, 9, 15))
    assert bot.run_weekly_research_campaign.call_count == 2


def test_interrupted_legacy_research_retries_after_restart(tmp_path):
    path = tmp_path / 'schedule.json'
    path.write_text('{"attempt_day": "2026-09-14"}')
    bot = Mock()
    bot.run_weekly_research_campaign.return_value = {}
    scheduler = PaperMaintenance(path, datetime(2026, 9, 14), background=False)
    scheduler.tick(bot, datetime(2026, 9, 14))
    bot.run_weekly_research_campaign.assert_called_once()


def test_paper_entrypoint_schedules_research_before_cycle(tmp_path, monkeypatch):
    import paper_trade_v3 as runner
    from core.broker import PaperBroker
    now = datetime(2026, 9, 14)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('TRADING_MODE', 'claude_hf')
    bot = Mock()
    bot.broker = PaperBroker(starting_cash=2000)
    bot.capital = 2000
    bot.run_weekly_research_campaign.return_value = {}
    bot.run_trading_cycle.side_effect = KeyboardInterrupt
    monkeypatch.setattr(runner, 'LLMTradingBot', lambda **kwargs: bot)
    monkeypatch.setattr(
        runner, 'PaperMaintenance',
        lambda: PaperMaintenance(tmp_path / 'schedule.json', now, background=False))
    runner.main()
    bot._initialize_core_accounting.assert_called_once()
    bot._startup_reconciliation.assert_called_once()
    bot.run_weekly_research_campaign.assert_called_once()
    names = [c[0] for c in bot.mock_calls]
    assert names.index('run_weekly_research_campaign') < names.index('run_trading_cycle')
    import os
    assert os.environ['TRADING_MODE'] == 'PAPER'


def test_paper_entrypoint_refuses_live_before_construction(monkeypatch):
    import paper_trade_v3 as runner
    constructor = Mock()
    monkeypatch.setattr(runner, 'LLMTradingBot', constructor)
    monkeypatch.setenv('TRADING_MODE', 'LIVE')
    with pytest.raises(RuntimeError, match='refuses'):
        runner.main()
    constructor.assert_not_called()


def test_paper_pid_claim_replaces_stale_and_blocks_duplicate(tmp_path):
    import os
    from paper_trade_v3 import _claim_pid, _release_pid
    path = tmp_path / 'bot.pid'
    path.write_text('99999999')
    claimed = _claim_pid(path)
    assert int(path.read_text()) == os.getpid()
    path.write_text('1')
    with pytest.raises(RuntimeError, match='already running'):
        _claim_pid(path)
    path.write_text(str(os.getpid()))
    _release_pid(claimed)
    assert not path.exists()


def test_bot_execute_trade_uses_paper_fill_and_persists(tmp_path, monkeypatch):
    import ultimate_bot_v3_llm as module
    import sqlite3
    from core.broker import PaperBroker
    from core.position_manager import PositionManager
    db_path = tmp_path / 'paper.sqlite'
    # Reproduce the production database's original schema, whose required
    # `signal` column exposed the post-fill persistence crash.
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE positions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "symbol TEXT NOT NULL, signal TEXT NOT NULL, size REAL NOT NULL, "
            "entry_price REAL NOT NULL, entry_time TIMESTAMP NOT NULL, "
            "status TEXT DEFAULT 'OPEN')")
    bot = module.LLMTradingBot.__new__(module.LLMTradingBot)
    bot.broker = PaperBroker(starting_cash=2000)
    bot.positions = PositionManager(db_path=str(db_path))
    bot._db_health = 'HEALTHY'
    bot._check_live_authorization = lambda _: (True, 'paper')
    bot._calculate_atr = lambda _: 1.0
    bot._position_size_scale = 1.0
    bot.drawdown_budgets = SimpleNamespace(allocation_multiplier=lambda *a: 1.0)
    bot.safety = Mock()
    bot.safety.check_can_trade.return_value = (True, 'OK')
    monkeypatch.setattr(module, 'fetch_latest_market_data', lambda *a, **k: candles())
    decision = dict(signal='BUY', price=100., size=100., strategy_signals=[],
                    strategies=['test'], stop_loss=98., take_profit=103., kronos_signal=None)
    pos_id = bot.execute_trade('TEST-USD', decision)
    assert pos_id is not None
    rows = bot.positions.get_open_positions()
    assert len(rows) == 1
    assert rows[0]['entry_price'] == pytest.approx(100.05)
    assert rows[0]['signal'] == 'BUY'
    bot.safety.on_position_open.assert_called_once()


def test_llm_learning_hooks_return_stable_local_summaries():
    llm = LLMOrchestrator()
    empty = llm.end_of_day_analysis([], {}, ['mean_reversion'])
    assert empty['performance']['trade_count'] == 0
    assert empty['failures'] == []

    review = llm.end_of_day_analysis(
        [{'pnl': 10}, {'pnl': -4}, {'pnl': 6}],
        {},
        ['mean_reversion'],
    )
    assert review['performance']['trade_count'] == 3
    assert review['performance']['average_pnl'] == pytest.approx(4.0)
    assert review['performance']['should_adapt'] is False
    assert review['successes']
