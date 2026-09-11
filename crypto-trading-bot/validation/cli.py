#!/usr/bin/env python3
"""
CLI entry points for validation, paper campaigns, and shadow mode.

Commands
--------
  python -m validation.cli campaign90      Start a 90-day paper validation campaign
  python -m validation.cli shadow          Start shadow mode (live data, no orders)
  python -m validation.cli report LABEL    Generate a validation report
  python -m validation.cli status          Show active campaigns + recent reports
  python -m validation.cli scorecard ID    Show daily scorecard for campaign ID

Examples
--------
  # Start a 90-day paper validation campaign
  python -m validation.cli campaign90 --capital 10000 --name "Baseline-90d"

  # Run shadow mode (live data, no orders submitted)
  python -m validation.cli shadow --mode SHADOW

  # Generate a validation report from trade history
  python -m validation.cli report "MyStrategy" --days 30

  # View status of all campaigns
  python -m validation.cli status
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure the project root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_campaign90(args):
    """Start a 90-day paper validation campaign."""
    from validation.paper_campaign import PaperCampaignManager

    mgr = PaperCampaignManager(db_path=args.db)

    # Load config snapshot
    config_snapshot = {
        'capital':       args.capital,
        'trading_mode':  'PAPER',
        'duration_days': 90,
        'freeze_timestamp': __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).isoformat(),
        'objective': (
            "PROVE: reliability, positive expectancy, risk control, "
            "execution accuracy, model stability, strategy stability, reproducibility"
        ),
    }

    import hashlib
    config_hash = hashlib.sha256(
        json.dumps(config_snapshot, sort_keys=True).encode()
    ).hexdigest()[:16]

    campaign = mgr.create(
        name=args.name,
        duration_days=90,
        starting_capital=args.capital,
        strategy_versions={},
        model_versions={},
        config_snapshot=config_snapshot,
        config_hash=config_hash,
        trading_mode='PAPER',
        notes=(
            "90-day validation campaign. Objective: prove edge, not maximize profit. "
            "DO NOT continuously modify strategy during this campaign."
        ),
    )
    mgr.start(campaign.campaign_id)

    print(f"\n{'='*60}")
    print(f"  90-DAY PAPER VALIDATION CAMPAIGN STARTED")
    print(f"{'='*60}")
    print(f"  Campaign ID:   {campaign.campaign_id}")
    print(f"  Config hash:   {config_hash}")
    print(f"  Capital:       ${args.capital:,.0f}")
    print(f"  Start:         {campaign.start_date}")
    print(f"  End:           {campaign.end_date}")
    print(f"{'='*60}")
    print(f"\nObjective: PROVE edge, not maximize profit.")
    print("Freeze the baseline config. New experiments run as CHALLENGERS.")
    print(f"\nCampaign ID for reference: {campaign.campaign_id}")


def cmd_shadow(args):
    """Start shadow mode — real decisions, no orders."""
    from core.shadow_mode import ShadowModeTracker, TradingMode

    tracker = ShadowModeTracker(db_path=args.db)
    mode = TradingMode[args.mode.upper()]
    tracker.set_mode(mode)

    print(f"\n{'='*60}")
    print(f"  SHADOW MODE ACTIVATED: {mode.value}")
    print(f"{'='*60}")
    print(f"  Live market data:  YES")
    print(f"  Real decisions:    YES")
    print(f"  Order submission:  {'YES' if tracker.should_submit_order() else 'NO'}")
    print(f"{'='*60}")

    if mode == TradingMode.SHADOW:
        print("\nSHADOW mode: full pipeline runs, NO orders submitted.")
        print("All decisions recorded for analysis.")
        print("\nNow run the bot:")
        print("  python ultimate_bot_v3_llm.py --mode SHADOW")
    elif mode == TradingMode.PAPER:
        print("\nPAPER mode: orders simulated locally.")
        print("\nNow run the bot:")
        print("  python ultimate_bot_v3_llm.py --mode PAPER")


def cmd_report(args):
    """Generate a validation report from trade history."""
    from validation.report import ValidationReportGenerator
    from validation.engine import Trade, _parse_dt
    import sqlite3

    print(f"\nGenerating validation report for '{args.label}' (last {args.days} days)...")

    db = args.db
    cutoff = (__import__('datetime').datetime.now(__import__('datetime').timezone.utc)
              - __import__('datetime').timedelta(days=args.days)).isoformat()

    try:
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM positions WHERE status='CLOSED' AND exit_time>=? "
                "ORDER BY exit_time",
                (cutoff,),
            ).fetchall()
        position_dicts = [dict(r) for r in rows]
    except Exception as e:
        print(f"ERROR: could not load positions from {db}: {e}")
        return

    gen = ValidationReportGenerator(capital=args.capital, db_path=db)
    ve  = gen._ve
    result = ve.evaluate_from_dicts(position_dicts, label=args.label)

    # Use evaluate_from_dicts internal Trade list
    trades = []
    for r in position_dicts:
        entry_t = _parse_dt(r.get('entry_time'))
        exit_t  = _parse_dt(r.get('exit_time'))
        if entry_t and exit_t:
            trades.append(Trade(
                entry_time=entry_t, exit_time=exit_t,
                pnl_net=float(r.get('net_pnl') or 0),
                pnl_gross=float(r.get('pnl') or 0),
                size=float(r.get('size') or 100),
                fees=float(r.get('fees') or 0),
                slippage=0.0,
                symbol=r.get('symbol', ''),
                strategy=r.get('strategy', ''),
            ))

    from validation.promotion_gates import PromotionStage
    report = gen.generate(
        label=args.label,
        report_type='strategy',
        trades=trades,
        promotion_stage=PromotionStage.BACKTEST_TO_WF,
        notes=f"Generated from {len(trades)} trades over last {args.days} days",
    )

    print("\n" + report.text_report())
    print(f"\nReport ID: {report.report_id}")
    print(f"Saved to:  {db}")


def cmd_status(args):
    """Show active campaigns and recent reports."""
    from validation.paper_campaign import PaperCampaignManager
    from validation.report import ValidationReportGenerator
    import sqlite3

    mgr = PaperCampaignManager(db_path=args.db)
    campaigns = mgr.list_all(limit=10)

    print(f"\n{'='*70}")
    print("  SYSTEM STATUS")
    print(f"{'='*70}")
    print(f"\n  CAMPAIGNS ({len(campaigns)}):")
    print(f"  {'ID':<10} {'Name':<20} {'Status':<12} {'Mode':<8} {'Start':<12} {'End':<12}")
    print("  " + "─" * 68)
    for c in campaigns:
        print(f"  {c['campaign_id'][:8]:<10} {c['name'][:20]:<20} {c['status']:<12} "
              f"{c['trading_mode']:<8} {c['start_date']:<12} {c['end_date']:<12}")

    try:
        gen = ValidationReportGenerator(db_path=args.db)
        reports = gen.list_reports(limit=5)
        print(f"\n  RECENT REPORTS ({len(reports)}):")
        for r in reports:
            status = "✓" if r['promotion_passed'] else "✗"
            print(f"  {status} {r['report_id'][:8]} {r['label'][:30]:<30} {r['generated_at'][:19]}")
    except Exception:
        pass

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Validation system CLI")
    parser.add_argument('--db', default='data/trade_memory.sqlite', help='SQLite database path')

    subs = parser.add_subparsers(dest='command')

    # campaign90
    p_c90 = subs.add_parser('campaign90', help='Start a 90-day paper validation campaign')
    p_c90.add_argument('--capital', type=float, default=10_000.0)
    p_c90.add_argument('--name', default='90d-Validation')

    # shadow
    p_sh = subs.add_parser('shadow', help='Start shadow/paper mode')
    p_sh.add_argument('--mode', default='SHADOW', choices=['SHADOW', 'PAPER', 'LIVE'])

    # report
    p_rp = subs.add_parser('report', help='Generate a validation report')
    p_rp.add_argument('label', help='Strategy label')
    p_rp.add_argument('--days', type=int, default=30)
    p_rp.add_argument('--capital', type=float, default=10_000.0)

    # status
    subs.add_parser('status', help='Show system status')

    args = parser.parse_args()

    if args.command == 'campaign90':
        cmd_campaign90(args)
    elif args.command == 'shadow':
        cmd_shadow(args)
    elif args.command == 'report':
        cmd_report(args)
    elif args.command == 'status':
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
