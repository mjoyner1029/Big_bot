"""Run a real research campaign across the available stock/ETF/crypto universe.

Usage:
    python scripts/run_research_campaign.py [--max-instruments N] [--period P]

Fetches real historical data via the existing fetcher (no fabrication),
runs the full discovery→validation funnel, populates the Alpha Library with
survivors (auto-PAPER, never live), and prints the funnel report.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("research_campaign")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-instruments", type=int, default=40)
    parser.add_argument("--period", default="2y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--db", default="data/trade_memory.sqlite")
    parser.add_argument("--universal", action="store_true",
                        help="run the full AutonomousResearchOrchestrator loop "
                             "(external intelligence + universal discovery + "
                             "market-neutral) instead of detectors only")
    args = parser.parse_args()

    from core.instruments import InstrumentUniverse
    from core.research_campaign import CampaignConfig, ResearchCampaignRunner
    from data.history import fetch_research_history

    universe = InstrumentUniverse()
    instruments = (universe.crypto()
                   + universe.etfs()[:12]
                   + universe.stocks()[:20])[: args.max_instruments]
    logger.info(f"Campaign universe: {len(instruments)} instruments "
                f"(crypto/ETF/stock), period={args.period}")

    data = {}
    for inst in instruments:
        try:
            df = fetch_research_history(inst.symbol, period=args.period,
                                          interval=args.interval)
            if df is not None and len(df) >= 200:
                data[inst.symbol] = df
            else:
                logger.info(f"  {inst.symbol}: insufficient history — skipped")
        except Exception as e:
            logger.warning(f"  {inst.symbol}: fetch failed ({e})")
    logger.info(f"Fetched usable history for {len(data)}/{len(instruments)} instruments")
    if not data:
        logger.error("No data available — campaign aborted (nothing fabricated)")
        return 1

    if args.universal:
        import json

        from core.research_orchestrator import AutonomousResearchOrchestrator
        from data.external_intelligence import ExternalSourceRegistry, RawEventStore
        from data.external_sources import USASpendingSource
        from data.research_interfaces import default_interface_sources

        registry = ExternalSourceRegistry(event_store=RawEventStore(args.db))
        registry.register(USASpendingSource())
        for src in default_interface_sources():
            registry.register(src)
        orchestrator = AutonomousResearchOrchestrator(
            db_path=args.db,
            config=None,
            source_registry=registry)
        report = orchestrator.run_campaign(data)
        print(ResearchCampaignRunner.format_report(report["campaign"]))
        summary = {k: v for k, v in report.items() if k != "campaign"}
        print(json.dumps(summary, indent=2, default=str))
        return 0

    runner = ResearchCampaignRunner(
        db_path=args.db,
        config=CampaignConfig(data_timeframe=args.interval),
    )
    report = runner.run(data, universe_has_membership_data=False)
    print(ResearchCampaignRunner.format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
