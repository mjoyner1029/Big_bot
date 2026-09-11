"""
PaperCampaignManager — reproducible, versioned paper trading campaigns.

PHASE 9

A campaign is a fixed-configuration paper trading run:
    - immutable start config
    - defined duration
    - tracked performance vs benchmarks
    - never overwrites historical campaigns

Campaign stages: SHADOW → PAPER → CANARY → LIMITED → NORMAL

Every campaign is reproducible via its config_hash.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc).isoformat()


class CampaignStatus(str, Enum):
    PLANNED   = "PLANNED"
    ACTIVE    = "ACTIVE"
    COMPLETED = "COMPLETED"
    ABORTED   = "ABORTED"


@dataclass
class Campaign:
    """Immutable configuration for a paper trading campaign."""
    campaign_id:        str
    name:               str
    status:             CampaignStatus
    trading_mode:       str    # PAPER | SHADOW | CANARY | LIMITED | LIVE
    start_date:         str
    end_date:           str
    duration_days:      int
    starting_capital:   float
    strategy_versions:  Dict[str, str]   # strategy_name → version
    model_versions:     Dict[str, str]   # model_name → version
    config_hash:        str
    config_snapshot:    Dict
    risk_config:        Dict
    broker_assumptions: Dict
    benchmark_config:   Dict
    created_at:         str = field(default_factory=_utcnow)
    ended_at:           Optional[str] = None
    abort_reason:       Optional[str] = None
    notes:              str = ''


@dataclass
class CampaignScorecard:
    """Daily scorecard for an active campaign."""
    campaign_id:        str
    date:               str
    portfolio_pnl:      float
    benchmark_pnl:      float
    alpha:              float
    drawdown:           float
    open_positions:     int
    trades_today:       int
    win_rate:           float
    expectancy:         float
    profit_factor:      float
    sharpe:             float
    execution_costs:    float
    slippage:           float
    reconciliation_errors: int
    system_uptime_pct:  float
    data_quality_incidents: int
    strategy_health:    Dict = field(default_factory=dict)
    model_health:       Dict = field(default_factory=dict)
    recorded_at:        str = field(default_factory=_utcnow)


_CREATE_CAMPAIGNS = """
CREATE TABLE IF NOT EXISTS paper_campaigns (
    campaign_id       TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    status            TEXT DEFAULT 'PLANNED',
    trading_mode      TEXT DEFAULT 'PAPER',
    start_date        TEXT NOT NULL,
    end_date          TEXT NOT NULL,
    duration_days     INTEGER NOT NULL,
    starting_capital  REAL NOT NULL,
    strategy_versions TEXT,
    model_versions    TEXT,
    config_hash       TEXT,
    config_snapshot   TEXT,
    risk_config       TEXT,
    broker_assumptions TEXT,
    benchmark_config  TEXT,
    created_at        TEXT NOT NULL,
    ended_at          TEXT,
    abort_reason      TEXT,
    notes             TEXT DEFAULT ''
)
"""

_CREATE_SCORECARDS = """
CREATE TABLE IF NOT EXISTS campaign_scorecards (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id             TEXT NOT NULL,
    date                    TEXT NOT NULL,
    portfolio_pnl           REAL DEFAULT 0,
    benchmark_pnl           REAL DEFAULT 0,
    alpha                   REAL DEFAULT 0,
    drawdown                REAL DEFAULT 0,
    open_positions          INTEGER DEFAULT 0,
    trades_today            INTEGER DEFAULT 0,
    win_rate                REAL DEFAULT 0,
    expectancy              REAL DEFAULT 0,
    profit_factor           REAL DEFAULT 0,
    sharpe                  REAL DEFAULT 0,
    execution_costs         REAL DEFAULT 0,
    slippage                REAL DEFAULT 0,
    reconciliation_errors   INTEGER DEFAULT 0,
    system_uptime_pct       REAL DEFAULT 1,
    data_quality_incidents  INTEGER DEFAULT 0,
    strategy_health         TEXT,
    model_health            TEXT,
    recorded_at             TEXT NOT NULL,
    UNIQUE(campaign_id, date)
)
"""


class PaperCampaignManager:
    """
    Manages reproducible, versioned paper trading campaigns.

    Rules:
        - Campaigns are never overwritten
        - Each campaign has an immutable config snapshot
        - Daily scorecards are recorded for every active campaign
        - 90-day validation campaigns freeze a baseline config
    """

    def __init__(self, db_path: str = "data/trade_memory.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_CAMPAIGNS)
            conn.execute(_CREATE_SCORECARDS)
            conn.commit()

    # ── Campaign lifecycle ────────────────────────────────────────────────────

    def create(
        self,
        name:              str,
        duration_days:     int,
        starting_capital:  float,
        strategy_versions: Dict[str, str],
        model_versions:    Dict[str, str],
        config_snapshot:   Dict,
        config_hash:       str = '',
        risk_config:       Dict = None,
        broker_assumptions: Dict = None,
        benchmark_config:  Dict = None,
        trading_mode:      str = 'PAPER',
        notes:             str = '',
    ) -> Campaign:
        """Create a new campaign. Config is frozen at creation time."""
        now        = datetime.now(timezone.utc)
        start_date = now.date().isoformat()
        end_date   = (now + timedelta(days=duration_days)).date().isoformat()

        campaign = Campaign(
            campaign_id=str(uuid.uuid4()),
            name=name,
            status=CampaignStatus.PLANNED,
            trading_mode=trading_mode,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            starting_capital=starting_capital,
            strategy_versions=strategy_versions,
            model_versions=model_versions,
            config_hash=config_hash,
            config_snapshot=config_snapshot,
            risk_config=risk_config or {},
            broker_assumptions=broker_assumptions or {
                'slippage_bps': 5,
                'spread_bps':   3,
                'fee_pct':      0.001,
                'fill_rate':    0.98,
            },
            benchmark_config=benchmark_config or {
                'benchmarks': ['CASH', 'BUY_HOLD_BTC', 'SPY'],
                'primary':    'BUY_HOLD_BTC',
            },
            notes=notes,
        )
        self._save(campaign)
        logger.info(
            f"PaperCampaign: created '{name}' [{campaign.campaign_id[:8]}] "
            f"{duration_days}d starting ${starting_capital:,.0f}"
        )
        return campaign

    def start(self, campaign_id: str) -> bool:
        """Mark campaign as ACTIVE."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE paper_campaigns SET status='ACTIVE' WHERE campaign_id=? AND status='PLANNED'",
                (campaign_id,),
            )
            conn.commit()
        logger.info(f"PaperCampaign: started [{campaign_id[:8]}]")
        return True

    def complete(self, campaign_id: str) -> None:
        now = _utcnow()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE paper_campaigns SET status='COMPLETED',ended_at=? WHERE campaign_id=?",
                (now, campaign_id),
            )
            conn.commit()
        logger.info(f"PaperCampaign: completed [{campaign_id[:8]}]")

    def abort(self, campaign_id: str, reason: str) -> None:
        now = _utcnow()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE paper_campaigns SET status='ABORTED',ended_at=?,abort_reason=? "
                "WHERE campaign_id=?",
                (now, reason, campaign_id),
            )
            conn.commit()
        logger.warning(f"PaperCampaign: ABORTED [{campaign_id[:8]}] — {reason}")

    # ── Scorecards ────────────────────────────────────────────────────────────

    def record_scorecard(self, scorecard: CampaignScorecard) -> None:
        """Record daily scorecard. Idempotent by (campaign_id, date)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO campaign_scorecards "
                "(campaign_id,date,portfolio_pnl,benchmark_pnl,alpha,drawdown,"
                "open_positions,trades_today,win_rate,expectancy,profit_factor,sharpe,"
                "execution_costs,slippage,reconciliation_errors,system_uptime_pct,"
                "data_quality_incidents,strategy_health,model_health,recorded_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (scorecard.campaign_id, scorecard.date,
                 scorecard.portfolio_pnl, scorecard.benchmark_pnl, scorecard.alpha,
                 scorecard.drawdown, scorecard.open_positions, scorecard.trades_today,
                 scorecard.win_rate, scorecard.expectancy, scorecard.profit_factor,
                 scorecard.sharpe, scorecard.execution_costs, scorecard.slippage,
                 scorecard.reconciliation_errors, scorecard.system_uptime_pct,
                 scorecard.data_quality_incidents,
                 json.dumps(scorecard.strategy_health), json.dumps(scorecard.model_health),
                 scorecard.recorded_at),
            )
            conn.commit()

    def get_scorecards(self, campaign_id: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM campaign_scorecards WHERE campaign_id=? ORDER BY date",
                (campaign_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_active(self) -> List[Campaign]:
        return self._load_by_status('ACTIVE')

    def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM paper_campaigns WHERE campaign_id=?", (campaign_id,)
            ).fetchone()
        return self._row_to_campaign(dict(row)) if row else None

    def list_all(self, limit: int = 20) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT campaign_id,name,status,trading_mode,start_date,end_date,"
                "duration_days,starting_capital,created_at FROM paper_campaigns "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def campaign_summary(self, campaign_id: str) -> str:
        """Human-readable summary of a campaign."""
        c  = self.get_campaign(campaign_id)
        sc = self.get_scorecards(campaign_id)
        if not c:
            return f"Campaign {campaign_id} not found"

        total_pnl = sum(s.get('portfolio_pnl', 0) for s in sc)
        lines = [
            f"Campaign: {c.name} [{c.campaign_id[:8]}]",
            f"Status:   {c.status.value}",
            f"Period:   {c.start_date} → {c.end_date} ({c.duration_days}d)",
            f"Capital:  ${c.starting_capital:,.0f}",
            f"Mode:     {c.trading_mode}",
            f"Days recorded: {len(sc)}",
            f"Cumulative PnL: ${total_pnl:,.2f}",
        ]
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, c: Campaign) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO paper_campaigns VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (c.campaign_id, c.name, c.status.value, c.trading_mode,
                 c.start_date, c.end_date, c.duration_days, c.starting_capital,
                 json.dumps(c.strategy_versions), json.dumps(c.model_versions),
                 c.config_hash, json.dumps(c.config_snapshot),
                 json.dumps(c.risk_config), json.dumps(c.broker_assumptions),
                 json.dumps(c.benchmark_config), c.created_at, c.ended_at,
                 c.abort_reason, c.notes),
            )
            conn.commit()

    def _load_by_status(self, status: str) -> List[Campaign]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM paper_campaigns WHERE status=? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        return [self._row_to_campaign(dict(r)) for r in rows]

    @staticmethod
    def _row_to_campaign(r: Dict) -> Campaign:
        return Campaign(
            campaign_id=r['campaign_id'],
            name=r['name'],
            status=CampaignStatus(r['status']),
            trading_mode=r['trading_mode'],
            start_date=r['start_date'],
            end_date=r['end_date'],
            duration_days=r['duration_days'],
            starting_capital=r['starting_capital'],
            strategy_versions=json.loads(r.get('strategy_versions') or '{}'),
            model_versions=json.loads(r.get('model_versions') or '{}'),
            config_hash=r.get('config_hash', ''),
            config_snapshot=json.loads(r.get('config_snapshot') or '{}'),
            risk_config=json.loads(r.get('risk_config') or '{}'),
            broker_assumptions=json.loads(r.get('broker_assumptions') or '{}'),
            benchmark_config=json.loads(r.get('benchmark_config') or '{}'),
            created_at=r.get('created_at', _utcnow()),
            ended_at=r.get('ended_at'),
            abort_reason=r.get('abort_reason'),
            notes=r.get('notes', ''),
        )
