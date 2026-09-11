import sqlite3
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB_PATH = ROOT / "data" / "trade_memory.sqlite"

st.set_page_config(page_title="liquidity13 Dashboard", page_icon="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%230b1220'/%3E%3Cpath d='M20 18h24v6H20zm0 11h24v6H20zm0 11h17v6H20z' fill='%23dfeaf8'/%3E%3C/svg%3E", layout="wide", initial_sidebar_state="expanded")

NAV_ITEMS = [
    "Overview",
    "Alpha Engine",
    "Paper Evidence",
    "System Health",
    "My account",
    "Active Stocks",
    "Dividend Insights",
    "Trading Stocks Chat",
    "Hybrid Funds",
    "Portfolio",
    "Settings",
    "History",
    "News",
    "Feedback",
]


def set_nav_page(page: str) -> None:
    st.session_state.nav_page = page


def get_current_page() -> str:
    page = st.query_params.get("page", "Overview")
    if isinstance(page, list):
        page = page[0] if page else "Overview"
    return page if page in NAV_ITEMS else "Overview"


def _table(query: str, params=()) -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(str(DB_PATH))
    try:
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def get_positions():
    if not DB_PATH.exists():
        return pd.DataFrame(columns=[
            "id", "symbol", "signal", "direction", "size", "entry_price", "entry_fill_price",
            "exit_price", "status", "pnl", "net_pnl", "strategy", "asset_class",
            "entry_time", "exit_time", "close_reason", "regime", "confidence",
            "stop_loss", "take_profit", "max_price"
        ])

    conn = sqlite3.connect(str(DB_PATH))
    try:
        query = """
            SELECT
                id, symbol, signal, direction, size, entry_price, entry_fill_price,
                exit_price, status, pnl, net_pnl, strategy, asset_class,
                entry_time, exit_time, close_reason, regime, confidence,
                stop_loss, take_profit, max_price
            FROM positions
            ORDER BY entry_time DESC
        """
        df = pd.read_sql_query(query, conn)
        if not df.empty:
            df["status"] = df["status"].fillna("UNKNOWN")
            df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
            df["net_pnl"] = pd.to_numeric(df["net_pnl"], errors="coerce").fillna(0.0)
            df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0.0)
        return df
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def get_trade_history(limit: int = 500):
    if not DB_PATH.exists():
        return pd.DataFrame(columns=[
            "id", "symbol", "strategy", "asset_class", "entry_price", "exit_price",
            "position_size_usd", "qty", "pnl", "pnl_pct", "win_loss", "hold_time_seconds",
            "entry_time", "exit_time", "is_paper", "market_regime", "reason_for_entry", "reason_for_exit"
        ])

    conn = sqlite3.connect(str(DB_PATH))
    try:
        query = """
            SELECT
                trade_id AS id,
                symbol,
                strategy_names AS strategy,
                asset_class,
                entry_price,
                exit_price,
                position_size_usd,
                qty,
                pnl,
                pnl_pct,
                win_loss,
                hold_time_seconds,
                timestamp_open AS entry_time,
                timestamp_close AS exit_time,
                is_paper,
                market_regime,
                reason_for_entry,
                reason_for_exit
            FROM trades
            ORDER BY timestamp_open DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(limit,))
        if not df.empty:
            df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0)
            df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce").fillna(0.0)
            df["position_size_usd"] = pd.to_numeric(df["position_size_usd"], errors="coerce").fillna(0.0)
            df["is_paper"] = df["is_paper"].fillna(0).astype(int)
            df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
            df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
        return df
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def get_summary():
    positions = get_positions()
    trades = get_trade_history(limit=1000)
    closed_pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0) if not trades.empty else pd.Series(dtype=float)
    paper_pnl = pd.to_numeric(trades.loc[trades["is_paper"] == 1, "pnl"], errors="coerce").fillna(0.0) if not trades.empty else pd.Series(dtype=float)
    live_pnl = pd.to_numeric(trades.loc[trades["is_paper"] == 0, "pnl"], errors="coerce").fillna(0.0) if not trades.empty else pd.Series(dtype=float)
    open_positions = positions[positions["status"].str.upper() == "OPEN"] if not positions.empty else pd.DataFrame()
    total_open_exposure = float(open_positions["size"].sum()) if not open_positions.empty else 0.0
    total_realized_pnl = float(closed_pnl.sum()) if not closed_pnl.empty else 0.0
    if trades.empty and positions.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "realized_pnl": 0.0,
            "paper_pnl": 0.0,
            "live_pnl": 0.0,
            "open_positions": 0,
            "open_exposure": 0.0,
        }
    win_rate = float((closed_pnl > 0).mean()) if not closed_pnl.empty else 0.0
    return {
        "total_trades": int(len(trades)),
        "win_rate": win_rate,
        "realized_pnl": float(total_realized_pnl),
        "paper_pnl": float(paper_pnl.sum()),
        "live_pnl": float(live_pnl.sum()),
        "open_positions": int(len(open_positions)),
        "open_exposure": total_open_exposure,
    }


RANGE_WINDOWS = {
    "1H": pd.Timedelta(hours=1),
    "1D": pd.Timedelta(days=1),
    "1M": pd.Timedelta(days=30),
    "6M": pd.Timedelta(days=182),
    "1Y": pd.Timedelta(days=365),
}


def build_earnings_series(trades: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["period", "earnings"])
    df = trades.dropna(subset=["entry_time"]).sort_values("entry_time")
    if df.empty:
        return pd.DataFrame(columns=["period", "earnings"])
    window = RANGE_WINDOWS.get(range_key)
    if window is not None:
        df = df[df["entry_time"] >= df["entry_time"].max() - window]
    if df.empty:
        return pd.DataFrame(columns=["period", "earnings"])
    return pd.DataFrame({"period": df["entry_time"], "earnings": df["pnl"].cumsum()})


BAR_FREQS = {"1H": "5min", "1D": "1h", "1M": "D", "6M": "W", "1Y": "W"}


def build_earnings_bars(trades: pd.DataFrame, range_key: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["period", "pnl"])
    df = trades.dropna(subset=["entry_time"]).sort_values("entry_time")
    if df.empty:
        return pd.DataFrame(columns=["period", "pnl"])
    window = RANGE_WINDOWS.get(range_key)
    if window is not None:
        df = df[df["entry_time"] >= df["entry_time"].max() - window]
    if df.empty:
        return pd.DataFrame(columns=["period", "pnl"])
    freq = BAR_FREQS.get(range_key, "W")
    grouped = df.groupby(df["entry_time"].dt.to_period(freq))["pnl"].sum()
    return pd.DataFrame({"period": grouped.index.to_timestamp(), "pnl": grouped.values})


def _chart_layout(fig: go.Figure, height: int = 280) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10), height=height,
        font=dict(color="#e7edf7"), showlegend=False,
        xaxis=dict(showgrid=False, linecolor="rgba(148,163,184,0.2)",
                   tickfont=dict(color="#9aa8bd", size=11), rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.12)",
                   linecolor="rgba(148,163,184,0.2)", tickfont=dict(color="#9aa8bd", size=11)),
    )
    return fig


def _sidebar_icon(path_svg: str):
    return f"<svg viewBox='0 0 24 24' aria-hidden='true'>{path_svg}</svg>"


def render_sidebar():
    summary = get_summary()
    current_page = get_current_page()
    nav_links = "".join(
        f"<a class='nav-link {'active' if item == current_page else ''}' href='?page={quote(item)}' target='_self'>{item}</a>"
        for item in NAV_ITEMS
    )
    st.markdown(
        f"""
        <div class='local-nav'>
        <div class='brand'>
            <div class='brand-mark'>
                <svg viewBox='0 0 24 24'><path d='M12 2.5a1.5 1.5 0 0 1 1.5 1.5v1.1A7.1 7.1 0 0 1 18.9 10h1.1a1.5 1.5 0 0 1 0 3h-1.1a7.1 7.1 0 0 1-5.4 5.4v1.1a1.5 1.5 0 0 1-3 0v-1.1A7.1 7.1 0 0 1 5.1 13H4a1.5 1.5 0 0 1 0-3h1.1A7.1 7.1 0 0 1 10.5 4.6V3.5A1.5 1.5 0 0 1 12 2.5Zm0 5.3a4.2 4.2 0 1 0 0 8.4 4.2 4.2 0 0 0 0-8.4Z' fill='currentColor'/></svg>
            </div>
            <span>liquidity13</span>
        </div>
        <div class='nav-caption'>Navigation</div>
        <div class='nav-links'>{nav_links}</div>
        <div class='nav-meta'>
            <div>Local portfolio monitor</div>
            <div>SQLite: {DB_PATH.name}</div>
            <div>Trades: {len(get_trade_history(limit=1000))}</div>
            <div>Open positions: {summary['open_positions']}</div>
            <div>Win rate: {summary['win_rate'] * 100:.1f}%</div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview():
    positions = get_positions()
    trade_history = get_trade_history(limit=1000)
    summary = get_summary()

    st.markdown(
        """
        <div class='topbar'>
            <div>
                <div class='subtle'>Dashboard</div>
                <h1 class='title'>Overview</h1>
            </div>
            <div class='badge'>Local monitoring</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    def pnl_tone(v: float) -> str:
        return "positive" if v >= 0 else "negative"

    cards = [
        ("Realized P&L", f"${summary['realized_pnl']:.2f}", f"{summary['total_trades']} closed trades", pnl_tone(summary["realized_pnl"])),
        ("Paper P&L", f"${summary['paper_pnl']:.2f}", "paper trading", pnl_tone(summary["paper_pnl"])),
        ("Live P&L", f"${summary['live_pnl']:.2f}", "live trading", pnl_tone(summary["live_pnl"])),
        ("Open Exposure", f"${summary['open_exposure']:.2f}", f"{summary['open_positions']} open positions", "neutral"),
    ]
    metric_html = "".join(
        f"<div class='metric-card'>"
        f"<div class='metric-head'><span>{label}</span><span class='metric-icon'><svg viewBox='0 0 24 24'><path d='M4 18h16v2H4zm1-2.5 4.4-4.4 3.2 3.2L18.5 6l1.5 1.5-7.8 8.9-3.6-3.6L5 15.5Z' fill='currentColor'/></svg></span></div>"
        f"<div class='metric-value {tone}'>{value}</div>"
        f"<div class='metric-foot'>{foot}</div>"
        f"</div>"
        for label, value, foot, tone in cards
    )
    st.markdown(f"<div class='metric-grid'>{metric_html}</div>", unsafe_allow_html=True)

    range_mode = st.segmented_control(
        "Time range",
        options=["1H", "1D", "1M", "6M", "1Y", "ALLTIME"],
        default="ALLTIME",
        selection_mode="single",
        label_visibility="collapsed",
    )
    earnings_df = build_earnings_series(trade_history, range_mode)
    bars_df = build_earnings_bars(trade_history, range_mode)

    left, right = st.columns([1.4, 0.9])
    with left:
        st.markdown(f"<div class='panel'><div class='panel-inner'><div class='panel-header'><div class='panel-title'>Portfolio earnings</div><div class='mini-chip'>{(range_mode or 'ALLTIME').lower()}</div></div></div></div>", unsafe_allow_html=True)
        if earnings_df.empty:
            st.write("No trades in this range.")
        else:
            line_tab, bar_tab = st.tabs(["Line", "Bars"])
            with line_tab:
                fig = go.Figure(go.Scatter(
                    x=earnings_df["period"], y=earnings_df["earnings"], mode="lines",
                    line=dict(color="#74f3b4", width=3),
                    fill="tozeroy", fillcolor="rgba(116,243,180,0.1)"))
                st.plotly_chart(_chart_layout(fig), width="stretch")
            with bar_tab:
                fig = go.Figure(go.Bar(
                    x=bars_df["period"], y=bars_df["pnl"],
                    marker_color=["#69e7af" if v >= 0 else "#ff8e9d" for v in bars_df["pnl"]]))
                st.plotly_chart(_chart_layout(fig), width="stretch")
    with right:
        st.markdown("<div class='panel'><div class='panel-inner'><div class='panel-header'><div class='panel-title'>Win rate</div><div class='mini-chip'>closed trades</div></div></div></div>", unsafe_allow_html=True)
        win_rate_pct = summary["win_rate"] * 100
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=win_rate_pct, domain={"x": [0, 1], "y": [0, 1]}, number={"suffix": "%", "font":{"color":"#e7edf7","size":28}}, gauge={"axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#6e7f9f"}, "bar":{"color":"#7ee7b8"}, "bgcolor":"rgba(0,0,0,0)", "steps":[{"range":[0,50],"color":"rgba(255,255,255,0.08)"},{"range":[50,80],"color":"rgba(126,231,184,0.18)"},{"range":[80,100],"color":"rgba(126,231,184,0.4)"}], "threshold":{"line":{"color":"#7ee7b8","width":3},"thickness":0.75,"value":50}}))
        gauge.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=220, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e7edf7"))
        st.plotly_chart(gauge, width="stretch")

    lower_left, lower_right = st.columns([1.1, 1.1])
    with lower_left:
        st.markdown("<div class='panel'><div class='panel-inner'><div class='panel-header'><div class='panel-title'>P&L by strategy</div><div class='mini-chip'>closed trades</div></div></div></div>", unsafe_allow_html=True)
        if trade_history.empty:
            st.write("No strategy data yet.")
        else:
            strategy_df = (
                trade_history.groupby("strategy", dropna=False)
                .agg(trades=("pnl", "size"), wins=("pnl", lambda s: int((s > 0).sum())), pnl=("pnl", "sum"))
                .reset_index()
                .sort_values("pnl", ascending=False)
            )
            strategy_df["win rate"] = (strategy_df["wins"] / strategy_df["trades"] * 100).map(lambda x: f"{x:.1f}%")
            strategy_df["pnl"] = strategy_df["pnl"].map(lambda x: f"${x:.2f}")
            st.dataframe(strategy_df[["strategy", "trades", "win rate", "pnl"]], hide_index=True, width="stretch")

    with lower_right:
        st.markdown("<div class='panel'><div class='panel-inner'><div class='panel-header'><div class='panel-title'>Top symbols by P&L</div><div class='mini-chip'>closed trades</div></div></div></div>", unsafe_allow_html=True)
        if trade_history.empty:
            st.write("No symbol data yet.")
        else:
            symbol_df = (
                trade_history.groupby("symbol", dropna=False)
                .agg(trades=("pnl", "size"), wins=("pnl", lambda s: int((s > 0).sum())), pnl=("pnl", "sum"))
                .reset_index()
                .sort_values("pnl", ascending=False)
                .head(8)
            )
            symbol_df["win rate"] = (symbol_df["wins"] / symbol_df["trades"] * 100).map(lambda x: f"{x:.1f}%")
            symbol_df["pnl"] = symbol_df["pnl"].map(lambda x: f"${x:.2f}")
            st.dataframe(symbol_df[["symbol", "trades", "win rate", "pnl"]], hide_index=True, width="stretch")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'><div class='panel-inner'><div class='panel-header'><div class='panel-title'>Recent trade log</div><div class='mini-chip'>local</div></div></div></div>", unsafe_allow_html=True)
    if trade_history.empty:
        st.write("No trade history has been recorded yet.")
    else:
        trade_view = trade_history[["symbol", "strategy", "pnl", "pnl_pct", "win_loss", "entry_time", "is_paper"]].copy()
        trade_view = trade_view.head(10)
        trade_view["pnl"] = trade_view["pnl"].map(lambda x: f"${x:.2f}")
        trade_view["pnl_pct"] = trade_view["pnl_pct"].map(lambda x: f"{x:.2f}%")
        trade_view["session"] = trade_view["is_paper"].map({1: "Paper", 0: "Live"})
        trade_view = trade_view.drop(columns=["is_paper"])
        st.dataframe(trade_view, hide_index=True, width="stretch")


def render_alpha_engine():
    st.markdown("<div class='topbar'><h1 class='title'>Alpha Engine</h1>"
                "<span class='badge'>research → validation → paper</span></div>",
                unsafe_allow_html=True)
    alphas = _table(
        "SELECT alpha_id, strategy_family, lifecycle_state, direction, "
        "universe, net_expectancy, edge_health_score, updated_at "
        "FROM alpha_library ORDER BY updated_at DESC")
    if alphas.empty:
        st.info("Alpha library empty — run a research campaign.")
    else:
        counts = alphas["lifecycle_state"].value_counts().to_dict()
        st.markdown(" ".join(
            f"<span class='mini-chip'>{k}: {v}</span>"
            for k, v in counts.items()), unsafe_allow_html=True)
        st.dataframe(alphas, hide_index=True, width="stretch", height=320)

    st.subheader("Search ledger (breadth feeding FDR)")
    ledger = _table("SELECT source, SUM(hypotheses) AS hypotheses "
                    "FROM search_ledger GROUP BY source ORDER BY 2 DESC")
    funnel = _table(
        "SELECT outcome, COUNT(*) AS n FROM campaign_hypotheses GROUP BY outcome")
    c1, c2 = st.columns(2)
    with c1:
        if not ledger.empty:
            st.dataframe(ledger, hide_index=True, width="stretch")
        else:
            st.caption("no search history yet")
    with c2:
        if not funnel.empty:
            st.dataframe(funnel, hide_index=True, width="stretch")
        else:
            st.caption("no campaign hypotheses yet")

    budgets = _table("SELECT family, budget, updated_at FROM research_budgets")
    if not budgets.empty:
        st.subheader("Gap-directed research budgets (next campaign)")
        st.dataframe(budgets, hide_index=True, width="stretch")


def render_paper_evidence():
    st.markdown("<div class='topbar'><h1 class='title'>Paper Evidence</h1>"
                "<span class='badge'>predicted vs realized</span></div>",
                unsafe_allow_html=True)
    try:
        from core.paper_evidence import PaperEvidenceTracker
        tracker = PaperEvidenceTracker(str(DB_PATH))
    except Exception as e:
        st.error(f"Evidence store unavailable: {e}")
        return
    st.code(tracker.summary_report(30), language=None)
    health = tracker.evidence_health_report()
    cols = st.columns(4)
    cols[0].metric("Resolved", health["status_counts"].get("RESOLVED", 0))
    cols[1].metric("Pending", health["status_counts"].get("PENDING", 0))
    cols[2].metric("Decay measured", health["decay_successful"])
    cols[3].metric("Decay pending", health["decay_pending"])
    with st.expander("Decay pipeline by status"):
        st.json(health["decay_by_status"] or {"note": "no measurements yet"})
    with st.expander("Model reliability (measurement only)"):
        st.json(tracker.model_reliability_report())
    snaps = _table("SELECT snapshot_date, resolved_count, ev_slope, brier, "
                   "exec_cost_error_bps, net_pnl_usd FROM calibration_snapshots "
                   "ORDER BY snapshot_date DESC LIMIT 30")
    if not snaps.empty:
        st.subheader("Daily calibration snapshots")
        st.dataframe(snaps, hide_index=True, width="stretch")


def render_system_health():
    st.markdown("<div class='topbar'><h1 class='title'>System Health</h1>"
                "<span class='badge'>fail-closed accounting</span></div>",
                unsafe_allow_html=True)
    try:
        from core.trade_history import DatabaseSchemaHealthCheck, data_integrity_scan
        report = DatabaseSchemaHealthCheck(str(DB_PATH)).run()
        state = "HEALTHY" if report["ok"] else "CRITICAL"
        (st.success if report["ok"] else st.error)(
            f"Core database: {state} (schema v{report.get('schema_version')})")
        st.json(report["tables"])
        integrity = data_integrity_scan(str(DB_PATH))
        if integrity["flagged"]:
            st.warning(f"{integrity['flagged']} flagged trade rows")
            st.json(integrity["issues"][:20])
        else:
            st.caption(f"Data integrity: {integrity['scanned']} rows scanned, "
                       "0 flagged")
    except Exception as e:
        st.error(f"Health check failed: {e}")
    sources = _table("SELECT source_name, last_success, last_failure, "
                     "consecutive_failures, disabled FROM source_health")
    if not sources.empty:
        st.subheader("External source health (optional — fail-soft)")
        st.dataframe(sources, hide_index=True, width="stretch")
    execs = _table("SELECT recorded_at, symbol, side, method, order_type, "
                   "resolution, net_ev, expected_round_trip_cost_bps "
                   "FROM execution_decisions ORDER BY id DESC LIMIT 25")
    if not execs.empty:
        st.subheader("Recent execution decisions")
        st.dataframe(execs, hide_index=True, width="stretch")


def _topbar(title: str, badge: str) -> None:
    st.markdown(
        f"<div class='topbar'><div><div class='subtle'>Dashboard</div>"
        f"<h1 class='title'>{title}</h1></div>"
        f"<span class='badge'>{badge}</span></div>",
        unsafe_allow_html=True,
    )


def render_my_account():
    _topbar("My account", "paper trading")
    trades = get_trade_history(limit=1000)
    summary = get_summary()
    fees = _table("SELECT COALESCE(SUM(fees),0) AS fees FROM trades")
    total_fees = float(fees["fees"].iloc[0]) if not fees.empty else 0.0
    c = st.columns(4)
    c[0].metric("Realized P&L", f"${summary['realized_pnl']:.2f}")
    c[1].metric("Total fees paid", f"${total_fees:.2f}")
    c[2].metric("Closed trades", summary["total_trades"])
    c[3].metric("Win rate", f"{summary['win_rate'] * 100:.1f}%")
    c = st.columns(4)
    c[0].metric("Open positions", summary["open_positions"])
    c[1].metric("Open exposure", f"${summary['open_exposure']:.2f}")
    c[2].metric("Paper P&L", f"${summary['paper_pnl']:.2f}")
    c[3].metric("Live P&L", f"${summary['live_pnl']:.2f}")
    if not trades.empty:
        st.subheader("Best / worst closed trades")
        cols = ["symbol", "strategy", "pnl", "pnl_pct", "entry_time", "exit_time"]
        b, w = st.columns(2)
        with b:
            st.caption("Top 5 winners")
            st.dataframe(trades.nlargest(5, "pnl")[cols], hide_index=True, width="stretch")
        with w:
            st.caption("Top 5 losers")
            st.dataframe(trades.nsmallest(5, "pnl")[cols], hide_index=True, width="stretch")
        recent = trades[trades["entry_time"] >= trades["entry_time"].max() - pd.Timedelta(days=30)]
        st.caption(
            f"Last 30 days: {len(recent)} trades, "
            f"${recent['pnl'].sum():.2f} P&L, "
            f"avg hold {recent['hold_time_seconds'].fillna(0).mean() / 3600:.1f}h"
        )
    else:
        st.info("No closed trades recorded yet.")


def render_active_stocks():
    _topbar("Active Stocks", "open + most traded")
    positions = get_positions()
    open_pos = positions[positions["status"].str.upper() == "OPEN"] if not positions.empty else pd.DataFrame()
    st.subheader(f"Open positions ({len(open_pos)})")
    if open_pos.empty:
        st.info("No open positions right now.")
    else:
        view = open_pos[["symbol", "direction", "size", "entry_price", "stop_loss",
                         "take_profit", "strategy", "confidence", "entry_time"]].copy()
        st.dataframe(view, hide_index=True, width="stretch")
    trades = get_trade_history(limit=1000)
    if not trades.empty:
        st.subheader("Most active symbols")
        active = (
            trades.groupby("symbol")
            .agg(trades=("pnl", "size"),
                 wins=("pnl", lambda s: int((s > 0).sum())),
                 pnl=("pnl", "sum"),
                 avg_size=("position_size_usd", "mean"))
            .reset_index()
            .sort_values("trades", ascending=False)
            .head(15)
        )
        active["win rate"] = (active["wins"] / active["trades"] * 100).map(lambda x: f"{x:.0f}%")
        active["pnl"] = active["pnl"].map(lambda x: f"${x:.2f}")
        active["avg_size"] = active["avg_size"].map(lambda x: f"${x:.0f}")
        st.dataframe(active[["symbol", "trades", "win rate", "pnl", "avg_size"]],
                     hide_index=True, width="stretch")


def render_dividend_insights():
    _topbar("Dividend Insights", "trading income")
    st.caption("Crypto pays no dividends — this shows realized trading income instead.")
    trades = get_trade_history(limit=1000)
    if trades.empty:
        st.info("No realized income yet.")
        return
    monthly = trades.dropna(subset=["exit_time"]).copy()
    monthly["month"] = monthly["exit_time"].dt.to_period("M").astype(str)
    income = monthly.groupby("month")["pnl"].sum().reset_index()
    fig = go.Figure(go.Bar(
        x=income["month"], y=income["pnl"],
        marker_color=["#69e7af" if v >= 0 else "#ff8e9d" for v in income["pnl"]]))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=280, margin=dict(l=10, r=10, t=10, b=10),
                      font=dict(color="#e7edf7"), showlegend=False,
                      yaxis=dict(gridcolor="rgba(148,163,184,0.12)"))
    st.subheader("Monthly realized income")
    st.plotly_chart(fig, width="stretch")
    st.subheader("Income by symbol")
    by_symbol = (trades.groupby("symbol")["pnl"].agg(["sum", "count"])
                 .reset_index().rename(columns={"sum": "income", "count": "trades"})
                 .sort_values("income", ascending=False))
    by_symbol["income"] = by_symbol["income"].map(lambda x: f"${x:.2f}")
    st.dataframe(by_symbol, hide_index=True, width="stretch")


def render_trading_chat():
    _topbar("Trading Stocks Chat", "bot decision feed")
    trades = get_trade_history(limit=30)
    if trades.empty:
        st.info("No trade decisions recorded yet.")
        return
    st.caption("Most recent trade reasoning, newest first.")
    for _, row in trades.iterrows():
        outcome = "won" if row["pnl"] > 0 else "lost"
        with st.chat_message("assistant"):
            st.markdown(
                f"**{row['symbol']}** · {row['strategy'] or 'n/a'} · "
                f"{outcome} **${row['pnl']:.2f}** ({row['pnl_pct']:.2f}%)"
            )
            if row.get("reason_for_entry"):
                st.markdown(f"*Entry:* {row['reason_for_entry']}")
            if row.get("reason_for_exit"):
                st.markdown(f"*Exit:* {row['reason_for_exit']}")
            st.caption(f"{row['entry_time']} → {row['exit_time']}")


def render_hybrid_funds():
    _topbar("Hybrid Funds", "sleeves by asset class")
    trades = get_trade_history(limit=1000)
    if trades.empty:
        st.info("No trades to build sleeve composition from.")
        return
    st.subheader("Performance by asset class")
    sleeve = (
        trades.groupby(trades["asset_class"].fillna("unknown"))
        .agg(trades=("pnl", "size"),
             wins=("pnl", lambda s: int((s > 0).sum())),
             pnl=("pnl", "sum"),
             notional=("position_size_usd", "sum"))
        .reset_index()
        .sort_values("pnl", ascending=False)
    )
    sleeve["win rate"] = (sleeve["wins"] / sleeve["trades"] * 100).map(lambda x: f"{x:.0f}%")
    sleeve["pnl"] = sleeve["pnl"].map(lambda x: f"${x:.2f}")
    sleeve["notional"] = sleeve["notional"].map(lambda x: f"${x:,.0f}")
    st.dataframe(sleeve[["asset_class", "trades", "win rate", "pnl", "notional"]],
                 hide_index=True, width="stretch")
    st.subheader("Strategy × asset class")
    mix = (
        trades.groupby([trades["asset_class"].fillna("unknown"), trades["strategy"].fillna("unknown")])
        .agg(trades=("pnl", "size"), pnl=("pnl", "sum"))
        .reset_index()
        .sort_values("pnl", ascending=False)
    )
    mix["pnl"] = mix["pnl"].map(lambda x: f"${x:.2f}")
    st.dataframe(mix, hide_index=True, width="stretch")


def render_portfolio():
    _topbar("Portfolio", "allocation + equity")
    positions = get_positions()
    trades = get_trade_history(limit=1000)
    open_pos = positions[positions["status"].str.upper() == "OPEN"] if not positions.empty else pd.DataFrame()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Current allocation")
        if open_pos.empty:
            st.info("No open positions — allocation is 100% cash.")
        else:
            pie = go.Figure(go.Pie(labels=open_pos["symbol"], values=open_pos["size"], hole=0.55))
            pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=280,
                              margin=dict(l=10, r=10, t=10, b=10), font=dict(color="#e7edf7"))
            st.plotly_chart(pie, width="stretch")
    with right:
        st.subheader("Cumulative P&L")
        earnings = build_earnings_series(trades, "ALLTIME")
        if earnings.empty:
            st.info("No closed trades yet.")
        else:
            fig = go.Figure(go.Scatter(x=earnings["period"], y=earnings["earnings"],
                                       mode="lines", line=dict(color="#74f3b4", width=2),
                                       fill="tozeroy", fillcolor="rgba(116,243,180,0.1)"))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=280, margin=dict(l=10, r=10, t=10, b=10),
                              font=dict(color="#e7edf7"), showlegend=False,
                              yaxis=dict(gridcolor="rgba(148,163,184,0.12)"))
            st.plotly_chart(fig, width="stretch")
    st.subheader("Recently closed positions")
    closed = positions[positions["status"].str.upper() == "CLOSED"].head(20) if not positions.empty else pd.DataFrame()
    if closed.empty:
        st.caption("no closed positions")
    else:
        st.dataframe(
            closed[["symbol", "direction", "size", "entry_price", "exit_price",
                    "net_pnl", "close_reason", "entry_time", "exit_time"]],
            hide_index=True, width="stretch")


def render_settings():
    import os
    _topbar("Settings", "runtime config (read-only)")
    st.caption("Secrets are never displayed. Values reflect this dashboard process's environment.")
    env_keys = [
        ("TRADING_MODE", "PAPER"),
        ("ALPHA_PIPELINE_MODE", "shadow"),
        ("META_ALPHA_MODE", "shadow"),
        ("PAPER_EVIDENCE_MODE", "true"),
        ("EXECUTION_OPTIMIZER_MODE", "active (paper) / shadow (live)"),
        ("ALPHA_RESEARCH_WEEKLY_MAX", "1000"),
    ]
    st.dataframe(
        pd.DataFrame(
            [(k, os.environ.get(k, f"(default: {d})")) for k, d in env_keys],
            columns=["setting", "value"],
        ),
        hide_index=True, width="stretch")
    st.subheader("Database")
    db_size = DB_PATH.stat().st_size / 1_048_576 if DB_PATH.exists() else 0
    counts = _table(
        "SELECT 'trades' AS tbl, COUNT(*) AS rows_ FROM trades "
        "UNION ALL SELECT 'positions', COUNT(*) FROM positions "
        "UNION ALL SELECT 'alpha_library', COUNT(*) FROM alpha_library "
        "UNION ALL SELECT 'paper_evidence', COUNT(*) FROM paper_evidence")
    st.caption(f"{DB_PATH} — {db_size:.1f} MB")
    if not counts.empty:
        st.dataframe(counts, hide_index=True, width="stretch")


def render_history():
    _topbar("History", "full trade log")
    trades = get_trade_history(limit=1000)
    if trades.empty:
        st.info("No trade history recorded yet.")
        return
    symbols = sorted(trades["symbol"].dropna().unique().tolist())
    c1, c2 = st.columns([2, 1])
    with c1:
        chosen = st.multiselect("Symbols", symbols, default=[])
    with c2:
        result = st.selectbox("Result", ["All", "Wins", "Losses"])
    view = trades.copy()
    if chosen:
        view = view[view["symbol"].isin(chosen)]
    if result == "Wins":
        view = view[view["pnl"] > 0]
    elif result == "Losses":
        view = view[view["pnl"] <= 0]
    st.caption(f"{len(view)} trades · P&L ${view['pnl'].sum():.2f}")
    st.dataframe(
        view[["symbol", "strategy", "asset_class", "entry_price", "exit_price",
              "position_size_usd", "pnl", "pnl_pct", "win_loss",
              "entry_time", "exit_time", "market_regime"]],
        hide_index=True, width="stretch", height=460)


def render_news():
    _topbar("News", "external intelligence")
    events = _table(
        "SELECT event_time, source_name, event_type, symbols, payload "
        "FROM external_events ORDER BY event_time DESC LIMIT 50")
    if events.empty:
        st.info("No external events ingested yet — external sources are optional and fail-soft.")
    else:
        st.dataframe(events, hide_index=True, width="stretch")
    sources = _table("SELECT source_name, last_success, last_failure, "
                     "consecutive_failures, disabled FROM source_health")
    if not sources.empty:
        st.subheader("Source health")
        st.dataframe(sources, hide_index=True, width="stretch")


FEEDBACK_PATH = ROOT / "logs" / "feedback.jsonl"


def render_feedback():
    import json
    from datetime import datetime, timezone
    _topbar("Feedback", "local notes")
    note = st.text_area("Note to self about bot behaviour, ideas, or issues", height=120)
    if st.button("Save note", type="primary") and note.strip():
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_PATH.open("a") as f:
            f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "note": note.strip()}) + "\n")
        st.success("Saved.")
    if FEEDBACK_PATH.exists():
        lines = FEEDBACK_PATH.read_text().strip().splitlines()
        if lines:
            st.subheader(f"Saved notes ({len(lines)})")
            for line in reversed(lines[-20:]):
                try:
                    item = json.loads(line)
                    st.markdown(f"- `{item['time'][:16]}` — {item['note']}")
                except Exception:
                    continue


def render_placeholder(title: str):
    st.markdown(
        f"""
        <div class='topbar'>
            <div><div class='subtle'>Dashboard</div><h1 class='title'>{title}</h1></div>
            <div class='badge'>Local</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class='panel'><div class='panel-inner'>
        <div class='panel-title'>This section is ready for local portfolio content.</div>
        </div></div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      :root {
        --bg: #070d15;
        --bg-2: #0d141d;
        --panel: rgba(18, 25, 34, 0.95);
        --panel-2: rgba(14, 19, 27, 0.9);
        --line: rgba(148, 163, 184, 0.12);
        --text: #eaf1f8;
        --muted: #8ea1be;
        --green: #69e7af;
        --green-2: #4fc98f;
        --cyan: #7de7ff;
        --red: #ff8e9d;
        --yellow: #f9d66c;
      }
      html, body, [data-testid="stAppViewContainer"] { background: linear-gradient(180deg, #04090f 0%, #090f17 100%); font-family: 'Inter', sans-serif; color: var(--text); }
      [data-testid="stAppViewContainer"] > .main { background: transparent; }
            .block-container { padding-top: 1.1rem; padding-left: 15rem; padding-right: 1.4rem; max-width: 1500px; }
            .local-nav { position: fixed; inset: 0 auto 0 0; width: 13rem; padding: 1.4rem 0.75rem; background: rgba(10, 15, 22, 0.96); border-right: 1px solid var(--line); z-index: 50; overflow-y: auto; }
      .brand { display: flex; align-items: center; gap: 0.7rem; padding: 0.4rem 0.5rem 1.1rem; font-size: 1.05rem; font-weight: 700; }
      .brand-mark { width: 28px; height: 28px; border-radius: 10px; display: grid; place-items: center; background: rgba(125,231,255,0.08); border: 1px solid rgba(125,231,255,0.22); color: var(--cyan); }
      .brand-mark svg { width: 16px; height: 16px; }
            .nav-caption { color: var(--muted); font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; margin: 0.2rem 0.45rem 0.45rem; }
            .nav-links { display: grid; gap: 0.42rem; }
            .nav-link { display: block; padding: 0.68rem 0.75rem; border-radius: 10px; color: var(--text); background: rgba(255,255,255,0.025); border: 1px solid transparent; text-decoration: none; font-size: 0.72rem; font-weight: 600; text-align: center; }
            .nav-link:hover { color: var(--text); border-color: rgba(125,231,255,0.2); background: rgba(125,231,255,0.055); text-decoration: none; }
            .nav-link.active { color: var(--text); border-color: rgba(125,231,255,0.25); background: rgba(125,231,255,0.11); }
            .nav-meta { margin: 1.35rem 0.45rem 0; display: grid; gap: 0.58rem; color: var(--muted); font-size: 0.68rem; }
      .stButton > button {
        border-radius: 12px; margin: 0.18rem 0; padding: 0.7rem 0.8rem; background: rgba(255,255,255,0.02); border: 1px solid transparent; color: var(--text); font-weight: 500; text-align: left; transition: 0.2s ease; }
      .stButton > button:hover { border-color: rgba(125,231,255,0.2); }
      .stButton > button[kind='primary'] { background: rgba(125,231,255,0.08); border-color: rgba(125,231,255,0.16); }
      .topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; }
      .title { margin: 0; font-size: 2rem; letter-spacing: -0.05em; font-weight: 700; }
      .subtle { color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem; }
      .badge { padding: 0.45rem 0.7rem; border-radius: 999px; border: 1px solid rgba(105,231,175,0.25); color: var(--green); background: rgba(105,231,175,0.08); font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; }
      .metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1rem; margin: 1rem 0 1.2rem; }
      .metric-card { background: linear-gradient(180deg, rgba(17,24,32,0.94), rgba(12,17,25,0.96)); border: 1px solid var(--line); border-radius: 16px; padding: 0.9rem 1rem; min-height: 124px; }
      .metric-head { display: flex; align-items: center; justify-content: space-between; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
      .metric-icon { width: 28px; height: 28px; display: grid; place-items: center; border-radius: 10px; background: rgba(125,231,255,0.08); color: var(--cyan); }
      .metric-icon svg { width: 15px; height: 15px; }
      .metric-value { font-size: clamp(1.8rem, 2vw, 2.4rem); line-height: 1.05; font-weight: 700; letter-spacing: -0.06em; margin-top: 0.8rem; }
      .metric-foot { margin-top: 0.45rem; color: var(--muted); font-size: 0.75rem; }
      .positive { color: var(--green); }
      .negative { color: #f87171; }
      .neutral { color: var(--yellow); }
      .panel { background: linear-gradient(180deg, rgba(15,22,30,0.96), rgba(10,17,23,0.96)); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 18px 40px rgba(4, 8, 12, 0.25); }
      .panel-inner { padding: 0.95rem 1rem 1rem; }
      .panel-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.7rem; }
      .panel-title { color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem; }
      .mini-chip { padding: 0.26rem 0.55rem; border-radius: 999px; background: rgba(255,255,255,0.02); border: 1px solid var(--line); color: var(--muted); font-size: 0.68rem; }
      .stDataFrame { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
      .stDataFrame th { color: var(--muted); text-transform: uppercase; font-size: 0.68rem; }
      .stDataFrame td { color: var(--text); }
      .stDataFrame { background: rgba(10,15,22,0.55); }
            @media (max-width: 1100px) { .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
            @media (max-width: 760px) { .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 11rem; } .local-nav { right: 0; bottom: auto; width: auto; height: 9.5rem; } .nav-links { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
    </style>
    """,
    unsafe_allow_html=True,
)

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Overview"

render_sidebar()

page = get_current_page()
PAGES = {
    "Overview": render_overview,
    "Alpha Engine": render_alpha_engine,
    "Paper Evidence": render_paper_evidence,
    "System Health": render_system_health,
    "My account": render_my_account,
    "Active Stocks": render_active_stocks,
    "Dividend Insights": render_dividend_insights,
    "Trading Stocks Chat": render_trading_chat,
    "Hybrid Funds": render_hybrid_funds,
    "Portfolio": render_portfolio,
    "Settings": render_settings,
    "History": render_history,
    "News": render_news,
    "Feedback": render_feedback,
}
PAGES.get(page, lambda: render_placeholder(page))()
