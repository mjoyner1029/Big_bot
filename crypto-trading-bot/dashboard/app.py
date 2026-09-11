import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "trade_memory.sqlite"

st.set_page_config(page_title="liquidity13 Dashboard", page_icon="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%230b1220'/%3E%3Cpath d='M20 18h24v6H20zm0 11h24v6H20zm0 11h17v6H20z' fill='%23dfeaf8'/%3E%3C/svg%3E", layout="wide")

NAV_ITEMS = [
    "Overview",
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


def _sidebar_icon(path_svg: str):
    return f"<svg viewBox='0 0 24 24' aria-hidden='true'>{path_svg}</svg>"


def render_sidebar():
    summary = get_summary()
    st.sidebar.markdown(
        """
        <div class='brand'>
            <div class='brand-mark'>
                <svg viewBox='0 0 24 24'><path d='M12 2.5a1.5 1.5 0 0 1 1.5 1.5v1.1A7.1 7.1 0 0 1 18.9 10h1.1a1.5 1.5 0 0 1 0 3h-1.1a7.1 7.1 0 0 1-5.4 5.4v1.1a1.5 1.5 0 0 1-3 0v-1.1A7.1 7.1 0 0 1 5.1 13H4a1.5 1.5 0 0 1 0-3h1.1A7.1 7.1 0 0 1 10.5 4.6V3.5A1.5 1.5 0 0 1 12 2.5Zm0 5.3a4.2 4.2 0 1 0 0 8.4 4.2 4.2 0 0 0 0-8.4Z' fill='currentColor'/></svg>
            </div>
            <span>liquidity13</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_page = st.session_state.get("nav_page", "Overview")
    if current_page not in NAV_ITEMS:
        current_page = "Overview"

    selected = st.sidebar.radio(
        "Navigation",
        NAV_ITEMS,
        index=NAV_ITEMS.index(current_page),
        key="nav_page",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    st.sidebar.caption("Local portfolio monitor")
    st.sidebar.caption(f"SQLite: {DB_PATH.name}")
    st.sidebar.caption(f"Trades: {len(get_trade_history(limit=1000))}")
    st.sidebar.caption(f"Open positions: {summary['open_positions']}")
    st.sidebar.caption(f"Win rate: {summary['win_rate'] * 100:.1f}%")


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
        "",
        options=["1H", "1D", "1M", "6M", "1Y", "ALLTIME"],
        default="ALLTIME",
        selection_mode="single",
        label_visibility="collapsed",
    )
    earnings_df = build_earnings_series(trade_history, range_mode)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=earnings_df["period"], y=earnings_df["earnings"], mode="lines", line=dict(color="#74f3b4", width=3), fill="tozeroy", fillcolor="rgba(116,243,180,0.1)"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10,r=10,t=10,b=10),
        height=280,
        xaxis=dict(showgrid=False, linecolor="rgba(148,163,184,0.2)", tickfont=dict(color="#9aa8bd", size=11)),
        yaxis=dict(showgrid=True, gridcolor="rgba(148,163,184,0.12)", linecolor="rgba(148,163,184,0.2)", tickfont=dict(color="#9aa8bd", size=11)),
        font=dict(color="#e7edf7"),
        showlegend=False,
    )

    left, right = st.columns([1.4, 0.9])
    with left:
        st.markdown(f"<div class='panel'><div class='panel-inner'><div class='panel-header'><div class='panel-title'>Cumulative P&L</div><div class='mini-chip'>{(range_mode or 'ALLTIME').lower()}</div></div></div></div>", unsafe_allow_html=True)
        if earnings_df.empty:
            st.write("No trades in this range.")
        else:
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("<div class='panel'><div class='panel-inner'><div class='panel-header'><div class='panel-title'>Win rate</div><div class='mini-chip'>closed trades</div></div></div></div>", unsafe_allow_html=True)
        win_rate_pct = summary["win_rate"] * 100
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=win_rate_pct, domain={"x": [0, 1], "y": [0, 1]}, number={"suffix": "%", "font":{"color":"#e7edf7","size":28}}, gauge={"axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#6e7f9f"}, "bar":{"color":"#7ee7b8"}, "bgcolor":"rgba(0,0,0,0)", "steps":[{"range":[0,50],"color":"rgba(255,255,255,0.08)"},{"range":[50,80],"color":"rgba(126,231,184,0.18)"},{"range":[80,100],"color":"rgba(126,231,184,0.4)"}], "threshold":{"line":{"color":"#7ee7b8","width":3},"thickness":0.75,"value":50}}))
        gauge.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=220, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e7edf7"))
        st.plotly_chart(gauge, use_container_width=True)

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
      .block-container { padding-top: 1.1rem; padding-left: 1.4rem; padding-right: 1.4rem; max-width: 1500px; }
      section[data-testid="stSidebar"] { background: rgba(10, 15, 22, 0.9); border-right: 1px solid var(--line); }
      .brand { display: flex; align-items: center; gap: 0.7rem; padding: 0.4rem 0.5rem 1.1rem; font-size: 1.05rem; font-weight: 700; }
      .brand-mark { width: 28px; height: 28px; border-radius: 10px; display: grid; place-items: center; background: rgba(125,231,255,0.08); border: 1px solid rgba(125,231,255,0.22); color: var(--cyan); }
      .brand-mark svg { width: 16px; height: 16px; }
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
    </style>
    """,
    unsafe_allow_html=True,
)

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Overview"

render_sidebar()

page = st.session_state.nav_page
if page == "Overview":
    render_overview()
else:
    render_placeholder(page)
