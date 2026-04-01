"""Reusable chart and table builders for the Streamlit dashboard.

All functions return Plotly figures or styled DataFrames — the caller
just passes them to ``st.plotly_chart`` / ``st.dataframe``.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any


# ── Colour palette ────────────────────────────────────────────────
GREEN = "#00c853"
RED   = "#ff1744"
BLUE  = "#2979ff"
GREY  = "#b0bec5"
BG    = "#0e1117"
CARD  = "#1e1e2f"


# ── Candlestick + overlays ───────────────────────────────────────

def build_candlestick(
    df: pd.DataFrame,
    symbol: str,
    show_ema: bool = True,
    show_bb: bool = True,
    show_volume: bool = True,
) -> go.Figure:
    """
    Build a candlestick chart with optional EMA / Bollinger-Band overlays
    and a volume sub-plot.
    """
    rows = 2 if show_volume else 1
    heights = [0.75, 0.25] if show_volume else [1.0]
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
    )

    # Candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Price",
        increasing_line_color=GREEN, decreasing_line_color=RED,
    ), row=1, col=1)

    # EMAs
    if show_ema:
        for col, colour, dash in [
            ("ema9", "#ffab00", "dot"),
            ("ema20", "#00e5ff", "dash"),
            ("ema50", "#d500f9", "solid"),
        ]:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col], name=col.upper(),
                    line=dict(color=colour, width=1, dash=dash),
                ), row=1, col=1)

    # Bollinger Bands
    if show_bb and "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"], name="BB Upper",
            line=dict(color=GREY, width=0.7, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"], name="BB Lower",
            line=dict(color=GREY, width=0.7, dash="dot"),
            fill="tonexty", fillcolor="rgba(176,190,197,0.07)",
        ), row=1, col=1)

    # Volume
    if show_volume and "Volume" in df.columns:
        colours = [GREEN if c >= o else RED for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=colours, opacity=0.5,
        ), row=2, col=1)

    fig.update_layout(
        title=f"{symbol} — Price Chart",
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=BG,
        xaxis_rangeslider_visible=False,
        height=560,
        margin=dict(l=50, r=20, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# ── Indicator sub-charts ─────────────────────────────────────────

def build_indicator_panel(df: pd.DataFrame) -> go.Figure:
    """RSI + MACD + Stochastic in stacked subplots."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["RSI (14)", "MACD", "Stochastic"],
        row_heights=[0.33, 0.33, 0.34],
    )

    # RSI
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi"], name="RSI",
            line=dict(color=BLUE, width=1.2),
        ), row=1, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=RED, row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=GREEN, row=1, col=1)

    # MACD
    for col, colour in [("macd", BLUE), ("macd_signal", "#ff6d00")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col.upper(),
                line=dict(color=colour, width=1),
            ), row=2, col=1)
    if "macd_hist" in df.columns:
        colours = [GREEN if v >= 0 else RED for v in df["macd_hist"]]
        fig.add_trace(go.Bar(
            x=df.index, y=df["macd_hist"], name="MACD Hist",
            marker_color=colours, opacity=0.6,
        ), row=2, col=1)

    # Stochastic
    for col, colour in [("stoch_k", BLUE), ("stoch_d", "#ff6d00")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col.upper(),
                line=dict(color=colour, width=1),
            ), row=3, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color=RED, row=3, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color=GREEN, row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=480, showlegend=False,
        margin=dict(l=50, r=20, t=30, b=20),
    )
    return fig


# ── Equity curve ──────────────────────────────────────────────────

def build_equity_curve(
    equity_data: List[Dict[str, Any]],
    starting_capital: float = 500,
) -> go.Figure:
    """
    Build an equity-over-time line from a list of
    ``{"timestamp": ..., "equity": ...}`` dicts.
    Falls back to a simple capital line if no data.
    """
    if not equity_data:
        fig = go.Figure()
        fig.add_annotation(text="No equity data yet", showarrow=False,
                           font=dict(size=18, color=GREY))
        fig.update_layout(template="plotly_dark", paper_bgcolor=BG,
                          plot_bgcolor=BG, height=300)
        return fig

    df = pd.DataFrame(equity_data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        x = df["timestamp"]
    else:
        x = list(range(len(df)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=df["equity"], name="Equity",
        fill="tozeroy", fillcolor="rgba(41,121,255,0.15)",
        line=dict(color=BLUE, width=2),
    ))
    fig.add_hline(y=starting_capital, line_dash="dash",
                  line_color=GREY, annotation_text="Starting capital")
    fig.update_layout(
        title="Portfolio Equity",
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=300,
        margin=dict(l=50, r=20, t=40, b=30),
    )
    return fig


# ── Positions table formatter ─────────────────────────────────────

def format_positions_df(positions: List[Dict], current_prices: Dict[str, float]) -> pd.DataFrame:
    """Convert raw position dicts into a display-ready DataFrame."""
    if not positions:
        return pd.DataFrame(columns=["Symbol", "Side", "Qty", "Entry", "Current", "P&L", "TP", "SL"])

    rows = []
    for p in positions:
        sym = p["symbol"]
        cur = current_prices.get(sym, p["entry_price"])
        entry = p["entry_price"]
        qty = p["qty"]
        side = p["side"]
        pnl = (cur - entry) * qty if side == "buy" else (entry - cur) * qty
        rows.append({
            "Symbol": sym,
            "Side": side.upper(),
            "Qty": round(qty, 6),
            "Entry": f"${entry:,.2f}",
            "Current": f"${cur:,.2f}",
            "P&L": f"${pnl:,.2f}",
            "TP": f"${p.get('take_profit_price', 0):,.2f}",
            "SL": f"${p.get('stop_loss_price', 0):,.2f}",
        })
    return pd.DataFrame(rows)


def format_trades_df(trades: List[Dict]) -> pd.DataFrame:
    """Convert closed-trade dicts into a display-ready DataFrame."""
    if not trades:
        return pd.DataFrame(columns=["Time", "Symbol", "Side", "Entry", "Exit", "P&L", "Result"])

    rows = []
    for t in trades[-50:]:  # last 50
        rows.append({
            "Time": t.get("closed_at", "")[:19],
            "Symbol": t["symbol"],
            "Side": t["side"].upper(),
            "Entry": f"${t['entry_price']:,.2f}",
            "Exit": f"${t.get('exit_price', 0):,.2f}",
            "P&L": f"${t.get('pnl', 0):,.2f}",
            "Result": t.get("result", ""),
        })
    return pd.DataFrame(rows[::-1])  # newest first


# ── Allocation donut ──────────────────────────────────────────────

def build_allocation_chart(
    positions: List[Dict],
    cash: float,
    current_prices: Dict[str, float],
) -> go.Figure:
    """Donut chart of portfolio allocation (cash + each position)."""
    labels = ["Cash"]
    values = [cash]
    colours = [GREY]
    palette = ["#2979ff", "#00e676", "#ff9100", "#e040fb", "#00e5ff",
               "#ffea00", "#ff5252", "#69f0ae", "#7c4dff", "#ffd740"]

    for i, p in enumerate(positions):
        sym = p["symbol"]
        cur = current_prices.get(sym, p["entry_price"])
        val = p["qty"] * cur
        labels.append(sym)
        values.append(round(val, 2))
        colours.append(palette[i % len(palette)])

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55, marker_colors=colours,
        textinfo="label+percent",
        textfont_size=11,
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig
