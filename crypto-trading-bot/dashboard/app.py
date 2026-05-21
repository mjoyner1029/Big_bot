"""
Multi-asset Trading Bot — Streamlit Dashboard

Launch:
    cd crypto-trading-bot
    streamlit run dashboard/app.py
"""
import sys, os

# Ensure the project root is importable
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import streamlit as st

from config.config import CONFIG, get_all_symbols, is_crypto
from config.symbols import CRYPTO_SYMBOLS, STOCK_SYMBOLS, get_all_available_symbols
from data.fetcher import fetch_latest_market_data
from indicators.ta_indicators import add_ta_indicators, get_latest_indicator_snapshot
from strategies.strategy_engine import generate_trade_signal
from strategies.thresholds import get_trade_thresholds
from trading.portfolio_manager import PortfolioManager
from trading.paper_trader import execute_paper_trade
from logs.trade_logger import log_trade

from dashboard.components import (
    build_candlestick,
    build_indicator_panel,
    build_equity_curve,
    build_allocation_chart,
    format_positions_df,
    format_trades_df,
)
from dashboard.autonomous_components import (
    load_autonomous_journal,
    load_cost_tracker,
    build_learning_metrics_card,
    build_symbol_preferences_chart,
    build_win_rate_by_confidence_chart,
    build_recent_decisions_table,
    build_velocity_gauges,
    build_cost_efficiency_card,
    build_cost_breakdown_chart,
    filter_autonomous_logs,
    load_recent_log_entries,
    build_world_events_card,
    build_learning_progress_chart,
)

# Page config
st.set_page_config(
    page_title="LIMITLESS - Autonomous AI Trading",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared state init ─────────────────────────────────────────────
if "portfolio" not in st.session_state:
    st.session_state.portfolio = PortfolioManager()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "market_cache" not in st.session_state:
    st.session_state.market_cache = {}       # sym -> df
if "cache_ts" not in st.session_state:
    st.session_state.cache_ts = 0.0
if "equity_history" not in st.session_state:
    st.session_state.equity_history = []


def _get_portfolio() -> PortfolioManager:
    return st.session_state.portfolio


def _current_prices() -> Dict[str, float]:
    """Build a {symbol: latest_close} map from cached data."""
    prices: Dict[str, float] = {}
    for sym, df in st.session_state.market_cache.items():
        if df is not None and not df.empty:
            prices[sym] = float(df["Close"].iloc[-1])
    return prices


def _refresh_market_data(symbols=None, force=False):
    """Fetch fresh OHLCV for all symbols (cached for 60 s)."""
    now = time.time()
    if not force and now - st.session_state.cache_ts < 60:
        return
    symbols = symbols or get_all_symbols()
    with st.spinner("Fetching market data..."):
        for sym in symbols:
            df = fetch_latest_market_data(ticker=sym)
            if df is not None and not df.empty:
                st.session_state.market_cache[sym] = df
    st.session_state.cache_ts = now

    # Record equity snapshot
    portfolio = _get_portfolio()
    prices = _current_prices()
    eq = portfolio.total_equity(prices)
    st.session_state.equity_history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "equity": eq,
    })


# Sidebar
with st.sidebar:
    st.title("LIMITLESS")
    st.caption("Autonomous AI Trading with Superhuman Intelligence")
    st.caption(f"Asset class: **{CONFIG['asset_class']}**")
    st.caption(f"Paper mode: **{CONFIG['use_paper_trading']}**")
    st.caption(f"LLM enabled: **{CONFIG.get('use_llm', False)}**")
    st.divider()

    if st.button("Refresh Data", use_container_width=True):
        _refresh_market_data(force=True)
        st.success("Data refreshed")

    if st.button("Save Portfolio", use_container_width=True):
        _get_portfolio().save_state()
        st.success("State saved")

    st.divider()
    st.caption("Powered by Streamlit + Plotly")


# Tab bar
tab_dash, tab_charts, tab_trade, tab_autonomous, tab_chat, tab_logs = st.tabs(
    ["Dashboard", "Charts", "Trade", "Autonomous AI", "Chat", "Logs"]
)

# Ensure we have data
_refresh_market_data()

# ══════════════════════════════════════════════════════════════════
#  TAB 1 — Dashboard
# ══════════════════════════════════════════════════════════════════
with tab_dash:
    portfolio = _get_portfolio()
    prices = _current_prices()
    summary = portfolio.summary(prices)

    # KPI row
    has_trades = summary["closed_trades"] > 0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Equity", f"${summary['equity']:,.2f}" if summary['equity'] else "--")
    c2.metric("Cash", f"${summary['cash']:,.2f}" if summary['cash'] else "--")
    c3.metric("Open Positions", summary["open_positions"])
    c4.metric("Realised P&L", f"${summary['total_realised_pnl']:,.2f}" if has_trades else "--")
    c5.metric("Win Rate", f"{summary['win_rate']*100:.1f}%" if has_trades else "--")

    # Equity curve + allocation donut side by side
    col_eq, col_alloc = st.columns([2, 1])
    with col_eq:
        fig_eq = build_equity_curve(
            st.session_state.equity_history,
            starting_capital=CONFIG["capital"],
        )
        st.plotly_chart(fig_eq, use_container_width=True)
    with col_alloc:
        fig_alloc = build_allocation_chart(
            portfolio.open_positions, portfolio.cash, prices,
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

    # Open positions
    st.subheader("Open Positions")
    pos_df = format_positions_df(portfolio.open_positions, prices)
    if pos_df.empty:
        st.info("No open positions")
    else:
        st.dataframe(pos_df, use_container_width=True, hide_index=True)

    # Recent closed trades
    st.subheader("Recent Trades")
    trades_df = format_trades_df(portfolio.closed_trades)
    if trades_df.empty:
        st.info("No closed trades yet")
    else:
        st.dataframe(trades_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 2 — Charts
# ══════════════════════════════════════════════════════════════════
with tab_charts:
    all_syms = get_all_available_symbols()
    chart_col1, chart_col2 = st.columns([1, 3])

    with chart_col1:
        selected_sym = st.selectbox("Symbol", all_syms, key="chart_symbol")
        chart_period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1, key="chart_period")
        chart_interval = st.selectbox("Interval", ["15m", "1h", "1d"], index=1, key="chart_interval")
        show_ema = st.checkbox("EMAs", value=True)
        show_bb = st.checkbox("Bollinger Bands", value=True)
        show_vol = st.checkbox("Volume", value=True)
        show_indicators = st.checkbox("Indicator Panel", value=True)

        if st.button("Load Chart", use_container_width=True):
            st.session_state._chart_reload = True

    with chart_col2:
        # Fetch data for selected symbol
        df_chart = fetch_latest_market_data(
            ticker=selected_sym, period=chart_period, interval=chart_interval,
        )
        if df_chart is not None and not df_chart.empty:
            df_ind = add_ta_indicators(df_chart)

            fig_candle = build_candlestick(
                df_ind, selected_sym,
                show_ema=show_ema, show_bb=show_bb, show_volume=show_vol,
            )
            st.plotly_chart(fig_candle, use_container_width=True)

            if show_indicators:
                fig_ind = build_indicator_panel(df_ind)
                st.plotly_chart(fig_ind, use_container_width=True)

            # Latest indicator snapshot
            with st.expander("Latest Indicator Values"):
                snap = get_latest_indicator_snapshot(df_chart)
                snap_clean = {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in snap.items()}
                st.json(snap_clean)
        else:
            st.warning(f"Could not fetch data for {selected_sym}")


# ══════════════════════════════════════════════════════════════════
#  TAB 3 — Manual Trade
# ══════════════════════════════════════════════════════════════════
with tab_trade:
    st.subheader("Manual Trade Entry")

    crypto_syms = CRYPTO_SYMBOLS
    stock_syms = STOCK_SYMBOLS

    trade_col1, trade_col2 = st.columns(2)

    with trade_col1:
        st.markdown("#### Select Asset")
        asset_type = st.radio("Asset type", ["Crypto", "Stock"], horizontal=True, key="trade_asset_type")
        if asset_type == "Crypto":
            trade_sym = st.selectbox("Crypto coin", crypto_syms, key="trade_crypto")
        else:
            trade_sym = st.selectbox("Stock ticker", stock_syms, key="trade_stock")

        trade_side = st.radio("Side", ["Buy", "Sell"], horizontal=True, key="trade_side")
        use_auto_signal = st.checkbox("Use bot signal (auto-analysis)", value=True, key="trade_auto")

    with trade_col2:
        st.markdown("#### Order Details")

        # Show current price
        df_trade = st.session_state.market_cache.get(trade_sym)
        if df_trade is None or df_trade.empty:
            df_trade = fetch_latest_market_data(ticker=trade_sym)
            if df_trade is not None:
                st.session_state.market_cache[trade_sym] = df_trade

        current_price = None
        price_ts = None
        if df_trade is not None and not df_trade.empty:
            current_price = float(df_trade["Close"].iloc[-1])
            # Extract the timestamp of the last bar
            if hasattr(df_trade.index, 'tz'):
                price_ts = df_trade.index[-1]
            else:
                price_ts = pd.Timestamp(df_trade.index[-1])
            st.metric("Current Price", f"${current_price:,.2f}")
            # Show how fresh the price is
            if price_ts is not None:
                try:
                    now_utc = pd.Timestamp.now(tz="UTC")
                    ts_utc = price_ts.tz_localize("UTC") if price_ts.tzinfo is None else price_ts
                    age = now_utc - ts_utc
                    age_min = int(age.total_seconds() // 60)
                    if age_min < 1:
                        age_str = "just now"
                    elif age_min < 60:
                        age_str = f"{age_min}m ago"
                    elif age_min < 1440:
                        age_str = f"{age_min // 60}h {age_min % 60}m ago"
                    else:
                        age_str = f"{age_min // 1440}d ago"
                    st.caption(f"Last bar: {ts_utc.strftime('%Y-%m-%d %H:%M %Z')} ({age_str})")
                except Exception:
                    st.caption(f"Last bar: {price_ts}")
        else:
            st.warning(f"Could not fetch price data for {trade_sym}")

        if not use_auto_signal:
            manual_confidence = st.slider("Confidence", 0.0, 1.0, 0.65, 0.05, key="trade_conf")
        else:
            manual_confidence = None

        manual_qty = st.number_input(
            "Quantity (0 = auto-size)",
            min_value=0.0, value=0.0, step=0.01,
            format="%.6f", key="trade_qty",
        )

    st.divider()

    exec_col1, exec_col2, exec_col3 = st.columns([1, 1, 2])

    with exec_col1:
        execute_btn = st.button(
            "Execute Trade", type="primary", use_container_width=True,
            disabled=(current_price is None),
        )
    with exec_col2:
        preview_btn = st.button("Preview Signal", use_container_width=True,
                                disabled=(current_price is None))

    # Signal generation / preview
    if preview_btn or execute_btn:
        if df_trade is None or df_trade.empty:
            st.error(f"No market data for {trade_sym}")
        else:
            if use_auto_signal:
                with st.spinner("Generating signal..."):
                    signal = generate_trade_signal(df_trade, symbol=trade_sym)
                if signal is None:
                    st.warning(f"Strategy returned no signal for {trade_sym} — try manual mode")
                else:
                    st.json(signal)
            else:
                # Build manual signal
                a_type = "crypto" if is_crypto(trade_sym) else "stock"
                thresholds = get_trade_thresholds(
                    current_price, manual_confidence,
                    side=trade_side.lower(), asset_type=a_type,
                )
                signal = {
                    "symbol": trade_sym,
                    "side": trade_side.lower(),
                    "asset_type": a_type,
                    "entry_price": current_price,
                    "confidence": round(manual_confidence, 4),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **thresholds,
                }
                st.json(signal)

            # Actually execute
            if execute_btn and signal is not None:
                portfolio = _get_portfolio()
                if not portfolio.can_open_position(signal):
                    st.error("Position rejected by portfolio risk gates")
                else:
                    qty = manual_qty if manual_qty > 0 else portfolio.compute_position_size(signal)
                    execute_paper_trade(signal, portfolio)
                    st.success(
                        f"{'Paper' if CONFIG['use_paper_trading'] else 'Live'} "
                        f"{signal['side'].upper()} {qty:.6f} {trade_sym} "
                        f"@ ${current_price:,.2f}"
                    )
                    st.rerun()


# ══════════════════════════════════════════════════════════════════
#  TAB 4 — Autonomous AI (Real-Time Learning & Analysis)
# ══════════════════════════════════════════════════════════════════
with tab_autonomous:
    st.header("Autonomous Trading System")
    st.caption("Real-time monitoring of autonomous decision-making, learning, and analysis")
    
    # Auto-refresh toggle
    col_refresh, col_interval = st.columns([3, 1])
    with col_refresh:
        auto_refresh = st.checkbox("Auto-refresh (every 30s)", value=False)
    with col_interval:
        if auto_refresh:
            st.info("Auto-refreshing...")
            time.sleep(30)
            st.rerun()
    
    st.divider()
    
    # Load data
    journal = load_autonomous_journal()
    cost_data = load_cost_tracker()
    portfolio = _get_portfolio()
    prices = _current_prices()
    summary = portfolio.summary(prices)
    
    # Section 1: Learning Metrics
    st.subheader("Learning & Performance")
    
    metrics = build_learning_metrics_card(journal)
    velocity = build_velocity_gauges(journal)
    
    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Decisions", metrics["total_decisions"])
    col2.metric("Trades Executed", metrics["executed"])
    col3.metric("Win Rate", f"{metrics['win_rate']:.1%}" if metrics['with_outcomes'] > 0 else "N/A")
    col4.metric("Trades/Hour", f"{velocity['current_velocity']:.1f}")
    col5.metric("Execution Rate", f"{velocity['execution_rate']:.1f}%")
    
    # Charts Row
    col_prog, col_conf = st.columns(2)
    with col_prog:
        st.plotly_chart(
            build_learning_progress_chart(journal),
            use_container_width=True,
        )
    with col_conf:
        st.plotly_chart(
            build_win_rate_by_confidence_chart(journal),
            use_container_width=True,
        )
    
    # Symbol Preferences
    st.plotly_chart(
        build_symbol_preferences_chart(journal),
        use_container_width=True,
    )
    
    st.divider()
    
    # Section 2: World Events & Market Analysis
    st.subheader("World Events Analysis")
    
    events = build_world_events_card()
    
    if events["available"]:
        col_sent, col_mag, col_summary = st.columns([1, 1, 2])
        
        with col_sent:
            sentiment_color = "normal"
            if events["sentiment"] > 0.3:
                sentiment_color = "normal"
            elif events["sentiment"] < -0.3:
                sentiment_color = "inverse"
            
            st.metric(
                "Market Sentiment",
                f"{events['sentiment']:+.2f}",
                delta="Bullish" if events["sentiment"] > 0 else "Bearish",
                delta_color=sentiment_color,
            )
        
        with col_mag:
            st.metric(
                "Event Magnitude",
                f"{events['magnitude']:.2f}",
                help="0 = negligible, 1 = major market-moving events"
            )
        
        with col_summary:
            st.info(f"**Summary:** {events['summary']}")
    else:
        st.warning("World events analysis not available. Ensure the bot is running with `enable_world_events_analysis: True`")
    
    st.divider()
    
    # Section 3: Cost Tracking & Efficiency
    st.subheader("Cost Tracking & Efficiency")
    
    cost_metrics = build_cost_efficiency_card(cost_data, summary["total_realised_pnl"])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Costs", f"${cost_metrics['total_costs']:.2f}")
    col2.metric("Net P&L", f"${cost_metrics['net_pnl']:.2f}")
    col3.metric("Cost Ratio", f"{cost_metrics['cost_ratio']:.1f}%")
    col4.metric("Efficiency", cost_metrics['efficiency'])
    
    col_breakdown, col_details = st.columns([1, 1])
    
    with col_breakdown:
        st.plotly_chart(
            build_cost_breakdown_chart(cost_data),
            use_container_width=True,
        )
    
    with col_details:
        st.write("**Cost Details:**")
        st.write(f"- Exchange Fees: ${cost_metrics['fees']:.2f}")
        st.write(f"- Slippage: ${cost_metrics['slippage']:.2f}")
        st.write(f"- API Costs: ${cost_metrics['api_costs']:.2f}")
        st.write(f"- Total Trades: {cost_metrics['total_trades']}")
        st.write(f"- Avg Cost/Trade: ${cost_metrics['total_costs'] / max(1, cost_metrics['total_trades']):.2f}")
        
        st.caption("**Efficiency Benchmarks:**")
        st.caption("- Excellent: < 5% (Claude-level)")
        st.caption("- Good: 5-15%")
        st.caption("- Acceptable: 15-25%")
        st.caption("- WARNING Poor: > 25%")
    
    st.divider()
    
    # Section 4: Recent Decisions & Reasoning
    st.subheader("Recent Autonomous Decisions")
    
    decisions_df = build_recent_decisions_table(journal, limit=30)
    
    if not decisions_df.empty:
        st.dataframe(
            decisions_df,
            use_container_width=True,
            height=400,
        )
    else:
        st.info("No decisions recorded yet. Start the bot to see autonomous decision-making in action.")
    
    st.divider()
    
    # Section 5: Live Decision Log
    st.subheader("Live Decision Log")
    st.caption("Real-time feed of autonomous decisions and reasoning")
    
    log_lines = load_recent_log_entries("logs/bot.log", lines=200)
    filtered_logs = filter_autonomous_logs(log_lines)
    
    if filtered_logs:
        log_text = "\n".join(filtered_logs[-30:])  # Last 30 relevant entries
        st.text_area(
            "Recent Autonomous Activity",
            value=log_text,
            height=400,
            disabled=True,
        )
    else:
        st.info("No autonomous activity logged yet. Ensure the bot is running.")
    
    # Quick actions
    st.divider()
    col_save, col_reset, col_status = st.columns(3)
    
    with col_save:
        if st.button("Save Learning State", use_container_width=True):
            try:
                from models.autonomous_agent import get_autonomous_agent
                agent = get_autonomous_agent()
                agent.save_journal()
                st.success("Learning state saved!")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col_reset:
        if st.button("Refresh Data", use_container_width=True):
            st.rerun()
    
    with col_status:
        if st.button("Get Full Status", use_container_width=True):
            try:
                from autonomous_mode import get_autonomous_status
                status = get_autonomous_status()
                st.json(status)
            except Exception as e:
                st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════
#  TAB 5 — Chat with Claude
# ══════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("Chat with Claude -- Market Analyst")

    api_key = CONFIG.get("anthropic_api_key", "")
    if not api_key:
        st.warning(
            "No `ANTHROPIC_API_KEY` set in `.env`. "
            "Add your key to enable the chat feature."
        )

    # System prompt with portfolio context
    def _build_system_prompt() -> str:
        portfolio = _get_portfolio()
        prices = _current_prices()
        summary = portfolio.summary(prices)
        positions_text = ""
        for p in portfolio.open_positions:
            sym = p["symbol"]
            cur = prices.get(sym, p["entry_price"])
            pnl = (cur - p["entry_price"]) * p["qty"] if p["side"] == "buy" else (p["entry_price"] - cur) * p["qty"]
            positions_text += (
                f"  - {p['side'].upper()} {p['qty']:.4f} {sym} "
                f"@ ${p['entry_price']:.2f}  current=${cur:.2f}  P&L=${pnl:.2f}\n"
            )
        if not positions_text:
            positions_text = "  (no open positions)\n"

        return (
            "You are an expert financial analyst and trading advisor embedded in a "
            "multi-asset trading bot that trades crypto and US stocks. "
            "You have access to the user's live portfolio state below.\n\n"
            f"Portfolio Summary:\n"
            f"  Equity: ${summary['equity']:,.2f}\n"
            f"  Cash: ${summary['cash']:,.2f}\n"
            f"  Open positions: {summary['open_positions']}\n"
            f"  Closed trades: {summary['closed_trades']}\n"
            f"  Realised P&L: ${summary['total_realised_pnl']:,.2f}\n"
            f"  Win rate: {summary['win_rate']*100:.1f}%\n\n"
            f"Open Positions:\n{positions_text}\n"
            f"Watchlist symbols: {get_all_symbols()}\n\n"
            "Answer the user's questions about markets, trading strategy, "
            "technical analysis, news, or their portfolio. Be concise and practical. "
            "When appropriate, suggest specific actions (buy/sell/hold) with reasoning."
        )

    # Chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask Claude about markets, your portfolio, strategy...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Call Claude
        with st.chat_message("assistant"):
            if not api_key:
                response = "No API key configured. Please add ANTHROPIC_API_KEY to your .env file."
                st.markdown(response)
            else:
                with st.spinner("Thinking..."):
                    try:
                        import anthropic
                        client = anthropic.Anthropic(api_key=api_key)

                        # Build messages list from history
                        messages = []
                        for m in st.session_state.chat_history:
                            messages.append({"role": m["role"], "content": m["content"]})

                        message = client.messages.create(
                            model=CONFIG.get("anthropic_model", "claude-3-5-sonnet-latest"),
                            max_tokens=2048,
                            temperature=0.4,
                            system=_build_system_prompt(),
                            messages=messages,
                        )
                        response = message.content[0].text
                    except Exception as e:
                        response = f"Error calling Claude: {e}"

                st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})


# ══════════════════════════════════════════════════════════════════
#  TAB 6 — Logs
# ══════════════════════════════════════════════════════════════════
with tab_logs:
    log_col1, log_col2 = st.columns(2)

    with log_col1:
        st.subheader("Trade Log (CSV)")
        log_path = CONFIG.get("trade_log_path", "logs/trade_log.csv")
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            try:
                log_df = pd.read_csv(log_path)
                st.dataframe(log_df.tail(100).iloc[::-1], use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Could not read trade log: {e}")
        else:
            st.info("No trades logged yet")

    with log_col2:
        st.subheader("Bot Log (last 80 lines)")
        bot_log_path = CONFIG.get("bot_log_path", "logs/bot.log")
        if os.path.exists(bot_log_path):
            try:
                with open(bot_log_path, "r") as f:
                    lines = f.readlines()
                tail = lines[-80:] if len(lines) > 80 else lines
                st.code("".join(tail), language="log")
            except Exception as e:
                st.warning(f"Could not read bot log: {e}")
        else:
            st.info("No bot log file yet")

    st.divider()

    # Portfolio state JSON viewer
    with st.expander("Raw Portfolio State (JSON)"):
        state_path = CONFIG.get("state_path", "logs/bot_state.json")
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            st.json(state)
        else:
            st.info("No saved state file")

    # Config viewer
    with st.expander("Bot Configuration"):
        safe_config = {k: v for k, v in CONFIG.items()
                       if "key" not in k.lower() and "secret" not in k.lower()
                       and "password" not in k.lower() and "passphrase" not in k.lower()}
        st.json(safe_config)
