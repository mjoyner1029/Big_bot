"""Dashboard components for autonomous trading features.

Real-time visualization of:
  • Autonomous agent learning and decisions
  • World events analysis
  • Velocity optimization metrics
  • Cost tracking and efficiency
  • Decision reasoning logs
"""
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Colour palette
GREEN = "#00c853"
RED = "#ff1744"
BLUE = "#2979ff"
ORANGE = "#ff6d00"
PURPLE = "#aa00ff"
GREY = "#b0bec5"
BG = "#0e1e17"
CARD = "#1e1e2f"


def load_autonomous_journal() -> Dict[str, Any]:
    """Load the autonomous agent's learning journal."""
    path = "logs/autonomous_journal.json"
    if not os.path.exists(path):
        return {
            "outcome_patterns": {},
            "win_rate_by_confidence": {},
            "win_rate_by_symbol": {},
            "symbol_preferences": {},
            "recent_decisions": [],
        }
    
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def load_cost_tracker() -> Dict[str, Any]:
    """Load cost tracking data."""
    path = "logs/cost_tracker.json"
    if not os.path.exists(path):
        return {
            "total_exchange_fees": 0,
            "total_slippage_cost": 0,
            "total_api_costs": 0,
            "total_trades": 0,
        }
    
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def build_learning_metrics_card(journal: Dict[str, Any]) -> Dict[str, Any]:
    """Build metrics card data for autonomous learning."""
    recent = journal.get("recent_decisions", [])
    executed = [d for d in recent if d.get("was_executed")]
    with_outcome = [d for d in executed if d.get("outcome") is not None]
    
    wins = sum(1 for d in with_outcome if d.get("outcome") == "win")
    win_rate = (wins / len(with_outcome)) if with_outcome else 0
    
    return {
        "total_decisions": len(recent),
        "executed": len(executed),
        "with_outcomes": len(with_outcome),
        "win_rate": win_rate,
        "wins": wins,
        "losses": len(with_outcome) - wins,
    }


def build_symbol_preferences_chart(journal: Dict[str, Any]) -> go.Figure:
    """Bar chart of learned symbol preferences."""
    prefs = journal.get("symbol_preferences", {})
    
    if not prefs:
        fig = go.Figure()
        fig.add_annotation(
            text="No symbol preferences learned yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=GREY)
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            height=300,
        )
        return fig
    
    # Sort by preference value
    sorted_prefs = sorted(prefs.items(), key=lambda x: x[1], reverse=True)
    symbols = [s for s, _ in sorted_prefs[:15]]  # Top 15
    values = [v for _, v in sorted_prefs[:15]]
    
    colors = [GREEN if v > 0 else RED for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=symbols,
            y=values,
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
        )
    ])
    
    fig.update_layout(
        title="Learned Symbol Preferences (Top 15)",
        xaxis_title="Symbol",
        yaxis_title="Confidence Adjustment",
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        height=350,
        margin=dict(l=50, r=20, t=50, b=80),
    )
    
    return fig


def build_win_rate_by_confidence_chart(journal: Dict[str, Any]) -> go.Figure:
    """Heatmap-style chart showing win rate at different confidence levels."""
    conf_stats = journal.get("win_rate_by_confidence", {})
    
    if not conf_stats:
        fig = go.Figure()
        fig.add_annotation(
            text="No confidence statistics yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=GREY)
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            height=250,
        )
        return fig
    
    # Parse stats
    data = []
    for conf_str, (wins, total) in conf_stats.items():
        if total > 0:
            data.append({
                "confidence": float(conf_str),
                "win_rate": wins / total,
                "total": total,
                "wins": wins,
            })
    
    if not data:
        fig = go.Figure()
        fig.add_annotation(
            text="No confidence statistics yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=GREY)
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            height=250,
        )
        return fig
    
    df = pd.DataFrame(data).sort_values("confidence")
    
    # Color gradient based on win rate
    colors = []
    for wr in df["win_rate"]:
        if wr >= 0.6:
            colors.append(GREEN)
        elif wr >= 0.5:
            colors.append(BLUE)
        elif wr >= 0.4:
            colors.append(ORANGE)
        else:
            colors.append(RED)
    
    fig = go.Figure(data=[
        go.Bar(
            x=df["confidence"],
            y=df["win_rate"],
            marker_color=colors,
            text=[f"{wr:.0%}<br>({w}/{t})" for wr, w, t in zip(df["win_rate"], df["wins"], df["total"])],
            textposition="outside",
            hovertemplate="Confidence: %{x:.2f}<br>Win Rate: %{y:.1%}<extra></extra>",
        )
    ])
    
    fig.update_layout(
        title="Win Rate by Confidence Level",
        xaxis_title="Confidence Threshold",
        yaxis_title="Win Rate",
        yaxis_tickformat=".0%",
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        height=300,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    
    # Add target line at 50%
    fig.add_hline(y=0.5, line_dash="dash", line_color=GREY, opacity=0.5)
    
    return fig


def build_recent_decisions_table(journal: Dict[str, Any], limit: int = 20) -> pd.DataFrame:
    """Table of recent autonomous decisions."""
    recent = journal.get("recent_decisions", [])[-limit:]
    
    if not recent:
        return pd.DataFrame(columns=["Time", "Symbol", "Decision", "Confidence", "Reasoning", "Outcome"])
    
    data = []
    for d in reversed(recent):  # Most recent first
        data.append({
            "Time": d.get("timestamp", "")[:19].replace("T", " "),
            "Symbol": d.get("symbol", ""),
            "Decision": "EXECUTE" if d.get("was_executed") else "SKIP",
            "Base Conf": f"{d.get('base_confidence', 0):.2f}",
            "Adj Conf": f"{d.get('adjusted_confidence', 0):.2f}",
            "Outcome": d.get("outcome", "pending") if d.get("was_executed") else "-",
            "P&L": f"${d.get('pnl', 0):.2f}" if d.get("pnl") is not None else "-",
            "Reasoning": (d.get("reasoning", "")[:60] + "...") if len(d.get("reasoning", "")) > 60 else d.get("reasoning", ""),
        })
    
    return pd.DataFrame(data)


def build_velocity_gauges(journal: Dict[str, Any]) -> Dict[str, Any]:
    """Build velocity metrics gauges."""
    # This will be populated from velocity_optimizer in real-time
    # For now, calculate from journal
    recent = journal.get("recent_decisions", [])
    if len(recent) < 2:
        return {
            "current_velocity": 0,
            "target_velocity": 2.5,
            "execution_rate": 0,
            "trades_today": 0,
        }
    
    # Calculate velocity from recent decisions (last hour)
    now = datetime.now(timezone.utc)
    one_hour_ago = now.timestamp() - 3600
    
    recent_hour = [
        d for d in recent
        if datetime.fromisoformat(d.get("timestamp", "")).timestamp() > one_hour_ago
    ]
    
    executed = [d for d in recent_hour if d.get("was_executed")]
    
    return {
        "current_velocity": len(executed),
        "target_velocity": 2.5,
        "execution_rate": (len(executed) / len(recent_hour) * 100) if recent_hour else 0,
        "trades_today": len([d for d in recent if d.get("was_executed")]),
    }


def build_cost_efficiency_card(cost_data: Dict[str, Any], total_pnl: float) -> Dict[str, Any]:
    """Build cost efficiency metrics."""
    total_fees = cost_data.get("total_exchange_fees", 0)
    total_slippage = cost_data.get("total_slippage_cost", 0)
    total_api = cost_data.get("total_api_costs", 0)
    total_costs = total_fees + total_slippage + total_api
    
    net_pnl = total_pnl - total_costs
    
    if total_pnl == 0:
        efficiency = "N/A"
        cost_ratio = 0
    else:
        cost_ratio = (total_costs / abs(total_pnl)) * 100
        if total_pnl > total_costs and cost_ratio < 5:
            efficiency = "Excellent"
        elif total_pnl > total_costs and cost_ratio < 15:
            efficiency = "Good"
        elif total_pnl > total_costs:
            efficiency = "Acceptable"
        else:
            efficiency = "WARNING Poor"
    
    return {
        "total_costs": total_costs,
        "fees": total_fees,
        "slippage": total_slippage,
        "api_costs": total_api,
        "gross_pnl": total_pnl,
        "net_pnl": net_pnl,
        "cost_ratio": cost_ratio,
        "efficiency": efficiency,
        "total_trades": cost_data.get("total_trades", 0),
    }


def build_cost_breakdown_chart(cost_data: Dict[str, Any]) -> go.Figure:
    """Pie chart of cost breakdown."""
    fees = cost_data.get("total_exchange_fees", 0)
    slippage = cost_data.get("total_slippage_cost", 0)
    api = cost_data.get("total_api_costs", 0)
    
    if fees + slippage + api == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No costs recorded yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=GREY)
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            height=300,
        )
        return fig
    
    fig = go.Figure(data=[go.Pie(
        labels=["Exchange Fees", "Slippage", "API Costs"],
        values=[fees, slippage, api],
        marker_colors=[RED, ORANGE, PURPLE],
        hole=0.4,
        textinfo="label+percent",
        textposition="outside",
    )])
    
    fig.update_layout(
        title="Cost Breakdown",
        template="plotly_dark",
        paper_bgcolor=BG,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
    )
    
    return fig


def load_recent_log_entries(log_path: str = "logs/bot.log", lines: int = 100) -> List[str]:
    """Load recent log entries."""
    if not os.path.exists(log_path):
        return ["Log file not found"]
    
    try:
        with open(log_path, "r") as f:
            all_lines = f.readlines()
            return all_lines[-lines:]
    except Exception as e:
        return [f"Error reading log: {e}"]


def filter_autonomous_logs(log_lines: List[str]) -> List[str]:
    """Filter log lines for autonomous-related entries."""
    keywords = ["[Autonomous]", "[WorldEvents]", "[Velocity]", "[CostTracker]", "EXECUTE", "REJECTED"]
    filtered = []
    
    for line in log_lines:
        if any(kw in line for kw in keywords):
            filtered.append(line.strip())
    
    return filtered[-50:]  # Last 50 relevant entries


def build_world_events_card() -> Dict[str, Any]:
    """Build world events analysis card (if available)."""
    # Try to get from autonomous agent's cached analysis
    try:
        from autonomous_mode import get_autonomous_status
        status = get_autonomous_status()
        events = status.get("world_events", {})
        
        if events:
            return {
                "sentiment": events.get("overall_sentiment", 0),
                "magnitude": events.get("magnitude", 0),
                "summary": events.get("summary", "No analysis available"),
                "available": True,
            }
    except Exception:
        pass
    
    return {
        "sentiment": 0,
        "magnitude": 0,
        "summary": "World events analysis not available",
        "available": False,
    }


def build_learning_progress_chart(journal: Dict[str, Any]) -> go.Figure:
    """Line chart showing learning progress over time (win rate trend)."""
    recent = journal.get("recent_decisions", [])
    executed = [d for d in recent if d.get("was_executed") and d.get("outcome") is not None]
    
    if len(executed) < 10:
        fig = go.Figure()
        fig.add_annotation(
            text="Need at least 10 trades for progress tracking",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=GREY)
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=BG,
            height=300,
        )
        return fig
    
    # Calculate rolling win rate (window=20)
    window = min(20, len(executed) // 2)
    rolling_data = []
    
    for i in range(window, len(executed) + 1):
        batch = executed[i-window:i]
        wins = sum(1 for d in batch if d.get("outcome") == "win")
        win_rate = wins / len(batch)
        rolling_data.append({
            "trade_num": i,
            "win_rate": win_rate,
        })
    
    df = pd.DataFrame(rolling_data)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df["trade_num"],
        y=df["win_rate"],
        mode="lines",
        line=dict(color=BLUE, width=2),
        fill="tozeroy",
        fillcolor="rgba(41, 121, 255, 0.1)",
        name="Win Rate",
    ))
    
    # Target line at 50%
    fig.add_hline(y=0.5, line_dash="dash", line_color=GREY, opacity=0.5, annotation_text="Target: 50%")
    
    fig.update_layout(
        title=f"Learning Progress (Rolling {window}-Trade Win Rate)",
        xaxis_title="Trade Number",
        yaxis_title="Win Rate",
        yaxis_tickformat=".0%",
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        height=300,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    
    return fig
