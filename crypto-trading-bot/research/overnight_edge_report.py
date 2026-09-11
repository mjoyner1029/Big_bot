from __future__ import annotations

from typing import Any, Dict, List


def build_overnight_edge_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a compact report summary from ranked overnight-edge candidates."""
    if not results:
        return {"total_candidates": 0, "top_candidates": []}

    ranked = sorted(results, key=lambda row: row.get("score", 0.0), reverse=True)
    top = []
    for row in ranked[:10]:
        metrics = row.get("metrics", {})
        top.append({
            "ticker": row.get("ticker"),
            "score": row.get("score"),
            "recommendation": row.get("recommendation"),
            "edge_type": row.get("edge_type"),
            "net_overnight_cagr": metrics.get("overnight_cagr", 0.0),
            "sharpe": metrics.get("overnight_sharpe", 0.0),
            "drawdown": metrics.get("overnight_max_drawdown", 0.0),
            "profit_factor": metrics.get("overnight_profit_factor", 0.0),
            "sample_size": metrics.get("sample_size", 0),
        })

    return {
        "total_candidates": len(ranked),
        "top_candidates": top,
        "best_score": ranked[0].get("score", 0.0),
        "best_ticker": ranked[0].get("ticker"),
    }
