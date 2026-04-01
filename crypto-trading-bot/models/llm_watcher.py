"""LLM Market Watcher — continuous background monitoring with Claude.

Runs on a configurable interval (default 15 min) and asks Claude to:
  1. Analyse current market conditions across the watchlist
  2. Identify emerging trends / momentum shifts
  3. Flag risk events (e.g., unusually high volatility, correlated selloffs)
  4. Score each watchlist symbol on a -1 (very bearish) to +1 (very bullish) scale

Results are stored in a thread-safe buffer so the strategy engine can query
the watcher's latest opinion for a symbol without blocking.

Usage:
    from models.llm_watcher import LLMWatcher
    watcher = LLMWatcher()
    watcher.start()
    ...
    opinion = watcher.get_opinion("BTC-USD")
    # {'bias': 'bullish', 'score': 0.65, 'reasoning': '...', 'updated': '...'}
"""
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config.config import CONFIG, get_all_symbols, is_crypto
from models.news_fetcher import fetch_all_headlines


class LLMWatcher:
    """Background thread that continuously analyses markets with Claude."""

    def __init__(self, interval_seconds: int = 900):
        self.interval = interval_seconds
        self._opinions: Dict[str, Dict[str, Any]] = {}
        self._market_summary: Optional[str] = None
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._running.set()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="llm-watcher",
        )
        self._thread.start()
        logging.info(f"[LLM Watcher] Started (interval={self.interval}s)")

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=10)
        logging.info("[LLM Watcher] Stopped")

    def get_opinion(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest LLM opinion for a symbol."""
        with self._lock:
            return self._opinions.get(symbol)

    def get_all_opinions(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._opinions)

    def get_market_summary(self) -> Optional[str]:
        with self._lock:
            return self._market_summary

    # ── Background loop ───────────────────────────────────────────

    def _run_loop(self) -> None:
        while self._running.is_set():
            try:
                self._analyse_markets()
            except Exception as e:
                logging.error(f"[LLM Watcher] Analysis cycle failed: {e}", exc_info=True)
            # Sleep in small increments so stop() is responsive
            for _ in range(self.interval):
                if not self._running.is_set():
                    return
                time.sleep(1)

    def _analyse_markets(self) -> None:
        """Run one full analysis cycle."""
        from models.llm_analyst import _call_claude

        # Check that LLM is configured
        if not CONFIG.get("anthropic_api_key"):
            logging.debug("[LLM Watcher] No API key — skipping cycle")
            return

        symbols = get_all_symbols()
        logging.info(f"[LLM Watcher] Analysing {len(symbols)} symbols...")

        # 1. Fetch headlines for market context
        crypto_headlines = fetch_all_headlines(is_crypto=True, max_total=50)
        stock_headlines = fetch_all_headlines(is_crypto=False, max_total=50)

        # 2. Fetch price context (from WebSocket feed or yfinance cache)
        price_context = self._get_price_context(symbols)

        # 3. Build the mega-prompt
        system = (
            "You are an elite financial market analyst monitoring real-time markets. "
            "You have access to current prices and recent news headlines. "
            "Your job is to:\n"
            "1. Provide a brief overall market summary (3-4 sentences)\n"
            "2. For each symbol, provide a JSON object with: symbol, bias "
            "(bullish/bearish/neutral), score (-1.0 to +1.0), and a one-sentence reasoning.\n\n"
            "Return ONLY valid JSON with two keys:\n"
            '  "market_summary": "string",\n'
            '  "symbols": [{"symbol": "...", "bias": "...", "score": 0.0, "reasoning": "..."}]\n'
        )

        user_parts = []
        if crypto_headlines:
            user_parts.append("=== CRYPTO NEWS ===")
            user_parts.extend(f"- {h}" for h in crypto_headlines[:30])
        if stock_headlines:
            user_parts.append("\n=== STOCK NEWS ===")
            user_parts.extend(f"- {h}" for h in stock_headlines[:30])
        if price_context:
            user_parts.append("\n=== CURRENT PRICES ===")
            user_parts.append(json.dumps(price_context, indent=2))

        user_parts.append(f"\n=== SYMBOLS TO ANALYSE ({len(symbols)}) ===")
        user_parts.append(", ".join(symbols))

        user_prompt = "\n".join(user_parts)

        # 4. Call Claude
        raw = _call_claude(system, user_prompt, max_tokens=4096, temperature=0.3)
        if raw is None:
            logging.warning("[LLM Watcher] Claude call returned None")
            return

        # 5. Parse response
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown fences
            import re
            match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
            if match:
                result = json.loads(match.group(1))
            else:
                logging.warning(f"[LLM Watcher] Could not parse response:\n{raw[:500]}")
                return

        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            self._market_summary = result.get("market_summary", "")
            for entry in result.get("symbols", []):
                sym = entry.get("symbol", "")
                if sym:
                    self._opinions[sym] = {
                        "bias": entry.get("bias", "neutral"),
                        "score": float(entry.get("score", 0)),
                        "reasoning": entry.get("reasoning", ""),
                        "updated": now,
                    }

        n_opinions = len(result.get("symbols", []))
        logging.info(
            f"[LLM Watcher] Analysis complete: {n_opinions} symbols scored  "
            f"summary: {self._market_summary[:100]}..."
        )

    def _get_price_context(self, symbols: List[str]) -> Dict[str, Any]:
        """Build a price context dict from available sources."""
        prices = {}

        # Try WebSocket feed first
        try:
            from data.websocket_feed import get_feed_manager
            feed = get_feed_manager()
            ws_prices = feed.get_latest_prices()
            prices.update(ws_prices)
        except Exception:
            pass

        # Fallback: quick yfinance snapshot for missing symbols
        missing = [s for s in symbols if s not in prices]
        if missing:
            try:
                import yfinance as yf
                for sym in missing[:20]:  # limit to avoid timeout
                    try:
                        t = yf.Ticker(sym)
                        p = getattr(t.fast_info, "last_price", None)
                        if p and p > 0:
                            prices[sym] = round(float(p), 2)
                    except Exception:
                        pass
            except Exception:
                pass

        return prices


# ── Module-level singleton ────────────────────────────────────────

_watcher: Optional[LLMWatcher] = None


def get_watcher() -> LLMWatcher:
    """Return the global LLM watcher singleton."""
    global _watcher
    if _watcher is None:
        interval = CONFIG.get("llm_watcher_interval", 900)
        _watcher = LLMWatcher(interval_seconds=interval)
    return _watcher
