"""Real-time WebSocket price feeds for crypto and stocks.

Supports:
  • Crypto – Coinbase Advanced Trade WebSocket (free, no auth for market data)
  • Stocks – Alpaca real-time WebSocket (requires API keys, free tier available)

Architecture:
  Each feed runs in a background thread and pushes price updates into an
  asyncio-safe queue.  The main loop (or dashboard) can consume updates
  via ``get_latest_prices()`` or register callbacks.

Usage:
    from data.websocket_feed import PriceFeedManager
    feed = PriceFeedManager()
    feed.start(crypto_symbols=["BTC-USD"], stock_symbols=["AAPL"])
    ...
    prices = feed.get_latest_prices()   # {"BTC-USD": 68441.5, "AAPL": 225.3}
    feed.stop()
"""
import json
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set

from config.config import CONFIG

# Type alias for price callbacks: fn(symbol, price, timestamp)
PriceCallback = Callable[[str, float, datetime], None]


class PriceFeedManager:
    """Manage real-time WebSocket connections for crypto and stock price feeds."""

    def __init__(self):
        self._prices: Dict[str, float] = {}          # sym -> last price
        self._timestamps: Dict[str, datetime] = {}    # sym -> last update time
        self._callbacks: List[PriceCallback] = []
        self._threads: List[threading.Thread] = []
        self._running = threading.Event()
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────

    def register_callback(self, fn: PriceCallback) -> None:
        """Register a function to be called on every price tick."""
        self._callbacks.append(fn)

    def get_latest_prices(self) -> Dict[str, float]:
        """Return a snapshot of the most recent prices."""
        with self._lock:
            return dict(self._prices)

    def get_price(self, symbol: str) -> Optional[float]:
        """Return the latest price for a single symbol."""
        with self._lock:
            return self._prices.get(symbol)

    def get_price_age_seconds(self, symbol: str) -> Optional[float]:
        """Return seconds since the last price update for *symbol*."""
        with self._lock:
            ts = self._timestamps.get(symbol)
            if ts is None:
                return None
            return (datetime.now(timezone.utc) - ts).total_seconds()

    def start(
        self,
        crypto_symbols: Optional[List[str]] = None,
        stock_symbols: Optional[List[str]] = None,
    ) -> None:
        """Start background WebSocket threads for the given symbols."""
        self._running.set()

        if crypto_symbols:
            t = threading.Thread(
                target=self._run_crypto_feed,
                args=(crypto_symbols,),
                daemon=True,
                name="ws-crypto",
            )
            t.start()
            self._threads.append(t)
            logging.info(f"[WS] Crypto feed started for {len(crypto_symbols)} symbols")

        if stock_symbols:
            t = threading.Thread(
                target=self._run_stock_feed,
                args=(stock_symbols,),
                daemon=True,
                name="ws-stocks",
            )
            t.start()
            self._threads.append(t)
            logging.info(f"[WS] Stock feed started for {len(stock_symbols)} symbols")

    def stop(self) -> None:
        """Signal all feed threads to stop."""
        self._running.clear()
        for t in self._threads:
            t.join(timeout=5)
        self._threads.clear()
        logging.info("[WS] All feeds stopped")

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    # ── Internal: push a price update ─────────────────────────────

    def _on_price(self, symbol: str, price: float, ts: Optional[datetime] = None) -> None:
        ts = ts or datetime.now(timezone.utc)
        with self._lock:
            self._prices[symbol] = price
            self._timestamps[symbol] = ts

        for cb in self._callbacks:
            try:
                cb(symbol, price, ts)
            except Exception as e:
                logging.warning(f"[WS] Callback error: {e}")

    # ── Crypto: Coinbase Advanced Trade WebSocket ─────────────────

    def _run_crypto_feed(self, symbols: List[str]) -> None:
        """Connect to Coinbase WebSocket and stream ticker data.

        Uses the public (unauthenticated) ``ticker`` channel which
        provides real-time last-trade prices.
        """
        try:
            import websocket  # websocket-client library
        except ImportError:
            logging.error("[WS] 'websocket-client' not installed — crypto feed unavailable")
            self._crypto_polling_fallback(symbols)
            return

        # Convert yfinance format (BTC-USD) to Coinbase format (BTC-USD is already correct)
        product_ids = [s.replace("-USD", "-USD") for s in symbols]  # keep as-is

        url = "wss://ws-feed.exchange.coinbase.com"
        subscribe_msg = json.dumps({
            "type": "subscribe",
            "product_ids": product_ids,
            "channels": ["ticker"],
        })

        reconnect_delay = 1

        while self._running.is_set():
            ws = None
            try:
                ws = websocket.create_connection(url, timeout=15)
                ws.send(subscribe_msg)
                logging.info(f"[WS Crypto] Connected to Coinbase for {len(product_ids)} products")
                reconnect_delay = 1  # reset on successful connect

                while self._running.is_set():
                    raw = ws.recv()
                    if not raw:
                        continue
                    data = json.loads(raw)
                    if data.get("type") == "ticker":
                        sym = data.get("product_id", "").replace("-", "-")  # already BTC-USD
                        price = float(data.get("price", 0))
                        ts_str = data.get("time", "")
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        except Exception:
                            ts = datetime.now(timezone.utc)
                        if price > 0:
                            self._on_price(sym, price, ts)

            except Exception as e:
                if self._running.is_set():
                    logging.warning(f"[WS Crypto] Connection error: {e} — reconnecting in {reconnect_delay}s")
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 60)
            finally:
                if ws:
                    try:
                        ws.close()
                    except Exception:
                        pass

    def _crypto_polling_fallback(self, symbols: List[str]) -> None:
        """Poll prices via REST if WebSocket library unavailable."""
        import requests
        logging.info("[WS Crypto] Falling back to REST polling (1s interval)")
        while self._running.is_set():
            for sym in symbols:
                try:
                    product = sym  # Coinbase uses BTC-USD
                    r = requests.get(
                        f"https://api.exchange.coinbase.com/products/{product}/ticker",
                        timeout=5,
                    )
                    if r.ok:
                        price = float(r.json().get("price", 0))
                        if price > 0:
                            self._on_price(sym, price)
                except Exception:
                    pass
            time.sleep(1)

    # ── Stocks: Alpaca WebSocket ──────────────────────────────────

    def _run_stock_feed(self, symbols: List[str]) -> None:
        """Connect to Alpaca's real-time stock WebSocket.

        Requires ALPACA_API_KEY and ALPACA_API_SECRET.
        Falls back to yfinance polling if keys are missing.
        """
        api_key = CONFIG.get("alpaca_api_key", "")
        api_secret = CONFIG.get("alpaca_api_secret", "")

        if not api_key or not api_secret:
            logging.warning("[WS Stocks] No Alpaca API keys — falling back to polling")
            self._stock_polling_fallback(symbols)
            return

        try:
            import websocket
        except ImportError:
            logging.error("[WS Stocks] 'websocket-client' not installed — stock feed unavailable")
            self._stock_polling_fallback(symbols)
            return

        # Alpaca uses iex or sip feed
        url = "wss://stream.data.alpaca.markets/v2/iex"

        auth_msg = json.dumps({
            "action": "auth",
            "key": api_key,
            "secret": api_secret,
        })
        subscribe_msg = json.dumps({
            "action": "subscribe",
            "trades": symbols,
        })

        reconnect_delay = 1

        while self._running.is_set():
            ws = None
            try:
                ws = websocket.create_connection(url, timeout=15)
                ws.send(auth_msg)
                auth_resp = json.loads(ws.recv())
                logging.info(f"[WS Stocks] Auth response: {auth_resp}")

                ws.send(subscribe_msg)
                logging.info(f"[WS Stocks] Subscribed to {len(symbols)} stocks")
                reconnect_delay = 1

                while self._running.is_set():
                    raw = ws.recv()
                    if not raw:
                        continue
                    messages = json.loads(raw)
                    if not isinstance(messages, list):
                        messages = [messages]
                    for msg in messages:
                        if msg.get("T") == "t":  # trade
                            sym = msg.get("S", "")
                            price = float(msg.get("p", 0))
                            ts_str = msg.get("t", "")
                            try:
                                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            except Exception:
                                ts = datetime.now(timezone.utc)
                            if price > 0 and sym:
                                self._on_price(sym, price, ts)

            except Exception as e:
                if self._running.is_set():
                    logging.warning(f"[WS Stocks] Connection error: {e} — reconnecting in {reconnect_delay}s")
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 60)
            finally:
                if ws:
                    try:
                        ws.close()
                    except Exception:
                        pass

    def _stock_polling_fallback(self, symbols: List[str]) -> None:
        """Poll stock prices via yfinance if WebSocket unavailable."""
        logging.info("[WS Stocks] Falling back to yfinance polling (5s interval)")
        while self._running.is_set():
            try:
                import yfinance as yf
                for sym in symbols:
                    try:
                        ticker = yf.Ticker(sym)
                        info = ticker.fast_info
                        price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
                        if price and price > 0:
                            self._on_price(sym, float(price))
                    except Exception:
                        pass
            except Exception:
                pass
            time.sleep(5)


# ── Module-level singleton ────────────────────────────────────────

_feed_manager: Optional[PriceFeedManager] = None


def get_feed_manager() -> PriceFeedManager:
    """Return the global feed manager singleton (creates on first call)."""
    global _feed_manager
    if _feed_manager is None:
        _feed_manager = PriceFeedManager()
    return _feed_manager
