"""Portfolio manager — tracks capital, open positions, and risk limits.

State is persisted to a JSON file so the bot can restart without losing
track of its positions.
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config.config import CONFIG
from logs.trade_logger import log_trade
from strategies.thresholds import get_trade_thresholds


class PortfolioManager:
    """
    In-memory portfolio with JSON persistence.

    Tracks:
      • available cash
      • list of open positions
      • closed-trade history (last 500)
    """

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = state_path or CONFIG.get("state_path", "logs/bot_state.json")
        configured_capital: float = CONFIG["capital"]
        if configured_capital > 0:
            self.starting_capital = configured_capital
        else:
            # No capital configured — try to fetch from broker
            self.starting_capital = self._fetch_broker_capital()
        self.cash: float = self.starting_capital
        self.open_positions: List[Dict[str, Any]] = []
        self.closed_trades: List[Dict[str, Any]] = []
        self._load_state()

    # ── Broker balance lookup ─────────────────────────────────────

    @staticmethod
    def _fetch_broker_capital() -> float:
        """Query the configured broker(s) for the real account balance.

        Tries Alpaca first (stocks), then ccxt (crypto).  Returns 0 if
        neither is configured or the calls fail — the dashboard will
        display '--' in that case.
        """
        # Alpaca
        api_key = CONFIG.get("alpaca_api_key", "")
        api_secret = CONFIG.get("alpaca_api_secret", "")
        if api_key and api_secret:
            try:
                import requests
                base = CONFIG.get("alpaca_base_url", "https://paper-api.alpaca.markets")
                resp = requests.get(
                    f"{base}/v2/account",
                    headers={
                        "APCA-API-KEY-ID": api_key,
                        "APCA-API-SECRET-KEY": api_secret,
                    },
                    timeout=10,
                )
                if resp.ok:
                    balance = float(resp.json().get("equity", 0))
                    if balance > 0:
                        logging.info(f"[Portfolio] Fetched Alpaca equity: ${balance:,.2f}")
                        return balance
            except Exception as e:
                logging.warning(f"[Portfolio] Alpaca balance fetch failed: {e}")

        # ccxt crypto exchange
        ex_key = CONFIG.get("coinbase_api_key", "")
        ex_secret = CONFIG.get("coinbase_api_secret", "")
        if ex_key and ex_secret:
            try:
                import ccxt
                exchange_id = CONFIG.get("exchange", "coinbase")
                exchange_cls = getattr(ccxt, exchange_id, None)
                if exchange_cls:
                    ex = exchange_cls({
                        "apiKey": ex_key,
                        "secret": ex_secret,
                        "password": CONFIG.get("coinbase_passphrase", ""),
                    })
                    bal = ex.fetch_balance()
                    total_usd = float(bal.get("total", {}).get("USD", 0))
                    if total_usd > 0:
                        logging.info(f"[Portfolio] Fetched {exchange_id} balance: ${total_usd:,.2f}")
                        return total_usd
            except Exception as e:
                logging.warning(f"[Portfolio] ccxt balance fetch failed: {e}")

        logging.warning(
            "[Portfolio] No capital configured and broker balance unavailable. "
            "Set the TRADING_CAPITAL env var or add broker API keys."
        )
        return 0.0

    # ── Persistence ───────────────────────────────────────────────

    def _load_state(self) -> None:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r") as f:
                    state = json.load(f)
                self.cash = state.get("cash", self.starting_capital)
                self.open_positions = state.get("open_positions", [])
                self.closed_trades = state.get("closed_trades", [])
                logging.info(
                    f"[Portfolio] Loaded state: cash=${self.cash:.2f}, "
                    f"{len(self.open_positions)} open, {len(self.closed_trades)} closed"
                )
            except Exception as e:
                logging.warning(f"[Portfolio] Could not load state: {e}")

    def save_state(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump({
                "cash": self.cash,
                "open_positions": self.open_positions,
                "closed_trades": self.closed_trades[-500:],  # keep last 500
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2, default=str)

    # ── Risk gates ────────────────────────────────────────────────

    def can_open_position(self, signal: Dict[str, Any]) -> bool:
        """Check whether a new position is allowed under risk limits."""
        # Max open positions
        if len(self.open_positions) >= CONFIG["max_open_positions"]:
            logging.warning("[Portfolio] Max open positions reached")
            return False

        # Already holding this symbol?
        sym = signal["symbol"]
        if any(p["symbol"] == sym for p in self.open_positions):
            logging.warning(f"[Portfolio] Already have an open position in {sym}")
            return False

        # Enough cash?
        size = self.compute_position_size(signal)
        cost = size * signal["entry_price"]
        if cost > self.cash:
            logging.warning(f"[Portfolio] Insufficient cash (${self.cash:.2f}) for ${cost:.2f}")
            return False

        # Max position % of total equity
        equity = self.total_equity(current_prices={})
        max_cost = equity * CONFIG["max_position_pct"]
        if cost > max_cost:
            logging.warning(f"[Portfolio] Position ${cost:.2f} exceeds {CONFIG['max_position_pct']*100}% limit")
            return False

        return True

    # ── Position sizing ───────────────────────────────────────────

    def compute_position_size(self, signal: Dict[str, Any]) -> float:
        """
        Compute the quantity to buy/sell based on the risk-per-trade rule.

        Uses fixed-fractional sizing: risk_pct * equity / distance_to_stop.
        If no stop-loss is in the signal, computes one dynamically from
        the thresholds module based on confidence and asset type.
        """
        equity = self.total_equity(current_prices={})
        risk_amt = equity * CONFIG["risk_per_trade_pct"]
        entry = signal["entry_price"]

        sl = signal.get("stop_loss_price")
        if sl is None or sl == 0:
            # Derive SL from the thresholds module using real confidence/asset data
            asset_type = signal.get("asset_type", "crypto")
            confidence = signal.get("confidence", 0.5)
            side = signal.get("side", "buy")
            thresh = get_trade_thresholds(entry, confidence, side=side, asset_type=asset_type)
            sl = thresh["stop_loss_price"]
            logging.info(
                f"[Portfolio] Computed SL from thresholds: ${sl:.2f} "
                f"(conf={confidence:.2f}, {asset_type}, {side})"
            )

        distance = abs(entry - sl)
        if distance < entry * 0.001:
            # SL too close to entry — use ATR-based minimum or threshold module
            asset_type = signal.get("asset_type", "crypto")
            confidence = signal.get("confidence", 0.5)
            side = signal.get("side", "buy")
            thresh = get_trade_thresholds(entry, confidence, side=side, asset_type=asset_type)
            distance = abs(entry - thresh["stop_loss_price"])
            logging.warning(
                f"[Portfolio] SL distance too small — recalculated from thresholds: {distance:.4f}"
            )

        qty = risk_amt / distance

        # Cap by max_position_pct
        max_cost = equity * CONFIG["max_position_pct"]
        max_qty = max_cost / entry if entry else 0
        qty = min(qty, max_qty)

        return round(qty, 6)

    # ── Position recording ────────────────────────────────────────

    def record_position(self, signal: Dict[str, Any],
                        order_id: Optional[str] = None,
                        qty: Optional[float] = None) -> None:
        """Add a new open position and deduct cash."""
        qty = qty or self.compute_position_size(signal)
        cost = qty * signal["entry_price"]
        self.cash -= cost

        position = {
            "symbol": signal["symbol"],
            "side": signal["side"],
            "asset_type": signal.get("asset_type", "crypto"),
            "entry_price": signal["entry_price"],
            "qty": qty,
            "cost": round(cost, 2),
            "take_profit_price": signal.get("take_profit_price", 0),
            "stop_loss_price": signal.get("stop_loss_price", 0),
            "trailing_stop_pct": signal.get("trailing_stop_pct", 0),
            "confidence": signal.get("confidence", 0),
            "opened_at": signal.get("timestamp", datetime.now(timezone.utc).isoformat()),
            "order_id": order_id,
        }
        self.open_positions.append(position)
        self.save_state()
        logging.info(
            f"[Portfolio] Opened {signal['side']} {qty:.6f} {signal['symbol']} "
            f"@ ${signal['entry_price']:.2f}  cash_remaining=${self.cash:.2f}"
        )

    def close_position(self, position: Dict[str, Any],
                       exit_price: float, result: str) -> None:
        """Close a position, compute P&L, log it, and return cash."""
        qty = position["qty"]
        entry = position["entry_price"]
        side = position["side"]

        if side == "buy":
            pnl = (exit_price - entry) * qty
        else:
            pnl = (entry - exit_price) * qty

        self.cash += qty * exit_price  # return proceeds

        trade_record = {
            **position,
            "exit_price": exit_price,
            "pnl": round(pnl, 2),
            "result": result,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.closed_trades.append(trade_record)

        if position in self.open_positions:
            self.open_positions.remove(position)

        self.save_state()

        log_trade(
            timestamp=trade_record["closed_at"],
            symbol=position["symbol"],
            side=side,
            entry_price=entry,
            tp_price=position.get("take_profit_price", 0),
            sl_price=position.get("stop_loss_price", 0),
            confidence=position.get("confidence", 0),
            qty=qty,
            result=result,
            exit_price=exit_price,
            pnl=pnl,
        )

        logging.info(
            f"[Portfolio] Closed {side} {qty:.6f} {position['symbol']} "
            f"@ ${exit_price:.2f}  PnL=${pnl:.2f}  result={result}"
        )

    # ── Equity & stats ────────────────────────────────────────────

    def total_equity(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total equity = cash + market value of open positions.
        Logs a warning for each position without a current price.
        """
        position_value = 0.0
        for pos in self.open_positions:
            sym = pos["symbol"]
            if sym in current_prices:
                price = current_prices[sym]
            else:
                price = pos["entry_price"]
                if current_prices:  # only warn if caller provided any prices
                    logging.warning(
                        f"[Portfolio] No current price for {sym} — "
                        f"using entry price ${price:.2f} (equity may be stale)"
                    )
            position_value += pos["qty"] * price
        return self.cash + position_value

    def summary(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Return a snapshot of the portfolio state."""
        prices = current_prices or {}
        equity = self.total_equity(prices)
        total_pnl = sum(t["pnl"] for t in self.closed_trades)
        wins = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        losses = sum(1 for t in self.closed_trades if t["pnl"] <= 0)
        return {
            "cash": round(self.cash, 2),
            "equity": round(equity, 2),
            "open_positions": len(self.open_positions),
            "closed_trades": len(self.closed_trades),
            "total_realised_pnl": round(total_pnl, 2),
            "win_rate": round(wins / (wins + losses), 4) if (wins + losses) else 0,
        }

    def rebalance(self, target_allocations: Optional[Dict[str, float]] = None,
                  current_prices: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """
        Rebalance portfolio to match target allocations.

        Generates a list of order dicts (side, symbol, qty, reason) that
        the caller should execute.  Does NOT execute trades itself.

        Args:
            target_allocations: {symbol: target_pct} where pct is 0-1.
            current_prices:     {symbol: current_price} for valuation.
        Returns:
            List of order dicts to execute.
        """
        if target_allocations is None:
            logging.info("[Portfolio] No target allocations provided — skipping rebalance")
            return []

        prices = current_prices or {}
        equity = self.total_equity(prices)
        if equity <= 0:
            logging.warning("[Portfolio] Zero equity — cannot rebalance")
            return []

        orders: List[Dict[str, Any]] = []

        for sym, target_pct in target_allocations.items():
            current = next((p for p in self.open_positions if p["symbol"] == sym), None)
            cur_price = prices.get(sym, current["entry_price"] if current else 0)
            if cur_price <= 0:
                logging.warning(f"[Rebalance] No price available for {sym} — skipping")
                continue

            current_val = current["qty"] * cur_price if current else 0
            current_pct = current_val / equity
            target_val = equity * target_pct
            delta_val = target_val - current_val
            delta_pct = target_pct - current_pct

            # Only act if the difference is meaningful (>1% of equity)
            if abs(delta_pct) < 0.01:
                logging.info(f"[Rebalance] {sym}: on-target ({current_pct*100:.1f}%)")
                continue

            if delta_val > 0:
                qty = delta_val / cur_price
                orders.append({
                    "symbol": sym,
                    "side": "buy",
                    "qty": round(qty, 6),
                    "reason": f"rebalance: {current_pct*100:.1f}% -> {target_pct*100:.1f}%",
                    "current_price": cur_price,
                })
                logging.info(
                    f"[Rebalance] {sym}: BUY {qty:.6f} @ ${cur_price:.2f}  "
                    f"({current_pct*100:.1f}% -> {target_pct*100:.1f}%)"
                )
            else:
                qty = abs(delta_val) / cur_price
                # Don't sell more than we hold
                if current:
                    qty = min(qty, current["qty"])
                orders.append({
                    "symbol": sym,
                    "side": "sell",
                    "qty": round(qty, 6),
                    "reason": f"rebalance: {current_pct*100:.1f}% -> {target_pct*100:.1f}%",
                    "current_price": cur_price,
                })
                logging.info(
                    f"[Rebalance] {sym}: SELL {qty:.6f} @ ${cur_price:.2f}  "
                    f"({current_pct*100:.1f}% -> {target_pct*100:.1f}%)"
                )

        return orders
