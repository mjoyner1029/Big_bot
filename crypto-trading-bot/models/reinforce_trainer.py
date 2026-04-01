"""
Reinforcement-learning trading agent.

Uses a simple tabular Q-learning approach that is practical to train on
historical OHLCV data without heavy deep-learning dependencies.

State representation — discretised bins of:
    RSI (5 bins), MACD-hist sign (+/-/~0), price-to-EMA20 (5 bins)

Actions:
    0 = hold, 1 = buy, 2 = sell

Reward:
    Realised % return when a position is closed; small negative for holding
    to discourage inaction.
"""
import logging
import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from indicators.ta_indicators import add_ta_indicators
from config.config import CONFIG


# ── Discretisation helpers ────────────────────────────────────────

RSI_BINS = [0, 20, 40, 60, 80, 100]
EMA_BINS = [-np.inf, -0.03, -0.01, 0.01, 0.03, np.inf]

def _discretise(value: float, bins: list) -> int:
    """Return the bin index for a continuous value."""
    for i in range(len(bins) - 1):
        if value <= bins[i + 1]:
            return i
    return len(bins) - 2


def _macd_sign(val: float, close: float = 1.0) -> int:
    """Classify MACD histogram as bullish/neutral/bearish.

    Normalises by close price so the same thresholds work for
    assets at any price level (BTC ~$60k vs SOL ~$20).
    """
    if close <= 0:
        close = 1.0
    normalised = val / close
    if normalised > 0.001:      # >0.1% of price
        return 2
    elif normalised < -0.001:
        return 0
    return 1


def _state_from_row(row: pd.Series) -> Optional[Tuple[int, int, int]]:
    """Build a discretised state tuple from indicator values.

    Returns None if any required indicator is missing.
    """
    rsi = row.get("rsi")
    macd_hist = row.get("macd_hist")
    close_to_ema = row.get("close_to_ema20")
    close = row.get("Close", 1.0)

    if rsi is None or pd.isna(rsi):
        return None
    if macd_hist is None or pd.isna(macd_hist):
        return None
    if close_to_ema is None or pd.isna(close_to_ema):
        return None

    rsi_bin = _discretise(float(rsi), RSI_BINS)
    macd_s = _macd_sign(float(macd_hist), float(close))
    ema_bin = _discretise(float(close_to_ema), EMA_BINS)
    return (rsi_bin, macd_s, ema_bin)


# ── Q-learning agent ─────────────────────────────────────────────

class QLearningTrader:
    """Tabular Q-learning trading agent."""

    ACTIONS = [0, 1, 2]  # hold, buy, sell

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table: dict = {}  # state -> [Q(hold), Q(buy), Q(sell)]

    def _get_q(self, state: tuple) -> list:
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]
        return self.q_table[state]

    def choose_action(self, state: tuple) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.choice(self.ACTIONS))
        q_vals = self._get_q(state)
        return int(np.argmax(q_vals))

    def update(self, state: tuple, action: int, reward: float, next_state: tuple) -> None:
        q = self._get_q(state)
        q_next = self._get_q(next_state)
        q[action] += self.alpha * (reward + self.gamma * max(q_next) - q[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ── Training loop ─────────────────────────────────────────────────

def train_reinforcement_model(
    df: pd.DataFrame,
    episodes: int = 50,
    save: bool = True,
    symbol: Optional[str] = None,
) -> QLearningTrader:
    """
    Train the Q-learning trading agent on historical OHLCV data.

    Args:
        df:       Raw OHLCV DataFrame.
        episodes: Number of full passes over the data.
        save:     Persist to disk when done.
    Returns:
        Trained QLearningTrader.
    """
    df = add_ta_indicators(df).dropna().reset_index(drop=True)
    agent = QLearningTrader()

    for ep in range(episodes):
        position: Optional[float] = None  # entry price when long
        total_reward = 0.0

        for i in range(len(df) - 1):
            state = _state_from_row(df.iloc[i])
            if state is None:
                continue  # skip bars with missing indicators
            next_state = _state_from_row(df.iloc[i + 1])
            if next_state is None:
                continue
            action = agent.choose_action(state)
            current_close = df.iloc[i]["Close"]
            next_close = df.iloc[i + 1]["Close"]

            reward = 0.0

            if action == 1:  # buy
                if position is None:
                    position = current_close
                    reward = -0.0001  # small cost for entering
                else:
                    reward = -0.001  # already in position
            elif action == 2:  # sell
                if position is not None:
                    pct_return = (current_close - position) / position
                    reward = pct_return
                    position = None
                else:
                    reward = -0.001  # nothing to sell
            else:  # hold
                if position is not None:
                    reward = (next_close - current_close) / current_close * 0.1
                else:
                    reward = -0.0001  # small cost for doing nothing

            agent.update(state, action, reward, next_state)
            total_reward += reward

        agent.decay_epsilon()
        if (ep + 1) % 10 == 0:
            logging.info(
                f"[RL] Episode {ep+1}/{episodes}  ε={agent.epsilon:.4f}  "
                f"total_reward={total_reward:.4f}  Q-states={len(agent.q_table)}"
            )

    if save:
        path = _rl_path(symbol)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(agent, f)
        logging.info(f"[RL] Agent saved to {path}")

    return agent


def _rl_path(symbol: Optional[str] = None) -> str:
    base = CONFIG.get("model_dir", "models/saved")
    if symbol:
        safe = symbol.replace("/", "_").replace("-", "_")
        return os.path.join(base, f"rl_{safe}.pkl")
    return CONFIG.get("rl_model_path", "models/saved/rl_agent.pkl")


def load_rl_agent(symbol: Optional[str] = None) -> Optional[QLearningTrader]:
    """Load a previously trained RL agent from disk."""
    path = _rl_path(symbol)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        agent = pickle.load(f)
    logging.info(f"[RL] Agent loaded from {path}")
    return agent


def rl_suggest_action(df: pd.DataFrame, symbol: Optional[str] = None) -> int:
    """
    Use the trained RL agent to suggest an action for the latest bar.

    Returns:
        0 (hold), 1 (buy), or 2 (sell).
    """
    agent = load_rl_agent(symbol=symbol)
    if agent is None:
        logging.warning(f"[RL] No trained agent for {symbol or 'default'} — defaulting to hold")
        return 0

    df_ind = add_ta_indicators(df).dropna()
    if df_ind.empty:
        logging.warning("[RL] No valid indicator data — defaulting to hold")
        return 0

    state = _state_from_row(df_ind.iloc[-1])
    if state is None:
        logging.warning("[RL] Required indicators missing from latest bar — defaulting to hold")
        return 0

    agent.epsilon = 0.0  # greedy at inference time
    action = agent.choose_action(state)
    labels = {0: "hold", 1: "buy", 2: "sell"}
    logging.info(f"[RL] Suggested action: {labels[action]}")
    return action
