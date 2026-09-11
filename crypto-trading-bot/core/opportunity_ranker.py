"""
Opportunity Ranker — final scoring and filtering before capital allocation.

Every candidate from all scanners passes through this ranker before the bot
decides whether to trade. The ranker is the last checkpoint before the
Portfolio Manager and Risk Engine.

Opportunity Score formula:
    score = (
        expected_return_score  * 0.25
        + confidence_score     * 0.20
        + strategy_reliability * 0.15
        + regime_score         * 0.15
        + liquidity_score      * 0.10
        + spread_score         * 0.05
        + cost_score           * 0.05
        + drawdown_penalty     * 0.05   (negative)
    )

NO_TRADE is always a valid decision. The ranker enforces a minimum score
threshold — candidates below it are rejected without execution.
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Configurable thresholds ────────────────────────────────────────────────────
MIN_OPPORTUNITY_SCORE       = 0.35   # below this → always NO_TRADE
PREFERRED_OPPORTUNITY_SCORE = 0.55   # above this → preferred candidate
MAX_CANDIDATES_PER_CYCLE    = 5      # evaluate at most this many per cycle


@dataclass
class RankedOpportunity:
    """A candidate with its final composite score."""
    symbol:             str
    asset_class:        str
    raw_score:          float           # from scanner
    ranked_score:       float           # after ranker adjustments
    expected_return:    float           # estimated net return %
    confidence:         float           # [0, 1]
    decision:           str             # 'TRADE' | 'NO_TRADE' | 'WATCH'
    reject_reason:      Optional[str]   = None
    score_breakdown:    Dict[str, float] = field(default_factory=dict)


class OpportunityRanker:
    """
    Ranks and filters candidates from all scanners.

    Takes into account:
        - Expected return (from strategy signals + Kronos + ML meta model)
        - Confidence (from weighted vote + individual signal confidences)
        - Strategy reliability (historical win rate × expectancy)
        - Market regime suitability
        - Liquidity (volume, spread)
        - Transaction costs
        - Current portfolio drawdown state
    """

    def __init__(
        self,
        min_score: float = MIN_OPPORTUNITY_SCORE,
        preferred_score: float = PREFERRED_OPPORTUNITY_SCORE,
    ):
        self.min_score       = min_score
        self.preferred_score = preferred_score

    def rank(
        self,
        candidates: List[Dict],         # list of raw candidate dicts
        regime: str = 'unknown',
        drawdown_pct: float = 0.0,
        strategy_stats: Optional[Dict[str, Dict]] = None,
    ) -> List[RankedOpportunity]:
        """
        Score and sort candidates. Returns RankedOpportunity list,
        best first.

        Args:
            candidates:      Raw candidate dicts (from scanner or analyze_trade_opportunity)
            regime:          Current market regime string
            drawdown_pct:    Current portfolio drawdown (0.0 → 1.0)
            strategy_stats:  Per-strategy win_rate / expectancy from kelly_wrapper
        """
        ranked = []
        for cand in candidates:
            ranked_opp = self._score(cand, regime, drawdown_pct, strategy_stats or {})
            ranked.append(ranked_opp)

        # Sort by ranked_score descending
        ranked.sort(key=lambda r: r.ranked_score, reverse=True)

        # Log top candidates
        for i, r in enumerate(ranked[:5]):
            logger.info(
                f"Ranker #{i+1}: {r.symbol} score={r.ranked_score:.3f} "
                f"decision={r.decision} "
                f"({r.reject_reason or 'approved'})"
            )

        return ranked

    def top_trades(
        self,
        candidates: List[Dict],
        regime: str = 'unknown',
        drawdown_pct: float = 0.0,
        strategy_stats: Optional[Dict] = None,
        top_n: int = MAX_CANDIDATES_PER_CYCLE,
    ) -> List[RankedOpportunity]:
        """Return only TRADE-decision candidates, best first, up to top_n."""
        ranked = self.rank(candidates, regime, drawdown_pct, strategy_stats)
        trades = [r for r in ranked if r.decision == 'TRADE']
        return trades[:top_n]

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score(
        self,
        cand: Dict,
        regime: str,
        drawdown_pct: float,
        strategy_stats: Dict,
    ) -> RankedOpportunity:
        """Compute composite opportunity score for one candidate."""
        symbol      = cand.get('symbol', '')
        asset_class = cand.get('asset_class', 'crypto')

        # ── Component scores ──────────────────────────────────────────────────

        # 1. Expected return (from momentum + Kronos confidence)
        momentum     = cand.get('momentum_score', 0.0)
        kronos_conf  = cand.get('kronos_confidence', 0.0)
        expected_ret = (abs(momentum) * 0.6 + kronos_conf * 0.4)
        expected_ret = max(0.0, min(1.0, expected_ret))

        # 2. Signal confidence (from strategy voting)
        signal_conf  = cand.get('signal_confidence', cand.get('confidence', 50.0))
        if isinstance(signal_conf, (int, float)):
            conf_score = signal_conf / 100.0 if signal_conf > 1.0 else signal_conf
        else:
            conf_score = 0.5

        # 3. Strategy reliability (win_rate × expectancy from historical stats)
        strat_name = cand.get('strategy', '')
        stats      = strategy_stats.get(strat_name, {})
        if stats and stats.get('trades', 0) >= 10:
            reliability = min(
                stats.get('win_rate', 0.5) * max(stats.get('expectancy', 0.0), 0.0) * 10,
                1.0
            )
        else:
            reliability = 0.4   # neutral for unknown strategies

        # 4. Regime score — how well does the signal fit the current regime?
        regime_score = self._regime_fit(cand, regime)

        # 5. Liquidity score
        volume_usd   = cand.get('volume_usd_24h', cand.get('volume_24h', 0.0))
        liq_score    = min(math.log10(max(volume_usd, 1)) / 9.0, 1.0)  # log10(1B) = 9

        # 6. Range score — uses REAL quote spread when supplied, else the
        #    intrabar range proxy (bar_range_pct). Never treats bar range as
        #    bid/ask spread.
        spread_pct = cand.get('spread_pct')          # quote-based only
        range_proxy = cand.get('bar_range_pct')
        liquidity_risk = spread_pct if spread_pct is not None else (
            range_proxy if range_proxy is not None else 0.001)
        spread_score = max(0.0, 1.0 - liquidity_risk * 100)

        # 7. Cost score (ATR relative to minimum profitable move)
        atr_pct      = cand.get('atr_pct', 0.02)
        cost_pct     = liquidity_risk * 2    # round-trip cost estimate
        cost_score   = max(0.0, 1.0 - cost_pct / max(atr_pct, 0.001))

        # 8. Drawdown penalty (reduce conviction when portfolio is hurting)
        drawdown_penalty = min(drawdown_pct * 2.0, 0.4)

        # Opportunity score composite
        breakdown = {
            'expected_return':    round(expected_ret, 3),
            'signal_confidence':  round(conf_score, 3),
            'strategy_reliability': round(reliability, 3),
            'regime_score':       round(regime_score, 3),
            'liquidity_score':    round(liq_score, 3),
            'spread_score':       round(spread_score, 3),
            'cost_score':         round(cost_score, 3),
            'drawdown_penalty':   round(-drawdown_penalty, 3),
        }

        score = (
            expected_ret    * 0.25
            + conf_score    * 0.20
            + reliability   * 0.15
            + regime_score  * 0.15
            + liq_score     * 0.10
            + spread_score  * 0.05
            + cost_score    * 0.05
            - drawdown_penalty * 0.10
        )
        score = round(max(0.0, min(1.0, score)), 4)

        # ── Decision ─────────────────────────────────────────────────────────
        if score < self.min_score:
            decision = 'NO_TRADE'
            reason   = f"Score {score:.3f} below minimum {self.min_score:.3f}"
        elif score >= self.preferred_score:
            decision = 'TRADE'
            reason   = None
        else:
            # In the middle band: WATCH (don't trade but continue monitoring)
            decision = 'WATCH'
            reason   = f"Score {score:.3f} in watch band ({self.min_score:.2f}–{self.preferred_score:.2f})"

        return RankedOpportunity(
            symbol=symbol,
            asset_class=asset_class,
            raw_score=cand.get('opportunity_score', score),
            ranked_score=score,
            expected_return=expected_ret,
            confidence=conf_score,
            decision=decision,
            reject_reason=reason,
            score_breakdown=breakdown,
        )

    def _regime_fit(self, cand: Dict, regime: str) -> float:
        """
        Score how well the candidate's signal fits the current market regime.

        Returns [0, 1] — 1.0 = perfect fit.
        """
        momentum   = cand.get('momentum_score', 0.0)
        signal_dir = 'bullish' if momentum > 0.1 else ('bearish' if momentum < -0.1 else 'neutral')

        regime_l = regime.lower()

        if 'bull' in regime_l:
            return 1.0 if signal_dir == 'bullish' else (0.4 if signal_dir == 'neutral' else 0.1)
        elif 'bear' in regime_l:
            return 1.0 if signal_dir == 'bearish' else (0.4 if signal_dir == 'neutral' else 0.1)
        elif 'sideways' in regime_l or 'range' in regime_l:
            return 0.7 if signal_dir == 'neutral' else 0.5
        else:
            return 0.5   # unknown regime → neutral score
