"""
Prediction Market Engine — probabilistic event trading.

ARCHITECTURE:
    Prediction markets are NOT treated like price trading.
    They are binary/multi-outcome markets on real-world events.

    The engine:
        1. Discovers active markets (via PredictionMarketScanner)
        2. Claude estimates the true probability of each outcome
        3. Calculates expected edge vs market probability
        4. Calculates expected value of a position
        5. Sizes bets using Kelly criterion
        6. Only bets when edge is statistically significant

Expected value formula (binary market):
    EV = (model_prob × win_payout) - (1 - model_prob) × stake
    where win_payout ≈ (1 / market_price) for YES positions

Kelly fraction:
    f* = (p * b - q) / b
    where p = model_prob, q = 1-p, b = net odds

Usage:
    engine = PredictionMarketEngine(capital=10_000, llm=llm_orchestrator)
    opportunities = engine.scan_and_analyze()
    for opp in opportunities:
        if opp.should_bet:
            size = opp.kelly_size_dollars
"""
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_utcnow = lambda: datetime.now(timezone.utc)

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_EDGE            = 0.05    # minimum edge (5%) to consider betting
MIN_EV              = 0.02    # minimum expected value per dollar staked (2%)
MAX_KELLY_FRACTION  = 0.05    # never bet more than 5% of capital on one market
MIN_LIQUIDITY_USD   = 5_000   # minimum market liquidity to participate
KELLY_SAFETY_FACTOR = 0.25    # use 25% of full Kelly (fractional Kelly)


@dataclass
class MarketAnalysis:
    """Claude + model analysis of a single prediction market."""
    market_id:      str
    question:       str
    end_date:       str
    market_prob:    float       # current market price for YES (≈ probability)
    model_prob:     float       # our estimated true probability
    edge:           float       # |model_prob - market_prob|
    ev:             float       # expected value per dollar staked
    kelly_fraction: float       # optimal Kelly fraction
    kelly_size_dollars: float   # actual dollar bet size
    should_bet:     bool
    reject_reason:  Optional[str] = None
    direction:      str = 'YES'  # 'YES' or 'NO'
    llm_reasoning:  str = ''
    confidence:     str = 'LOW'  # 'LOW' | 'MEDIUM' | 'HIGH'
    liquidity_usd:  float = 0.0
    time_to_close_days: float = 0.0


class PredictionMarketEngine:
    """
    Analyzes prediction markets and sizes bets using Kelly criterion.

    Claude is used to form the model probability estimate.
    The engine never bets without a statistically significant edge.
    """

    def __init__(
        self,
        capital: float,
        llm=None,                        # LLMOrchestrator instance
        max_market_exposure_pct: float = 0.10,  # max 10% of capital in prediction markets
    ):
        self.capital                 = capital
        self.llm                     = llm
        self.max_market_exposure_pct = max_market_exposure_pct
        self._current_exposure: float = 0.0   # total dollars currently in prediction markets

        logger.info(
            f"PredictionMarketEngine ready: capital=${capital:,.0f} "
            f"max_exposure={max_market_exposure_pct:.0%}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def scan_and_analyze(self, markets: Optional[List[Dict]] = None) -> List[MarketAnalysis]:
        """
        Scan prediction markets and analyze each one.

        Args:
            markets: Optional list of market dicts from PredictionMarketScanner.
                     If None, fetches fresh markets.

        Returns list of MarketAnalysis objects sorted by expected value.
        """
        if markets is None:
            from core.scanners import PredictionMarketScanner
            scanner = PredictionMarketScanner(min_volume_usd=MIN_LIQUIDITY_USD)
            raw     = scanner.scan(top_n=20)
            markets = [{'conditionId': c.meta.get('market_id', ''),
                        'question': c.symbol,
                        'outcomePrices': [c.price],
                        'volume': c.volume_usd_24h,
                        'endDate': c.meta.get('end_date', ''),
                        'liquidityClob': c.volume_usd_24h * 0.1}
                       for c in raw]

        analyses = []
        for market in markets:
            analysis = self._analyze_market(market)
            if analysis is not None:
                analyses.append(analysis)

        # Sort by EV descending
        analyses.sort(key=lambda a: a.ev, reverse=True)
        return analyses

    def get_bettable_opportunities(
        self, markets: Optional[List[Dict]] = None
    ) -> List[MarketAnalysis]:
        """Return only markets where should_bet is True, sorted by EV."""
        all_analyses = self.scan_and_analyze(markets)
        return [a for a in all_analyses if a.should_bet]

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _analyze_market(self, market: Dict) -> Optional[MarketAnalysis]:
        """Analyze a single market. Returns None if insufficient data."""
        try:
            market_id  = market.get('conditionId', '')
            question   = market.get('question', '')
            end_date   = market.get('endDate', '')
            prices     = market.get('outcomePrices', [])
            volume     = float(market.get('volume', 0))
            liquidity  = float(market.get('liquidityClob', 0))

            if not question or not prices:
                return None

            market_prob = float(prices[0]) if prices else 0.5
            if not (0.02 < market_prob < 0.98):
                return None   # too extreme — market already very confident

            if liquidity < MIN_LIQUIDITY_USD:
                return None

            # Time to close
            days_to_close = self._days_until(end_date)
            if days_to_close is not None and days_to_close < 1:
                return None   # market closing too soon

            # Get model probability via Claude (or use market prob as placeholder)
            model_prob, reasoning, confidence = self._get_model_probability(
                question, market_prob, end_date
            )

            # Determine direction: bet YES if model_prob > market_prob, NO otherwise
            if model_prob > market_prob:
                direction     = 'YES'
                stake_price   = market_prob         # cost to buy YES = market_prob
                win_payout    = 1.0 - stake_price   # net win per dollar staked on YES
            else:
                direction     = 'NO'
                stake_price   = 1.0 - market_prob   # cost to buy NO
                win_payout    = 1.0 - stake_price

            p = model_prob if direction == 'YES' else (1.0 - model_prob)
            q = 1.0 - p
            b = win_payout / stake_price if stake_price > 0 else 1.0

            edge = abs(model_prob - market_prob)
            ev   = p * b - q   # Kelly EV formula

            kelly = max((p * b - q) / b, 0.0) * KELLY_SAFETY_FACTOR
            kelly = min(kelly, MAX_KELLY_FRACTION)

            # Scale by maximum market exposure
            max_bet    = self.capital * self.max_market_exposure_pct
            remaining  = max_bet - self._current_exposure
            kelly_size = min(kelly * self.capital, remaining, max_bet * 0.3)

            should_bet = (
                edge >= MIN_EDGE
                and ev >= MIN_EV
                and kelly_size >= 10.0
                and confidence in ('MEDIUM', 'HIGH')
            )

            reject_reason = None
            if not should_bet:
                if edge < MIN_EDGE:
                    reject_reason = f"Edge {edge:.1%} below minimum {MIN_EDGE:.0%}"
                elif ev < MIN_EV:
                    reject_reason = f"EV {ev:.1%} below minimum {MIN_EV:.0%}"
                elif confidence not in ('MEDIUM', 'HIGH'):
                    reject_reason = f"Low confidence in probability estimate"
                else:
                    reject_reason = f"Kelly size ${kelly_size:.2f} too small"

            return MarketAnalysis(
                market_id=market_id,
                question=question,
                end_date=end_date,
                market_prob=round(market_prob, 4),
                model_prob=round(model_prob, 4),
                edge=round(edge, 4),
                ev=round(ev, 4),
                kelly_fraction=round(kelly, 4),
                kelly_size_dollars=round(kelly_size, 2),
                should_bet=should_bet,
                reject_reason=reject_reason,
                direction=direction,
                llm_reasoning=reasoning,
                confidence=confidence,
                liquidity_usd=liquidity,
                time_to_close_days=days_to_close or 30.0,
            )

        except Exception as e:
            logger.warning(f"PredictionMarketEngine: analysis error for {market.get('question','?')}: {e}")
            return None

    def _get_model_probability(
        self,
        question: str,
        market_prob: float,
        end_date: str,
    ) -> tuple:
        """
        Ask Claude to estimate the probability of the event.

        Returns (model_prob, reasoning, confidence_level).
        Falls back to market probability if Claude is unavailable.
        """
        if self.llm is None:
            return market_prob, "No LLM available — using market probability", "LOW"

        try:
            prompt = f"""You are analyzing a prediction market question.

Question: {question}
Market resolution date: {end_date}
Current market probability (YES): {market_prob:.1%}

Based on your knowledge of real-world events, news, and base rates:
1. What is your estimated probability that the YES outcome occurs?
2. How confident are you in this estimate? (LOW / MEDIUM / HIGH)
3. Brief reasoning (2-3 sentences max)

Respond in this exact format:
PROBABILITY: 0.XX
CONFIDENCE: MEDIUM
REASONING: [your reasoning]"""

            response = self.llm.quick_analysis(prompt)
            if not response:
                return market_prob, "LLM returned no response", "LOW"

            # Parse structured response
            lines = response.strip().split('\n')
            prob  = market_prob
            conf  = 'LOW'
            reasoning = response

            for line in lines:
                if line.startswith('PROBABILITY:'):
                    try:
                        prob = float(line.split(':')[1].strip())
                        prob = max(0.02, min(0.98, prob))
                    except ValueError:
                        pass
                elif line.startswith('CONFIDENCE:'):
                    conf = line.split(':')[1].strip().upper()
                    if conf not in ('LOW', 'MEDIUM', 'HIGH'):
                        conf = 'LOW'
                elif line.startswith('REASONING:'):
                    reasoning = line.split(':', 1)[1].strip()

            return prob, reasoning, conf

        except Exception as e:
            logger.warning(f"PredictionMarketEngine: LLM analysis error: {e}")
            return market_prob, f"LLM error: {e}", "LOW"

    @staticmethod
    def _days_until(date_str: str) -> Optional[float]:
        """Parse end date and return days until closing."""
        if not date_str:
            return None
        formats = ["%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]
        for fmt in formats:
            try:
                end_dt = datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
                now    = datetime.now(timezone.utc)
                return (end_dt - now).total_seconds() / 86400
            except ValueError:
                continue
        return None

    def on_position_closed(self, size_dollars: float) -> None:
        """Call when a prediction market position closes."""
        self._current_exposure = max(0.0, self._current_exposure - size_dollars)

    def on_position_opened(self, size_dollars: float) -> None:
        """Call when a prediction market position is opened."""
        self._current_exposure += size_dollars
