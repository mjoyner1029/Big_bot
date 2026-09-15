"""LLM Strategy Orchestrator - AI-Powered Strategy Selection

Instead of running all strategies, use Claude to analyze:
- Current market conditions
- GEX regime
- Recent performance
- Failure patterns
- Economic context

Then DECIDE which 3-5 strategies to use and how to combine them.

This is the "brain" - like Vertus.ai's meta-optimizer.
"""
import logging
import json
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    """Use LLM to intelligently select and combine strategies."""
    
    def __init__(self, anthropic_api_key: Optional[str] = None):
        self.api_key = anthropic_api_key
        self.last_decision = None
        self.decision_history = []
        
    def analyze_and_decide(
        self,
        market_conditions: Dict,
        gex_regime: Dict,
        recent_performance: Dict,
        failure_patterns: List,
        available_strategies: List[str],
        market_data: Dict = None
    ) -> Dict:
        """Use LLM to analyze conditions and select best strategies.
        
        Returns:
            {
                'selected_strategies': ['mean_reversion_zscore', 'breakout', ...],
                'strategy_weights': {'mean_reversion_zscore': 3.0, ...},
                'reasoning': 'Bull market with high volatility...',
                'confidence': 0.85
            }
        """
        
        # Build context for LLM (include actual chart data)
        context = self._build_market_context(
            market_conditions, gex_regime, recent_performance, failure_patterns, market_data
        )
        
        # If no API key, use rule-based fallback
        if not self.api_key:
            return self._rule_based_selection(context, available_strategies)
        
        # Call Claude to decide
        try:
            decision = self._call_claude(context, available_strategies)
            self.last_decision = decision
            self.decision_history.append({
                'timestamp': datetime.now().isoformat(),
                'decision': decision
            })
            return decision
        except Exception as e:
            logger.warning(f"[LLMOrchestrator] Claude call failed: {e}, using fallback")
            return self._rule_based_selection(context, available_strategies)
    
    def _build_market_context(
        self,
        market_conditions: Dict,
        gex_regime: Dict,
        recent_performance: Dict,
        failure_patterns: List,
        market_data: Dict = None
    ) -> str:
        """Build human-readable market analysis for LLM with actual chart data."""
        
        context = f"""CURRENT MARKET CONDITIONS:

Daily Regime: {market_conditions.get('daily_regime', 'unknown')}
Daily Confidence: {market_conditions.get('regime_confidence', 0)*100:.0f}%
Trading Bias: {market_conditions.get('bias', 'balanced')}

GEX Regime: {gex_regime.get('regime', 'unknown')}
Volatility: {gex_regime.get('volatility', 'normal')}
GEX Total: ${gex_regime.get('total_gex', 0):,.0f}

BTC Price Action:
- 1h change: {market_conditions.get('btc_1h', 0)*100:+.1f}%
- 24h change: {market_conditions.get('btc_24h', 0)*100:+.1f}%
- 7d change: {market_conditions.get('btc_7d', 0)*100:+.1f}%

RECENT PERFORMANCE (last 50 trades):
"""
        
        if recent_performance and isinstance(recent_performance, dict):
            for strategy, perf in recent_performance.items():
                if isinstance(perf, dict):
                    context += f"\n{strategy}:"
                    context += f"\n  - Win rate: {perf.get('win_rate', 0)*100:.0f}%"
                    context += f"\n  - Avg P&L: ${perf.get('avg_pnl', 0):.2f}"
                    context += f"\n  - Trades: {perf.get('trade_count', 0)}"
        else:
            context += "\nNo recent performance data available (first run)"
        
        if failure_patterns:
            context += f"\n\nFAILURE PATTERNS DETECTED:"
            for pattern in failure_patterns[:5]:
                context += f"\n  - {pattern.get('description', 'Unknown')}"
        
        # Add actual chart data so Claude can see what's happening
        if market_data:
            context += self._build_chart_analysis(market_data)
        
        return context
    
    def _build_chart_analysis(self, market_data: Dict) -> str:
        """Analyze actual price charts and add to context."""
        import pandas as pd
        
        chart_context = "\n\nACTUAL CHART ANALYSIS (Top 5 symbols):"
        
        # Analyze top 5 liquid symbols (BTC, ETH, etc.)
        symbols_to_analyze = []
        for symbol in ['BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'DOGE-USD']:
            if symbol in market_data:
                symbols_to_analyze.append(symbol)
        
        # Fallback to first 5 symbols if specific ones not found
        if not symbols_to_analyze:
            symbols_to_analyze = list(market_data.keys())[:5]
        
        for symbol in symbols_to_analyze:
            df = market_data.get(symbol)
            if df is None or not isinstance(df, pd.DataFrame) or df.empty or len(df) < 20:
                continue
            
            try:
                # Get close prices
                close_col = None
                for col in ['close', 'Close', 'CLOSE']:
                    if col in df.columns:
                        close_col = col
                        break
                
                if close_col is None:
                    continue
                
                close = pd.to_numeric(df[close_col], errors='coerce').dropna()
                if len(close) < 20:
                    continue
                
                price = float(close.iloc[-1])
                
                # Calculate key indicators
                sma20 = close.rolling(20).mean().iloc[-1] if len(close) >= 20 else price
                sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else price
                
                # RSI calculation
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_now = float(rsi.iloc[-1]) if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50.0
                
                # Recent price action
                recent_prices = close.tail(10).tolist()
                pct_change_1h = ((price / close.iloc[-12]) - 1) * 100 if len(close) >= 12 else 0
                pct_change_24h = ((price / close.iloc[-288]) - 1) * 100 if len(close) >= 288 else 0
                
                # Trend determination
                if price > sma20 and price > sma50:
                    trend = "STRONG UPTREND"
                elif price > sma20:
                    trend = "UPTREND"
                elif price < sma20 and price < sma50:
                    trend = "STRONG DOWNTREND"
                elif price < sma20:
                    trend = "DOWNTREND"
                else:
                    trend = "RANGING"
                
                # RSI signal
                if rsi_now > 70:
                    rsi_signal = "OVERBOUGHT"
                elif rsi_now < 30:
                    rsi_signal = "OVERSOLD"
                elif rsi_now > 55:
                    rsi_signal = "BULLISH"
                elif rsi_now < 45:
                    rsi_signal = "BEARISH"
                else:
                    rsi_signal = "NEUTRAL"
                
                chart_context += f"\n\n{symbol}:"
                chart_context += f"\n  Price: ${price:.2f}"
                chart_context += f"\n  Trend: {trend} (SMA20: ${sma20:.2f}, SMA50: ${sma50:.2f})"
                chart_context += f"\n  RSI(14): {rsi_now:.1f} - {rsi_signal}"
                chart_context += f"\n  Recent: 1h {pct_change_1h:+.1f}%, 24h {pct_change_24h:+.1f}%"
                chart_context += f"\n  Last 10 bars: {', '.join([f'${p:.2f}' for p in recent_prices])}"
                
            except Exception as e:
                logger.debug(f"[LLMOrchestrator] Error analyzing {symbol}: {e}")
                continue
        
        return chart_context
    
    def _call_claude(self, context: str, available_strategies: List[str]) -> Dict:
        """Call Claude API to get strategy selection."""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            prompt = f"""{context}

AVAILABLE STRATEGIES:
{', '.join(available_strategies)}

TASK:
Analyze the market conditions above and select the 3-5 best strategies to use RIGHT NOW.
Consider:
1. Market regime (bull/bear/neutral)
2. GEX volatility regime
3. Recent strategy performance
4. Known failure patterns

Return JSON with:
{{
    "selected_strategies": ["strategy1", "strategy2", ...],
    "strategy_weights": {{"strategy1": 3.0, "strategy2": 2.0, ...}},
    "reasoning": "Your analysis in 2-3 sentences",
    "confidence": 0.0-1.0
}}

Select strategies that COMPLEMENT each other (e.g., mean reversion + breakout covers both ranging and trending).
Weight based on current conditions (higher weight = more allocation).
"""
            
            message = client.messages.create(
                model="claude-sonnet-4-6",  # Latest Sonnet 4 (works with user's API key)
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Parse Claude's response
            response_text = message.content[0].text
            
            # Extract JSON (Claude might wrap in markdown)
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            
            decision = json.loads(json_str)
            
            logger.info(
                f"[LLMOrchestrator] Claude selected {len(decision['selected_strategies'])} strategies: "
                f"{', '.join(decision['selected_strategies'])}"
            )
            logger.info(f"[LLMOrchestrator] Reasoning: {decision['reasoning']}")
            
            return decision
            
        except Exception as e:
            logger.error(f"[LLMOrchestrator] Claude API error: {e}")
            raise
    
    def _rule_based_selection(self, context: str, available_strategies: List[str]) -> Dict:
        """Fallback rule-based strategy selection if no LLM."""
        
        # Parse context for key indicators
        is_bull = 'bull' in context.lower() and 'bear' not in context.lower()
        is_bear = 'bear' in context.lower() and 'bull' not in context.lower()
        high_vol = 'negative' in context.lower() or 'high' in context.lower()
        
        selected = []
        weights = {}
        
        # Always include mean reversion (proven winner)
        selected.append('mean_reversion')
        weights['mean_reversion'] = 3.0
        
        if is_bull and high_vol:
            # Bull + high vol = momentum + breakouts
            selected.extend(['momentum', 'breakout', 'correlation_lag'])
            weights.update({
                'momentum': 2.5,
                'breakout': 2.0,
                'correlation_lag': 1.5
            })
            reasoning = "Bull market with high volatility - favoring momentum and breakout strategies"
            
        elif is_bear and high_vol:
            # Bear + high vol = mean reversion + stat arb
            selected.extend(['mean_reversion', 'stat_arb'])
            weights.update({
                'mean_reversion': 3.0,
                'stat_arb': 2.0
            })
            reasoning = "Bear market with high volatility - favoring mean reversion and statistical arbitrage"
            
        elif high_vol:
            # High vol, neutral = breakouts + momentum
            selected.extend(['breakout', 'momentum', 'ema_trend_follow'])
            weights.update({
                'breakout': 2.5,
                'momentum': 2.0,
                'ema_trend_follow': 1.5
            })
            reasoning = "High volatility - favoring breakouts and momentum"
            
        else:
            # Low vol / ranging = mean reversion + pairs
            selected.extend(['vwap_reversion', 'stat_arb'])
            weights.update({
                'vwap_reversion': 2.0,
                'stat_arb': 1.5
            })
            reasoning = "Low volatility / ranging - favoring mean reversion strategies"
        
        logger.info(f"[LLMOrchestrator] Rule-based selection: {', '.join(selected)}")
        logger.info(f"[LLMOrchestrator] Reasoning: {reasoning}")
        
        return {
            'selected_strategies': selected,
            'strategy_weights': weights,
            'reasoning': reasoning,
            'confidence': 0.65
        }
    
    def get_last_decision(self) -> Optional[Dict]:
        """Get the most recent strategy decision."""
        return self.last_decision

    @staticmethod
    def _trade_pnls(trades: List) -> List[float]:
        """Extract finite realized P&L values from dicts or trade objects."""
        import math

        values = []
        for trade in trades or []:
            raw = trade.get('pnl', 0.0) if isinstance(trade, dict) else getattr(trade, 'pnl', 0.0)
            try:
                pnl = float(raw or 0.0)
            except (TypeError, ValueError):
                continue
            if math.isfinite(pnl):
                values.append(pnl)
        return values

    def analyze_trade_performance(
        self,
        trades: List,
        current_strategies: Optional[List[str]] = None,
        market_data: Optional[Dict] = None,
    ) -> Dict:
        """Return a dependable review payload for the intraday learning job.

        Adaptations remain disabled here. Parameter changes must continue through
        the experiment and validation controls owned by the trading bot.
        """
        pnls = self._trade_pnls(trades)
        if not pnls:
            return {
                'trade_count': 0,
                'win_rate': 0.0,
                'average_pnl': 0.0,
                'recommendations': 'No resolved trades are available to review.',
                'should_adapt': False,
                'adaptations': {},
            }

        wins = sum(pnl > 0 for pnl in pnls)
        win_rate = wins / len(pnls)
        average_pnl = sum(pnls) / len(pnls)
        if len(pnls) < 5:
            recommendation = 'Collect more resolved paper trades before proposing changes.'
        elif win_rate < 0.4 or average_pnl < 0:
            recommendation = 'Review losing signals and execution evidence before proposing a validated experiment.'
        else:
            recommendation = 'Recent paper results are stable; continue collecting evidence.'

        return {
            'trade_count': len(pnls),
            'win_rate': win_rate,
            'average_pnl': average_pnl,
            'recommendations': recommendation,
            'should_adapt': False,
            'adaptations': {},
        }

    def end_of_day_analysis(
        self,
        trades: List,
        daily_decision: Optional[Dict] = None,
        strategies_used: Optional[List[str]] = None,
    ) -> Dict:
        """Build the structured learning record expected by the paper runner."""
        performance = self.analyze_trade_performance(
            trades=trades,
            current_strategies=strategies_used,
        )
        count = performance['trade_count']
        if count == 0:
            return {
                'successes': [],
                'failures': [],
                'insights': 'No resolved paper trades were available for end-of-day analysis.',
                'tomorrow_plan': 'Keep paper trading and collect validated execution evidence.',
                'performance': performance,
            }

        win_rate = performance['win_rate']
        average_pnl = performance['average_pnl']
        successes = [f"{win_rate:.0%} win rate across {count} resolved trades"] if win_rate >= 0.5 else []
        failures = []
        if win_rate < 0.4:
            failures.append(f"Low win rate: {win_rate:.0%}")
        if average_pnl < 0:
            failures.append(f"Negative average P&L: ${average_pnl:.2f}")
        insight = (
            f"Reviewed {count} resolved trades with average P&L ${average_pnl:.2f}."
        )
        return {
            'successes': successes,
            'failures': failures,
            'insights': insight,
            'tomorrow_plan': performance['recommendations'],
            'performance': performance,
        }
