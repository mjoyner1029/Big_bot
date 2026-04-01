"""
LLM-powered market analyst using Anthropic Claude.

Provides three capabilities:
  1. Deep sentiment analysis of news headlines (beyond VADER)
  2. Technical-analysis interpretation (reads indicator values and reasons)
  3. Trade-signal validation (sanity-checks a proposed trade)

All calls are optional — if no API key is configured the functions
gracefully return neutral / no-opinion defaults so the bot still works.

Prompts are asset-class–aware: the system automatically adapts phrasing
for crypto vs. stock assets based on the symbol being analysed.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from config.config import CONFIG, is_crypto


def _asset_label(symbol: Optional[str] = None) -> str:
    """Return a human-readable asset-class label for use in prompts."""
    if symbol is None:
        return "financial"
    return "cryptocurrency" if is_crypto(symbol) else "stock / equity"

_client = None


def _get_client():
    """Lazy-init the Anthropic client (avoids import error if lib missing)."""
    global _client
    if _client is not None:
        return _client

    api_key = CONFIG.get("anthropic_api_key", "")
    if not api_key:
        return None

    try:
        import anthropic
        _client = anthropic.Anthropic(api_key=api_key)
        return _client
    except Exception as e:
        logging.warning(f"[LLM] Could not initialise Anthropic client: {e}")
        return None


def _call_claude(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> Optional[str]:
    """
    Send a single request to Claude and return the text response.
    Returns None if the API key is missing or the call fails.
    """
    client = _get_client()
    if client is None:
        return None

    model = CONFIG.get("anthropic_model", "claude-sonnet-4-20250514")
    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text
    except Exception as e:
        logging.error(f"[LLM] Claude API call failed: {e}", exc_info=True)
        return None


# ── 1. Deep sentiment analysis ───────────────────────────────────

def llm_sentiment_analysis(headlines: List[str], symbol: Optional[str] = None) -> Optional[float]:
    """
    Ask Claude to score a batch of headlines on a -1 → +1 scale.

    Returns:
        Float in [-1, 1] or None if LLM unavailable.
    """
    if not headlines:
        return None

    asset = _asset_label(symbol)
    system = (
        f"You are a {asset} market sentiment analyst. "
        "Given a list of recent news headlines, return ONLY a JSON object "
        "with two keys: \"score\" (a float from -1.0 bearish to +1.0 bullish) "
        "and \"reasoning\" (one sentence). No other text."
    )
    # Send at most 40 headlines to stay within context budget
    sample = headlines[:40]
    user = f"Rate the overall {asset} market sentiment from these headlines:\n\n"
    user += "\n".join(f"- {h}" for h in sample)

    raw = _call_claude(system, user, max_tokens=256)
    if raw is None:
        return None

    try:
        parsed = json.loads(raw)
        score = float(parsed["score"])
        reasoning = parsed.get("reasoning", "")
        logging.info(f"[LLM Sentiment] score={score:.3f}  reason={reasoning}")
        return max(-1.0, min(1.0, score))
    except Exception as e:
        logging.warning(f"[LLM Sentiment] Could not parse response: {e}\nRaw: {raw}")
        return None


# ── 2. Technical-analysis interpretation ─────────────────────────

def llm_interpret_indicators(indicator_snapshot: Dict[str, Any], symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Ask Claude to interpret a set of TA indicator values and return a
    structured opinion: bias (bullish/bearish/neutral), confidence 0-1,
    and a brief reasoning string.

    Returns:
        Dict with keys: bias, confidence, reasoning — or None.
    """
    asset = _asset_label(symbol)
    system = (
        f"You are an expert {asset} technical analyst. "
        f"Given a snapshot of technical indicator values for a {asset} asset, "
        "return ONLY a JSON object with: "
        "\"bias\" (one of \"bullish\", \"bearish\", \"neutral\"), "
        "\"confidence\" (float 0.0 to 1.0), "
        "\"reasoning\" (2-3 sentences). No other text."
    )
    user = "Here are the latest indicator values:\n\n"
    user += json.dumps(
        {k: round(v, 6) if isinstance(v, float) else v for k, v in indicator_snapshot.items()},
        indent=2,
    )

    raw = _call_claude(system, user, max_tokens=512)
    if raw is None:
        return None

    try:
        parsed = json.loads(raw)
        logging.info(
            f"[LLM TA] bias={parsed.get('bias')}  "
            f"confidence={parsed.get('confidence')}  "
            f"reason={parsed.get('reasoning', '')[:80]}"
        )
        return parsed
    except Exception as e:
        logging.warning(f"[LLM TA] Could not parse response: {e}\nRaw: {raw}")
        return None


# ── 3. Trade-signal validation ────────────────────────────────────

def llm_validate_trade(signal: Dict[str, Any], indicator_snapshot: Dict[str, Any], symbol: Optional[str] = None) -> Dict[str, Any]:
    """
    Ask Claude to review a proposed trade signal given the current indicators.

    Returns:
        Dict with keys: approved (bool), adjusted_confidence (float),
        reasoning (str).  If the LLM is not configured (no API key),
        passes through with the original confidence.  If the LLM is
        configured but the call fails, REJECTS the trade for safety.
    """
    client = _get_client()
    if client is None:
        # LLM not configured — pass through (user hasn't opted into validation)
        return {
            "approved": True,
            "adjusted_confidence": signal.get("confidence", 0),
            "reasoning": "LLM not configured — validation skipped",
        }

    asset = _asset_label(symbol)
    system = (
        f"You are a risk-aware {asset} trading advisor. "
        "Given a proposed trade signal and the current technical indicators, "
        "decide whether the trade should proceed. "
        "Return ONLY a JSON object with: "
        "\"approved\" (bool), "
        "\"adjusted_confidence\" (float 0-1, your revised confidence), "
        "\"reasoning\" (2-3 sentences). No other text."
    )
    user = (
        f"Proposed trade:\n{json.dumps(signal, indent=2, default=str)}\n\n"
        f"Current indicators:\n{json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in indicator_snapshot.items()}, indent=2)}"
    )

    raw = _call_claude(system, user, max_tokens=512)
    if raw is None:
        # LLM was configured but call failed — reject for safety
        logging.warning("[LLM Validate] Claude call failed — rejecting trade for safety")
        return {
            "approved": False,
            "adjusted_confidence": 0.0,
            "reasoning": "LLM validation call failed — trade rejected for safety",
        }

    try:
        parsed = json.loads(raw)
        logging.info(
            f"[LLM Validate] approved={parsed.get('approved')}  "
            f"adj_conf={parsed.get('adjusted_confidence')}  "
            f"reason={parsed.get('reasoning', '')[:80]}"
        )
        return parsed
    except Exception as e:
        logging.warning(f"[LLM Validate] Could not parse response: {e}\nRaw: {raw}")
        return {
            "approved": False,
            "adjusted_confidence": 0.0,
            "reasoning": f"LLM response unparseable — trade rejected for safety",
        }
