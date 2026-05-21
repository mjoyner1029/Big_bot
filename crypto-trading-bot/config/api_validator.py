"""
API Key Validator
Real-time validation of API keys by making test calls to each service.
"""

import os
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class APIStatus(Enum):
    """API validation status"""
    VALID = "valid"
    INVALID = "invalid"
    NOT_CONFIGURED = "not_configured"
    RATE_LIMITED = "rate_limited"
    SERVICE_ERROR = "service_error"


@dataclass
class APIValidationResult:
    """Result of API key validation"""
    service: str
    status: APIStatus
    message: str
    details: Optional[str] = None
    latency_ms: Optional[float] = None

    def __str__(self) -> str:
        icon = {
            APIStatus.VALID: "OK",
            APIStatus.INVALID: "FAIL",
            APIStatus.NOT_CONFIGURED: " ",
            APIStatus.RATE_LIMITED: "RATE",
            APIStatus.SERVICE_ERROR: "ERROR"
        }[self.status]
        
        msg = f"{icon} {self.service}: {self.message}"
        if self.latency_ms:
            msg += f" ({self.latency_ms:.0f}ms)"
        if self.details:
            msg += f"\n   {self.details}"
        return msg


class APIKeyValidator:
    """Validates API keys by making actual API calls"""

    def __init__(self, config: Dict):
        self.config = config
        self.results: Dict[str, APIValidationResult] = {}

    def validate_all(self) -> Dict[str, APIValidationResult]:
        """Validate all configured API keys"""
        self.results = {}
        
        # Validate each service
        self.results["anthropic"] = self._validate_anthropic()
        self.results["coinbase"] = self._validate_coinbase()
        self.results["alpaca"] = self._validate_alpaca()
        self.results["news_api"] = self._validate_news_api()
        
        return self.results

    def _validate_anthropic(self) -> APIValidationResult:
        """Validate Anthropic Claude API key"""
        api_key = self.config.get("anthropic_api_key", "")
        
        if not api_key:
            return APIValidationResult(
                service="Anthropic Claude",
                status=APIStatus.NOT_CONFIGURED,
                message="No API key configured",
                details="Set ANTHROPIC_API_KEY to enable LLM features"
            )
        
        try:
            import anthropic
            
            start = time.time()
            client = anthropic.Anthropic(api_key=api_key)
            
            # Make minimal test call
            response = client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            
            latency = (time.time() - start) * 1000
            
            return APIValidationResult(
                service="Anthropic Claude",
                status=APIStatus.VALID,
                message="API key valid",
                details=f"Model: {self.config.get('anthropic_model', 'claude-sonnet-4')}",
                latency_ms=latency
            )
            
        except ImportError:
            return APIValidationResult(
                service="Anthropic Claude",
                status=APIStatus.SERVICE_ERROR,
                message="Anthropic package not installed",
                details="Run: pip install anthropic"
            )
        except Exception as e:
            error_msg = str(e).lower()
            
            if "invalid" in error_msg or "authentication" in error_msg or "401" in error_msg:
                return APIValidationResult(
                    service="Anthropic Claude",
                    status=APIStatus.INVALID,
                    message="Invalid API key",
                    details="Check your key at console.anthropic.com"
                )
            elif "rate" in error_msg or "429" in error_msg:
                return APIValidationResult(
                    service="Anthropic Claude",
                    status=APIStatus.RATE_LIMITED,
                    message="Rate limited",
                    details="Wait a moment and try again"
                )
            else:
                return APIValidationResult(
                    service="Anthropic Claude",
                    status=APIStatus.SERVICE_ERROR,
                    message="API error",
                    details=str(e)[:100]
                )

    def _validate_coinbase(self) -> APIValidationResult:
        """Validate Coinbase API credentials"""
        api_key = self.config.get("coinbase_api_key", "")
        api_secret = self.config.get("coinbase_api_secret", "")
        
        if not api_key or not api_secret:
            return APIValidationResult(
                service="Coinbase",
                status=APIStatus.NOT_CONFIGURED,
                message="No API credentials configured",
                details="Set COINBASE_API_KEY and COINBASE_API_SECRET for live crypto trading"
            )
        
        try:
            import ccxt
            
            start = time.time()
            exchange = ccxt.coinbase({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True
            })
            
            # Test call - fetch balance
            balance = exchange.fetch_balance()
            latency = (time.time() - start) * 1000
            
            # Check if we got valid data
            if balance and 'total' in balance:
                total_assets = len([k for k, v in balance['total'].items() if v > 0])
                return APIValidationResult(
                    service="Coinbase",
                    status=APIStatus.VALID,
                    message="API credentials valid",
                    details=f"Account active, {total_assets} assets with balance",
                    latency_ms=latency
                )
            else:
                return APIValidationResult(
                    service="Coinbase",
                    status=APIStatus.VALID,
                    message="API credentials valid",
                    details="Account connected successfully",
                    latency_ms=latency
                )
                
        except ImportError:
            return APIValidationResult(
                service="Coinbase",
                status=APIStatus.SERVICE_ERROR,
                message="CCXT package not installed",
                details="Run: pip install ccxt"
            )
        except Exception as e:
            error_msg = str(e).lower()
            
            if "invalid" in error_msg or "authentication" in error_msg or "401" in error_msg:
                return APIValidationResult(
                    service="Coinbase",
                    status=APIStatus.INVALID,
                    message="Invalid API credentials",
                    details="Verify your API key and secret at coinbase.com"
                )
            elif "permission" in error_msg or "403" in error_msg:
                return APIValidationResult(
                    service="Coinbase",
                    status=APIStatus.INVALID,
                    message="Insufficient permissions",
                    details="Enable trading permissions for your API key"
                )
            elif "rate" in error_msg or "429" in error_msg:
                return APIValidationResult(
                    service="Coinbase",
                    status=APIStatus.RATE_LIMITED,
                    message="Rate limited",
                    details="Too many requests, wait a moment"
                )
            else:
                return APIValidationResult(
                    service="Coinbase",
                    status=APIStatus.SERVICE_ERROR,
                    message="API error",
                    details=str(e)[:100]
                )

    def _validate_alpaca(self) -> APIValidationResult:
        """Validate Alpaca API credentials"""
        api_key = self.config.get("alpaca_api_key", "")
        api_secret = self.config.get("alpaca_api_secret", "")
        base_url = self.config.get("alpaca_base_url", "https://paper-api.alpaca.markets")
        
        if not api_key or not api_secret:
            return APIValidationResult(
                service="Alpaca",
                status=APIStatus.NOT_CONFIGURED,
                message="No API credentials configured",
                details="Set ALPACA_API_KEY and ALPACA_API_SECRET for stock trading"
            )
        
        try:
            import alpaca_trade_api as tradeapi
            
            start = time.time()
            api = tradeapi.REST(
                key_id=api_key,
                secret_key=api_secret,
                base_url=base_url
            )
            
            # Test call - get account
            account = api.get_account()
            latency = (time.time() - start) * 1000
            
            is_paper = "paper" in base_url.lower()
            mode = "Paper" if is_paper else "Live"
            
            return APIValidationResult(
                service="Alpaca",
                status=APIStatus.VALID,
                message=f"API credentials valid ({mode} mode)",
                details=f"Account: {account.status}, Buying power: ${float(account.buying_power):,.2f}",
                latency_ms=latency
            )
            
        except ImportError:
            return APIValidationResult(
                service="Alpaca",
                status=APIStatus.SERVICE_ERROR,
                message="alpaca-trade-api package not installed",
                details="Run: pip install alpaca-trade-api"
            )
        except Exception as e:
            error_msg = str(e).lower()
            
            if "invalid" in error_msg or "authentication" in error_msg or "401" in error_msg:
                return APIValidationResult(
                    service="Alpaca",
                    status=APIStatus.INVALID,
                    message="Invalid API credentials",
                    details="Check your keys at alpaca.markets"
                )
            elif "forbidden" in error_msg or "403" in error_msg:
                return APIValidationResult(
                    service="Alpaca",
                    status=APIStatus.INVALID,
                    message="Access forbidden",
                    details="Verify API key permissions and account status"
                )
            elif "rate" in error_msg or "429" in error_msg:
                return APIValidationResult(
                    service="Alpaca",
                    status=APIStatus.RATE_LIMITED,
                    message="Rate limited",
                    details="Too many requests, wait a moment"
                )
            else:
                return APIValidationResult(
                    service="Alpaca",
                    status=APIStatus.SERVICE_ERROR,
                    message="API error",
                    details=str(e)[:100]
                )

    def _validate_news_api(self) -> APIValidationResult:
        """Validate News API key"""
        api_key = self.config.get("news_api_key", "")
        
        if not api_key:
            return APIValidationResult(
                service="News API",
                status=APIStatus.NOT_CONFIGURED,
                message="No API key configured",
                details="Optional: Get free key at newsapi.org for sentiment analysis"
            )
        
        try:
            import requests
            
            start = time.time()
            
            # Test call to News API
            response = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": "bitcoin",
                    "apiKey": api_key,
                    "pageSize": 1
                },
                timeout=10
            )
            
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    return APIValidationResult(
                        service="News API",
                        status=APIStatus.VALID,
                        message="API key valid",
                        details=f"Plan: {data.get('totalResults', 'N/A')} results available",
                        latency_ms=latency
                    )
            
            elif response.status_code == 401:
                return APIValidationResult(
                    service="News API",
                    status=APIStatus.INVALID,
                    message="Invalid API key",
                    details="Check your key at newsapi.org/account"
                )
            
            elif response.status_code == 429:
                return APIValidationResult(
                    service="News API",
                    status=APIStatus.RATE_LIMITED,
                    message="Rate limited",
                    details="Daily limit reached or too many requests"
                )
            
            else:
                return APIValidationResult(
                    service="News API",
                    status=APIStatus.SERVICE_ERROR,
                    message=f"HTTP {response.status_code}",
                    details=response.text[:100]
                )
                
        except ImportError:
            return APIValidationResult(
                service="News API",
                status=APIStatus.SERVICE_ERROR,
                message="requests package not installed",
                details="Run: pip install requests"
            )
        except Exception as e:
            return APIValidationResult(
                service="News API",
                status=APIStatus.SERVICE_ERROR,
                message="API error",
                details=str(e)[:100]
            )

    def get_summary(self) -> str:
        """Generate summary of all API validations"""
        if not self.results:
            return "No API validations run yet"
        
        lines = []
        lines.append("=" * 70)
        lines.append("API KEY VALIDATION RESULTS")
        lines.append("=" * 70)
        lines.append("")
        
        # Count statuses
        valid = sum(1 for r in self.results.values() if r.status == APIStatus.VALID)
        invalid = sum(1 for r in self.results.values() if r.status == APIStatus.INVALID)
        not_configured = sum(1 for r in self.results.values() if r.status == APIStatus.NOT_CONFIGURED)
        errors = sum(1 for r in self.results.values() if r.status == APIStatus.SERVICE_ERROR)
        
        lines.append(f"Valid: {valid}  |  Invalid: {invalid}  |  Not Configured: {not_configured}  |  Errors: {errors}")
        lines.append("")
        
        # Show each result
        for service, result in self.results.items():
            lines.append(str(result))
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


def validate_apis(config: Dict, verbose: bool = True) -> Dict[str, APIValidationResult]:
    """
    Convenience function to validate all API keys.
    
    Args:
        config: Configuration dictionary
        verbose: Print summary if True
        
    Returns:
        Dictionary of validation results by service
    """
    validator = APIKeyValidator(config)
    results = validator.validate_all()
    
    if verbose:
        print(validator.get_summary())
    
    return results


if __name__ == "__main__":
    # Test with current config
    from config import CONFIG
    validate_apis(CONFIG)
