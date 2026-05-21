"""
Configuration Validator
Comprehensive validation framework for all bot configuration settings.
Validates types, ranges, dependencies, and API keys.
"""

import os
import re
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """Severity levels for validation issues"""
    ERROR = "error"      # Will prevent bot from starting
    WARNING = "warning"  # Should fix but not blocking
    INFO = "info"        # Informational only


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    field: str
    level: ValidationLevel
    message: str
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        msg = f"[{self.level.value.upper()}] {self.field}: {self.message}"
        if self.suggestion:
            msg += f"\n  → Suggestion: {self.suggestion}"
        return msg


class ConfigValidator:
    """Validates bot configuration with comprehensive checks"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: List[ValidationResult] = []

    def validate_all(self) -> Tuple[bool, List[ValidationResult]]:
        """
        Run all validation checks.
        
        Returns:
            (is_valid, results) - True if no errors, list of all validation results
        """
        self.results = []
        
        # Core configuration
        self._validate_capital()
        self._validate_risk_parameters()
        self._validate_trading_mode()
        self._validate_asset_class()
        
        # Exchange/Broker configuration
        self._validate_exchange_config()
        
        # API keys and integrations
        self._validate_api_keys()
        
        # Strategy parameters
        self._validate_strategy_weights()
        self._validate_thresholds()
        
        # Autonomous features
        self._validate_autonomous_config()
        
        # Risk management (discipline layer)
        self._validate_discipline_params()
        
        # Scheduling
        self._validate_scheduler()
        
        # Cross-field validations
        self._validate_dependencies()
        
        # Check for errors
        has_errors = any(r.level == ValidationLevel.ERROR for r in self.results)
        return (not has_errors, self.results)

    def _validate_capital(self):
        """Validate capital configuration"""
        capital = self.config.get("capital", 0)
        use_paper = self.config.get("use_paper_trading", True)
        
        if capital == 0 and use_paper:
            self.results.append(ValidationResult(
                field="capital",
                level=ValidationLevel.WARNING,
                message="No capital configured. Paper trading will use simulated balance.",
                suggestion="Set TRADING_CAPITAL environment variable or update config.py"
            ))
        elif capital == 0 and not use_paper:
            self.results.append(ValidationResult(
                field="capital",
                level=ValidationLevel.ERROR,
                message="No capital configured for live trading",
                suggestion="Set TRADING_CAPITAL environment variable before enabling live trading"
            ))
        elif capital < 0:
            self.results.append(ValidationResult(
                field="capital",
                level=ValidationLevel.ERROR,
                message=f"Invalid capital: {capital} (must be >= 0)",
                suggestion="Set a positive value for TRADING_CAPITAL"
            ))
        elif capital > 0 and capital < 100:
            self.results.append(ValidationResult(
                field="capital",
                level=ValidationLevel.WARNING,
                message=f"Low capital: ${capital:,.2f}. Some positions may be too small.",
                suggestion="Consider using at least $500 for meaningful position sizes"
            ))

    def _validate_risk_parameters(self):
        """Validate risk management parameters"""
        mode = self.config.get("trading_mode", "balanced")
        
        # Get mode-specific risk parameter
        risk_key = f"risk_per_trade_{mode}" if mode != "balanced" else "risk_per_trade_pct"
        risk_pct = self.config.get(risk_key, self.config.get("risk_per_trade_pct", 0.02))
        
        if not 0 < risk_pct <= 0.05:
            self.results.append(ValidationResult(
                field="risk_per_trade",
                level=ValidationLevel.WARNING,
                message=f"Risk per trade {risk_pct*100:.1f}% outside recommended range (0.5%-5%)",
                suggestion="Typical risk per trade is 1-3% for most strategies"
            ))
        
        # Validate max positions
        pos_key = f"max_open_positions_{mode}" if mode != "balanced" else "max_open_positions"
        max_pos = self.config.get(pos_key, self.config.get("max_open_positions", 5))
        
        if max_pos < 1:
            self.results.append(ValidationResult(
                field="max_open_positions",
                level=ValidationLevel.ERROR,
                message=f"Invalid max_open_positions: {max_pos} (must be >= 1)",
                suggestion="Set to at least 3 for diversification"
            ))
        elif max_pos > 50:
            self.results.append(ValidationResult(
                field="max_open_positions",
                level=ValidationLevel.WARNING,
                message=f"Very high max_open_positions: {max_pos}",
                suggestion="High position counts increase management complexity"
            ))
        
        # Validate max position percentage
        max_pos_pct_key = f"max_position_pct_{mode}" if mode != "balanced" else "max_position_pct"
        max_pos_pct = self.config.get(max_pos_pct_key, self.config.get("max_position_pct", 0.25))
        
        if not 0 < max_pos_pct <= 1.0:
            self.results.append(ValidationResult(
                field="max_position_pct",
                level=ValidationLevel.ERROR,
                message=f"Invalid max_position_pct: {max_pos_pct} (must be 0-1)",
                suggestion="Set between 0.1 (10%) and 0.5 (50%)"
            ))

    def _validate_trading_mode(self):
        """Validate trading mode"""
        mode = self.config.get("trading_mode", "balanced")
        valid_modes = ["conservative", "balanced", "aggressive", "claude_hf"]
        
        if mode not in valid_modes:
            self.results.append(ValidationResult(
                field="trading_mode",
                level=ValidationLevel.ERROR,
                message=f"Invalid trading mode: '{mode}'",
                suggestion=f"Use one of: {', '.join(valid_modes)}"
            ))

    def _validate_asset_class(self):
        """Validate asset class configuration"""
        asset_class = self.config.get("asset_class", "both")
        valid_classes = ["crypto", "stocks", "both"]
        
        if asset_class not in valid_classes:
            self.results.append(ValidationResult(
                field="asset_class",
                level=ValidationLevel.ERROR,
                message=f"Invalid asset_class: '{asset_class}'",
                suggestion=f"Use one of: {', '.join(valid_classes)}"
            ))
        
        # Check watchlists
        if asset_class in ["crypto", "both"]:
            crypto_list = self.config.get("crypto_watchlist", [])
            if not crypto_list:
                self.results.append(ValidationResult(
                    field="crypto_watchlist",
                    level=ValidationLevel.WARNING,
                    message="Empty crypto watchlist",
                    suggestion="Add crypto symbols to watch (e.g., BTC-USD, ETH-USD)"
                ))
        
        if asset_class in ["stocks", "both"]:
            stock_list = self.config.get("stock_watchlist", [])
            if not stock_list:
                self.results.append(ValidationResult(
                    field="stock_watchlist",
                    level=ValidationLevel.WARNING,
                    message="Empty stock watchlist",
                    suggestion="Add stock symbols to watch (e.g., AAPL, MSFT)"
                ))

    def _validate_exchange_config(self):
        """Validate exchange/broker configuration"""
        use_paper = self.config.get("use_paper_trading", True)
        asset_class = self.config.get("asset_class", "both")
        
        # Check crypto exchange config
        if asset_class in ["crypto", "both"]:
            coinbase_key = self.config.get("coinbase_api_key", "")
            if not use_paper and not coinbase_key:
                self.results.append(ValidationResult(
                    field="coinbase_api_key",
                    level=ValidationLevel.ERROR,
                    message="Live crypto trading requires Coinbase API key",
                    suggestion="Set COINBASE_API_KEY or enable paper trading"
                ))
        
        # Check stock broker config
        if asset_class in ["stocks", "both"]:
            alpaca_key = self.config.get("alpaca_api_key", "")
            if not use_paper and not alpaca_key:
                self.results.append(ValidationResult(
                    field="alpaca_api_key",
                    level=ValidationLevel.ERROR,
                    message="Live stock trading requires Alpaca API key",
                    suggestion="Set ALPACA_API_KEY or enable paper trading"
                ))

    def _validate_api_keys(self):
        """Validate API key format and presence"""
        # Anthropic Claude
        use_llm = self.config.get("use_llm", True)
        anthropic_key = self.config.get("anthropic_api_key", "")
        
        if use_llm and not anthropic_key:
            self.results.append(ValidationResult(
                field="anthropic_api_key",
                level=ValidationLevel.WARNING,
                message="LLM features enabled but no Anthropic API key configured",
                suggestion="Set ANTHROPIC_API_KEY or disable use_llm"
            ))
        
        if anthropic_key and not anthropic_key.startswith("sk-ant-"):
            self.results.append(ValidationResult(
                field="anthropic_api_key",
                level=ValidationLevel.WARNING,
                message="Anthropic API key format looks incorrect (should start with 'sk-ant-')",
                suggestion="Verify your API key from console.anthropic.com"
            ))
        
        # News API
        news_key = self.config.get("news_api_key", "")
        if not news_key:
            self.results.append(ValidationResult(
                field="news_api_key",
                level=ValidationLevel.INFO,
                message="No News API key configured. Sentiment analysis will be limited.",
                suggestion="Get free key from newsapi.org for enhanced sentiment analysis"
            ))

    def _validate_strategy_weights(self):
        """Validate strategy component weights"""
        ta_weight = self.config.get("ta_weight", 0.30)
        ml_weight = self.config.get("ml_weight", 0.30)
        sentiment_weight = self.config.get("sentiment_weight", 0.15)
        llm_weight = self.config.get("llm_weight", 0.25)
        
        total = ta_weight + ml_weight + sentiment_weight + llm_weight
        
        if abs(total - 1.0) > 0.01:
            self.results.append(ValidationResult(
                field="strategy_weights",
                level=ValidationLevel.WARNING,
                message=f"Strategy weights sum to {total:.2f}, not 1.0",
                suggestion="Weights should sum to 1.0 for proper confidence calculation"
            ))
        
        # Individual weight checks
        for name, weight in [
            ("ta_weight", ta_weight),
            ("ml_weight", ml_weight),
            ("sentiment_weight", sentiment_weight),
            ("llm_weight", llm_weight)
        ]:
            if not 0 <= weight <= 1:
                self.results.append(ValidationResult(
                    field=name,
                    level=ValidationLevel.ERROR,
                    message=f"Invalid {name}: {weight} (must be 0-1)",
                    suggestion="Set weight between 0 and 1"
                ))

    def _validate_thresholds(self):
        """Validate trading thresholds"""
        conf_threshold = self.config.get("confidence_threshold", 0.55)
        
        if not 0 < conf_threshold < 1:
            self.results.append(ValidationResult(
                field="confidence_threshold",
                level=ValidationLevel.ERROR,
                message=f"Invalid confidence_threshold: {conf_threshold} (must be 0-1)",
                suggestion="Set between 0.5 and 0.8 for balanced trading"
            ))
        elif conf_threshold < 0.50:
            self.results.append(ValidationResult(
                field="confidence_threshold",
                level=ValidationLevel.WARNING,
                message=f"Low confidence threshold: {conf_threshold}",
                suggestion="Values below 0.5 may generate too many low-quality signals"
            ))
        elif conf_threshold > 0.80:
            self.results.append(ValidationResult(
                field="confidence_threshold",
                level=ValidationLevel.WARNING,
                message=f"High confidence threshold: {conf_threshold}",
                suggestion="Values above 0.8 may filter out too many good trades"
            ))

    def _validate_autonomous_config(self):
        """Validate autonomous trading features"""
        auto_learning = self.config.get("enable_autonomous_learning", True)
        world_events = self.config.get("enable_world_events_analysis", True)
        use_llm = self.config.get("use_llm", True)
        anthropic_key = self.config.get("anthropic_api_key", "")
        
        if (auto_learning or world_events) and not use_llm:
            self.results.append(ValidationResult(
                field="autonomous_features",
                level=ValidationLevel.WARNING,
                message="Autonomous features enabled but use_llm is False",
                suggestion="Enable use_llm to use autonomous learning and world events"
            ))
        
        if (auto_learning or world_events) and not anthropic_key:
            self.results.append(ValidationResult(
                field="autonomous_features",
                level=ValidationLevel.WARNING,
                message="Autonomous features require Anthropic API key",
                suggestion="Set ANTHROPIC_API_KEY to enable autonomous learning"
            ))
        
        # Validate intervals
        reflection_interval = self.config.get("autonomous_reflection_interval", 12)
        if reflection_interval < 1:
            self.results.append(ValidationResult(
                field="autonomous_reflection_interval",
                level=ValidationLevel.ERROR,
                message=f"Invalid reflection interval: {reflection_interval}",
                suggestion="Set to at least 1 cycle"
            ))

    def _validate_discipline_params(self):
        """Validate risk discipline parameters"""
        max_dd = self.config.get("max_daily_drawdown_pct", 0.03)
        
        if not 0 < max_dd <= 0.20:
            self.results.append(ValidationResult(
                field="max_daily_drawdown_pct",
                level=ValidationLevel.WARNING,
                message=f"Daily drawdown limit {max_dd*100:.1f}% outside typical range (1-20%)",
                suggestion="Conservative: 3-5%, Moderate: 5-10%, Aggressive: 10-15%"
            ))
        
        # Validate correlation limits
        max_cluster_exp = self.config.get("max_cluster_exposure_pct", 0.40)
        if not 0 < max_cluster_exp <= 1.0:
            self.results.append(ValidationResult(
                field="max_cluster_exposure_pct",
                level=ValidationLevel.ERROR,
                message=f"Invalid cluster exposure: {max_cluster_exp}",
                suggestion="Set between 0.2 (20%) and 0.6 (60%)"
            ))
        
        # Validate loss streak
        max_losses = self.config.get("max_consecutive_losses_pause", 4)
        if max_losses < 2:
            self.results.append(ValidationResult(
                field="max_consecutive_losses_pause",
                level=ValidationLevel.WARNING,
                message=f"Very aggressive loss streak limit: {max_losses}",
                suggestion="Consider 3-5 for better risk management"
            ))

    def _validate_scheduler(self):
        """Validate scheduling parameters"""
        mode = self.config.get("trading_mode", "balanced")
        interval_key = f"loop_interval_{mode}" if mode != "balanced" else "loop_interval_seconds"
        interval = self.config.get(interval_key, self.config.get("loop_interval_seconds", 300))
        
        if interval < 10:
            self.results.append(ValidationResult(
                field="loop_interval",
                level=ValidationLevel.WARNING,
                message=f"Very short loop interval: {interval}s",
                suggestion="Intervals under 30s may cause rate limiting and high API costs"
            ))
        elif interval > 3600:
            self.results.append(ValidationResult(
                field="loop_interval",
                level=ValidationLevel.INFO,
                message=f"Long loop interval: {interval/60:.0f} minutes",
                suggestion="Consider if this matches your trading strategy timeframe"
            ))

    def _validate_dependencies(self):
        """Cross-field dependency validation"""
        # If using stocks, should enforce market hours
        asset_class = self.config.get("asset_class", "both")
        enforce_hours = self.config.get("enforce_market_hours", True)
        
        if asset_class in ["stocks", "both"] and not enforce_hours:
            self.results.append(ValidationResult(
                field="enforce_market_hours",
                level=ValidationLevel.WARNING,
                message="Trading stocks without market hours enforcement",
                suggestion="Enable enforce_market_hours to avoid trading during closed hours"
            ))
        
        # If using multi-timeframe, interval should be appropriate
        use_mtf = self.config.get("use_multi_timeframe", True)
        interval = self.config.get("interval", "1h")
        
        if use_mtf and interval in ["1d", "1w"]:
            self.results.append(ValidationResult(
                field="multi_timeframe",
                level=ValidationLevel.WARNING,
                message="Multi-timeframe analysis on daily data may not be useful",
                suggestion="Use hourly or minute intervals for multi-timeframe analysis"
            ))

    def get_summary(self, results: List[ValidationResult]) -> str:
        """Generate human-readable summary"""
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        infos = [r for r in results if r.level == ValidationLevel.INFO]
        
        summary = []
        summary.append("=" * 70)
        summary.append("CONFIGURATION VALIDATION SUMMARY")
        summary.append("=" * 70)
        summary.append("")
        
        if not results:
            summary.append("SUCCESS All configuration checks passed!")
            return "\n".join(summary)
        
        summary.append(f"Found {len(errors)} errors, {len(warnings)} warnings, {len(infos)} info messages")
        summary.append("")
        
        if errors:
            summary.append("ERRORS (must fix):")
            summary.append("-" * 70)
            for r in errors:
                summary.append(str(r))
                summary.append("")
        
        if warnings:
            summary.append("WARNING  WARNINGS (should review):")
            summary.append("-" * 70)
            for r in warnings:
                summary.append(str(r))
                summary.append("")
        
        if infos:
            summary.append("ℹ️  INFORMATION:")
            summary.append("-" * 70)
            for r in infos:
                summary.append(str(r))
                summary.append("")
        
        summary.append("=" * 70)
        
        return "\n".join(summary)


def validate_config(config: Dict[str, Any], verbose: bool = True) -> bool:
    """
    Convenience function to validate configuration.
    
    Args:
        config: Configuration dictionary to validate
        verbose: Print summary if True
        
    Returns:
        True if valid (no errors), False otherwise
    """
    validator = ConfigValidator(config)
    is_valid, results = validator.validate_all()
    
    if verbose:
        print(validator.get_summary(results))
    
    return is_valid


if __name__ == "__main__":
    # Test with current config
    from config import CONFIG
    validate_config(CONFIG)
