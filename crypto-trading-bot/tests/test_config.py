"""
Test Suite for Configuration Module
Comprehensive tests for config loading, validation, and API key checking.
"""

import unittest
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import CONFIG, get_all_symbols, is_crypto
from config.validator import ConfigValidator, ValidationLevel, validate_config
from config.health_check import SystemHealthChecker, HealthStatus, check_system_health


class TestConfigLoading(unittest.TestCase):
    """Test configuration loading and basic functions"""
    
    def test_config_is_dict(self):
        """Config should be a dictionary"""
        self.assertIsInstance(CONFIG, dict)
    
    def test_required_keys_exist(self):
        """All required configuration keys should exist"""
        required_keys = [
            "capital",
            "trading_mode",
            "asset_class",
            "use_paper_trading",
            "risk_per_trade_pct",
            "max_open_positions",
            "confidence_threshold",
            "anthropic_api_key",
            "loop_interval_seconds"
        ]
        
        for key in required_keys:
            self.assertIn(key, CONFIG, f"Required key '{key}' missing from CONFIG")
    
    def test_trading_mode_valid(self):
        """Trading mode should be one of the valid options"""
        valid_modes = ["conservative", "balanced", "aggressive", "claude_hf"]
        self.assertIn(CONFIG["trading_mode"], valid_modes)
    
    def test_capital_is_numeric(self):
        """Capital should be a number"""
        self.assertIsInstance(CONFIG["capital"], (int, float))
        self.assertGreaterEqual(CONFIG["capital"], 0)
    
    def test_risk_parameters_valid(self):
        """Risk parameters should be in valid ranges"""
        # Test each mode's risk parameter
        for mode in ["conservative", "balanced", "aggressive", "claude_hf"]:
            key = f"risk_per_trade_{mode}"
            if key in CONFIG:
                risk = CONFIG[key]
                self.assertGreater(risk, 0, f"{key} should be > 0")
                self.assertLessEqual(risk, 0.10, f"{key} should be <= 10%")
    
    def test_get_all_symbols_returns_list(self):
        """get_all_symbols should return a list"""
        symbols = get_all_symbols()
        self.assertIsInstance(symbols, list)
        self.assertGreater(len(symbols), 0, "Should return at least one symbol")
    
    def test_is_crypto_detection(self):
        """is_crypto should correctly identify crypto symbols"""
        # Crypto symbols
        self.assertTrue(is_crypto("BTC-USD"))
        self.assertTrue(is_crypto("ETH-USD"))
        self.assertTrue(is_crypto("BTCUSDT"))
        
        # Stock symbols
        self.assertFalse(is_crypto("AAPL"))
        self.assertFalse(is_crypto("MSFT"))
        self.assertFalse(is_crypto("SPY"))


class TestConfigValidator(unittest.TestCase):
    """Test configuration validator"""
    
    def setUp(self):
        """Set up test config"""
        self.test_config = {
            "capital": 1000,
            "trading_mode": "balanced",
            "asset_class": "both",
            "use_paper_trading": True,
            "crypto_watchlist": ["BTC-USD", "ETH-USD"],
            "stock_watchlist": ["AAPL", "MSFT"],
            "risk_per_trade_pct": 0.02,
            "max_open_positions": 5,
            "max_position_pct": 0.25,
            "ta_weight": 0.30,
            "ml_weight": 0.30,
            "sentiment_weight": 0.15,
            "llm_weight": 0.25,
            "confidence_threshold": 0.55,
            "use_llm": True,
            "anthropic_api_key": "",
            "enable_autonomous_learning": True,
            "autonomous_reflection_interval": 12,
            "max_daily_drawdown_pct": 0.03,
            "max_cluster_exposure_pct": 0.40,
            "max_consecutive_losses_pause": 4,
            "loop_interval_seconds": 300,
            "enforce_market_hours": True,
            "use_multi_timeframe": True,
            "interval": "1h",
            "coinbase_api_key": "",
            "alpaca_api_key": "",
            "news_api_key": "",
        }
    
    def test_validator_initialization(self):
        """Validator should initialize correctly"""
        validator = ConfigValidator(self.test_config)
        self.assertEqual(len(validator.results), 0)
    
    def test_valid_config_passes(self):
        """A valid configuration should pass validation"""
        validator = ConfigValidator(self.test_config)
        is_valid, results = validator.validate_all()
        
        # Should have results but no errors
        self.assertGreater(len(results), 0)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        self.assertEqual(len(errors), 0, f"Should have no errors, got: {errors}")
    
    def test_invalid_capital_detected(self):
        """Negative capital should be detected"""
        bad_config = self.test_config.copy()
        bad_config["capital"] = -100
        
        validator = ConfigValidator(bad_config)
        is_valid, results = validator.validate_all()
        
        self.assertFalse(is_valid)
        errors = [r for r in results if r.level == ValidationLevel.ERROR and "capital" in r.field.lower()]
        self.assertGreater(len(errors), 0, "Should detect negative capital")
    
    def test_invalid_trading_mode_detected(self):
        """Invalid trading mode should be detected"""
        bad_config = self.test_config.copy()
        bad_config["trading_mode"] = "invalid_mode"
        
        validator = ConfigValidator(bad_config)
        is_valid, results = validator.validate_all()
        
        self.assertFalse(is_valid)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        self.assertGreater(len(errors), 0)
    
    def test_weights_sum_validation(self):
        """Strategy weights should sum to 1.0"""
        bad_config = self.test_config.copy()
        bad_config["ta_weight"] = 0.50
        bad_config["ml_weight"] = 0.50
        bad_config["sentiment_weight"] = 0.50
        bad_config["llm_weight"] = 0.50
        
        validator = ConfigValidator(bad_config)
        is_valid, results = validator.validate_all()
        
        warnings = [r for r in results if "weight" in r.field.lower()]
        self.assertGreater(len(warnings), 0, "Should warn about weights not summing to 1.0")
    
    def test_risk_parameter_ranges(self):
        """Risk parameters outside valid ranges should be flagged"""
        bad_config = self.test_config.copy()
        bad_config["risk_per_trade_pct"] = 0.50  # 50% - way too high
        
        validator = ConfigValidator(bad_config)
        is_valid, results = validator.validate_all()
        
        warnings = [r for r in results if "risk" in r.field.lower()]
        self.assertGreater(len(warnings), 0, "Should warn about excessive risk")
    
    def test_missing_api_key_warning(self):
        """Missing API keys for enabled features should generate warnings"""
        config = self.test_config.copy()
        config["use_llm"] = True
        config["anthropic_api_key"] = ""
        
        validator = ConfigValidator(config)
        is_valid, results = validator.validate_all()
        
        warnings = [r for r in results if "anthropic" in r.field.lower()]
        self.assertGreater(len(warnings), 0, "Should warn about missing Anthropic key when LLM enabled")
    
    def test_summary_generation(self):
        """Validator should generate readable summary"""
        validator = ConfigValidator(self.test_config)
        is_valid, results = validator.validate_all()
        summary = validator.get_summary(results)
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0
)
        self.assertIn("VALIDATION", summary.upper())


class TestHealthChecker(unittest.TestCase):
    """Test system health checker"""
    
    def setUp(self):
        """Set up test config"""
        self.test_config = CONFIG.copy()
    
    def test_health_checker_initialization(self):
        """Health checker should initialize correctly"""
        checker = SystemHealthChecker(self.test_config)
        self.assertEqual(len(checker.checks), 0)
    
    def test_health_check_runs(self):
        """Health check should run without errors"""
        checker = SystemHealthChecker(self.test_config)
        is_healthy, checks = checker.check_all()
        
        self.assertIsInstance(is_healthy, bool)
        self.assertIsInstance(checks, list)
        self.assertGreater(len(checks), 0, "Should perform at least one check")
    
    def test_python_version_check(self):
        """Python version check should run"""
        checker = SystemHealthChecker(self.test_config)
        checker._check_python_version()
        
        self.assertEqual(len(checker.checks), 1)
        self.assertEqual(checker.checks[0].component, "Python Version")
    
    def test_required_packages_check(self):
        """Required packages check should identify missing packages"""
        checker = SystemHealthChecker(self.test_config)
        checker._check_required_packages()
        
        self.assertGreater(len(checker.checks), 0)
        check = checker.checks[0]
        self.assertEqual(check.component, "Required Packages")
    
    def test_directories_are_created(self):
        """Health check should create missing directories"""
        checker = SystemHealthChecker(self.test_config)
        checker._check_directories()
        
        # Check that logs and models directories exist or were created
        project_root = Path(__file__).parent.parent
        logs_dir = project_root / "logs"
        models_dir = project_root / "models" / "saved"
        
        self.assertTrue(logs_dir.exists() or any("Directories" in c.component for c in checker.checks))
        self.assertTrue(models_dir.exists() or any("Directories" in c.component for c in checker.checks))
    
    def test_summary_generation(self):
        """Health checker should generate readable summary"""
        checker = SystemHealthChecker(self.test_config)
        is_healthy, checks = checker.check_all()
        summary = checker.get_summary()
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertIn("HEALTH", summary.upper())
    
    def test_quick_status(self):
        """Health checker should generate quick status"""
        checker = SystemHealthChecker(self.test_config)
        checker.check_all()
        status = checker.get_quick_status()
        
        self.assertIsInstance(status, str)
        self.assertGreater(len(status), 0)


class TestConfigIntegration(unittest.TestCase):
    """Integration tests for complete config system"""
    
    def test_validate_config_convenience_function(self):
        """validate_config convenience function should work"""
        # Should run without raising exceptions
        result = validate_config(CONFIG, verbose=False)
        self.assertIsInstance(result, bool)
    
    def test_check_system_health_convenience_function(self):
        """check_system_health convenience function should work"""
        # Should run without raising exceptions
        result = check_system_health(CONFIG, verbose=False)
        self.assertIsInstance(result, bool)
    
    def test_full_validation_pipeline(self):
        """Complete validation pipeline should work"""
        # Run config validation
        config_valid = validate_config(CONFIG, verbose=False)
        
        # Run health check
        system_healthy = check_system_health(CONFIG, verbose=False)
        
        # Both should return boolean results
        self.assertIsInstance(config_valid, bool)
        self.assertIsInstance(system_healthy, bool)


class TestConfigModes(unittest.TestCase):
    """Test different trading mode configurations"""
    
    def test_conservative_mode_parameters(self):
        """Conservative mode should have appropriate parameters"""
        test_config = CONFIG.copy()
        test_config["trading_mode"] = "conservative"
        
        # Conservative should have lower risk
        self.assertLessEqual(
            CONFIG.get("risk_per_trade_conservative", 0.01),
            CONFIG.get("risk_per_trade_balanced", 0.02)
        )
    
    def test_aggressive_mode_parameters(self):
        """Aggressive mode should have appropriate parameters"""
        test_config = CONFIG.copy()
        test_config["trading_mode"] = "aggressive"
        
        # Aggressive should have higher risk and more positions
        self.assertGreater(
            CONFIG.get("risk_per_trade_aggressive", 0.03),
            CONFIG.get("risk_per_trade_conservative", 0.01)
        )
        
        self.assertGreater(
            CONFIG.get("max_open_positions_aggressive", 8),
            CONFIG.get("max_open_positions_conservative", 3)
        )
    
    def test_claude_hf_mode_parameters(self):
        """Claude HF mode should have high-frequency parameters"""
        # Should have shorter intervals
        self.assertLess(
            CONFIG.get("loop_interval_claude_hf", 30),
            CONFIG.get("loop_interval_balanced", 300)
        )
        
        # Should allow more trades per day
        self.assertGreater(
            CONFIG.get("max_trades_per_day_claude_hf", 500),
            CONFIG.get("max_trades_per_day_balanced", 20)
        )


def run_all_tests():
    """Run all tests with verbose output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfigLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestHealthChecker))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigModes))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
