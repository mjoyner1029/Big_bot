"""
Test Suite for API Validation
Tests API key validators with mocked responses.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.api_validator import APIKeyValidator, APIStatus, validate_apis


class TestAPIKeyValidator(unittest.TestCase):
    """Test API key validation with mocked API calls"""
    
    def setUp(self):
        """Set up test configuration"""
        self.test_config = {
            "anthropic_api_key": "sk-ant-test123",
            "anthropic_model": "claude-3-5-sonnet-20241022",
            "coinbase_api_key": "test_coinbase_key",
            "coinbase_api_secret": "test_coinbase_secret",
            "alpaca_api_key": "test_alpaca_key",
            "alpaca_api_secret": "test_alpaca_secret",
            "alpaca_base_url": "https://paper-api.alpaca.markets",
            "news_api_key": "test_news_key"
        }
    
    def test_validator_initialization(self):
        """Validator should initialize correctly"""
        validator = APIKeyValidator(self.test_config)
        self.assertEqual(len(validator.results), 0)
    
    def test_not_configured_detection(self):
        """Should detect when API keys are not configured"""
        empty_config = {
            "anthropic_api_key": "",
            "coinbase_api_key": "",
            "coinbase_api_secret": "",
            "alpaca_api_key": "",
            "alpaca_api_secret": "",
            "alpaca_base_url": "",
            "news_api_key": ""
        }
        
        validator = APIKeyValidator(empty_config)
        
        # Test Anthropic
        result = validator._validate_anthropic()
        self.assertEqual(result.status, APIStatus.NOT_CONFIGURED)
        
        # Test Coinbase
        result = validator._validate_coinbase()
        self.assertEqual(result.status, APIStatus.NOT_CONFIGURED)
        
        # Test Alpaca
        result = validator._validate_alpaca()
        self.assertEqual(result.status, APIStatus.NOT_CONFIGURED)
        
        # Test News API
        result = validator._validate_news_api()
        self.assertEqual(result.status, APIStatus.NOT_CONFIGURED)
    
    @patch('config.api_validator.anthropic.Anthropic')
    def test_anthropic_valid_key(self, mock_anthropic_class):
        """Should validate correct Anthropic API key"""
        # Mock successful API call
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client
        
        validator = APIKeyValidator(self.test_config)
        result = validator._validate_anthropic()
        
        self.assertEqual(result.status, APIStatus.VALID)
        self.assertIn("valid", result.message.lower())
        self.assertIsNotNone(result.latency_ms)
    
    @patch('config.api_validator.anthropic.Anthropic')
    def test_anthropic_invalid_key(self, mock_anthropic_class):
        """Should detect invalid Anthropic API key"""
        # Mock authentication error
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("401 invalid authentication")
        mock_anthropic_class.return_value = mock_client
        
        validator = APIKeyValidator(self.test_config)
        result = validator._validate_anthropic()
        
        self.assertEqual(result.status, APIStatus.INVALID)
        self.assertIn("invalid", result.message.lower())
    
    @patch('config.api_validator.anthropic.Anthropic')
    def test_anthropic_rate_limit(self, mock_anthropic_class):
        """Should detect rate limiting"""
        # Mock rate limit error
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("429 rate limit exceeded")
        mock_anthropic_class.return_value = mock_client
        
        validator = APIKeyValidator(self.test_config)
        result = validator._validate_anthropic()
        
        self.assertEqual(result.status, APIStatus.RATE_LIMITED)
    
    @patch('config.api_validator.ccxt.coinbase')
    def test_coinbase_valid_key(self, mock_coinbase_class):
        """Should validate correct Coinbase API key"""
        # Mock successful API call
        mock_exchange = MagicMock()
        mock_exchange.fetch_balance.return_value = {
            'total': {'BTC': 1.0, 'USD': 1000.0}
        }
        mock_coinbase_class.return_value = mock_exchange
        
        validator = APIKeyValidator(self.test_config)
        result = validator._validate_coinbase()
        
        self.assertEqual(result.status, APIStatus.VALID)
        self.assertIn("valid", result.message.lower())
    
    @patch('config.api_validator.ccxt.coinbase')
    def test_coinbase_invalid_key(self, mock_coinbase_class):
        """Should detect invalid Coinbase API key"""
        # Mock authentication error
        mock_exchange = MagicMock()
        mock_exchange.fetch_balance.side_effect = Exception("401 authentication failed")
        mock_coinbase_class.return_value = mock_exchange
        
        validator = APIKeyValidator(self.test_config)
        result = validator._validate_coinbase()
        
        self.assertEqual(result.status, APIStatus.INVALID)
    
    @patch('config.api_validator.tradeapi.REST')
    def test_alpaca_valid_key(self, mock_rest_class):
        """Should validate correct Alpaca API key"""
        # Mock successful API call
        mock_api = MagicMock()
        mock_account = MagicMock()
        mock_account.status = "ACTIVE"
        mock_account.buying_power = "10000.00"
        mock_api.get_account.return_value = mock_account
        mock_rest_class.return_value = mock_api
        
        validator = APIKeyValidator(self.test_config)
        result = validator._validate_alpaca()
        
        self.assertEqual(result.status, APIStatus.VALID)
        self.assertIn("valid", result.message.lower())
        self.assertIn("paper" if "paper" in self.test_config["alpaca_base_url"].lower() else "live", result.message.lower())
    
    @patch('config.api_validator.tradeapi.REST')
    def test_alpaca_invalid_key(self, mock_rest_class):
        """Should detect invalid Alpaca API key"""
        # Mock authentication error
        mock_api = MagicMock()
        mock_api.get_account.side_effect = Exception("401 unauthorized")
        mock_rest_class.return_value = mock_api
        
        validator = APIKeyValidator(self.test_config)
        result = validator._validate_alpaca()
        
        self.assertEqual(result.status, APIStatus.INVALID)
    
    @patch('config.api_validator.requests.get')
    def test_news_api_valid_key(self, mock_get):
        """Should validate correct News API key"""
        # Mock successful API call
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "ok",
            "totalResults": 100
        }
        mock_get.return_value = mock_response
        
        validator = APIKeyValidator(self.test_config)
        result = validator._validate_news_api()
        
        self.assertEqual(result.status, APIStatus.VALID)
        self.assertIn("valid", result.message.lower())
    
    @patch('config.api_validator.requests.get')
    def test_news_api_invalid_key(self, mock_get):
        """Should detect invalid News API key"""
        # Mock authentication error
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"
        mock_get.return_value = mock_response
        
        validator = APIKeyValidator(self.test_config)
        result = validator._validate_news_api()
        
        self.assertEqual(result.status, APIStatus.INVALID)
    
    @patch('config.api_validator.requests.get')
    def test_news_api_rate_limit(self, mock_get):
        """Should detect News API rate limiting"""
        # Mock rate limit error
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_get.return_value = mock_response
        
        validator = APIKeyValidator(self.test_config)
        result = validator._validate_news_api()
        
        self.assertEqual(result.status, APIStatus.RATE_LIMITED)
    
    def test_validate_all_structure(self):
        """validate_all should return results for all services"""
        validator = APIKeyValidator(self.test_config)
        results = validator.validate_all()
        
        # Should have results for all services
        self.assertIn("anthropic", results)
        self.assertIn("coinbase", results)
        self.assertIn("alpaca", results)
        self.assertIn("news_api", results)
        
        # Each result should have correct structure
        for service, result in results.items():
            self.assertIsNotNone(result.service)
            self.assertIsInstance(result.status, APIStatus)
            self.assertIsNotNone(result.message)
    
    def test_summary_generation(self):
        """Validator should generate readable summary"""
        validator = APIKeyValidator(self.test_config)
        validator.validate_all()
        summary = validator.get_summary()
        
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertIn("API", summary.upper())
    
    def test_api_result_string_representation(self):
        """API validation results should have readable string representation"""
        from config.api_validator import APIValidationResult
        
        result = APIValidationResult(
            service="Test Service",
            status=APIStatus.VALID,
            message="Test message",
            details="Test details",
            latency_ms=123.45
        )
        
        result_str = str(result)
        self.assertIn("Test Service", result_str)
        self.assertIn("Test message", result_str)
        self.assertIn("123ms", result_str)


class TestAPIValidationIntegration(unittest.TestCase):
    """Integration tests for API validation"""
    
    def test_validate_apis_convenience_function(self):
        """validate_apis convenience function should work"""
        test_config = {
            "anthropic_api_key": "",
            "coinbase_api_key": "",
            "coinbase_api_secret": "",
            "alpaca_api_key": "",
            "alpaca_api_secret": "",
            "alpaca_base_url": "",
            "news_api_key": ""
        }
        
        # Should run without raising exceptions
        results = validate_apis(test_config, verbose=False)
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)


def run_all_tests():
    """Run all API validation tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestAPIKeyValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIValidationIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("API VALIDATION TEST SUMMARY")
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
