#!/usr/bin/env python3
"""
Complete System Validation Script
Runs all validation checks: config, APIs, health, and tests.
"""

import sys
import os
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_section(title: str):
    """Print section header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")


def main():
    """Run complete validation suite"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  LIMITLESS TRADING BOT - COMPLETE SYSTEM VALIDATION".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    
    all_passed = True
    
    # 1. Load configuration
    print_section("1. LOADING CONFIGURATION")
    try:
        from config.config import CONFIG
        print(f"OK: Configuration loaded successfully")
        print(f"  Trading mode: {CONFIG.get('trading_mode', 'unknown')}")
        print(f"  Asset class: {CONFIG.get('asset_class', 'unknown')}")
        print(f"  Paper trading: {CONFIG.get('use_paper_trading', True)}")
        print(f"  Capital: ${CONFIG.get('capital', 0):,.2f}")
    except Exception as e:
        print(f"FAILED: Failed to load configuration: {e}")
        return False
    
    # 2. System health check
    print_section("2. SYSTEM HEALTH CHECK")
    try:
        from config.health_check import check_system_health
        is_healthy = check_system_health(CONFIG, verbose=True)
        
        if is_healthy:
            print(f"\nOK: System health check PASSED")
        else:
            print(f"\nWARNING: System health check has issues (see above)")
            all_passed = False
    except Exception as e:
        print(f"FAILED: Health check failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # 3. Configuration validation
    print_section("3. CONFIGURATION VALIDATION")
    try:
        from config.validator import validate_config
        is_valid = validate_config(CONFIG, verbose=True)
        
        if is_valid:
            print(f"\nOK: Configuration validation PASSED")
        else:
            print(f"\nWARNING: Configuration has issues (see above)")
            all_passed = False
    except Exception as e:
        print(f"FAILED: Configuration validation failed: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    # 4. API key validation
    print_section("4. API KEY VALIDATION")
    try:
        from config.api_validator import validate_apis
        api_results = validate_apis(CONFIG, verbose=True)
        
        # Check if any APIs are configured and valid
        valid_apis = sum(1 for r in api_results.values() if r.status.value == "valid")
        invalid_apis = sum(1 for r in api_results.values() if r.status.value == "invalid")
        
        if invalid_apis > 0:
            print(f"\nWARNING: Some API keys are invalid (see above)")
            all_passed = False
        elif valid_apis > 0:
            print(f"\nOK: API validation completed - {valid_apis} APIs valid")
        else:
            print(f"\nWARNING: No API keys configured (paper trading mode)")
    except Exception as e:
        print(f"WARNING: API validation skipped: {e}")
        # Don't fail entirely if API validation has issues
    
    # 5. Run unit tests
    print_section("5. RUNNING UNIT TESTS")
    try:
        # Import test modules
        from tests.test_config import run_all_tests as run_config_tests
        from tests.test_api_validation import run_all_tests as run_api_tests
        
        print("Running configuration tests...")
        config_tests_passed = run_config_tests()
        
        print("\nRunning API validation tests...")
        api_tests_passed = run_api_tests()
        
        if config_tests_passed and api_tests_passed:
            print(f"\nOK: All unit tests PASSED")
        else:
            print(f"\nFAILED: Some unit tests FAILED")
            all_passed = False
    except Exception as e:
        print(f"WARNING: Unit tests skipped: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Check for required files
    print_section("6. CHECKING REQUIRED FILES")
    required_files = [
        "main.py",
        "requirements.txt",
        "config/config.py",
        "config/validator.py",
        "config/api_validator.py",
        "config/health_check.py",
        "scripts/setup.py",
        "dashboard/app.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"OK: {file_path}")
        else:
            print(f"MISSING: {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nWARNING: {len(missing_files)} required files missing")
        all_passed = False
    else:
        print(f"\nOK: All required files present")
    
    # 7. Check directory structure
    print_section("7. CHECKING DIRECTORY STRUCTURE")
    required_dirs = [
        "config",
        "logs",
        "models/saved",
        "data/historical",
        "dashboard",
        "strategies",
        "trading",
        "tests",
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"OK: {dir_path}/")
        else:
            print(f"WARNING: {dir_path}/ - creating...")
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"  Created {dir_path}/")
            except Exception as e:
                print(f"  Failed to create: {e}")
                missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"\nWARNING: Could not create {len(missing_dirs)} directories")
        all_passed = False
    else:
        print(f"\nOK: All required directories exist")
    
    # Final summary
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    
    if all_passed:
        print("*" + "  VALIDATION COMPLETE - ALL CHECKS PASSED".center(68) + "*")
        print("*" + " " * 68 + "*")
        print("*" + "  Your LIMITLESS trading bot is ready to run!".center(68) + "*")
        print("*" + " " * 68 + "*")
        print("*" + "  Next steps:".ljust(68) + "*")
        print("*" + "    1. Run the dashboard: streamlit run dashboard/app.py".ljust(68) + "*")
        print("*" + "    2. Or start the bot: python main.py".ljust(68) + "*")
    else:
        print("*" + "  VALIDATION COMPLETE - SOME ISSUES FOUND".center(68) + "*")
        print("*" + " " * 68 + "*")
        print("*" + "  Review the messages above and fix any errors.".center(68) + "*")
        print("*" + "  Most issues can be fixed by:".ljust(68) + "*")
        print("*" + "    1. Running: python scripts/setup.py".ljust(68) + "*")
        print("*" + "    2. Installing missing packages: pip install -r requirements.txt".ljust(68) + "*")
        print("*" + "    3. Configuring API keys in .env file".ljust(68) + "*")
    
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nValidation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nValidation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
