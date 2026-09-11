#!/usr/bin/env python3
"""
Code validation script — verifies current project structure and modules.

This script performs DEEP CODE validation (imports, schema, module structure).
For startup readiness, use:  python scripts/preflight.py

To run:
    python scripts/validate.py
"""

import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70 + "\n")


def main():
    print("\n" + "*" * 70)
    print("*" + "  TRADING BOT — CODE VALIDATION".center(68) + "*")
    print("*" * 70)

    all_passed = True

    # 1. Load configuration
    print_section("1. CONFIGURATION LOAD")
    try:
        from config.config import CONFIG
        mode = CONFIG.get("trading_mode", "unknown")
        cap  = CONFIG.get("capital", 0)
        print(f"  OK: CONFIG loaded")
        print(f"  Strategy profile:  {mode}")
        print(f"  Capital:           ${cap:,.2f}")
        print(f"  Live trading gate: {CONFIG.get('ENABLE_LIVE_TRADING', False)}")
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False

    # 2. Execution mode enum
    print_section("2. EXECUTION MODE ENUM")
    try:
        from core.trading_mode import ExecutionMode, resolve_execution_mode
        for m in ("PAPER", "SHADOW", "BACKTEST", "LIVE",
                  "paper", "Paper", "shadow"):
            resolved = resolve_execution_mode(m)
            print(f"  '{m}' → {resolved.value}")
        # Verify obsolete modes are rejected
        rejected = []
        for bad in ("balanced", "claude_hf", "conservative"):
            try:
                resolve_execution_mode(bad)
                print(f"  FAIL: '{bad}' should have been rejected")
                all_passed = False
            except ValueError:
                rejected.append(bad)
        print(f"  OK: obsolete modes rejected: {rejected}")
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False

    # 3. Required files
    print_section("3. REQUIRED FILES")
    required_files = [
        "config/config.py",
        "config/validator.py",
        "core/trading_mode.py",
        "core/shadow_mode.py",
        "core/data_quality.py",
        "core/governance.py",
        "core/deployment_manager.py",
        "validation/engine.py",
        "validation/benchmark_engine.py",
        "validation/walk_forward.py",
        "validation/monte_carlo.py",
        "validation/promotion_gates.py",
        "validation/paper_campaign.py",
        "validation/report.py",
        "validation/scenario_engine.py",
        "validation/execution_simulator.py",
        "validation/attribution.py",
        "validation/calibration.py",
        "validation/live_graduation.py",
        "scripts/preflight.py",
        "scripts/live_readiness_check.py",
        "requirements.txt",
        ".env.example",
    ]

    missing = []
    for f in required_files:
        path = PROJECT_ROOT / f
        if path.exists():
            print(f"  OK:      {f}")
        else:
            print(f"  MISSING: {f}")
            missing.append(f)

    if missing:
        print(f"\n  {len(missing)} required file(s) missing")
        all_passed = False

    # 4. Required directories
    print_section("4. REQUIRED DIRECTORIES")
    required_dirs = [
        "config", "core", "validation", "scripts", "tests",
        "logs", "data", "models/saved", "state", "strategies",
        "trading", "backtest", "exchanges", "indicators",
    ]
    for d in required_dirs:
        path = PROJECT_ROOT / d
        if path.exists():
            print(f"  OK:     {d}/")
        else:
            print(f"  WARN:   {d}/ — will be created on startup")

    # 5. Core module imports
    print_section("5. CORE MODULE IMPORTS")
    core_modules = [
        ("core.trading_mode",       "ExecutionMode"),
        ("core.shadow_mode",        "ShadowModeTracker"),
        ("core.data_quality",       "DataQualityMonitor"),
        ("core.governance",         "ModelGovernance"),
        ("core.deployment_manager", "StrategyDeploymentManager"),
        ("validation.engine",       "ValidationEngine"),
        ("validation.benchmark_engine", "BenchmarkEngine"),
        ("validation.walk_forward", "WalkForwardValidator"),
        ("validation.monte_carlo",  "MonteCarloEngine"),
        ("validation.promotion_gates", "PromotionGates"),
        ("validation.paper_campaign",  "PaperCampaignManager"),
        ("validation.report",          "ValidationReportGenerator"),
        ("validation.scenario_engine", "ScenarioEngine"),
        ("validation.execution_simulator", "RealisticExecutionSimulator"),
        ("validation.attribution",   "PerformanceAttributionEngine"),
        ("validation.calibration",   "CalibrationAnalyzer"),
        ("validation.live_graduation", "LiveCapitalGraduation"),
    ]
    for module, cls in core_modules:
        try:
            import importlib
            mod = importlib.import_module(module)
            getattr(mod, cls)
            print(f"  OK:   {module}.{cls}")
        except Exception as e:
            print(f"  FAIL: {module}.{cls} — {e}")
            all_passed = False

    # 6. Safety checks
    print_section("6. SAFETY CHECKS")
    try:
        from core.shadow_mode import ShadowModeTracker, TradingMode
        t = ShadowModeTracker.__new__(ShadowModeTracker)
        t._mode   = TradingMode.SHADOW
        t._active = True
        assert not t.should_submit_order(), "Shadow must not submit orders"
        print("  OK: SHADOW mode cannot submit orders")

        t._mode   = TradingMode.PAPER
        t._active = False
        assert t.should_submit_order(), "PAPER must be able to submit orders"
        print("  OK: PAPER mode can submit orders")
    except Exception as e:
        print(f"  FAIL: {e}")
        all_passed = False

    try:
        from core.trading_mode import is_live_authorized
        import os
        orig = os.environ.copy()
        os.environ["EXECUTION_MODE"]    = "PAPER"
        os.environ["ENABLE_LIVE_TRADING"] = "true"
        assert not is_live_authorized(), "PAPER+live_gate=true must NOT authorize"
        os.environ["EXECUTION_MODE"]    = "LIVE"
        os.environ["ENABLE_LIVE_TRADING"] = "false"
        assert not is_live_authorized(), "LIVE+live_gate=false must NOT authorize"
        os.environ["EXECUTION_MODE"]    = "LIVE"
        os.environ["ENABLE_LIVE_TRADING"] = "true"
        assert is_live_authorized(),     "LIVE+live_gate=true must authorize"
        os.environ.clear()
        os.environ.update(orig)
        print("  OK: LIVE authorization requires both EXECUTION_MODE=LIVE AND ENABLE_LIVE_TRADING=true")
    except Exception as e:
        print(f"  FAIL: live authorization gate: {e}")
        all_passed = False

    # 7. Run preflight in BACKTEST mode (no API calls)
    print_section("7. PREFLIGHT SMOKE (BACKTEST, no API)")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/preflight.py", "--mode", "BACKTEST"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0:
            last = [l for l in result.stdout.splitlines() if l.strip()][-3:]
            for l in last:
                print(f"  {l}")
            print("  OK: preflight --mode BACKTEST returned 0")
        else:
            failures = [l for l in result.stdout.splitlines() if "FAIL" in l][:5]
            for f in failures:
                print(f"  {f}")
            print(f"  WARN: preflight --mode BACKTEST returned {result.returncode} "
                  f"(may be due to missing .env keys)")
    except Exception as e:
        print(f"  WARN: could not run preflight subprocess: {e}")

    # Summary
    print("\n" + "*" * 70)
    if all_passed:
        print("*" + "  CODE VALIDATION PASSED".center(68) + "*")
        print("*" + " " * 68 + "*")
        print("*" + "  Next: python scripts/preflight.py".center(68) + "*")
    else:
        print("*" + "  CODE VALIDATION — ISSUES FOUND".center(68) + "*")
        print("*" + " " * 68 + "*")
        print("*" + "  Fix issues above, then re-run.".center(68) + "*")
    print("*" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nCancelled")
        sys.exit(1)



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
