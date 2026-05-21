#!/usr/bin/env python3
"""
Interactive Setup Wizard
Guided configuration for the LIMITLESS trading bot.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class SetupWizard:
    """Interactive setup wizard for bot configuration"""

    def __init__(self):
        self.config = {}
        self.env_vars = {}
        self.project_root = Path(__file__).parent.parent
        self.env_path = self.project_root / ".env"

    def run(self):
        """Run the complete setup wizard"""
        self._print_header()
        
        # Check if .env exists
        if self.env_path.exists():
            print(f"\n{Colors.YELLOW}Found existing .env file{Colors.ENDC}")
            overwrite = self._ask_yes_no("Do you want to reconfigure? (This will backup your existing .env)")
            if not overwrite:
                print(f"\n{Colors.CYAN}Setup cancelled. To edit manually: nano {self.env_path}{Colors.ENDC}")
                return
            
            # Backup existing .env
            backup_path = self.env_path.with_suffix('.env.backup')
            import shutil
            shutil.copy(self.env_path, backup_path)
            print(f"{Colors.GREEN}Backed up to {backup_path}{Colors.ENDC}")
        
        # Run setup steps
        print(f"\n{Colors.BOLD}Let's set up your LIMITLESS trading bot!{Colors.ENDC}\n")
        
        self._setup_trading_capital()
        self._setup_trading_mode()
        self._setup_paper_trading()
        self._setup_anthropic()
        self._setup_exchanges()
        self._setup_news_api()
        
        # Write configuration
        self._write_env_file()
        
        # Run validation
        self._validate_setup()
        
        # Print next steps
        self._print_next_steps()

    def _print_header(self):
        """Print wizard header"""
        print(f"\n{Colors.CYAN}{'='*70}")
        print(f"{Colors.BOLD}LIMITLESS TRADING BOT - SETUP WIZARD{Colors.ENDC}{Colors.CYAN}")
        print(f"{'='*70}{Colors.ENDC}\n")

    def _ask_yes_no(self, question: str, default: bool = False) -> bool:
        """Ask yes/no question"""
        default_str = "Y/n" if default else "y/N"
        while True:
            response = input(f"{question} [{default_str}]: ").strip().lower()
            if response == '':
                return default
            if response in ['y', 'yes']:
                return True
            if response in ['n', 'no']:
                return False
            print(f"{Colors.RED}Please answer 'y' or 'n'{Colors.ENDC}")

    def _ask_string(self, question: str, default: str = "", secret: bool = False) -> str:
        """Ask for string input"""
        if default:
            prompt = f"{question} [{default}]: "
        else:
            prompt = f"{question}: "
        
        if secret:
            import getpass
            response = getpass.getpass(prompt)
        else:
            response = input(prompt).strip()
        
        return response if response else default

    def _ask_choice(self, question: str, choices: Dict[str, str], default: str) -> str:
        """Ask user to choose from options"""
        print(f"\n{question}")
        for key, description in choices.items():
            marker = " (default)" if key == default else ""
            print(f"  {Colors.CYAN}{key}{Colors.ENDC}: {description}{marker}")
        
        while True:
            response = input(f"\nChoice [{default}]: ").strip()
            if response == '':
                return default
            if response in choices:
                return response
            print(f"{Colors.RED}Invalid choice. Please select from: {', '.join(choices.keys())}{Colors.ENDC}")

    def _ask_number(self, question: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        """Ask for numeric input"""
        while True:
            response = input(f"{question} [{default}]: ").strip()
            if response == '':
                return default
            
            try:
                value = float(response)
                if min_val is not None and value < min_val:
                    print(f"{Colors.RED}Value must be at least {min_val}{Colors.ENDC}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"{Colors.RED}Value must be at most {max_val}{Colors.ENDC}")
                    continue
                return value
            except ValueError:
                print(f"{Colors.RED}Please enter a valid number{Colors.ENDC}")

    def _setup_trading_capital(self):
        """Configure initial trading capital"""
        print(f"\n{Colors.BOLD}1. TRADING CAPITAL{Colors.ENDC}")
        print(f"{Colors.CYAN}Set your starting capital (can be changed later){Colors.ENDC}")
        print(f"For paper trading, this is your simulated balance.")
        print(f"For live trading, this should match your exchange balance.\n")
        
        capital = self._ask_number(
            "Starting capital (USD)",
            default=1000.0,
            min_val=10.0
        )
        
        self.env_vars['TRADING_CAPITAL'] = str(capital)
        print(f"{Colors.GREEN}OK: Capital set to ${capital:,.2f}{Colors.ENDC}")

    def _setup_trading_mode(self):
        """Configure trading mode"""
        print(f"\n{Colors.BOLD}2. TRADING MODE{Colors.ENDC}")
        
        modes = {
            "conservative": "Safe - 1% risk, 10 trades/day max, 10min intervals",
            "balanced": "Moderate - 2% risk, 20 trades/day max, 5min intervals (RECOMMENDED)",
            "aggressive": "High - 3% risk, 50 trades/day max, 2min intervals",
            "claude_hf": "AI High-Frequency - 0.5% risk, 500 trades/day max, 30sec intervals"
        }
        
        mode = self._ask_choice(
            "Select trading mode:",
            modes,
            default="balanced"
        )
        
        self.config['trading_mode'] = mode
        print(f"{Colors.GREEN}OK: Trading mode: {mode}{Colors.ENDC}")

    def _setup_paper_trading(self):
        """Configure paper vs live trading"""
        print(f"\n{Colors.BOLD}3. TRADING MODE (Paper vs Live){Colors.ENDC}")
        print(f"{Colors.YELLOW}WARNING: Always test with paper trading first!{Colors.ENDC}\n")
        
        use_paper = self._ask_yes_no(
            "Enable paper trading (simulated trades)?",
            default=True
        )
        
        self.config['use_paper_trading'] = use_paper
        
        if use_paper:
            print(f"{Colors.GREEN}OK: Paper trading enabled - safe for testing{Colors.ENDC}")
        else:
            print(f"{Colors.RED}WARNING: Live trading enabled - real money will be used!{Colors.ENDC}")
            confirm = self._ask_yes_no(f"{Colors.RED}Are you absolutely sure?{Colors.ENDC}", default=False)
            if not confirm:
                print(f"{Colors.YELLOW}Reverting to paper trading for safety{Colors.ENDC}")
                self.config['use_paper_trading'] = True

    def _setup_anthropic(self):
        """Configure Anthropic Claude API"""
        print(f"\n{Colors.BOLD}4. ANTHROPIC CLAUDE (AI Brain){Colors.ENDC}")
        print(f"{Colors.CYAN}Claude provides autonomous market analysis and decision-making.{Colors.ENDC}")
        print(f"  • ~$5 free credit (usually 1-2 weeks of trading)")
        print(f"  • ~$0.02-0.10 per day after that")
        print(f"  • Get key at: {Colors.UNDERLINE}https://console.anthropic.com{Colors.ENDC}\n")
        
        has_key = self._ask_yes_no("Do you have an Anthropic API key?", default=False)
        
        if has_key:
            api_key = self._ask_string(
                "Enter your Anthropic API key (starts with 'sk-ant-')",
                secret=True
            )
            
            if api_key:
                # Basic format validation
                if api_key.startswith('sk-ant-'):
                    self.env_vars['ANTHROPIC_API_KEY'] = api_key
                    print(f"{Colors.GREEN}OK: API key configured{Colors.ENDC}")
                else:
                    print(f"{Colors.YELLOW}WARNING: Key format looks incorrect, but saving anyway{Colors.ENDC}")
                    self.env_vars['ANTHROPIC_API_KEY'] = api_key
            else:
                print(f"{Colors.YELLOW}Skipped - you can add this later to .env{Colors.ENDC}")
        else:
            print(f"{Colors.CYAN}No problem! The bot will run without AI features.{Colors.ENDC}")
            print(f"You can add a key later by editing {self.env_path}")

    def _setup_exchanges(self):
        """Configure exchange API keys"""
        use_paper = self.config.get('use_paper_trading', True)
        
        if use_paper:
            print(f"\n{Colors.BOLD}5. EXCHANGE APIS{Colors.ENDC}")
            print(f"{Colors.CYAN}Paper trading doesn't require exchange API keys.{Colors.ENDC}")
            print(f"Skipping exchange setup. You can add keys later for live trading.\n")
            return
        
        print(f"\n{Colors.BOLD}5. EXCHANGE APIS (Live Trading){Colors.ENDC}")
        print(f"{Colors.YELLOW}WARNING: These keys will have access to your funds!{Colors.ENDC}")
        print(f"{Colors.YELLOW}WARNING: Use API keys with 2FA and withdrawal restrictions!{Colors.ENDC}\n")
        
        # Coinbase (crypto)
        setup_coinbase = self._ask_yes_no("Set up Coinbase (crypto trading)?", default=False)
        if setup_coinbase:
            print(f"\nGet API keys from: {Colors.UNDERLINE}https://www.coinbase.com/settings/api{Colors.ENDC}")
            
            api_key = self._ask_string("Coinbase API Key", secret=True)
            api_secret = self._ask_string("Coinbase API Secret", secret=True)
            passphrase = self._ask_string("Coinbase Passphrase", secret=True)
            
            if api_key:
                self.env_vars['COINBASE_API_KEY'] = api_key
            if api_secret:
                self.env_vars['COINBASE_API_SECRET'] = api_secret
            if passphrase:
                self.env_vars['COINBASE_PASSPHRASE'] = passphrase
            
            print(f"{Colors.GREEN}OK: Coinbase configured{Colors.ENDC}")
        
        # Alpaca (stocks)
        setup_alpaca = self._ask_yes_no("Set up Alpaca (stock trading)?", default=False)
        if setup_alpaca:
            print(f"\nGet API keys from: {Colors.UNDERLINE}https://alpaca.markets{Colors.ENDC}")
            
            mode_choice = self._ask_choice(
                "Alpaca mode:",
                {
                    "paper": "Paper trading (safe, free)",
                    "live": "Live trading (real money)"
                },
                default="paper"
            )
            
            api_key = self._ask_string("Alpaca API Key", secret=True)
            api_secret = self._ask_string("Alpaca API Secret", secret=True)
            
            if api_key:
                self.env_vars['ALPACA_API_KEY'] = api_key
            if api_secret:
                self.env_vars['ALPACA_API_SECRET'] = api_secret
            
            if mode_choice == "live":
                self.env_vars['ALPACA_BASE_URL'] = "https://api.alpaca.markets"
            else:
                self.env_vars['ALPACA_BASE_URL'] = "https://paper-api.alpaca.markets"
            
            print(f"{Colors.GREEN}OK: Alpaca configured ({mode_choice} mode){Colors.ENDC}")

    def _setup_news_api(self):
        """Configure News API for sentiment analysis"""
        print(f"\n{Colors.BOLD}6. NEWS API (Optional){Colors.ENDC}")
        print(f"{Colors.CYAN}Enhances sentiment analysis with news data.{Colors.ENDC}")
        print(f"  • Free tier: 100 requests/day")
        print(f"  • Get key at: {Colors.UNDERLINE}https://newsapi.org{Colors.ENDC}\n")
        
        setup_news = self._ask_yes_no("Set up News API?", default=False)
        
        if setup_news:
            api_key = self._ask_string("News API Key")
            if api_key:
                self.env_vars['NEWS_API_KEY'] = api_key
                print(f"{Colors.GREEN}OK: News API configured{Colors.ENDC}")
        else:
            print(f"{Colors.CYAN}Skipped - sentiment analysis will use fallback methods{Colors.ENDC}")

    def _write_env_file(self):
        """Write .env file with configured values"""
        print(f"\n{Colors.BOLD}Writing configuration...{Colors.ENDC}")
        
        # Read existing .env template if it exists
        template_lines = []
        if self.env_path.exists():
            with open(self.env_path, 'r') as f:
                template_lines = f.readlines()
        
        # Build new .env content
        lines = []
        lines.append("# ── LIMITLESS Trading Bot Configuration ───────────────────────\n")
        lines.append(f"# Generated by setup wizard\n")
        lines.append(f"# Edit this file to update settings\n\n")
        
        # Trading capital
        lines.append("# ── Trading Capital ───────────────────────────────────────────\n")
        lines.append(f"TRADING_CAPITAL={self.env_vars.get('TRADING_CAPITAL', '0')}\n\n")
        
        # Crypto Exchange
        lines.append("# ── Crypto Exchange (Coinbase / any ccxt-supported) ──────────\n")
        lines.append(f"COINBASE_API_KEY={self.env_vars.get('COINBASE_API_KEY', '')}\n")
        lines.append(f"COINBASE_API_SECRET={self.env_vars.get('COINBASE_API_SECRET', '')}\n")
        lines.append(f"COINBASE_PASSPHRASE={self.env_vars.get('COINBASE_PASSPHRASE', '')}\n\n")
        
        # Stock Broker
        lines.append("# ── Stock Broker (Alpaca) ────────────────────────────────────\n")
        lines.append(f"ALPACA_API_KEY={self.env_vars.get('ALPACA_API_KEY', '')}\n")
        lines.append(f"ALPACA_API_SECRET={self.env_vars.get('ALPACA_API_SECRET', '')}\n")
        lines.append(f"ALPACA_BASE_URL={self.env_vars.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')}\n\n")
        
        # Anthropic Claude
        lines.append("# ── Anthropic Claude LLM ─────────────────────────────────────\n")
        lines.append(f"ANTHROPIC_API_KEY={self.env_vars.get('ANTHROPIC_API_KEY', '')}\n\n")
        
        # News API
        lines.append("# ── News / Sentiment ─────────────────────────────────────────\n")
        lines.append(f"NEWS_API_KEY={self.env_vars.get('NEWS_API_KEY', '')}\n\n")
        
        # Notifications
        lines.append("# ── Notifications (optional) ─────────────────────────────────\n")
        lines.append(f"DISCORD_WEBHOOK=\n")
        
        # Write to file
        with open(self.env_path, 'w') as f:
            f.writelines(lines)
        
        print(f"{Colors.GREEN}OK: Configuration saved to {self.env_path}{Colors.ENDC}")

    def _validate_setup(self):
        """Validate the setup"""
        print(f"\n{Colors.BOLD}Validating configuration...{Colors.ENDC}\n")
        
        try:
            # Import and run validators
            sys.path.insert(0, str(self.project_root))
            from config.config import CONFIG
            from config.validator import validate_config
            
            # Run validation
            is_valid = validate_config(CONFIG, verbose=True)
            
            if is_valid:
                print(f"\n{Colors.GREEN}{Colors.BOLD}SUCCESS: Configuration is valid!{Colors.ENDC}")
            else:
                print(f"\n{Colors.YELLOW}WARNING: Configuration has some issues (see above){Colors.ENDC}")
                print(f"{Colors.CYAN}You can still run the bot, but some features may be limited.{Colors.ENDC}")
        
        except Exception as e:
            print(f"{Colors.YELLOW}Could not run full validation: {e}{Colors.ENDC}")
            print(f"{Colors.CYAN}Configuration was written successfully. You can test it by running the bot.{Colors.ENDC}")

    def _print_next_steps(self):
        """Print next steps for user"""
        print(f"\n{Colors.CYAN}{'='*70}")
        print(f"{Colors.BOLD}SETUP COMPLETE!{Colors.ENDC}{Colors.CYAN}")
        print(f"{'='*70}{Colors.ENDC}\n")
        
        print(f"{Colors.BOLD}Next steps:{Colors.ENDC}\n")
        
        print(f"1. {Colors.CYAN}Run the dashboard:{Colors.ENDC}")
        print(f"   streamlit run dashboard/app.py\n")
        
        print(f"2. {Colors.CYAN}Or start the trading bot:{Colors.ENDC}")
        print(f"   python main.py\n")
        
        print(f"3. {Colors.CYAN}To update configuration later:{Colors.ENDC}")
        print(f"   nano {self.env_path}")
        print(f"   or run this wizard again: python scripts/setup.py\n")
        
        if not self.env_vars.get('ANTHROPIC_API_KEY'):
            print(f"{Colors.YELLOW}TIP: Add an Anthropic API key to enable AI features!{Colors.ENDC}")
            print(f"   Get $5 free credit at: https://console.anthropic.com\n")
        
        print(f"{Colors.GREEN}Happy trading!{Colors.ENDC}\n")


def main():
    """Main entry point"""
    wizard = SetupWizard()
    try:
        wizard.run()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Setup cancelled by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Setup error: {e}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
