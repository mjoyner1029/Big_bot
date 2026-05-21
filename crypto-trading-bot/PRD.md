# PRD: LIMITLESS — Autonomous AI Trading Bot (Production Deployment)

## Overview

LIMITLESS is a personal autonomous AI trading bot for crypto and US equities. The codebase is substantially built. This PRD defines the remaining work to get it from a local dev environment to a 24/7 production deployment on AWS with validated paper trading performance.

---

## Goal

Deploy the bot to AWS so it runs continuously without human intervention, validated against a paper trading baseline before committing real capital.

---

## Phase 1: API Key Configuration & Environment Validation

### Required API Keys (3 total)

| Service | Purpose | Paper Trading Available |
|---|---|---|
| Coinbase Advanced Trade | Crypto trading | Yes |
| Alpaca Markets | US equities trading | Yes (free) |
| Anthropic (Claude Sonnet) | LLM signal validation layer | N/A (paid per token) |

### Tasks

1. **Audit `config/config.py`** — confirm all API key fields are present and correctly named for all three services.
2. **Audit `config/api_validator.py`** — ensure validation logic covers Coinbase, Alpaca, and Anthropic keys at startup.
3. **Create a `.env.example` file** — document all required environment variables so configuration is explicit.
4. **Move all secrets out of `config/config.py`** into environment variables loaded via `python-dotenv`. No API keys should be hardcoded or committed.
5. **Run `check_api_keys.py` and `check_coinbase.py`** — fix any failures against live sandbox/paper endpoints.

---

## Phase 2: Paper Trading Validation

### Objective

Run the bot in paper trading mode across both crypto and equities for a minimum of **2 weeks** and measure real performance before any live capital is deployed.

### Starting Conditions

- Paper trading capital: **$100 simulated**
- Markets: Crypto (Coinbase sandbox) + US Equities (Alpaca paper)
- Mode: `use_paper_trading: True` in config

### Go/No-Go Threshold

| Metric | Minimum Required |
|---|---|
| Return over 2-week period | 10–15% |
| Max drawdown | < 20% |
| Win rate | > 50% |
| Circuit breaker triggers | < 3 in 2 weeks |

> **Note:** 10-15% bi-weekly (equivalent to ~260-390% annually) is significantly above market benchmarks. If the bot fails to hit this threshold consistently, lower the bar to 3-5% bi-weekly before concluding the strategy is broken. A single strong 2-week run should not be sufficient to go live — require **two consecutive** passing periods.

### Tasks

1. **Run `python main.py --backtest`** — validate backtest module runs without errors on available historical data in `data/historical/`.
2. **Run `python main.py`** in paper trading mode — confirm full cycle completes without exceptions.
3. **Verify `logs/trade_log.csv`** is being written correctly with entry price, exit price, P&L per trade.
4. **Implement a validation summary script** (`scripts/validate_paper_run.py`) that reads `trade_log.csv` and outputs: total return %, max drawdown, win rate, and number of circuit breaker events.

---

## Phase 3: AWS Deployment

### Infrastructure

- **Provider:** AWS
- **Recommended service:** EC2 t3.small (or t3.medium if LLM calls are frequent) in `us-east-1`
- **OS:** Ubuntu 22.04 LTS
- **Process management:** `systemd` service or `supervisord` to restart bot on crash
- **Uptime target:** 24/7, auto-restart on failure

### Tasks

1. **Create `deploy/` directory** with the following:
   - `requirements.txt` pinned versions
   - `Dockerfile` for containerized deployment (optional but preferred)
   - `systemd/limitless.service` unit file to run `main.py` as a daemon
   - `systemd/limitless-dashboard.service` unit file to run `streamlit run dashboard/app.py`

2. **Expose Streamlit dashboard publicly:**
   - Bind to `0.0.0.0:8501`
   - Add AWS Security Group rule: inbound TCP 8501 from your IP only (not public internet)
   - Configure Streamlit with a login password via `~/.streamlit/secrets.toml`

3. **Environment variable injection on EC2:**
   - Store API keys in **AWS Secrets Manager** or EC2 instance environment variables
   - Never store secrets in the repo or on disk in plaintext

4. **Log persistence:**
   - Mount `logs/` and `state/` to an EBS volume or S3 sync so data survives instance restarts

---

## Phase 4: SMS Alerting

### Trigger Conditions (send SMS immediately)

| Event | Description |
|---|---|
| Kill switch triggered | Multi-asset market crash detected |
| Circuit breaker fired | 20% total loss OR 5 rapid losses |
| Position health failure | Underwater position force-closed |
| Bot process crash | systemd detects `main.py` exited unexpectedly |
| Daily P&L below -5% | End-of-day loss threshold |

### Implementation

- **Service:** Twilio (recommended) or AWS SNS
- **Integration point:** Add an `alerts/sms_notifier.py` module with a single `send_alert(message: str)` function
- Hook into: `strategies/killswitch.py`, `strategies/discipline.py`, `trading/position_health.py`, and the main loop exception handler in `main.py`
- Add `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, `ALERT_PHONE_NUMBER` to environment variables

---

## Phase 5: Pre-Live Checklist

Before switching `use_paper_trading` to `False`:

- [ ] Two consecutive 2-week paper trading periods meet go/no-go threshold
- [ ] All API keys validated against live (non-sandbox) endpoints
- [ ] SMS alerts tested end-to-end with a manual trigger
- [ ] Streamlit dashboard accessible from mobile browser
- [ ] Bot running on AWS with auto-restart confirmed (kill process manually, verify it restarts)
- [ ] `state/` and `logs/` persisted to EBS or S3
- [ ] Kill switch tested in paper mode by simulating a price crash

---

## Out of Scope

- Strategy changes or new signal sources
- Multi-user access or SaaS features
- Leverage or margin trading
- Solana integration (documented in `SOLANA_BOT_INTEGRATION.md` but not part of this deployment)

---

## File Map (Relevant to This PRD)

| File | Relevance |
|---|---|
| `config/config.py` | API key config — move to env vars |
| `config/api_validator.py` | Startup key validation |
| `main.py` | Entry point — add crash alert hook |
| `strategies/killswitch.py` | Hook SMS alert on trigger |
| `strategies/discipline.py` | Hook SMS alert on circuit breaker |
| `trading/position_health.py` | Hook SMS alert on force-close |
| `dashboard/app.py` | Streamlit — expose on EC2 with auth |
| `logs/trade_log.csv` | Source of truth for paper trading validation |
| `scripts/validate.py` | Extend to produce paper trading summary report |

---

## Success Definition

The project is "done" when:
> The bot runs continuously on AWS, trades both crypto and equities in paper mode, sends SMS alerts on critical events, is accessible via Streamlit on mobile, and has produced two consecutive 2-week periods at ≥10% return before any real capital is deposited.
