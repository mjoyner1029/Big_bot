# LIMITLESS AWS Deployment Guide

## EC2 Setup

1. **Launch EC2** — Ubuntu 22.04 LTS, t3.small (or t3.medium if LLM calls are frequent), us-east-1
2. **Security Group inbound rules:**
   - TCP 22 (SSH) — your IP only
   - TCP 8501 (Streamlit) — your IP only
   - No public internet access to 8501

## First-Time Server Setup

```bash
# On EC2 instance
sudo apt update && sudo apt install -y python3.11 python3.11-venv git

cd ~
git clone <your-repo-url> limitless
cd limitless

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy your .env file (never commit this)
nano .env   # paste contents from .env.example with real values
```

## Install systemd Services

```bash
# Copy service files
sudo cp deploy/systemd/limitless.service /etc/systemd/system/
sudo cp deploy/systemd/limitless-dashboard.service /etc/systemd/system/

# Reload and enable
sudo systemctl daemon-reload
sudo systemctl enable limitless
sudo systemctl enable limitless-dashboard

# Start
sudo systemctl start limitless
sudo systemctl start limitless-dashboard

# Check status
sudo systemctl status limitless
sudo systemctl status limitless-dashboard
```

## Verify Auto-Restart

```bash
# Kill the process manually — confirm it restarts within 30 seconds
sudo kill $(pgrep -f "python main.py")
sleep 35
sudo systemctl status limitless   # should show "active (running)"
```

## View Logs

```bash
tail -f ~/limitless/logs/bot.log
tail -f ~/limitless/logs/trade_log.csv
journalctl -u limitless -f
```

## Access Dashboard from Phone

Navigate to: `http://<EC2-PUBLIC-IP>:8501`

Restrict to your IP only via the Security Group — do NOT open to 0.0.0.0/0.

## Persist Logs Across Reboots

Option A — EBS volume (recommended):
```bash
# Mount EBS to /data, symlink logs and state directories
sudo mkfs.ext4 /dev/xvdf
sudo mount /dev/xvdf /data
sudo chown ubuntu:ubuntu /data
mv ~/limitless/logs /data/logs && ln -s /data/logs ~/limitless/logs
mv ~/limitless/state /data/state && ln -s /data/state ~/limitless/state
# Add to /etc/fstab for auto-mount on reboot
```

Option B — S3 sync (simpler):
```bash
# Add to crontab: sync logs to S3 every 5 minutes
*/5 * * * * aws s3 sync ~/limitless/logs s3://your-bucket/limitless/logs/
*/5 * * * * aws s3 sync ~/limitless/state s3://your-bucket/limitless/state/
```

## Going Live (after paper trading passes Go/No-Go)

1. Run `python scripts/validate_paper_run.py` — must show `>>> GO <<<` twice
2. Edit `.env` — set `ALPACA_BASE_URL=https://api.alpaca.markets`
3. Edit `config/config.py` — set `"use_paper_trading": False`
4. Restart: `sudo systemctl restart limitless`
