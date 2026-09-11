#!/usr/bin/env python3
"""
Start the paper trading bot as a fully detached subprocess (new session).
Run:  .venv/bin/python scripts/start_bot_daemon.py [--capital N]
Stop: pkill -f run_intraday_evidence
Log:  logs/paper_training_console.log
PID:  logs/daemon.pid
"""
import contextlib
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(PROJECT_ROOT, "logs", "paper_training_console.log")
PID_FILE = os.path.join(PROJECT_ROOT, "logs", "daemon.pid")
PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--capital", type=int, default=10000)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "state"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "reports", "paper_training"), exist_ok=True)

    cmd = [
        PYTHON, "-u",
        os.path.join(PROJECT_ROOT, "scripts", "run_intraday_evidence.py"),
        "--capital", str(args.capital),
        "--asset-class", "both",
        "--trading-mode", "balanced",
        "--confidence-threshold", "0.50",
        "--risk-per-trade-pct", "0.01",
        "--max-open-positions", "8",
        "--max-position-pct", "0.15",
        "--target-closed-trades", "99999",
        "--check-every-cycles", "20",
        "--flatten-every-cycles", "10",
        "--max-cycles", "99999",
        "--enable-alt-data-intel",
        "--trade-log-path", os.path.join(PROJECT_ROOT, "logs", "trade_log_paper_live.csv"),
        "--bot-log-path", os.path.join(PROJECT_ROOT, "logs", "bot_paper_training.log"),
        "--state-path", os.path.join(PROJECT_ROOT, "state", "paper_training_state.json"),
        "--snapshot-out-dir", os.path.join(PROJECT_ROOT, "reports", "paper_training"),
    ]

    # Fork #1 — parent exits immediately so the terminal returns
    pid = os.fork()
    if pid > 0:
        # We are the original process — print and exit fast
        sys.stdout.write(f"[daemon] Forking bot... PID will be in {PID_FILE}\n")
        sys.stdout.write(f"[daemon] Log: {LOG_FILE}  |  Stop: pkill -f run_intraday_evidence\n")
        sys.stdout.flush()
        os._exit(0)   # skip Python cleanup — exit immediately

    # ── We are Child 1 ──
    os.setsid()   # new session — detach from terminal completely

    # Fork #2 — child1 exits, grandchild becomes the bot
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)   # child1 exits immediately

    # ── We are the Grandchild (actual daemon) ──
    os.chdir(PROJECT_ROOT)

    # Redirect stdin → /dev/null, stdout/stderr → log (append, line-buffered)
    log_fd = os.open(LOG_FILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    null_fd = os.open("/dev/null", os.O_RDONLY)
    os.dup2(null_fd, 0)   # stdin  → /dev/null
    os.dup2(log_fd, 1)    # stdout → log
    os.dup2(log_fd, 2)    # stderr → log
    os.close(log_fd)
    os.close(null_fd)

    # Close any inherited fds > 2 so no lingering pipe/pty fds survive into bot
    try:
        max_fd = os.sysconf("SC_OPEN_MAX")
    except (AttributeError, ValueError):
        max_fd = 256
    max_fd = min(max_fd, 256)   # cap at 256 — avoids macOS 10240-fd scan hang
    for fd_num in range(3, max_fd):
        with contextlib.suppress(OSError):
            os.close(fd_num)

    # Write PID file (we are the final daemon PID)
    os.write(1, (f"[daemon] PID {os.getpid()} starting bot\n").encode())
    pid_bytes = (str(os.getpid()) + "\n").encode()
    pid_fd = os.open(PID_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(pid_fd, pid_bytes)
    os.close(pid_fd)

    # Replace this process with the actual bot
    os.execv(PYTHON, cmd)


if __name__ == "__main__":
    main()
