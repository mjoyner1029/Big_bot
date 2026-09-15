"""Daily maintenance and durable weekly research scheduling for paper runs."""
import json
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


class PaperMaintenance:
    def __init__(self, state_path="state/paper_maintenance.json", now=None,
                 background=True):
        self.path = Path(state_path)
        self.last_day = (now or datetime.now()).date()
        self.background = background
        self._thread = None
        try:
            self.state = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.state = {}

    def tick(self, bot, now=None):
        now = now or datetime.now()
        day = now.date().isoformat()
        week = list(now.isocalendar()[:2])
        if now.date() != self.last_day:
            bot.end_of_day_learning()
            bot.daily_regime_analysis()
            self.last_day = now.date()
        if self.state.get("completed_week") == week:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        retry_after = self.state.get("retry_after")
        if retry_after:
            try:
                if now < datetime.fromisoformat(retry_after):
                    return
            except ValueError:
                pass
        if self.state.get("status") == "running":
            owner_pid = int(self.state.get("owner_pid") or 0)
            if owner_pid and self._pid_alive(owner_pid):
                return
            logger.warning("Retrying research interrupted by a stopped paper runner")
        elif self.state.get("attempt_day") == day and not self.state.get("status"):
            # Migrate the old schedule format. An attempt without a completion
            # record means the process was interrupted mid-campaign.
            logger.warning("Retrying interrupted research from legacy schedule state")
        # Persist before invoking research: a restart cannot create a tight retry loop.
        self.state["attempt_day"] = day
        self.state["status"] = "running"
        self.state["owner_pid"] = os.getpid()
        self.state["retry_after"] = None
        self._save()
        if self.background:
            self._thread = threading.Thread(
                target=self._run_weekly, args=(bot, week, now),
                name="paper-weekly-research", daemon=True)
            self._thread.start()
            return
        self._run_weekly(bot, week, now)

    def _run_weekly(self, bot, week, attempted_at):
        try:
            bot.weekly_reviewer.llm = bot.llm
            bot.weekly_reviewer.run()
        except Exception:
            logger.warning("Paper weekly review failed", exc_info=True)
        logger.info("Paper scheduled research starting for ISO week %s", week)
        try:
            report = bot.run_weekly_research_campaign()
        except Exception:
            report = None
            logger.warning("Paper scheduled research failed", exc_info=True)
        if report is not None:
            self.state["completed_week"] = week
            self.state["status"] = "completed"
            self.state["owner_pid"] = None
            self.state["retry_after"] = None
            self._save()
            logger.info("Paper scheduled research completed; validation controls promotion")
        else:
            self.state["status"] = "failed"
            self.state["owner_pid"] = None
            self.state["retry_after"] = (
                attempted_at + timedelta(days=1)
            ).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            self._save()
            logger.warning("Paper research incomplete; next retry tomorrow")

    @staticmethod
    def _pid_alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state))
        temp.replace(self.path)
