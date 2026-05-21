"""SMS Alert Notifier

Sends SMS alerts via Twilio when critical bot events occur.
Requires TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER,
and ALERT_PHONE_NUMBER to be set as environment variables.

If any variable is missing, alerts are logged as warnings but do NOT crash the bot.
"""
import logging
import os


def send_alert(message: str) -> bool:
    """Send an SMS alert. Returns True on success, False otherwise.

    Silently skips (with a warning log) if Twilio env vars are not configured.
    Never raises — callers wrap this in try/except as a safety net, but this
    function is designed to fail gracefully on its own.
    """
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
    from_number = os.getenv("TWILIO_FROM_NUMBER", "")
    to_number = os.getenv("ALERT_PHONE_NUMBER", "")

    if not all([account_sid, auth_token, from_number, to_number]):
        logging.warning(
            f"[SMS] Alert not sent (Twilio not configured): {message}"
        )
        return False

    try:
        from twilio.rest import Client  # type: ignore

        client = Client(account_sid, auth_token)
        client.messages.create(
            body=f"LIMITLESS BOT: {message}",
            from_=from_number,
            to=to_number,
        )
        logging.info(f"[SMS] Alert sent: {message}")
        return True

    except ImportError:
        logging.warning(
            "[SMS] twilio package not installed. Run: pip install twilio"
        )
        return False
    except Exception as e:
        logging.error(f"[SMS] Failed to send alert: {e}")
        return False
