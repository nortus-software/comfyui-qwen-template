import logging
import time

import requests

log = logging.getLogger("webhook")

MAX_ATTEMPTS = 3
BACKOFF_BASE = 1  # seconds — exponential: 1, 2, 4


def send_webhook(url: str, payload: dict) -> bool:
    """POST payload to url. Retries with exponential backoff. Returns True if delivered."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            log.info("Webhook attempt %d/%d to %s", attempt + 1, MAX_ATTEMPTS, url)
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            log.info("Webhook delivered successfully")
            return True
        except Exception as exc:
            log.warning("Webhook attempt %d failed: %s", attempt + 1, exc)
            if attempt < MAX_ATTEMPTS - 1:
                delay = BACKOFF_BASE * (2 ** attempt)
                time.sleep(delay)
    log.error("Webhook failed after %d attempts", MAX_ATTEMPTS)
    return False
