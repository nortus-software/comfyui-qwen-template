import time
from unittest.mock import patch, MagicMock
import pytest


@patch("src.webhook.requests.post")
def test_send_webhook_success(mock_post):
    """send_webhook returns True on 200 response."""
    mock_post.return_value = MagicMock(status_code=200)
    mock_post.return_value.raise_for_status = MagicMock()

    from src.webhook import send_webhook

    result = send_webhook("https://example.com/hook", {"status": "completed"})

    assert result is True
    mock_post.assert_called_once_with(
        "https://example.com/hook",
        json={"status": "completed"},
        timeout=10,
    )


@patch("src.webhook.time.sleep")
@patch("src.webhook.requests.post")
def test_send_webhook_retries_on_failure(mock_post, mock_sleep):
    """send_webhook retries 3 times with exponential backoff, returns False if all fail."""
    mock_post.side_effect = Exception("connection refused")

    from src.webhook import send_webhook

    result = send_webhook("https://example.com/hook", {"status": "failed", "error": "boom"})

    assert result is False
    assert mock_post.call_count == 3
    mock_sleep.assert_any_call(1)
    mock_sleep.assert_any_call(2)


@patch("src.webhook.time.sleep")
@patch("src.webhook.requests.post")
def test_send_webhook_succeeds_on_retry(mock_post, mock_sleep):
    """send_webhook returns True if a retry succeeds."""
    mock_post.side_effect = [
        Exception("timeout"),
        MagicMock(status_code=200, raise_for_status=MagicMock()),
    ]

    from src.webhook import send_webhook

    result = send_webhook("https://example.com/hook", {"status": "completed"})

    assert result is True
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once_with(1)


@patch("src.webhook.time.sleep")
@patch("src.webhook.requests.post")
def test_send_webhook_retries_on_http_error(mock_post, mock_sleep):
    """send_webhook retries on non-2xx HTTP status."""
    import requests as req

    bad_response = MagicMock(status_code=500)
    bad_response.raise_for_status.side_effect = req.HTTPError("500 Server Error")

    ok_response = MagicMock(status_code=200)
    ok_response.raise_for_status = MagicMock()

    mock_post.side_effect = [bad_response, ok_response]

    from src.webhook import send_webhook

    result = send_webhook("https://example.com/hook", {"status": "completed"})

    assert result is True
    assert mock_post.call_count == 2
