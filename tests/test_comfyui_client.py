import json
from unittest.mock import patch, MagicMock
import pytest


def test_upload_image_sends_multipart():
    """upload_image should POST file as multipart to ComfyUI /upload/image."""
    from src.comfyui_client import ComfyUIClient

    client = ComfyUIClient("http://localhost:8188")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"name": "test.png", "subfolder": "", "type": "input"}

    with patch("requests.post", return_value=mock_response) as mock_post:
        result = client.upload_image(b"fake-image-bytes", "test.png")

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "/upload/image" in call_kwargs[0][0]
    assert result["name"] == "test.png"


def test_upload_image_raises_on_failure():
    """upload_image should raise on non-200 response."""
    from src.comfyui_client import ComfyUIClient

    client = ComfyUIClient("http://localhost:8188")

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.raise_for_status.side_effect = Exception("500 Server Error")

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(Exception):
            client.upload_image(b"fake-image-bytes", "test.png")


def test_submit_prompt_returns_prompt_id():
    """submit_prompt should POST workflow and return prompt_id."""
    from src.comfyui_client import ComfyUIClient

    client = ComfyUIClient("http://localhost:8188")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"prompt_id": "abc-123"}

    workflow = {"1": {"class_type": "LoadImage"}}

    with patch("requests.post", return_value=mock_response) as mock_post:
        prompt_id = client.submit_prompt(workflow)

    assert prompt_id == "abc-123"
    call_args = mock_post.call_args
    body = json.loads(call_args[1]["data"])
    assert body["prompt"] == workflow


def test_poll_until_complete_returns_outputs():
    """poll_until_complete should poll /history until prompt_id appears."""
    from src.comfyui_client import ComfyUIClient

    client = ComfyUIClient("http://localhost:8188")

    # First call: not ready. Second call: ready with outputs.
    not_ready = MagicMock()
    not_ready.status_code = 200
    not_ready.json.return_value = {}

    ready = MagicMock()
    ready.status_code = 200
    ready.json.return_value = {
        "abc-123": {
            "outputs": {
                "35": {
                    "images": [{"filename": "output.png", "subfolder": "", "type": "output"}]
                }
            }
        }
    }

    with patch("requests.get", side_effect=[not_ready, ready]):
        with patch("time.sleep"):  # skip actual sleep
            outputs = client.poll_until_complete("abc-123", poll_interval=0)

    assert "35" in outputs
    assert outputs["35"]["images"][0]["filename"] == "output.png"


def test_get_output_image_returns_bytes():
    """get_output_image should GET /view with correct params."""
    from src.comfyui_client import ComfyUIClient

    client = ComfyUIClient("http://localhost:8188")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"png-bytes"

    with patch("requests.get", return_value=mock_response) as mock_get:
        data = client.get_output_image("output.png", "", "output")

    assert data == b"png-bytes"
    call_url = mock_get.call_args[0][0]
    assert "/view" in call_url
