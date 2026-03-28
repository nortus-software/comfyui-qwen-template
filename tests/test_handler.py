import json
from unittest.mock import patch, MagicMock, mock_open
import pytest


MOCK_WORKFLOW = {
    "37": {
        "class_type": "LoadImage",
        "inputs": {},
        "widgets_values": ["placeholder.png", "image"]
    },
    "43": {
        "class_type": "VideoFrameExtractorNode",
        "inputs": {},
        "widgets_values": ["placeholder.mp4", 0, 10, 1, 0, "image"]
    }
}


@patch("src.handler.GCSClient")
@patch("src.handler.ComfyUIClient")
@patch("src.handler.download_source")
@patch("src.handler.load_workflow", return_value=MOCK_WORKFLOW)
def test_handler_image_job(mock_load_wf, mock_download, mock_comfyui_cls, mock_gcs_cls):
    """Handler should process an image job end-to-end."""
    # Setup mocks
    mock_download.return_value = (b"image-bytes", ".png")

    mock_comfyui = MagicMock()
    mock_comfyui_cls.return_value = mock_comfyui
    mock_comfyui.upload_image.return_value = {"name": "ref.png"}
    mock_comfyui.submit_prompt.return_value = "prompt-123"
    mock_comfyui.poll_until_complete.return_value = {
        "35": {"images": [{"filename": "output.png", "subfolder": "", "type": "output"}]}
    }
    mock_comfyui.get_output_image.return_value = b"output-bytes"

    mock_gcs = MagicMock()
    mock_gcs_cls.return_value = mock_gcs
    mock_gcs.get_signed_url.return_value = "https://storage.googleapis.com/signed"

    from src.handler import handler

    event = {
        "input": {
            "type": "image",
            "source": "https://example.com/photo.png",
            "gcs_output_path": "outputs/job1/"
        }
    }

    with patch.dict("os.environ", {"GCS_BUCKET": "test-bucket"}):
        result = handler(event)

    assert result["output_url"] == "https://storage.googleapis.com/signed"
    mock_comfyui.upload_image.assert_called_once()
    mock_comfyui.submit_prompt.assert_called_once()
    mock_gcs.upload_bytes.assert_called_once()


@patch("src.handler.GCSClient")
@patch("src.handler.ComfyUIClient")
@patch("src.handler.download_source")
@patch("src.handler.load_workflow", return_value=MOCK_WORKFLOW)
def test_handler_video_job(mock_load_wf, mock_download, mock_comfyui_cls, mock_gcs_cls):
    """Handler should process a video job with frame parameters."""
    mock_download.return_value = (b"video-bytes", ".mp4")

    mock_comfyui = MagicMock()
    mock_comfyui_cls.return_value = mock_comfyui
    mock_comfyui.upload_image.return_value = {"name": "clip.mp4"}
    mock_comfyui.submit_prompt.return_value = "prompt-456"
    mock_comfyui.poll_until_complete.return_value = {
        "35": {"images": [{"filename": "output.png", "subfolder": "", "type": "output"}]}
    }
    mock_comfyui.get_output_image.return_value = b"output-bytes"

    mock_gcs = MagicMock()
    mock_gcs_cls.return_value = mock_gcs
    mock_gcs.get_signed_url.return_value = "https://storage.googleapis.com/signed"

    from src.handler import handler

    event = {
        "input": {
            "type": "video",
            "source": "gs://bucket/media/videos/clip.mp4",
            "gcs_output_path": "outputs/job2/",
            "frame_start": 5,
            "frame_end": 15,
            "frame_step": 2
        }
    }

    with patch.dict("os.environ", {"GCS_BUCKET": "test-bucket"}):
        result = handler(event)

    assert result["output_url"] == "https://storage.googleapis.com/signed"


def test_handler_missing_input_raises():
    """Handler should return error for missing required fields."""
    from src.handler import handler

    event = {"input": {}}

    result = handler(event)
    assert "error" in result
