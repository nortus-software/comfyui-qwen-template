import json
from unittest.mock import patch, MagicMock, mock_open
import pytest


MOCK_WORKFLOW = {
    "25": {
        "class_type": "ClownsharKSampler_Beta",
        "inputs": {"denoise": 0.8, "cfg": 1},
    },
    "37": {
        "class_type": "LoadImage",
        "inputs": {},
        "widgets_values": ["placeholder.png", "image"]
    },
    "43": {
        "class_type": "VideoFrameExtractorNode",
        "inputs": {"start_second": 0, "frame_count": 10, "selected_frame": 0},
        "widgets_values": ["placeholder.mp4", 0, 10, 1, 0, "image"]
    },
    "44": {
        "class_type": "Nortus_Prompter_NodeInput",
        "inputs": {"trigger_word": "", "model_size": "auto", "character_details": ""},
    },
}


@patch("src.handler.os.remove")
@patch("src.handler.os.symlink")
@patch("src.handler.os.path.lexists", return_value=True)
@patch("src.handler.get_lora_path", return_value=("/tmp/lora_cache/blobs/abc.safetensors", "style-v2.safetensors"))
@patch("src.handler.inject_lora", side_effect=lambda wf, **kw: wf)
@patch("src.handler.GCSClient")
@patch("src.handler.ComfyUIClient")
@patch("src.handler.download_source")
@patch("src.handler.load_workflow", return_value=MOCK_WORKFLOW)
def test_handler_image_job(mock_load_wf, mock_download, mock_comfyui_cls, mock_gcs_cls,
                           mock_inject_lora, mock_get_lora, mock_lexists, mock_symlink, mock_remove):
    """Handler should process an image job end-to-end with LoRA."""
    # download_source called for reference image and source
    mock_download.side_effect = [(b"ref-bytes", ".png"), (b"image-bytes", ".png")]

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
            "reference_image": "https://example.com/ref.png",
            "lora": "gs://bucket/media/loras/style-v2.safetensors",
            "gcs_output_path": "outputs/job1/"
        }
    }

    with patch.dict("os.environ", {"GCS_BUCKET": "test-bucket"}):
        result = handler(event)

    assert result["output_url"] == "https://storage.googleapis.com/signed"
    assert mock_comfyui.upload_image.call_count == 2
    mock_comfyui.submit_prompt.assert_called_once()
    mock_gcs.upload_bytes.assert_called_once()
    mock_inject_lora.assert_called_once()
    assert mock_inject_lora.call_args.kwargs["lora_name"] == "style-v2.safetensors"
    # LoRA was resolved via cache and symlinked, not copied
    mock_get_lora.assert_called_once()
    mock_symlink.assert_called_once()
    sym_args = mock_symlink.call_args.args
    assert sym_args[0] == "/tmp/lora_cache/blobs/abc.safetensors"
    assert sym_args[1].endswith("style-v2.safetensors")
    # Symlink cleanup runs in finally
    assert mock_remove.called


@patch("src.handler.os.remove")
@patch("src.handler.os.symlink")
@patch("src.handler.os.path.lexists", return_value=True)
@patch("src.handler.get_lora_path", return_value=("/tmp/lora_cache/blobs/abc.safetensors", "style-v2.safetensors"))
@patch("src.handler.inject_lora", side_effect=lambda wf, **kw: wf)
@patch("src.handler.GCSClient")
@patch("src.handler.ComfyUIClient")
@patch("src.handler.download_source")
@patch("src.handler.load_workflow", return_value=MOCK_WORKFLOW)
def test_handler_video_job(mock_load_wf, mock_download, mock_comfyui_cls, mock_gcs_cls,
                           mock_inject_lora, mock_get_lora, mock_lexists, mock_symlink, mock_remove):
    """Handler should process a video job with frame parameters and LoRA."""
    mock_download.side_effect = [(b"ref-bytes", ".png"), (b"video-bytes", ".mp4")]

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
            "reference_image": "https://example.com/ref.png",
            "lora": "gs://bucket/media/loras/style-v2.safetensors",
            "gcs_output_path": "outputs/job2/",
            "frame_start": 5,
            "frame_end": 15,
            "frame_step": 2
        }
    }

    with patch.dict("os.environ", {"GCS_BUCKET": "test-bucket"}):
        result = handler(event)

    assert result["output_url"] == "https://storage.googleapis.com/signed"


@patch("src.handler.os.remove")
@patch("src.handler.os.symlink")
@patch("src.handler.os.path.lexists", return_value=True)
@patch("src.handler.get_lora_path", return_value=("/tmp/lora_cache/blobs/abc.safetensors", "style-v2.safetensors"))
@patch("src.handler.inject_video_settings", side_effect=lambda wf, **kw: wf)
@patch("src.handler.inject_prompter", side_effect=lambda wf, **kw: wf)
@patch("src.handler.inject_ksampler", side_effect=lambda wf, **kw: wf)
@patch("src.handler.inject_lora", side_effect=lambda wf, **kw: wf)
@patch("src.handler.GCSClient")
@patch("src.handler.ComfyUIClient")
@patch("src.handler.download_source")
@patch("src.handler.load_workflow", return_value=MOCK_WORKFLOW)
def test_handler_threads_settings_block(mock_load_wf, mock_download, mock_comfyui_cls, mock_gcs_cls,
                                        mock_inject_lora, mock_inject_ks, mock_inject_pr, mock_inject_vs,
                                        mock_get_lora, mock_lexists, mock_symlink, mock_remove):
    """Handler should pass each settings sub-block to its injector."""
    mock_download.side_effect = [(b"ref-bytes", ".png"), (b"video-bytes", ".mp4")]

    mock_comfyui = MagicMock()
    mock_comfyui_cls.return_value = mock_comfyui
    mock_comfyui.submit_prompt.return_value = "prompt-789"
    mock_comfyui.poll_until_complete.return_value = {
        "35": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}
    }
    mock_comfyui.get_output_image.return_value = b"output-bytes"

    mock_gcs = MagicMock()
    mock_gcs_cls.return_value = mock_gcs
    mock_gcs.get_signed_url.return_value = "https://signed"

    from src.handler import handler

    event = {
        "input": {
            "type": "video",
            "source": "gs://bucket/clip.mp4",
            "reference_image": "https://example.com/ref.png",
            "lora": "gs://bucket/lora.safetensors",
            "settings": {
                "ksampler": {"denoise": 0.4, "cfg": 3.0},
                "prompter": {"trigger_word": "trig", "model_size": "large", "character_details": "x"},
                "video": {"start_second": 2, "frame_count": 12, "selected_frame": 1},
            },
        }
    }

    with patch.dict("os.environ", {"GCS_BUCKET": "test-bucket"}):
        handler(event)

    mock_inject_ks.assert_called_once()
    assert mock_inject_ks.call_args.kwargs == {"denoise": 0.4, "cfg": 3.0}
    mock_inject_pr.assert_called_once()
    assert mock_inject_pr.call_args.kwargs == {"trigger_word": "trig", "model_size": "large", "character_details": "x"}
    mock_inject_vs.assert_called_once()
    assert mock_inject_vs.call_args.kwargs == {"start_second": 2, "frame_count": 12, "selected_frame": 1}


@patch("src.handler.os.remove")
@patch("src.handler.os.symlink")
@patch("src.handler.os.path.lexists", return_value=True)
@patch("src.handler.get_lora_path", return_value=("/tmp/lora_cache/blobs/abc.safetensors", "style-v2.safetensors"))
@patch("src.handler.inject_video_settings", side_effect=lambda wf, **kw: wf)
@patch("src.handler.inject_prompter", side_effect=lambda wf, **kw: wf)
@patch("src.handler.inject_ksampler", side_effect=lambda wf, **kw: wf)
@patch("src.handler.inject_lora", side_effect=lambda wf, **kw: wf)
@patch("src.handler.GCSClient")
@patch("src.handler.ComfyUIClient")
@patch("src.handler.download_source")
@patch("src.handler.load_workflow", return_value=MOCK_WORKFLOW)
def test_handler_no_settings_calls_injectors_with_empty_kwargs(
    mock_load_wf, mock_download, mock_comfyui_cls, mock_gcs_cls,
    mock_inject_lora, mock_inject_ks, mock_inject_pr, mock_inject_vs,
    mock_get_lora, mock_lexists, mock_symlink, mock_remove
):
    """Without a settings block, injectors are called with empty kwargs."""
    mock_download.side_effect = [(b"ref-bytes", ".png"), (b"img-bytes", ".png")]

    mock_comfyui = MagicMock()
    mock_comfyui_cls.return_value = mock_comfyui
    mock_comfyui.submit_prompt.return_value = "prompt-000"
    mock_comfyui.poll_until_complete.return_value = {
        "35": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}
    }
    mock_comfyui.get_output_image.return_value = b"output-bytes"

    mock_gcs = MagicMock()
    mock_gcs_cls.return_value = mock_gcs
    mock_gcs.get_signed_url.return_value = "https://signed"

    from src.handler import handler

    event = {
        "input": {
            "type": "image",
            "source": "https://example.com/p.png",
            "reference_image": "https://example.com/ref.png",
            "lora": "gs://bucket/lora.safetensors",
        }
    }

    with patch.dict("os.environ", {"GCS_BUCKET": "test-bucket"}):
        handler(event)

    assert mock_inject_ks.call_args.kwargs == {}
    assert mock_inject_pr.call_args.kwargs == {}
    assert mock_inject_vs.call_args.kwargs == {}


def test_handler_missing_input_raises():
    """Handler should return error for missing required fields."""
    from src.handler import handler

    event = {"input": {}}

    result = handler(event)
    assert "error" in result


def test_handler_missing_lora_returns_error():
    """Handler should return error when lora field is missing."""
    from src.handler import handler

    event = {
        "input": {
            "type": "image",
            "source": "https://example.com/photo.png",
        }
    }

    result = handler(event)
    assert "error" in result
    assert "lora" in result["error"].lower()
