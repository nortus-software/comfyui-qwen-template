from unittest.mock import MagicMock, patch
import pytest


def _make_ctx():
    """Build a JobContext stub with mocked deps."""
    from src.handler import JobContext
    ctx = JobContext(
        job_id="job-1",
        config=MagicMock(comfyui_dir="/ComfyUI", gcs_signed_url_expiry=3600),
        comfyui=MagicMock(),
        gcs=MagicMock(),
        gcs_output_path="outputs/",
        webhook_url=None,
    )
    return ctx


@patch("src.pipeline.download_source", return_value=(b"img-bytes", ".png"))
def test_download_and_upload_image_uploads_with_uuid_filename(mock_download):
    from src.pipeline import download_and_upload_image

    ctx = _make_ctx()
    filename = download_and_upload_image("https://x.test/img.png", "ref", ctx)

    mock_download.assert_called_once()
    assert filename.startswith("ref_")
    assert filename.endswith(".png")
    ctx.comfyui.upload_image.assert_called_once()
    upload_args = ctx.comfyui.upload_image.call_args.args
    assert upload_args[0] == b"img-bytes"
    assert upload_args[1] == filename


@patch("src.pipeline.os.symlink")
@patch("src.pipeline.os.path.lexists", return_value=False)
@patch("src.pipeline.get_lora_path", return_value=("/tmp/lora_cache/blobs/abc.safetensors", "style.safetensors"))
def test_setup_lora_symlinks_into_comfyui(mock_get, mock_lexists, mock_symlink):
    from src.pipeline import setup_lora

    ctx = _make_ctx()
    filename, dest_path = setup_lora("gs://b/lora.safetensors", ctx)

    assert filename == "style.safetensors"
    assert dest_path == "/ComfyUI/models/loras/style.safetensors"
    mock_symlink.assert_called_once_with("/tmp/lora_cache/blobs/abc.safetensors", dest_path)


@patch("src.pipeline.os.remove")
@patch("src.pipeline.os.symlink")
@patch("src.pipeline.os.path.lexists", return_value=True)
@patch("src.pipeline.get_lora_path", return_value=("/tmp/lora_cache/blobs/abc.safetensors", "style.safetensors"))
def test_setup_lora_replaces_existing_symlink(mock_get, mock_lexists, mock_symlink, mock_remove):
    from src.pipeline import setup_lora

    ctx = _make_ctx()
    setup_lora("gs://b/lora.safetensors", ctx)

    mock_remove.assert_called_once_with("/ComfyUI/models/loras/style.safetensors")
    mock_symlink.assert_called_once()


def test_submit_and_fetch_output_returns_first_image_bytes():
    from src.pipeline import submit_and_fetch_output

    ctx = _make_ctx()
    ctx.comfyui.submit_prompt.return_value = "p-1"
    ctx.comfyui.poll_until_complete.return_value = {
        "35": {"images": [{"filename": "out.png", "subfolder": "", "type": "output"}]}
    }
    ctx.comfyui.get_output_image.return_value = b"out-bytes"

    out = submit_and_fetch_output({"workflow": "stub"}, ctx)

    assert out == b"out-bytes"
    ctx.comfyui.submit_prompt.assert_called_once_with({"workflow": "stub"})
    ctx.comfyui.get_output_image.assert_called_once_with("out.png", "", "output")


def test_submit_and_fetch_output_raises_when_node_missing():
    from src.pipeline import submit_and_fetch_output

    ctx = _make_ctx()
    ctx.comfyui.submit_prompt.return_value = "p-1"
    ctx.comfyui.poll_until_complete.return_value = {"99": {}}

    with pytest.raises(ValueError, match="No output found at node 35"):
        submit_and_fetch_output({}, ctx)


def test_submit_and_fetch_output_raises_when_no_images():
    from src.pipeline import submit_and_fetch_output

    ctx = _make_ctx()
    ctx.comfyui.submit_prompt.return_value = "p-1"
    ctx.comfyui.poll_until_complete.return_value = {"35": {"images": []}}

    with pytest.raises(ValueError, match="No images in output"):
        submit_and_fetch_output({}, ctx)


def test_upload_output_returns_signed_url_and_path():
    from src.pipeline import upload_output

    ctx = _make_ctx()
    ctx.gcs.get_signed_url.return_value = "https://signed"

    result = upload_output(b"bytes", ctx)

    assert result["output_url"] == "https://signed"
    assert result["gcs_output_path"] == "outputs/output_job-1.png"
    ctx.gcs.upload_bytes.assert_called_once_with(
        b"bytes", "outputs/output_job-1.png", content_type="image/png"
    )


def test_upload_output_raises_without_gcs():
    from src.pipeline import upload_output

    ctx = _make_ctx()
    ctx.gcs = None

    with pytest.raises(ValueError, match="GCS_BUCKET not configured"):
        upload_output(b"bytes", ctx)
