from unittest.mock import MagicMock, patch
import pytest


def _ctx():
    from src.handler import JobContext
    return JobContext(
        job_id="job-img-1",
        config=MagicMock(comfyui_dir="/ComfyUI", workflows_dir="/wf/", gcs_signed_url_expiry=3600),
        comfyui=MagicMock(),
        gcs=MagicMock(),
        gcs_output_path="outputs/",
        webhook_url=None,
    )


# ---------- validation ----------

def test_missing_reference_image_returns_error():
    from src.workflows import process_first_frame_image

    result = process_first_frame_image(
        {"model_reference": "x", "lora": "y"}, _ctx()
    )
    assert "error" in result
    assert "reference_image" in result["error"]


def test_missing_model_reference_returns_error():
    from src.workflows import process_first_frame_image

    result = process_first_frame_image(
        {"reference_image": "x", "lora": "y"}, _ctx()
    )
    assert "error" in result
    assert "model_reference" in result["error"]


def test_missing_lora_returns_error():
    from src.workflows import process_first_frame_image

    result = process_first_frame_image(
        {"reference_image": "x", "model_reference": "y"}, _ctx()
    )
    assert "error" in result
    assert "lora" in result["error"]


def test_strict_rejects_type_field():
    from src.workflows import process_first_frame_image

    result = process_first_frame_image(
        {
            "reference_image": "r", "model_reference": "m", "lora": "l",
            "type": "image",
        },
        _ctx(),
    )
    assert "error" in result
    assert "type" in result["error"].lower()


def test_strict_rejects_source_field():
    from src.workflows import process_first_frame_image

    result = process_first_frame_image(
        {
            "reference_image": "r", "model_reference": "m", "lora": "l",
            "source": "https://x.test/y.png",
        },
        _ctx(),
    )
    assert "error" in result
    assert "source" in result["error"].lower()


def test_strict_rejects_settings_video():
    from src.workflows import process_first_frame_image

    result = process_first_frame_image(
        {
            "reference_image": "r", "model_reference": "m", "lora": "l",
            "settings": {"video": {"start_second": 0}},
        },
        _ctx(),
    )
    assert "error" in result
    assert "video" in result["error"].lower()


# ---------- happy path ----------

@patch("src.workflows.upload_output", return_value={"output_url": "https://signed", "gcs_output_path": "outputs/output_job-img-1.png"})
@patch("src.workflows.submit_and_fetch_output", return_value=b"out-bytes")
@patch("src.workflows.setup_lora", return_value=("lora.safetensors", "/ComfyUI/models/loras/lora.safetensors"))
@patch("src.workflows.download_and_upload_image", side_effect=["model_def.png", "ref_abc.png"])
@patch("src.workflows.load_workflow", return_value={
    "37": {"class_type": "LoadImage", "inputs": {"image": ""}},
    "43": {"class_type": "LoadImage", "inputs": {"image": ""}},
    "25": {"class_type": "ClownsharKSampler_Beta", "inputs": {"denoise": 0.8, "cfg": 1}},
    "44": {"class_type": "Nortus_Prompter_NodeInput", "inputs": {"trigger_word": "", "model_size": "auto", "character_details": ""}},
    "40": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "x.safetensors", "strength_model": 0.85}},
})
def test_happy_path_injects_into_correct_nodes(
    mock_load, mock_dl_upload, mock_lora, mock_submit, mock_upload
):
    from src.workflows import process_first_frame_image

    job_input = {
        "reference_image": "gs://b/char.jpg",
        "model_reference": "gs://b/model.jpg",
        "lora": "gs://b/lora.safetensors",
        "settings": {
            "ksampler": {"denoise": 0.7, "cfg": 1.5},
            "prompter": {"trigger_word": "Sadie01", "character_details": "x"},
        },
    }
    result = process_first_frame_image(job_input, _ctx())

    assert result["output_url"] == "https://signed"
    # download order: model first (→ node 37), then reference (→ node 43)
    assert mock_dl_upload.call_count == 2
    assert mock_dl_upload.call_args_list[0].args[0] == "gs://b/model.jpg"
    assert mock_dl_upload.call_args_list[0].args[1] == "model"
    assert mock_dl_upload.call_args_list[1].args[0] == "gs://b/char.jpg"
    assert mock_dl_upload.call_args_list[1].args[1] == "ref"
    mock_lora.assert_called_once()
    mock_submit.assert_called_once()
    submitted_workflow = mock_submit.call_args.args[0]
    # model_reference (character) → 37, reference_image (driving) → 43
    assert submitted_workflow["37"]["inputs"]["image"] == "model_def.png"
    assert submitted_workflow["43"]["inputs"]["image"] == "ref_abc.png"
    # ksampler / prompter / lora applied
    assert submitted_workflow["25"]["inputs"]["denoise"] == 0.7
    assert submitted_workflow["25"]["inputs"]["cfg"] == 1.5
    assert submitted_workflow["44"]["inputs"]["trigger_word"] == "Sadie01"
    assert submitted_workflow["40"]["inputs"]["lora_name"] == "lora.safetensors"


@patch("src.workflows.upload_output", return_value={})
@patch("src.workflows.submit_and_fetch_output", return_value=b"")
@patch("src.workflows.setup_lora", return_value=("lora.safetensors", "/x"))
@patch("src.workflows.download_and_upload_image", side_effect=["model.png", "ref.png"])
@patch("src.workflows.load_workflow", return_value={
    "37": {"class_type": "LoadImage", "inputs": {"image": ""}},
    "43": {"class_type": "LoadImage", "inputs": {"image": ""}},
    "25": {"class_type": "ClownsharKSampler_Beta", "inputs": {}},
    "44": {"class_type": "Nortus_Prompter_NodeInput", "inputs": {}},
    "40": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "", "strength_model": 0.85}},
})
def test_uses_correct_workflow_filename(
    mock_load, mock_dl, mock_lora, mock_submit, mock_upload
):
    """Loads workflow_first_frame_image_api.json, not the legacy one."""
    from src.workflows import process_first_frame_image

    process_first_frame_image(
        {
            "reference_image": "r", "model_reference": "m", "lora": "l",
        },
        _ctx(),
    )

    load_path = mock_load.call_args.args[0]
    assert load_path.endswith("workflow_first_frame_image_api.json")
