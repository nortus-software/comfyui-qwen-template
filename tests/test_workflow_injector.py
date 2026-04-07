import json
import copy
import pytest


# Minimal workflow stub with just the nodes we care about
STUB_WORKFLOW = {
    "37": {
        "class_type": "LoadImage",
        "inputs": {"image": "placeholder.png"},
    },
    "43": {
        "class_type": "VideoFrameExtractorNode",
        "inputs": {
            "video": "placeholder.mp4",
            "start_second": 0,
            "frame_count": 10,
            "frame_interval": 1,
        },
    }
}


def test_inject_image_sets_load_image_node():
    """Should set the filename in node 37 (LoadImage)."""
    from src.workflow_injector import inject_reference

    workflow = copy.deepcopy(STUB_WORKFLOW)
    result = inject_reference(workflow, media_type="image", filename="uploaded_ref.png")

    assert result["37"]["inputs"]["image"] == "uploaded_ref.png"
    # Video node should be unchanged
    assert result["43"]["inputs"]["video"] == "placeholder.mp4"


def test_inject_video_sets_video_node():
    """Should set filename and frame params in node 43 (VideoFrameExtractorNode)."""
    from src.workflow_injector import inject_reference

    workflow = copy.deepcopy(STUB_WORKFLOW)
    result = inject_reference(
        workflow,
        media_type="video",
        filename="uploaded_clip.mp4",
        frame_start=5,
        frame_end=20,
        frame_step=2,
    )

    assert result["43"]["inputs"]["video"] == "uploaded_clip.mp4"
    assert result["43"]["inputs"]["start_second"] == 5
    assert result["43"]["inputs"]["frame_count"] == 20
    assert result["43"]["inputs"]["frame_interval"] == 2


def test_inject_unknown_type_raises():
    """Should raise ValueError for unknown media type."""
    from src.workflow_injector import inject_reference

    workflow = copy.deepcopy(STUB_WORKFLOW)
    with pytest.raises(ValueError, match="Unsupported media type"):
        inject_reference(workflow, media_type="audio", filename="test.wav")


# --- inject_lora tests ---

STUB_WORKFLOW_WITH_LORA = {
    "42": {
        "class_type": "UNETLoader",
        "inputs": {},
    },
    "40": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"model": ["42", 0], "lora_name": "linaZ.safetensors", "strength_model": 0.85},
    },
    "25": {
        "class_type": "ClownsharKSampler_Beta",
        "inputs": {"model": ["40", 0]},
    },
}


def test_inject_lora_updates_existing_node():
    """Should update the existing LoRA loader's filename and strength."""
    from src.workflow_injector import inject_lora

    workflow = copy.deepcopy(STUB_WORKFLOW_WITH_LORA)
    result = inject_lora(workflow, lora_name="style-v2.safetensors")

    assert result["40"]["inputs"]["lora_name"] == "style-v2.safetensors"
    assert result["40"]["inputs"]["strength_model"] == 0.85


def test_inject_lora_custom_strength():
    """Should respect custom strength_model parameter."""
    from src.workflow_injector import inject_lora

    workflow = copy.deepcopy(STUB_WORKFLOW_WITH_LORA)
    result = inject_lora(workflow, lora_name="style-v2.safetensors", strength_model=0.6)

    assert result["40"]["inputs"]["lora_name"] == "style-v2.safetensors"
    assert result["40"]["inputs"]["strength_model"] == 0.6


def test_inject_lora_no_lora_node_raises():
    """Should raise ValueError if no LoRA loader node found."""
    from src.workflow_injector import inject_lora

    workflow = {"20": {"class_type": "KSampler", "inputs": {}}}
    with pytest.raises(ValueError, match="No LoRA loader node"):
        inject_lora(workflow, lora_name="style.safetensors")


def test_inject_lora_does_not_mutate_original():
    """Should return a new dict, not mutate the input."""
    from src.workflow_injector import inject_lora

    workflow = copy.deepcopy(STUB_WORKFLOW_WITH_LORA)
    result = inject_lora(workflow, lora_name="style-v2.safetensors")

    # Original should be unchanged
    assert workflow["40"]["inputs"]["lora_name"] == "linaZ.safetensors"
    assert result["40"]["inputs"]["lora_name"] == "style-v2.safetensors"
