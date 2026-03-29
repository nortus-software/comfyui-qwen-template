import json
import copy
import pytest


# Minimal workflow stub with just the nodes we care about
STUB_WORKFLOW = {
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


def test_inject_image_sets_load_image_node():
    """Should set the filename in node 37 (LoadImage)."""
    from src.workflow_injector import inject_reference

    workflow = copy.deepcopy(STUB_WORKFLOW)
    result = inject_reference(workflow, media_type="image", filename="uploaded_ref.png")

    assert result["37"]["widgets_values"][0] == "uploaded_ref.png"
    # Video node should be unchanged
    assert result["43"]["widgets_values"][0] == "placeholder.mp4"


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

    assert result["43"]["widgets_values"][0] == "uploaded_clip.mp4"
    assert result["43"]["widgets_values"][1] == 5
    assert result["43"]["widgets_values"][2] == 20
    assert result["43"]["widgets_values"][3] == 2


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
        "widgets_values": ["z_image_turbo_bf16.safetensors", "default"],
    },
    "40": {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"model": ["42", 0]},
        "widgets_values": ["linaZ.safetensors", 0.85],
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

    assert result["40"]["widgets_values"][0] == "style-v2.safetensors"
    assert result["40"]["widgets_values"][1] == 0.85  # default strength


def test_inject_lora_custom_strength():
    """Should respect custom strength_model parameter."""
    from src.workflow_injector import inject_lora

    workflow = copy.deepcopy(STUB_WORKFLOW_WITH_LORA)
    result = inject_lora(workflow, lora_name="style-v2.safetensors", strength_model=0.6)

    assert result["40"]["widgets_values"][0] == "style-v2.safetensors"
    assert result["40"]["widgets_values"][1] == 0.6


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
    assert workflow["40"]["widgets_values"][0] == "linaZ.safetensors"
    assert result["40"]["widgets_values"][0] == "style-v2.safetensors"
