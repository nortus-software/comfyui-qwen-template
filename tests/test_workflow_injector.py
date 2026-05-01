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
            "selected_frame": 0,
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


def test_inject_video_sets_filename_only():
    """inject_reference for video sets filename only — no frame params."""
    from src.workflow_injector import inject_reference

    workflow = copy.deepcopy(STUB_WORKFLOW)
    result = inject_reference(workflow, media_type="video", filename="uploaded_clip.mp4")

    assert result["43"]["inputs"]["video"] == "uploaded_clip.mp4"
    assert result["43"]["inputs"]["start_second"] == 0
    assert result["43"]["inputs"]["frame_count"] == 10
    assert result["43"]["inputs"]["selected_frame"] == 0


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


# --- inject_ksampler tests ---

STUB_WORKFLOW_KSAMPLER = {
    "25": {
        "class_type": "ClownsharKSampler_Beta",
        "inputs": {"denoise": 0.8, "cfg": 1, "steps": 8},
    },
}


def test_inject_ksampler_sets_provided_fields():
    from src.workflow_injector import inject_ksampler

    workflow = copy.deepcopy(STUB_WORKFLOW_KSAMPLER)
    result = inject_ksampler(workflow, denoise=0.5, cfg=2.5)

    assert result["25"]["inputs"]["denoise"] == 0.5
    assert result["25"]["inputs"]["cfg"] == 2.5
    assert result["25"]["inputs"]["steps"] == 8


def test_inject_ksampler_partial_only_writes_provided():
    from src.workflow_injector import inject_ksampler

    workflow = copy.deepcopy(STUB_WORKFLOW_KSAMPLER)
    result = inject_ksampler(workflow, denoise=0.3)

    assert result["25"]["inputs"]["denoise"] == 0.3
    assert result["25"]["inputs"]["cfg"] == 1


def test_inject_ksampler_no_args_unchanged():
    from src.workflow_injector import inject_ksampler

    workflow = copy.deepcopy(STUB_WORKFLOW_KSAMPLER)
    result = inject_ksampler(workflow)

    assert result["25"]["inputs"] == STUB_WORKFLOW_KSAMPLER["25"]["inputs"]


def test_inject_ksampler_does_not_mutate_original():
    from src.workflow_injector import inject_ksampler

    workflow = copy.deepcopy(STUB_WORKFLOW_KSAMPLER)
    inject_ksampler(workflow, denoise=0.1)

    assert workflow["25"]["inputs"]["denoise"] == 0.8


# --- inject_prompter tests ---

STUB_WORKFLOW_PROMPTER = {
    "44": {
        "class_type": "Nortus_Prompter_NodeInput",
        "inputs": {
            "trigger_word": "old_trigger",
            "model_size": "auto",
            "character_details": "",
        },
    },
}


def test_inject_prompter_sets_all_fields():
    from src.workflow_injector import inject_prompter

    workflow = copy.deepcopy(STUB_WORKFLOW_PROMPTER)
    result = inject_prompter(
        workflow,
        trigger_word="new_trig",
        model_size="large",
        character_details="tall, blonde",
    )

    inputs = result["44"]["inputs"]
    assert inputs["trigger_word"] == "new_trig"
    assert inputs["model_size"] == "large"
    assert inputs["character_details"] == "tall, blonde"


def test_inject_prompter_partial_only_writes_provided():
    from src.workflow_injector import inject_prompter

    workflow = copy.deepcopy(STUB_WORKFLOW_PROMPTER)
    result = inject_prompter(workflow, trigger_word="trig_only")

    assert result["44"]["inputs"]["trigger_word"] == "trig_only"
    assert result["44"]["inputs"]["model_size"] == "auto"
    assert result["44"]["inputs"]["character_details"] == ""


def test_inject_prompter_no_args_unchanged():
    from src.workflow_injector import inject_prompter

    workflow = copy.deepcopy(STUB_WORKFLOW_PROMPTER)
    result = inject_prompter(workflow)

    assert result["44"]["inputs"] == STUB_WORKFLOW_PROMPTER["44"]["inputs"]


def test_inject_prompter_does_not_mutate_original():
    from src.workflow_injector import inject_prompter

    workflow = copy.deepcopy(STUB_WORKFLOW_PROMPTER)
    inject_prompter(workflow, trigger_word="x")

    assert workflow["44"]["inputs"]["trigger_word"] == "old_trigger"


# --- inject_video_settings tests ---

def test_inject_video_settings_sets_all_fields():
    from src.workflow_injector import inject_video_settings

    workflow = copy.deepcopy(STUB_WORKFLOW)
    result = inject_video_settings(
        workflow, start_second=5, frame_count=20, selected_frame=3
    )

    inputs = result["43"]["inputs"]
    assert inputs["start_second"] == 5
    assert inputs["frame_count"] == 20
    assert inputs["selected_frame"] == 3
    assert inputs["video"] == "placeholder.mp4"


def test_inject_video_settings_partial():
    from src.workflow_injector import inject_video_settings

    workflow = copy.deepcopy(STUB_WORKFLOW)
    result = inject_video_settings(workflow, frame_count=15)

    assert result["43"]["inputs"]["frame_count"] == 15
    assert result["43"]["inputs"]["start_second"] == 0
    assert result["43"]["inputs"]["selected_frame"] == 0


def test_inject_video_settings_no_args_unchanged():
    from src.workflow_injector import inject_video_settings

    workflow = copy.deepcopy(STUB_WORKFLOW)
    result = inject_video_settings(workflow)

    assert result["43"]["inputs"] == STUB_WORKFLOW["43"]["inputs"]


def test_inject_video_settings_does_not_mutate_original():
    from src.workflow_injector import inject_video_settings

    workflow = copy.deepcopy(STUB_WORKFLOW)
    inject_video_settings(workflow, start_second=99)

    assert workflow["43"]["inputs"]["start_second"] == 0


# --- inject_node43_image tests ---

STUB_WORKFLOW_MODEL_REF = {
    "37": {
        "class_type": "LoadImage",
        "inputs": {"image": "char.png"},
    },
    "43": {
        "class_type": "LoadImage",
        "inputs": {"image": ""},
    },
}


def test_inject_node43_image_sets_node_43_image():
    from src.workflow_injector import inject_node43_image

    workflow = copy.deepcopy(STUB_WORKFLOW_MODEL_REF)
    result = inject_node43_image(workflow, filename="model.png")

    assert result["43"]["inputs"]["image"] == "model.png"
    # Node 37 (character ref) must remain untouched
    assert result["37"]["inputs"]["image"] == "char.png"


def test_inject_node43_image_does_not_mutate_original():
    from src.workflow_injector import inject_node43_image

    workflow = copy.deepcopy(STUB_WORKFLOW_MODEL_REF)
    inject_node43_image(workflow, filename="model.png")

    assert workflow["43"]["inputs"]["image"] == ""
