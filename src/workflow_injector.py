import copy
import json

# Node IDs from the workflow
LOAD_IMAGE_NODE = "37"
VIDEO_FRAME_NODE = "43"
KSAMPLER_NODE = "25"
PROMPTER_NODE = "44"


def load_workflow(path: str) -> dict:
    """Load workflow JSON (API format) from disk."""
    with open(path) as f:
        return json.load(f)


def inject_reference(workflow: dict, media_type: str, filename: str) -> dict:
    """Inject a reference image or video filename into the workflow."""
    workflow = copy.deepcopy(workflow)

    if media_type == "image":
        workflow[LOAD_IMAGE_NODE]["inputs"]["image"] = filename
    elif media_type == "video":
        workflow[VIDEO_FRAME_NODE]["inputs"]["video"] = filename
    else:
        raise ValueError(f"Unsupported media type: {media_type}")

    return workflow


def inject_lora(workflow: dict, lora_name: str, strength_model: float = 0.85) -> dict:
    """Update the LoRA filename in the workflow's existing LoRA loader node."""
    workflow = copy.deepcopy(workflow)

    for node in workflow.values():
        if "LoraLoader" in node.get("class_type", ""):
            node["inputs"]["lora_name"] = lora_name
            node["inputs"]["strength_model"] = strength_model
            return workflow

    raise ValueError("No LoRA loader node found in workflow")


def inject_ksampler(workflow: dict, denoise=None, cfg=None, **_ignored) -> dict:
    """Override ksampler params. Only provided fields are written."""
    workflow = copy.deepcopy(workflow)
    inputs = workflow[KSAMPLER_NODE]["inputs"]
    if denoise is not None:
        inputs["denoise"] = denoise
    if cfg is not None:
        inputs["cfg"] = cfg
    return workflow


def inject_prompter(
    workflow: dict,
    trigger_word=None,
    model_size=None,
    character_details=None,
    **_ignored,
) -> dict:
    """Override prompter params. Only provided fields are written."""
    workflow = copy.deepcopy(workflow)
    inputs = workflow[PROMPTER_NODE]["inputs"]
    if trigger_word is not None:
        inputs["trigger_word"] = trigger_word
    if model_size is not None:
        inputs["model_size"] = model_size
    if character_details is not None:
        inputs["character_details"] = character_details
    return workflow


def inject_video_settings(
    workflow: dict,
    start_second=None,
    frame_count=None,
    selected_frame=None,
    **_ignored,
) -> dict:
    """Override video frame extractor params. Only provided fields are written."""
    workflow = copy.deepcopy(workflow)
    inputs = workflow[VIDEO_FRAME_NODE]["inputs"]
    if start_second is not None:
        inputs["start_second"] = start_second
    if frame_count is not None:
        inputs["frame_count"] = frame_count
    if selected_frame is not None:
        inputs["selected_frame"] = selected_frame
    return workflow
