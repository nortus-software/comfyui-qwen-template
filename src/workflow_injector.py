import copy
import json

# Node IDs from the workflow
LOAD_IMAGE_NODE = "37"
VIDEO_FRAME_NODE = "43"


def load_workflow(path: str) -> dict:
    """Load workflow JSON (API format) from disk."""
    with open(path) as f:
        return json.load(f)


def inject_reference(
    workflow: dict,
    media_type: str,
    filename: str,
    frame_start: int = 0,
    frame_end: int = 10,
    frame_step: int = 1,
) -> dict:
    """Inject a reference image or video filename into the workflow."""
    workflow = copy.deepcopy(workflow)

    if media_type == "image":
        workflow[LOAD_IMAGE_NODE]["inputs"]["image"] = filename
    elif media_type == "video":
        inputs = workflow[VIDEO_FRAME_NODE]["inputs"]
        inputs["video"] = filename
        inputs["start_second"] = frame_start
        inputs["frame_count"] = frame_end
        inputs["frame_interval"] = frame_step
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
