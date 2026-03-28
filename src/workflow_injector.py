import copy
import json

# Node IDs from the workflow
LOAD_IMAGE_NODE = "37"
VIDEO_FRAME_NODE = "43"


def load_workflow(path: str) -> dict:
    """Load workflow JSON from disk."""
    with open(path) as f:
        data = json.load(f)
    # ComfyUI API expects prompt format keyed by node ID.
    # If the workflow has a "nodes" array (UI format), we need to convert.
    if "nodes" in data:
        return _convert_ui_to_api(data)
    return data


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
        workflow[LOAD_IMAGE_NODE]["widgets_values"][0] = filename
    elif media_type == "video":
        wv = workflow[VIDEO_FRAME_NODE]["widgets_values"]
        wv[0] = filename
        wv[1] = frame_start
        wv[2] = frame_end
        wv[3] = frame_step
    else:
        raise ValueError(f"Unsupported media type: {media_type}")

    return workflow


def _convert_ui_to_api(ui_workflow: dict) -> dict:
    """Convert ComfyUI UI-format workflow to API-format prompt.

    UI format has a "nodes" array with objects containing "id", "class_type", etc.
    API format is a dict keyed by string node IDs.
    """
    api_prompt = {}
    for node in ui_workflow["nodes"]:
        node_id = str(node["id"])
        api_prompt[node_id] = {
            "class_type": node["type"],
            "inputs": {inp["name"]: inp.get("link") for inp in node.get("inputs", [])},
            "widgets_values": node.get("widgets_values", []),
        }
    return api_prompt
