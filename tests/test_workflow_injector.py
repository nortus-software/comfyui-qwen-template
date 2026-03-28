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
