import pytest


def test_registry_contains_first_frame():
    from src.workflows import WORKFLOWS

    assert "first_frame" in WORKFLOWS
    wf = WORKFLOWS["first_frame"]
    assert wf.name == "first_frame"
    assert wf.filename == "workflow_first_frame_api.json"
    assert callable(wf.process)


def test_registry_contains_first_frame_image():
    from src.workflows import WORKFLOWS

    assert "first_frame_image" in WORKFLOWS
    wf = WORKFLOWS["first_frame_image"]
    assert wf.name == "first_frame_image"
    assert wf.filename == "workflow_first_frame_image_api.json"
    assert callable(wf.process)


def test_default_workflow_is_first_frame():
    from src.workflows import DEFAULT_WORKFLOW

    assert DEFAULT_WORKFLOW == "first_frame"


def test_get_workflow_def_returns_for_known_name():
    from src.workflows import get_workflow_def

    wf = get_workflow_def("first_frame_image")
    assert wf.name == "first_frame_image"


def test_get_workflow_def_unknown_raises():
    from src.workflows import get_workflow_def

    with pytest.raises(ValueError, match="Unknown workflow"):
        get_workflow_def("does_not_exist")


def test_get_workflow_def_default_when_none():
    from src.workflows import get_workflow_def

    wf = get_workflow_def(None)
    assert wf.name == "first_frame"
