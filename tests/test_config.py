import os
from unittest.mock import patch
import pytest


def test_config_loads_defaults():
    """Config should have sensible defaults when env vars are not set."""
    # Clear any existing env vars
    env_vars = ["GCS_BUCKET", "GCS_SIGNED_URL_EXPIRY", "WORKFLOWS_DIR", "COMFYUI_URL"]
    original = {k: os.environ.pop(k, None) for k in env_vars}

    try:
        # Re-import to pick up cleared env
        import importlib
        import src.config as config_module
        importlib.reload(config_module)
        from src.config import Config

        config = Config()
        assert config.comfyui_url == "http://127.0.0.1:8188"
        assert config.workflows_dir == "/ComfyUI/user/default/workflows/"
        assert config.gcs_signed_url_expiry == 3600
        assert config.gcs_bucket is None
        assert config.comfyui_dir == "/ComfyUI"
    finally:
        for k, v in original.items():
            if v is not None:
                os.environ[k] = v


def test_config_reads_env_vars():
    """Config should read values from environment variables."""
    os.environ["GCS_BUCKET"] = "test-bucket"
    os.environ["GCS_SIGNED_URL_EXPIRY"] = "7200"
    os.environ["WORKFLOWS_DIR"] = "/custom/wf/"
    os.environ["COMFYUI_URL"] = "http://localhost:9999"

    try:
        import importlib
        import src.config as config_module
        importlib.reload(config_module)
        from src.config import Config

        config = Config()
        assert config.gcs_bucket == "test-bucket"
        assert config.gcs_signed_url_expiry == 7200
        assert config.workflows_dir == "/custom/wf/"
        assert config.comfyui_url == "http://localhost:9999"
    finally:
        for k in ["GCS_BUCKET", "GCS_SIGNED_URL_EXPIRY", "WORKFLOWS_DIR", "COMFYUI_URL"]:
            os.environ.pop(k, None)


def test_config_workflows_dir_default():
    from src.config import Config
    with patch.dict("os.environ", {}, clear=True):
        cfg = Config()
        assert cfg.workflows_dir == "/ComfyUI/user/default/workflows/"


def test_config_workflows_dir_from_env():
    from src.config import Config
    with patch.dict("os.environ", {"WORKFLOWS_DIR": "/tmp/wf/"}, clear=True):
        cfg = Config()
        assert cfg.workflows_dir == "/tmp/wf/"


def test_config_no_longer_has_workflow_path():
    """Old single-workflow path attribute is removed."""
    from src.config import Config
    cfg = Config()
    assert not hasattr(cfg, "workflow_path")
