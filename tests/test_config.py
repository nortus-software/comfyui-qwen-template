import os
import pytest


def test_config_loads_defaults():
    """Config should have sensible defaults when env vars are not set."""
    # Clear any existing env vars
    env_vars = ["GCS_BUCKET", "GCS_SIGNED_URL_EXPIRY", "WORKFLOW_PATH", "COMFYUI_URL"]
    original = {k: os.environ.pop(k, None) for k in env_vars}

    try:
        # Re-import to pick up cleared env
        import importlib
        import src.config as config_module
        importlib.reload(config_module)
        from src.config import Config

        config = Config()
        assert config.comfyui_url == "http://127.0.0.1:8188"
        assert config.workflow_path == "/ComfyUI/user/default/workflows/workflow.json"
        assert config.gcs_signed_url_expiry == 3600
        assert config.gcs_bucket is None
    finally:
        for k, v in original.items():
            if v is not None:
                os.environ[k] = v


def test_config_reads_env_vars():
    """Config should read values from environment variables."""
    os.environ["GCS_BUCKET"] = "test-bucket"
    os.environ["GCS_SIGNED_URL_EXPIRY"] = "7200"
    os.environ["WORKFLOW_PATH"] = "/custom/workflow.json"
    os.environ["COMFYUI_URL"] = "http://localhost:9999"

    try:
        import importlib
        import src.config as config_module
        importlib.reload(config_module)
        from src.config import Config

        config = Config()
        assert config.gcs_bucket == "test-bucket"
        assert config.gcs_signed_url_expiry == 7200
        assert config.workflow_path == "/custom/workflow.json"
        assert config.comfyui_url == "http://localhost:9999"
    finally:
        for k in ["GCS_BUCKET", "GCS_SIGNED_URL_EXPIRY", "WORKFLOW_PATH", "COMFYUI_URL"]:
            os.environ.pop(k, None)
