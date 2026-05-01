import logging
import os
from dataclasses import dataclass, field

log = logging.getLogger("config")


@dataclass
class Config:
    comfyui_url: str = field(default_factory=lambda: os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188"))
    workflows_dir: str = field(default_factory=lambda: os.environ.get("WORKFLOWS_DIR", "/ComfyUI/user/default/workflows/"))
    gcs_bucket: str | None = field(default_factory=lambda: os.environ.get("GCS_BUCKET"))
    gcs_signed_url_expiry: int = field(default_factory=lambda: int(os.environ.get("GCS_SIGNED_URL_EXPIRY", "3600")))
    comfyui_dir: str = field(default_factory=lambda: os.environ.get("COMFYUI_DIR", "/ComfyUI"))

    def __post_init__(self):
        if os.environ.get("WORKFLOW_PATH"):
            log.warning(
                "WORKFLOW_PATH is set but no longer used. "
                "Set WORKFLOWS_DIR (default %s) and select via the 'workflow' field on the job input.",
                self.workflows_dir,
            )
