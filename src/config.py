import os
from dataclasses import dataclass, field


@dataclass
class Config:
    comfyui_url: str = field(default_factory=lambda: os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188"))
    workflow_path: str = field(default_factory=lambda: os.environ.get("WORKFLOW_PATH", "/ComfyUI/user/default/workflows/workflow.json"))
    gcs_bucket: str | None = field(default_factory=lambda: os.environ.get("GCS_BUCKET"))
    gcs_signed_url_expiry: int = field(default_factory=lambda: int(os.environ.get("GCS_SIGNED_URL_EXPIRY", "3600")))
