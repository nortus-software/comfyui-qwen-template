import logging
import os
from dataclasses import dataclass
from typing import Optional

from comfyui_client import ComfyUIClient
from config import Config
from gcs import GCSClient
from webhook import send_webhook
from workflows import WORKFLOWS, get_workflow_def

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("handler")


@dataclass
class JobContext:
    job_id: str
    config: object
    comfyui: object
    gcs: object
    gcs_output_path: str
    webhook_url: Optional[str]
    lora_dest_path: Optional[str] = None  # set by processors via setup_lora; cleaned in finally


def handler(event: dict) -> dict:
    """RunPod serverless handler. Routes to the workflow's processor."""
    job_input = event.get("input", {}) or {}
    job_id = event.get("id", "unknown")
    webhook_url = job_input.get("webhook_url")
    log.info("Job input: %s", job_input)

    def _notify(result: dict, success: bool) -> dict:
        if not webhook_url:
            return result
        if success:
            payload = {
                "status": "completed",
                "output_url": result["output_url"],
                "metadata": {
                    "job_id": job_id,
                    "workflow": job_input.get("workflow") or "first_frame",
                    "media_type": job_input.get("type"),
                    "gcs_output_path": result.get("gcs_output_path", ""),
                },
            }
        else:
            payload = {"status": "failed", "error": result["error"]}
        if not send_webhook(webhook_url, payload):
            result["webhook_failed"] = True
        return result

    ctx = None
    try:
        try:
            wf_def = get_workflow_def(job_input.get("workflow"))
        except ValueError as e:
            return _notify({"error": str(e)}, success=False)

        config = Config()
        comfyui = ComfyUIClient(config.comfyui_url)
        gcs = GCSClient(config.gcs_bucket) if config.gcs_bucket else None

        ctx = JobContext(
            job_id=job_id,
            config=config,
            comfyui=comfyui,
            gcs=gcs,
            gcs_output_path=job_input.get("gcs_output_path", "outputs/"),
            webhook_url=webhook_url,
        )

        result = wf_def.process(job_input, ctx)
        if "error" in result:
            return _notify(result, success=False)
        return _notify(result, success=True)

    except Exception as e:
        log.exception("Handler failed")
        return _notify({"error": str(e)}, success=False)

    finally:
        if ctx and ctx.lora_dest_path and os.path.lexists(ctx.lora_dest_path):
            os.remove(ctx.lora_dest_path)


# RunPod entry point
if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
