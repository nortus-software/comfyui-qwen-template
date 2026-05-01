import logging
import os
import uuid

from source_downloader import download_source
from lora_cache import get_lora_path

log = logging.getLogger("pipeline")

OUTPUT_NODE_ID = "35"


def download_and_upload_image(uri: str, prefix: str, ctx) -> str:
    """Download a source URI and upload it to ComfyUI under a uuid-tagged filename."""
    log.info("Downloading %s: %s", prefix, uri)
    data, ext = download_source(uri, gcs_client=ctx.gcs)
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}{ext}"
    ctx.comfyui.upload_image(data, filename)
    log.info("Uploaded as %s (%d bytes)", filename, len(data))
    return filename


def setup_lora(uri: str, ctx) -> tuple[str, str]:
    """Resolve a LoRA via the cache and symlink it into ComfyUI's loras dir.

    Returns (lora_filename, symlink_path). Caller owns cleanup of the symlink.
    """
    log.info("Resolving LoRA: %s", uri)
    cached_path, lora_filename = get_lora_path(uri, gcs_client=ctx.gcs)
    dest_path = os.path.join(ctx.config.comfyui_dir, "models", "loras", lora_filename)
    if os.path.lexists(dest_path):
        os.remove(dest_path)
    os.symlink(cached_path, dest_path)
    log.info("LoRA symlinked to %s", dest_path)
    return lora_filename, dest_path


def submit_and_fetch_output(workflow: dict, ctx) -> bytes:
    """Submit a workflow, poll, and return the first output image bytes from node 35."""
    prompt_id = ctx.comfyui.submit_prompt(workflow)
    log.info("Submitted prompt: %s", prompt_id)

    outputs = ctx.comfyui.poll_until_complete(prompt_id)
    log.info("ComfyUI completed. Output nodes: %s", list(outputs.keys()))

    if OUTPUT_NODE_ID not in outputs:
        raise ValueError(
            f"No output found at node {OUTPUT_NODE_ID}. Available: {list(outputs.keys())}"
        )

    images = outputs[OUTPUT_NODE_ID].get("images", [])
    if not images:
        raise ValueError("No images in output")

    first = images[0]
    out_bytes = ctx.comfyui.get_output_image(
        first["filename"], first.get("subfolder", ""), first.get("type", "output")
    )
    log.info("Got output image: %d bytes", len(out_bytes))
    return out_bytes


def upload_output(output_bytes: bytes, ctx) -> dict:
    """Upload output bytes to GCS and return a dict with signed URL and key."""
    if not ctx.gcs:
        raise ValueError("GCS_BUCKET not configured")
    output_key = f"{ctx.gcs_output_path.rstrip('/')}/output_{ctx.job_id}.png"
    ctx.gcs.upload_bytes(output_bytes, output_key, content_type="image/png")
    signed_url = ctx.gcs.get_signed_url(output_key, expiry=ctx.config.gcs_signed_url_expiry)
    log.info("Uploaded to GCS: %s", output_key)
    return {"output_url": signed_url, "gcs_output_path": output_key}
