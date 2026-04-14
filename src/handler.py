import logging
import os
import uuid

from comfyui_client import ComfyUIClient
from config import Config
from gcs import GCSClient
from lora_cache import get_lora_path
from source_downloader import download_source
from webhook import send_webhook
from workflow_injector import (
    inject_ksampler,
    inject_lora,
    inject_prompter,
    inject_reference,
    inject_video_settings,
    load_workflow,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("handler")

# Output node ID in the workflow (PreviewImage / SaveImage)
OUTPUT_NODE_ID = "35"


def handler(event: dict) -> dict:
    """RunPod serverless handler. Processes one image/video generation job."""
    lora_dest_path = None

    try:
        job_input = event.get("input", {})
        job_id = event.get("id", "unknown")
        log.info("Job input: %s", job_input)

        media_type = job_input.get("type")
        source = job_input.get("source")
        reference_image_uri = job_input.get("reference_image")
        lora_uri = job_input.get("lora")
        gcs_output_path = job_input.get("gcs_output_path", "outputs/")
        settings = job_input.get("settings") or {}
        webhook_url = job_input.get("webhook_url")

        def _notify(result: dict, success: bool) -> dict:
            if not webhook_url:
                return result
            if success:
                payload = {
                    "status": "completed",
                    "output_url": result["output_url"],
                    "metadata": {
                        "job_id": job_id,
                        "media_type": media_type,
                        "gcs_output_path": result.get("gcs_output_path", ""),
                    },
                }
            else:
                payload = {"status": "failed", "error": result["error"]}
            if not send_webhook(webhook_url, payload):
                result["webhook_failed"] = True
            return result

        missing = [k for k, v in {
            "type": media_type,
            "source": source,
            "reference_image": reference_image_uri,
            "lora": lora_uri,
        }.items() if not v]
        if missing:
            return _notify({"error": f"Missing required fields: {missing}"}, success=False)
        if media_type not in ("image", "video"):
            return _notify({"error": f"Invalid type: {media_type}. Must be 'image' or 'video'"}, success=False)

        config = Config()
        comfyui = ComfyUIClient(config.comfyui_url)
        gcs = GCSClient(config.gcs_bucket) if config.gcs_bucket else None

        # 1. Download reference image (always goes to node 37)
        log.info("Downloading reference image: %s", reference_image_uri)
        ref_bytes, ref_ext = download_source(reference_image_uri, gcs_client=gcs)
        log.info("Reference image: %d bytes", len(ref_bytes))

        # 2. Download source (image or video, goes to matching node)
        log.info("Downloading source (%s): %s", media_type, source)
        source_bytes, source_ext = download_source(source, gcs_client=gcs)
        log.info("Source: %d bytes", len(source_bytes))

        # 3. Resolve LoRA via cache (downloads on miss, hits warm cache otherwise)
        log.info("Resolving LoRA: %s", lora_uri)
        cached_lora_path, lora_filename = get_lora_path(lora_uri, gcs_client=gcs)
        lora_dest_path = os.path.join(config.comfyui_dir, "models", "loras", lora_filename)
        if os.path.lexists(lora_dest_path):
            os.remove(lora_dest_path)
        os.symlink(cached_lora_path, lora_dest_path)
        log.info("LoRA symlinked to %s", lora_dest_path)

        # 4. Upload reference image and source to ComfyUI
        ref_filename = f"ref_{uuid.uuid4().hex[:8]}{ref_ext}"
        comfyui.upload_image(ref_bytes, ref_filename)
        log.info("Uploaded reference image as %s", ref_filename)

        source_filename = f"src_{uuid.uuid4().hex[:8]}{source_ext}"
        comfyui.upload_image(source_bytes, source_filename)
        log.info("Uploaded source as %s", source_filename)

        # 5. Load and inject workflow
        workflow = load_workflow(config.workflow_path)
        # Reference image always goes into node 37 (character reference)
        workflow = inject_reference(workflow, media_type="image", filename=ref_filename)
        # Source goes into matching node (image → 37 override, video → 43)
        workflow = inject_reference(
            workflow,
            media_type=media_type,
            filename=source_filename,
        )
        workflow = inject_lora(workflow, lora_name=lora_filename)
        workflow = inject_ksampler(workflow, **settings.get("ksampler", {}))
        workflow = inject_prompter(workflow, **settings.get("prompter", {}))
        workflow = inject_video_settings(workflow, **settings.get("video", {}))
        log.info("Workflow loaded and injected")

        # 5. Submit to ComfyUI
        prompt_id = comfyui.submit_prompt(workflow)
        log.info("Submitted prompt: %s", prompt_id)

        # 6. Poll for completion
        outputs = comfyui.poll_until_complete(prompt_id)
        log.info("ComfyUI completed. Output nodes: %s", list(outputs.keys()))

        # 7. Get output image
        if OUTPUT_NODE_ID not in outputs:
            return _notify({"error": f"No output found at node {OUTPUT_NODE_ID}. Available: {list(outputs.keys())}"}, success=False)

        output_images = outputs[OUTPUT_NODE_ID].get("images", [])
        if not output_images:
            return _notify({"error": "No images in output"}, success=False)

        first_output = output_images[0]
        output_bytes = comfyui.get_output_image(
            first_output["filename"],
            first_output.get("subfolder", ""),
            first_output.get("type", "output"),
        )
        log.info("Got output image: %d bytes", len(output_bytes))

        # 8. Upload to GCS
        if gcs:
            output_key = f"{gcs_output_path.rstrip('/')}/output_{job_id}.png"
            gcs.upload_bytes(output_bytes, output_key, content_type="image/png")
            signed_url = gcs.get_signed_url(output_key, expiry=config.gcs_signed_url_expiry)
            log.info("Uploaded to GCS: %s", output_key)
            return _notify({"output_url": signed_url, "gcs_output_path": output_key}, success=True)

        return _notify({"error": "GCS_BUCKET not configured"}, success=False)

    except Exception as e:
        log.exception("Handler failed")
        return _notify({"error": str(e)}, success=False)

    finally:
        # Cleanup the symlink in ComfyUI's models dir; cache blob persists for reuse
        if lora_dest_path and os.path.lexists(lora_dest_path):
            os.remove(lora_dest_path)


# RunPod entry point — only runs when executed directly
if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
