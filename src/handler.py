import os
import shutil
import uuid

from comfyui_client import ComfyUIClient
from config import Config
from gcs import GCSClient
from source_downloader import download_source
from workflow_injector import inject_lora, inject_reference, load_workflow

# Output node ID in the workflow (PreviewImage / SaveImage)
OUTPUT_NODE_ID = "35"


def handler(event: dict) -> dict:
    """RunPod serverless handler. Processes one image/video generation job."""
    lora_local_path = None
    lora_dest_path = None

    try:
        job_input = event.get("input", {})

        media_type = job_input.get("type")
        source = job_input.get("source")
        gcs_output_path = job_input.get("gcs_output_path", "outputs/")

        if not media_type or not source:
            return {"error": "Missing required fields: 'type' and 'source'"}

        # LoRA is required
        lora_uri = job_input.get("lora")
        if not lora_uri:
            return {"error": "Missing required field: 'lora'"}

        config = Config()
        comfyui = ComfyUIClient(config.comfyui_url)
        gcs = GCSClient(config.gcs_bucket) if config.gcs_bucket else None

        # 1. Download source media
        media_bytes, ext = download_source(source, gcs_client=gcs)

        # 2. Download LoRA from GCS and copy to ComfyUI models dir
        lora_bytes, _ = download_source(lora_uri, gcs_client=gcs)
        lora_filename = os.path.basename(lora_uri)
        lora_local_path = os.path.join("/tmp", "loras", lora_filename)
        os.makedirs(os.path.dirname(lora_local_path), exist_ok=True)
        with open(lora_local_path, "wb") as f:
            f.write(lora_bytes)

        lora_dest_path = os.path.join(config.comfyui_dir, "models", "loras", lora_filename)
        shutil.copy2(lora_local_path, lora_dest_path)

        # 3. Upload source to ComfyUI
        filename = f"ref_{uuid.uuid4().hex[:8]}{ext}"
        comfyui.upload_image(media_bytes, filename)

        # 4. Load and inject workflow
        workflow = load_workflow(config.workflow_path)
        workflow = inject_reference(
            workflow,
            media_type=media_type,
            filename=filename,
            frame_start=job_input.get("frame_start", 0),
            frame_end=job_input.get("frame_end", 10),
            frame_step=job_input.get("frame_step", 1),
        )
        workflow = inject_lora(workflow, lora_name=lora_filename)

        # 5. Submit to ComfyUI
        prompt_id = comfyui.submit_prompt(workflow)

        # 6. Poll for completion
        outputs = comfyui.poll_until_complete(prompt_id)

        # 7. Get output image
        if OUTPUT_NODE_ID not in outputs:
            return {"error": f"No output found at node {OUTPUT_NODE_ID}"}

        output_images = outputs[OUTPUT_NODE_ID].get("images", [])
        if not output_images:
            return {"error": "No images in output"}

        first_output = output_images[0]
        output_bytes = comfyui.get_output_image(
            first_output["filename"],
            first_output.get("subfolder", ""),
            first_output.get("type", "output"),
        )

        # 8. Upload to GCS
        if gcs:
            output_key = f"{gcs_output_path.rstrip('/')}/output_{prompt_id}.png"
            gcs.upload_bytes(output_bytes, output_key, content_type="image/png")
            signed_url = gcs.get_signed_url(output_key, expiry=config.gcs_signed_url_expiry)
            return {"output_url": signed_url}

        return {"error": "GCS_BUCKET not configured"}

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Cleanup LoRA files to avoid filling disk across jobs
        if lora_local_path and os.path.exists(lora_local_path):
            os.remove(lora_local_path)
        if lora_dest_path and os.path.exists(lora_dest_path):
            os.remove(lora_dest_path)


# RunPod entry point — only runs when executed directly
if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
