import logging
import os
import shutil
import uuid

from comfyui_client import ComfyUIClient
from config import Config
from gcs import GCSClient
from source_downloader import download_source
from workflow_injector import inject_lora, inject_reference, load_workflow

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("handler")

# Output node ID in the workflow (PreviewImage / SaveImage)
OUTPUT_NODE_ID = "35"


def handler(event: dict) -> dict:
    """RunPod serverless handler. Processes one image/video generation job."""
    lora_local_path = None
    lora_dest_path = None

    try:
        job_input = event.get("input", {})
        log.info("Job input: %s", job_input)

        media_type = job_input.get("type")
        source = job_input.get("source")
        reference_image_uri = job_input.get("reference_image")
        lora_uri = job_input.get("lora")
        gcs_output_path = job_input.get("gcs_output_path", "outputs/")

        missing = [k for k, v in {
            "type": media_type,
            "source": source,
            "reference_image": reference_image_uri,
            "lora": lora_uri,
        }.items() if not v]
        if missing:
            return {"error": f"Missing required fields: {missing}"}
        if media_type not in ("image", "video"):
            return {"error": f"Invalid type: {media_type}. Must be 'image' or 'video'"}

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

        # 3. Download LoRA and copy to ComfyUI models dir
        log.info("Downloading LoRA: %s", lora_uri)
        lora_bytes, _ = download_source(lora_uri, gcs_client=gcs)
        lora_filename = os.path.basename(lora_uri)
        lora_local_path = os.path.join("/tmp", "loras", lora_filename)
        os.makedirs(os.path.dirname(lora_local_path), exist_ok=True)
        with open(lora_local_path, "wb") as f:
            f.write(lora_bytes)

        lora_dest_path = os.path.join(config.comfyui_dir, "models", "loras", lora_filename)
        shutil.copy2(lora_local_path, lora_dest_path)
        log.info("LoRA copied to %s", lora_dest_path)

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
            frame_start=job_input.get("frame_start", 0),
            frame_end=job_input.get("frame_end", 2),
            frame_step=job_input.get("frame_step", 1),
        )
        workflow = inject_lora(workflow, lora_name=lora_filename)
        log.info("Workflow loaded and injected")

        # 5. Submit to ComfyUI
        prompt_id = comfyui.submit_prompt(workflow)
        log.info("Submitted prompt: %s", prompt_id)

        # 6. Poll for completion
        outputs = comfyui.poll_until_complete(prompt_id)
        log.info("ComfyUI completed. Output nodes: %s", list(outputs.keys()))

        # 7. Get output image
        if OUTPUT_NODE_ID not in outputs:
            return {"error": f"No output found at node {OUTPUT_NODE_ID}. Available: {list(outputs.keys())}"}

        output_images = outputs[OUTPUT_NODE_ID].get("images", [])
        if not output_images:
            return {"error": "No images in output"}

        first_output = output_images[0]
        output_bytes = comfyui.get_output_image(
            first_output["filename"],
            first_output.get("subfolder", ""),
            first_output.get("type", "output"),
        )
        log.info("Got output image: %d bytes", len(output_bytes))

        # 8. Upload to GCS
        if gcs:
            output_key = f"{gcs_output_path.rstrip('/')}/output_{prompt_id}.png"
            gcs.upload_bytes(output_bytes, output_key, content_type="image/png")
            signed_url = gcs.get_signed_url(output_key, expiry=config.gcs_signed_url_expiry)
            log.info("Uploaded to GCS: %s", output_key)
            return {"output_url": signed_url}

        return {"error": "GCS_BUCKET not configured"}

    except Exception as e:
        log.exception("Handler failed")
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
