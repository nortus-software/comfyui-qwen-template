import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

from workflow_injector import (
    inject_ksampler,
    inject_lora,
    inject_model_reference,
    inject_prompter,
    inject_reference,
    inject_video_settings,
    load_workflow,
)
from pipeline import (
    download_and_upload_image,
    setup_lora,
    submit_and_fetch_output,
    upload_output,
)

log = logging.getLogger("workflows")


@dataclass(frozen=True)
class WorkflowDef:
    name: str
    filename: str
    process: Callable[[dict, object], dict]


def process_first_frame(job_input: dict, ctx) -> dict:
    """Run the original first-frame workflow (image or video source)."""
    media_type = job_input.get("type")
    source = job_input.get("source")
    reference_image = job_input.get("reference_image")
    lora_uri = job_input.get("lora")
    settings = job_input.get("settings") or {}

    missing = [k for k, v in {
        "type": media_type,
        "source": source,
        "reference_image": reference_image,
        "lora": lora_uri,
    }.items() if not v]
    if missing:
        return {"error": f"Missing required fields: {missing}"}
    if media_type not in ("image", "video"):
        return {"error": f"Invalid type: {media_type}. Must be 'image' or 'video'"}

    ref_filename = download_and_upload_image(reference_image, "ref", ctx)
    src_filename = download_and_upload_image(source, "src", ctx)
    lora_filename, lora_dest = setup_lora(lora_uri, ctx)
    ctx.lora_dest_path = lora_dest

    wf_def = WORKFLOWS["first_frame"]
    workflow_path = os.path.join(ctx.config.workflows_dir, wf_def.filename)
    workflow = load_workflow(workflow_path)
    workflow = inject_reference(workflow, media_type="image", filename=ref_filename)
    workflow = inject_reference(workflow, media_type=media_type, filename=src_filename)
    workflow = inject_lora(workflow, lora_name=lora_filename)
    workflow = inject_ksampler(workflow, **settings.get("ksampler", {}))
    workflow = inject_prompter(workflow, **settings.get("prompter", {}))
    workflow = inject_video_settings(workflow, **settings.get("video", {}))

    output_bytes = submit_and_fetch_output(workflow, ctx)
    return upload_output(output_bytes, ctx)


def process_first_frame_image(job_input: dict, ctx) -> dict:
    """Run the image-to-image first-frame workflow."""
    reference_image = job_input.get("reference_image")
    model_reference = job_input.get("model_reference")
    lora_uri = job_input.get("lora")
    settings = job_input.get("settings") or {}

    missing = [k for k, v in {
        "reference_image": reference_image,
        "model_reference": model_reference,
        "lora": lora_uri,
    }.items() if not v]
    if missing:
        return {"error": f"Missing required fields: {missing}"}

    # Strict: reject fields that don't apply to this workflow.
    forbidden = []
    if "type" in job_input:
        forbidden.append("type")
    if "source" in job_input:
        forbidden.append("source")
    if "video" in settings:
        forbidden.append("settings.video")
    if forbidden:
        return {
            "error": (
                f"Workflow 'first_frame_image' does not accept fields: {forbidden}. "
                "Use 'first_frame' for image/video sources or 'settings.video'."
            )
        }

    ref_filename = download_and_upload_image(reference_image, "ref", ctx)
    model_filename = download_and_upload_image(model_reference, "model", ctx)
    lora_filename, lora_dest = setup_lora(lora_uri, ctx)
    ctx.lora_dest_path = lora_dest

    wf_def = WORKFLOWS["first_frame_image"]
    workflow_path = os.path.join(ctx.config.workflows_dir, wf_def.filename)
    workflow = load_workflow(workflow_path)
    workflow = inject_reference(workflow, media_type="image", filename=ref_filename)
    workflow = inject_model_reference(workflow, filename=model_filename)
    workflow = inject_lora(workflow, lora_name=lora_filename)
    workflow = inject_ksampler(workflow, **settings.get("ksampler", {}))
    workflow = inject_prompter(workflow, **settings.get("prompter", {}))

    output_bytes = submit_and_fetch_output(workflow, ctx)
    return upload_output(output_bytes, ctx)


WORKFLOWS = {
    "first_frame": WorkflowDef(
        name="first_frame",
        filename="workflow_first_frame_api.json",
        process=process_first_frame,
    ),
    "first_frame_image": WorkflowDef(
        name="first_frame_image",
        filename="workflow_first_frame_image_api.json",
        process=process_first_frame_image,
    ),
}

DEFAULT_WORKFLOW = "first_frame"


def get_workflow_def(name: Optional[str]) -> WorkflowDef:
    """Return the WorkflowDef for `name`, falling back to DEFAULT_WORKFLOW when None."""
    key = name or DEFAULT_WORKFLOW
    if key not in WORKFLOWS:
        raise ValueError(f"Unknown workflow: {key!r}. Known: {sorted(WORKFLOWS)}")
    return WORKFLOWS[key]
