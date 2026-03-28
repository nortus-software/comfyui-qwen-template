import io
import json
import time
import requests


class ComfyUIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def upload_image(self, image_bytes: bytes, filename: str) -> dict:
        """Upload an image to ComfyUI. Returns {"name", "subfolder", "type"}."""
        resp = requests.post(
            f"{self.base_url}/upload/image",
            files={"image": (filename, io.BytesIO(image_bytes), "image/png")},
            data={"overwrite": "true"},
        )
        resp.raise_for_status()
        return resp.json()

    def submit_prompt(self, workflow: dict) -> str:
        """Submit a workflow to ComfyUI. Returns prompt_id."""
        resp = requests.post(
            f"{self.base_url}/prompt",
            data=json.dumps({"prompt": workflow}),
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()["prompt_id"]

    def poll_until_complete(self, prompt_id: str, poll_interval: float = 2.0, timeout: float = 600.0) -> dict:
        """Poll /history/{prompt_id} until the job completes. Returns outputs dict."""
        start = time.time()
        while time.time() - start < timeout:
            resp = requests.get(f"{self.base_url}/history/{prompt_id}")
            resp.raise_for_status()
            history = resp.json()
            if prompt_id in history:
                return history[prompt_id]["outputs"]
            time.sleep(poll_interval)
        raise TimeoutError(f"ComfyUI job {prompt_id} did not complete within {timeout}s")

    def get_output_image(self, filename: str, subfolder: str, output_type: str) -> bytes:
        """Download an output image from ComfyUI."""
        resp = requests.get(
            f"{self.base_url}/view",
            params={"filename": filename, "subfolder": subfolder, "type": output_type},
        )
        resp.raise_for_status()
        return resp.content
