import base64
import os
import re
import requests


def download_source(source: str, gcs_client=None) -> tuple[bytes, str]:
    """
    Download media from a source URL/URI/base64 string.
    Returns (bytes, file_extension).
    """
    if source.startswith("data:"):
        # base64 encoded: data:image/png;base64,<data>
        match = re.match(r"data:(\w+)/(\w+);base64,(.+)", source)
        if not match:
            raise ValueError(f"Invalid data URI: {source[:50]}")
        ext = f".{match.group(2)}"
        data = base64.b64decode(match.group(3))
        return data, ext

    if source.startswith("gs://"):
        if gcs_client is None:
            raise ValueError("GCS client required for gs:// sources")
        # gs://bucket/path/to/file.ext -> path/to/file.ext
        path = source.split("/", 3)[3]  # skip gs://bucket/
        ext = os.path.splitext(source)[1]
        data = gcs_client.download_bytes(path)
        return data, ext

    if source.startswith("http://") or source.startswith("https://"):
        resp = requests.get(source)
        resp.raise_for_status()
        ext = os.path.splitext(source.split("?")[0])[1] or ".png"
        return resp.content, ext

    raise ValueError(f"Unsupported source format: {source[:50]}")


def download_source_to_file(source: str, dest_path: str, gcs_client=None) -> str:
    """Stream-download a source to a file. Returns file extension."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    if source.startswith("gs://"):
        if gcs_client is None:
            raise ValueError("GCS client required for gs:// sources")
        path = source.split("/", 3)[3]
        ext = os.path.splitext(source)[1]
        gcs_client.download_to_file(path, dest_path)
        return ext

    if source.startswith("http://") or source.startswith("https://"):
        resp = requests.get(source, stream=True)
        resp.raise_for_status()
        ext = os.path.splitext(source.split("?")[0])[1] or ".bin"
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return ext

    raise ValueError(f"Unsupported source format for streaming: {source[:50]}")
