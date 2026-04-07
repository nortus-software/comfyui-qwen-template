import fcntl
import hashlib
import json
import logging
import os
import time

try:
    from source_downloader import download_source_to_file
except ImportError:
    from src.source_downloader import download_source_to_file

log = logging.getLogger("lora_cache")

CACHE_DIR = "/tmp/lora_cache"
BLOBS_DIR = os.path.join(CACHE_DIR, "blobs")
INDEX_PATH = os.path.join(CACHE_DIR, "index.json")
LOCK_PATH = os.path.join(CACHE_DIR, "index.lock")
BUDGET_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


def _ensure_dirs():
    os.makedirs(BLOBS_DIR, exist_ok=True)


def _key_for(uri: str) -> str:
    return hashlib.sha256(uri.encode("utf-8")).hexdigest()


def _ext_for(uri: str) -> str:
    return os.path.splitext(uri.split("?")[0])[1] or ".safetensors"


def _load_index() -> dict:
    if not os.path.exists(INDEX_PATH):
        return {}
    try:
        with open(INDEX_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        log.warning("Cache index corrupt; rebuilding")
        return _rebuild_index()


def _save_index(index: dict):
    tmp = INDEX_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(index, f)
    os.replace(tmp, INDEX_PATH)


def _rebuild_index() -> dict:
    index = {}
    if not os.path.isdir(BLOBS_DIR):
        return index
    for name in os.listdir(BLOBS_DIR):
        full = os.path.join(BLOBS_DIR, name)
        if not os.path.isfile(full):
            continue
        key = os.path.splitext(name)[0]
        st = os.stat(full)
        index[key] = {
            "uri": "",
            "filename": name,
            "size": st.st_size,
            "last_used_ts": st.st_mtime,
        }
    return index


class _Lock:
    def __enter__(self):
        _ensure_dirs()
        self.fh = open(LOCK_PATH, "w")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def _evict_for(index: dict, incoming_size: int):
    if incoming_size > BUDGET_BYTES:
        raise ValueError(
            f"LoRA size {incoming_size} exceeds cache budget {BUDGET_BYTES}"
        )
    current = sum(e["size"] for e in index.values())
    while current + incoming_size > BUDGET_BYTES and index:
        victim_key = min(index, key=lambda k: index[k]["last_used_ts"])
        victim = index.pop(victim_key)
        victim_ext = os.path.splitext(victim["filename"])[1]
        victim_path = os.path.join(BLOBS_DIR, f"{victim_key}{victim_ext}")
        if os.path.exists(victim_path):
            os.remove(victim_path)
        current -= victim["size"]
        log.info("Evicted LoRA from cache: %s", victim["filename"])


def get_lora_path(uri: str, gcs_client=None) -> tuple[str, str]:
    """Return (cached_blob_path, original_filename) for the LoRA at `uri`."""
    _ensure_dirs()
    key = _key_for(uri)
    ext = _ext_for(uri)
    blob_path = os.path.join(BLOBS_DIR, f"{key}{ext}")
    filename = os.path.basename(uri.split("?")[0])

    with _Lock():
        index = _load_index()
        if key in index and os.path.exists(blob_path):
            index[key]["last_used_ts"] = time.time()
            _save_index(index)
            log.info("LoRA cache hit: %s", uri)
            return blob_path, filename

    log.info("LoRA cache miss: %s", uri)
    tmp_path = f"{blob_path}.tmp.{os.getpid()}"
    download_source_to_file(uri, tmp_path, gcs_client=gcs_client)
    size = os.path.getsize(tmp_path)

    if size > BUDGET_BYTES:
        os.remove(tmp_path)
        raise ValueError(f"LoRA size {size} exceeds cache budget {BUDGET_BYTES}")

    with _Lock():
        index = _load_index()
        _evict_for(index, size)
        os.replace(tmp_path, blob_path)
        index[key] = {
            "uri": uri,
            "filename": filename,
            "size": size,
            "last_used_ts": time.time(),
        }
        _save_index(index)

    return blob_path, filename
