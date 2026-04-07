import json
import os
import time
from unittest.mock import patch
import pytest

from src import lora_cache


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "lora_cache"
    monkeypatch.setattr(lora_cache, "CACHE_DIR", str(d))
    monkeypatch.setattr(lora_cache, "BLOBS_DIR", str(d / "blobs"))
    monkeypatch.setattr(lora_cache, "INDEX_PATH", str(d / "index.json"))
    monkeypatch.setattr(lora_cache, "LOCK_PATH", str(d / "index.lock"))
    return d


def _fake_download(uri, dest_path, gcs_client=None):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(b"X" * 1024)
    return os.path.splitext(uri)[1] or ".safetensors"


def test_cache_miss_downloads_and_returns_path(cache_dir):
    with patch("src.lora_cache.download_source_to_file", side_effect=_fake_download) as dl:
        path, filename = lora_cache.get_lora_path("https://example.com/foo.safetensors")
    assert os.path.exists(path)
    assert filename == "foo.safetensors"
    dl.assert_called_once()
    index = json.loads((cache_dir / "index.json").read_text())
    assert len(index) == 1


def test_cache_hit_skips_download(cache_dir):
    with patch("src.lora_cache.download_source_to_file", side_effect=_fake_download):
        lora_cache.get_lora_path("https://example.com/foo.safetensors")
    with patch("src.lora_cache.download_source_to_file", side_effect=_fake_download) as dl:
        path, filename = lora_cache.get_lora_path("https://example.com/foo.safetensors")
    assert os.path.exists(path)
    assert filename == "foo.safetensors"
    dl.assert_not_called()


def test_lru_evicts_oldest_when_over_budget(cache_dir, monkeypatch):
    monkeypatch.setattr(lora_cache, "BUDGET_BYTES", 2048)

    def dl(uri, dest_path, gcs_client=None):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(b"X" * 1024)
        return ".safetensors"

    with patch("src.lora_cache.download_source_to_file", side_effect=dl):
        p1, _ = lora_cache.get_lora_path("https://example.com/a.safetensors")
        p2, _ = lora_cache.get_lora_path("https://example.com/b.safetensors")
        assert os.path.exists(p1) and os.path.exists(p2)
        time.sleep(0.01)
        lora_cache.get_lora_path("https://example.com/a.safetensors")
        time.sleep(0.01)
        lora_cache.get_lora_path("https://example.com/c.safetensors")

    assert os.path.exists(p1)
    assert not os.path.exists(p2)


def test_incoming_larger_than_budget_raises(cache_dir, monkeypatch):
    monkeypatch.setattr(lora_cache, "BUDGET_BYTES", 512)

    def dl(uri, dest_path, gcs_client=None):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(b"X" * 1024)
        return ".safetensors"

    with patch("src.lora_cache.download_source_to_file", side_effect=dl):
        with pytest.raises(ValueError, match="exceeds cache budget"):
            lora_cache.get_lora_path("https://example.com/big.safetensors")


def test_corrupt_index_rebuilds_from_filesystem(cache_dir):
    with patch("src.lora_cache.download_source_to_file", side_effect=_fake_download):
        lora_cache.get_lora_path("https://example.com/foo.safetensors")

    (cache_dir / "index.json").write_text("{not valid json")

    with patch("src.lora_cache.download_source_to_file", side_effect=_fake_download) as dl:
        path, _ = lora_cache.get_lora_path("https://example.com/foo.safetensors")

    assert os.path.exists(path)
    dl.assert_not_called()
