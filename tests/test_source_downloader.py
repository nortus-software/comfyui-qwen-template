import base64
from unittest.mock import patch, MagicMock
import pytest


def test_download_http_source():
    """Should download bytes from an HTTP URL."""
    from src.source_downloader import download_source

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"http-image-data"
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        data, ext = download_source("https://example.com/photo.jpg")

    assert data == b"http-image-data"
    assert ext == ".jpg"


def test_download_base64_source():
    """Should decode base64 data."""
    from src.source_downloader import download_source

    encoded = base64.b64encode(b"raw-image").decode()
    source = f"data:image/png;base64,{encoded}"

    data, ext = download_source(source)

    assert data == b"raw-image"
    assert ext == ".png"


def test_download_gcs_source():
    """Should download from GCS URI."""
    from src.source_downloader import download_source

    mock_gcs = MagicMock()
    mock_gcs.download_bytes.return_value = b"gcs-image-data"

    data, ext = download_source("gs://bucket/media/images/ref.png", gcs_client=mock_gcs)

    assert data == b"gcs-image-data"
    assert ext == ".png"
    mock_gcs.download_bytes.assert_called_with("media/images/ref.png")


def test_download_unknown_scheme_raises():
    """Should raise ValueError for unknown source format."""
    from src.source_downloader import download_source

    with pytest.raises(ValueError, match="Unsupported source"):
        download_source("ftp://example.com/file.png")


def test_download_source_to_file_https(tmp_path):
    from src.source_downloader import download_source_to_file

    dest = tmp_path / "out.safetensors"
    fake_resp = MagicMock()
    fake_resp.iter_content.return_value = [b"abc", b"def"]
    fake_resp.raise_for_status = MagicMock()
    with patch("src.source_downloader.requests.get", return_value=fake_resp) as mget:
        ext = download_source_to_file("https://example.com/x.safetensors", str(dest))
    assert ext == ".safetensors"
    assert dest.read_bytes() == b"abcdef"
    mget.assert_called_once_with("https://example.com/x.safetensors", stream=True)


def test_download_source_to_file_gcs(tmp_path):
    from src.source_downloader import download_source_to_file

    dest = tmp_path / "out.safetensors"
    gcs = MagicMock()
    ext = download_source_to_file(
        "gs://bucket/path/to/x.safetensors", str(dest), gcs_client=gcs
    )
    assert ext == ".safetensors"
    gcs.download_to_file.assert_called_once_with("path/to/x.safetensors", str(dest))


def test_download_source_to_file_unsupported(tmp_path):
    from src.source_downloader import download_source_to_file

    with pytest.raises(ValueError):
        download_source_to_file("ftp://nope", str(tmp_path / "x"))
