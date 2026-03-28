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
