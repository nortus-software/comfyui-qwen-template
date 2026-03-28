from unittest.mock import patch, MagicMock
import pytest


def test_upload_bytes_calls_gcs():
    """upload_bytes should upload data to the correct GCS path."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    with patch("google.cloud.storage.Client", mock_client):
        from src.gcs import GCSClient

        client = GCSClient("test-bucket")
        client.upload_bytes(b"image-data", "outputs/job1/output.png", "image/png")

    mock_bucket.blob.assert_called_with("outputs/job1/output.png")
    mock_blob.upload_from_string.assert_called_with(b"image-data", content_type="image/png")


def test_get_signed_url_returns_url():
    """get_signed_url should return a signed URL string."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    mock_blob.generate_signed_url.return_value = "https://storage.googleapis.com/signed"

    with patch("google.cloud.storage.Client", mock_client):
        from src.gcs import GCSClient

        client = GCSClient("test-bucket")
        url = client.get_signed_url("outputs/job1/output.png", expiry=3600)

    assert url == "https://storage.googleapis.com/signed"
    mock_blob.generate_signed_url.assert_called_once()


def test_download_bytes_returns_content():
    """download_bytes should return file content from GCS."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    mock_blob.download_as_bytes.return_value = b"file-content"

    with patch("google.cloud.storage.Client", mock_client):
        from src.gcs import GCSClient

        client = GCSClient("test-bucket")
        data = client.download_bytes("media/images/ref.png")

    assert data == b"file-content"
