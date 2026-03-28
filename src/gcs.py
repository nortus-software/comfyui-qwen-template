import datetime
from google.cloud import storage


class GCSClient:
    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def upload_bytes(self, data: bytes, destination_path: str, content_type: str = "application/octet-stream"):
        """Upload bytes to a GCS path."""
        blob = self.bucket.blob(destination_path)
        blob.upload_from_string(data, content_type=content_type)

    def get_signed_url(self, path: str, expiry: int = 3600) -> str:
        """Generate a signed download URL."""
        blob = self.bucket.blob(path)
        return blob.generate_signed_url(
            version="v4",
            expiration=datetime.timedelta(seconds=expiry),
            method="GET",
        )

    def download_bytes(self, path: str) -> bytes:
        """Download file content from GCS."""
        blob = self.bucket.blob(path)
        return blob.download_as_bytes()
