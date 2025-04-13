"""Utility functions for Google Cloud Storage (GCS) operations."""

from google.api_core import exceptions
from google.cloud import logging as google_cloud_logging
from google.cloud import storage
from loguru import logger

# Initialize Google Cloud Logging
logging_client = google_cloud_logging.Client()
gcloud_logger = logging_client.logger(__name__)

# Add a custom sink to forward loguru logs to Google Cloud Logging


class GoogleCloudSink:
    """Custom sink to forward loguru logs to Google Cloud Logging."""

    def write(self, message: str) -> None:
        """Write a log message to Google Cloud Logging."""
        record = message.strip()
        gcloud_logger.log_text(record)


logger.add(GoogleCloudSink(), level='INFO')  # Forward loguru logs to Google Cloud Logging


def create_bucket_if_not_exists(bucket_name: str, project: str, location: str) -> None:
    """Creates a new bucket if it doesn't already exist.

    Args:
        bucket_name: Name of the bucket to create
        project: Google Cloud project ID
        location: Location to create the bucket in (defaults to us-central1)
    """
    storage_client = storage.Client(project=project)

    bucket_name = bucket_name.removeprefix('gs://')
    try:
        storage_client.get_bucket(bucket_name)
        logger.info('Bucket already exists', bucket_name=bucket_name)
    except exceptions.NotFound:
        bucket = storage_client.create_bucket(bucket_name,
                                              location=location,
                                              project=project)
        logger.info('Created bucket in location',
                    bucket_name=bucket.name,
                    bucket_location=bucket.location)
