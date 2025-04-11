"""Utility functions for Google Cloud Storage (GCS) operations."""


import logging

from google.api_core import exceptions
from google.cloud import storage

logger = logging.getLogger(__name__)


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
        logger.info('Bucket already exists', extra={'bucket_name': bucket_name})
    except exceptions.NotFound:
        bucket = storage_client.create_bucket(bucket_name,
                                              location=location,
                                              project=project)
        logger.info('Created bucket in location',
                    extra={'bucket_name': bucket.name,
                           'bucket_location': bucket.location})
