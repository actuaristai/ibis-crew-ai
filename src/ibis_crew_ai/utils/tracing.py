"""This module provides a custom OpenTelemetry span exporter for Google Cloud Trace and Logging.

It extends the functionality of the CloudTraceSpanExporter to log span data to Google Cloud Logging
and handle large attribute values by storing them in Google Cloud Storage.
It is designed to work with the Google Cloud Python client libraries and OpenTelemetry SDK.
"""

import json
from collections.abc import Sequence
from typing import Any

from google.cloud import logging as google_cloud_logging
from google.cloud import storage
from loguru import logger
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult


class CloudTraceLoggingSpanExporter(CloudTraceSpanExporter):
    """An extended version of CloudTraceSpanExporter that logs span data to Google Cloud Logging.

    Also handles large attribute values by storing them in Google Cloud Storage.
    This class helps bypass the 256 character limit of Cloud Trace for attribute values
    by leveraging Cloud Logging (which has a 256KB limit) and Cloud Storage for larger payloads.
    """

    def __init__(self,
                 logging_client: google_cloud_logging.Client | None = None,
                 storage_client: storage.Client | None = None,
                 bucket_name: str | None = None,
                 *,
                 debug: bool = False,
                 **kwargs: dict[str, Any],
                 ) -> None:
        """Initialize the exporter with Google Cloud clients and configuration.

        Args:
            logging_client (google.cloud.logging.Client, optional): Google Cloud Logging client.
            storage_client (google.cloud.storage.Client, optional): Google Cloud Storage client.
            bucket_name (str, optional): Name of the GCS bucket to store large payloads.
            debug (bool): Enable debug mode for additional logging.
            **kwargs (dict): Additional arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        self.debug = debug
        self.logging_client = logging_client or google_cloud_logging.Client(project=self.project_id)
        self.logger = self.logging_client.logger(__name__)
        self.storage_client = storage_client or storage.Client(project=self.project_id)
        self.bucket_name = (bucket_name or f'{self.project_id}-ibis-crew-ai-logs-data')
        self.bucket = self.storage_client.bucket(self.bucket_name)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export the spans to Google Cloud Logging and Cloud Trace.

        Args:
            spans (Sequence[ReadableSpan]): A sequence of spans to export.

        Returns:
            SpanExportResult: The result of the export operation.
        """
        for span in spans:
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, 'x')
            span_id = format(span_context.span_id, 'x')
            span_dict = json.loads(span.to_json())

            span_dict['trace'] = f'projects/{self.project_id}/traces/{trace_id}'
            span_dict['span_id'] = span_id

            span_dict = self._process_large_attributes(span_dict=span_dict, span_id=span_id)

            if self.debug:
                self.logger.debug(span_dict)

            # Log the span data to Google Cloud Logging
            self.logger.log_struct(span_dict, severity='INFO')

        # Export spans to Google Cloud Trace using the parent class method
        return super().export(spans)

    def store_in_gcs(self, content: str, span_id: str) -> str:
        """Initiate storing large content in Google Cloud Storage/.

        Args:
            content (str): The content to store.
            span_id (str): The ID of the span.

        Returns:
            str: The GCS URI of the stored content.
        """
        if not self.storage_client.bucket(self.bucket_name).exists():
            logger.warning('Bucket not found. Unable to store span attributes in GCS.',
                           bucket_name=self.bucket_name)
            return 'GCS bucket not found'

        blob_name = f'spans/{span_id}.json'
        blob = self.bucket.blob(blob_name)

        blob.upload_from_string(content, 'application/json')
        return f'gs://{self.bucket_name}/{blob_name}'

    def _process_large_attributes(self, span_dict: dict, span_id: str) -> dict:
        """Process large attribute values by storing them in GCS.

        If they exceed the size limit of Google Cloud Logging.

        Args:
            span_dict (dict): The span data dictionary.
            span_id (str): The span ID.

        Returns:
            dict: The updated span dictionary.
        """
        attributes = span_dict.get('attributes', {})
        attributes_size = len(json.dumps(attributes).encode())

        if attributes_size > 255 * 1024:  # 250 KB
            logger.info(f'Attributes size ({attributes_size} bytes) exceeds 250 KB, '
                        'storing in GCS to avoid large log entry errors.')

            # Separate large payload from other attributes
            attributes_payload = dict(attributes.items())
            attributes_retain = dict(attributes.items())

            # Store large payload in GCS
            gcs_uri = self.store_in_gcs(json.dumps(attributes_payload), span_id)
            attributes_retain['uri_payload'] = gcs_uri
            attributes_retain['url_payload'] = (
                f'https://storage.mtls.cloud.google.com/'
                f'{self.bucket_name}/spans/{span_id}.json'
            )

            span_dict['attributes'] = attributes_retain
        else:
            logger.debug(f'Attributes size ({attributes_size} bytes) is within the limit, '
                         'no need to store in GCS.')

        return span_dict
