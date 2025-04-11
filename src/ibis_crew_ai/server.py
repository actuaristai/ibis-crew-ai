"""API server using fast api."""

import os
from collections.abc import Generator

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse
from google.cloud import logging as google_cloud_logging
from langchain_core.runnables import RunnableConfig
from loguru import logger  # Import loguru logger
from traceloop.sdk import Instruments, Traceloop

from ibis_crew_ai.agent import agent
from ibis_crew_ai.utils.tracing import CloudTraceLoggingSpanExporter
from ibis_crew_ai.utils.typing import Feedback, InputChat, Request, dumps, ensure_valid_config

# Initialize FastAPI app and logging
app = FastAPI(title='ibis-crew-ai',
              description='API for interacting with the Agent ibis-crew-ai')

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

# Initialize Telemetry
try:
    Traceloop.init(app_name=app.title,
                   disable_batch=False,
                   exporter=CloudTraceLoggingSpanExporter(),
                   instruments={Instruments.LANGCHAIN, Instruments.CREW})
except (ValueError, RuntimeError) as e:  # Replace with specific exceptions
    logger.exception('Failed to initialize Telemetry: {}', e)


def set_tracing_properties(config: RunnableConfig) -> None:
    """Sets tracing association properties for the current request.

    Args:
        config: Optional RunnableConfig containing request metadata
    """
    try:
        Traceloop.set_association_properties({'log_type': 'tracing',
                                              'run_id': str(config.get('run_id', 'None')),
                                              'user_id': config['metadata'].pop('user_id', 'None'),
                                              'session_id': config['metadata'].pop('session_id', 'None'),
                                              'commit_sha': os.environ.get('COMMIT_SHA', 'None')})
    except Exception as e:  # noqa: BLE001
        logger.error('Error setting tracing properties: {}', e)


def stream_messages(input_msg: InputChat,
                    config: RunnableConfig | None = None) -> Generator[str, None, None]:
    """Stream events in response to an input chat.

    Args:
        input_msg: The input chat messages
        config: Optional configuration for the runnable

    Yields:
        JSON serialized event data
    """
    config = ensure_valid_config(config=config)
    set_tracing_properties(config)
    input_dict = input_msg.model_dump()

    for data in agent.stream(input_dict, config=config, stream_mode='messages'):
        yield dumps(data) + '\n'


# Routes
@app.get('/', response_class=RedirectResponse)
def redirect_root_to_docs() -> RedirectResponse:
    """Redirect the root URL to the API documentation."""
    return RedirectResponse(url='/docs')


@app.post('/feedback')
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logger.info('Feedback received: {}', feedback.model_dump())
    return {'status': 'success'}


@app.post('/stream_messages')
def stream_chat_events(request: Request) -> StreamingResponse:
    """Stream chat events in response to an input request.

    Args:
        request: The chat request containing input and config

    Returns:
        Streaming response of chat events
    """
    logger.info('Streaming chat events for session: {}', request.config.get('session_id', 'unknown'))
    return StreamingResponse(stream_messages(input_msg=request.input, config=request.config),
                             media_type='text/event-stream')


# Main execution
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='localhost', port=8000)
