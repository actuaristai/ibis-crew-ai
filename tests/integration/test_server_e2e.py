"""Integration tests for the FastAPI server."""  # noqa: INP001

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
import requests
from loguru import logger
from requests.exceptions import RequestException

BASE_URL = 'http://127.0.0.1:8000/'
STREAM_URL = BASE_URL + 'stream_messages'
FEEDBACK_URL = BASE_URL + 'feedback'

HEADERS = {'Content-Type': 'application/json'}


def log_output(pipe: Any, log_func: Any) -> None:  # noqa: ANN401
    """Log the output from the given pipe."""
    for line in iter(pipe.readline, ''):
        log_func(line.strip())


def start_server() -> subprocess.Popen[str]:
    """Start the FastAPI server using subprocess and log its output."""
    command = [sys.executable,
               '-m',
               'uvicorn',
               'app.server:app',
               '--host',
               'localhost',
               '--port',
               '8000']
    env = os.environ.copy()
    env['INTEGRATION_TEST'] = 'TRUE'
    process = subprocess.Popen(command,  # noqa: S603
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True,
                               bufsize=1,
                               env=env)

    # Start threads to log stdout and stderr in real-time
    threading.Thread(target=log_output, args=(process.stdout, logger.info), daemon=True).start()
    threading.Thread(target=log_output, args=(process.stderr, logger.error), daemon=True).start()

    return process


def wait_for_server(timeout: int = 60, interval: int = 1) -> bool:
    """Wait for the server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get('http://127.0.0.1:8000/docs', timeout=10)
            if response.status_code == 200:  # noqa: PLR2004
                logger.info('Server is ready')
                return True
        except RequestException:
            pass
        time.sleep(interval)
    logger.error(f'Server did not become ready within {timeout} seconds')
    return False


@pytest.fixture(scope='session')
def server_fixture(request: Any) -> Iterator[subprocess.Popen[str]]:  # noqa: ANN401
    """Pytest fixture to start and stop the server for testing."""
    logger.info('Starting server process')
    server_process = start_server()
    if not wait_for_server():
        pytest.fail('Server failed to start')
    logger.info('Server process started')

    def stop_server() -> None:
        logger.info('Stopping server process')
        server_process.terminate()
        server_process.wait()
        logger.info('Server process stopped')

    request.addfinalizer(stop_server)  # noqa: PT021
    return server_process


def test_chat_stream() -> None:
    """Test the chat stream functionality."""
    logger.info('Starting chat stream test')

    data = {'input': {'messages': [{'type': 'human',
                                    'content': 'Hello, AI!'},
                                   {'type': 'ai',
                                    'content': 'Hello!'},
                                   {'type': 'human',
                                    'content': 'What is the weather in NY?'}]},
            'config': {'metadata': {'user_id': 'test-user',
                                    'session_id': 'test-session'}}}

    response = requests.post(STREAM_URL, headers=HEADERS, json=data, stream=True, timeout=10)
    assert response.status_code == 200  # noqa: PLR2004

    events = [json.loads(line) for line in response.iter_lines() if line]
    assert events, 'No events received from stream'

    # Verify each event is a tuple of message and metadata
    for event in events:
        assert isinstance(event, list), 'Event should be a list'
        assert len(event) == 2, 'Event should contain message and metadata'  # noqa: PLR2004
        message, _ = event

        # Verify message structure
        assert isinstance(message, dict), 'Message should be a dictionary'
        assert message['type'] == 'constructor'
        assert 'kwargs' in message, 'Constructor message should have kwargs'

    # Verify at least one message has content
    has_content = False
    for event in events:
        message = event[0]
        if message.get('type') == 'constructor' and 'content' in message['kwargs']:
            has_content = True
            break
    assert has_content, 'At least one message should have content'


def test_chat_stream_error_handling() -> None:
    """Test the chat stream error handling."""
    logger.info('Starting chat stream error handling test')

    data = {'input': {'messages': [{'type': 'invalid_type',
                                    'content': 'Cause an error'}]}}
    response = requests.post(STREAM_URL, headers=HEADERS, json=data, stream=True, timeout=10)

    assert response.status_code == 422, (  # noqa: PLR2004
        f'Expected status code 422, got {response.status_code}')
    logger.info('Error handling test completed successfully')


def test_collect_feedback() -> None:
    """Test the feedback collection endpoint (/feedback) to ensure it properly logs the received feedback."""
    # Create sample feedback data
    feedback_data = {'score': 4,
                     'run_id': str(uuid.uuid4()),
                     'text': 'Great response!'}

    response = requests.post(FEEDBACK_URL, json=feedback_data, headers=HEADERS, timeout=10)
    assert response.status_code == 200  # noqa: PLR2004
