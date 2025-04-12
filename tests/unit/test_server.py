"""Unit tests for the FastAPI server endpoints."""  # noqa: INP001

import json
import os
import sys
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from google.auth.credentials import Credentials
from ibis_crew_ai.utils.typing import InputChat
from langchain_core.messages import HumanMessage


@pytest.fixture(autouse=True)
def mock_google_cloud_credentials() -> Generator[None, None, None]:
    """Mock Google Cloud credentials for testing."""
    with patch.dict(os.environ,
                    {'GOOGLE_APPLICATION_CREDENTIALS': '/path/to/mock/credentials.json',
                     'GOOGLE_CLOUD_PROJECT_ID': 'mock-project-id'}):
        yield


@pytest.fixture(autouse=True)
def mock_google_auth_default() -> Generator[None, None, None]:
    """Mock the google.auth.default function for testing."""
    mock_credentials = MagicMock(spec=Credentials)
    mock_project = 'mock-project-id'

    with patch('google.auth.default', return_value=(mock_credentials, mock_project)):
        yield


@pytest.fixture
def sample_input_chat() -> InputChat:
    """Fixture to create a sample input chat for testing."""
    return InputChat(messages=[HumanMessage(content='What is the meaning of life?')])


def test_redirect_root_to_docs() -> None:
    """Test that the root endpoint (/) redirects to the Swagger UI documentation."""
    # Mock the agent module before importing server
    mock_agent = MagicMock()
    with patch.dict(sys.modules, {'app.agent': mock_agent}):
        # Now import server after the mock is in place
        from ibis_crew_ai.server import app

        client = TestClient(app)
        response = client.get('/')
        assert response.status_code == 200  # noqa: PLR2004
        assert 'Swagger UI' in response.text

