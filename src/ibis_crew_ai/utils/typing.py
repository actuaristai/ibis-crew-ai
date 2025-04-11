"""This module provides utility functions and classes for handling chat input.

It includes functionality for creating chat input requests, collecting
feedback, and serialization in the context of a chat application.
It includes classes for representing chat input and feedback, as well as functions
for ensuring valid configurations and serializing objects to JSON.
It is designed to work with LangChain objects and Pydantic models.
It is designed to be used in a Python application that requires chat input handling,
feedback collection, and serialization of objects to JSON format.
"""

import json
import uuid
from typing import Annotated, Any, Literal

from langchain_core.load.serializable import Serializable
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class InputChat(BaseModel):
    """Represents the input for a chat session."""

    messages: list[
        Annotated[HumanMessage | AIMessage | ToolMessage, Field(discriminator='type')]
    ] = Field(
        ..., description='The chat messages representing the current conversation.'
    )


class Request(BaseModel):
    """Represents the input for a chat request with optional configuration.

    Attributes:
        input: The chat input containing messages and other chat-related data
        config: Optional configuration for the runnable, including tags, callbacks, etc.
    """

    input: InputChat
    config: RunnableConfig | None = None


class Feedback(BaseModel):
    """Represents feedback for a conversation."""

    score: int | float
    text: str | None = ""
    run_id: str
    log_type: Literal['feedback'] = 'feedback'
    service_name: Literal['ibis-crew-ai'] = 'ibis-crew-ai'


def ensure_valid_config(config: RunnableConfig | None) -> RunnableConfig:
    """Ensures a valid RunnableConfig by setting defaults for missing fields."""
    if config is None:
        config = RunnableConfig()
    if config.get('run_id') is None:
        config['run_id'] = uuid.uuid4()
    if config.get('metadata') is None:
        config['metadata'] = {}
    return config


def default_serialization(obj: Any) -> Any:  # noqa: ANN401
    """Default serialization for LangChain objects.

    Converts BaseModel instances to JSON strings.
    """
    if isinstance(obj, Serializable):
        return obj.to_json()
    return None


def dumps(obj: Any) -> str:  # noqa: ANN401
    """Serialize an object to a JSON string.

    For LangChain objects (BaseModel instances), it converts them to
    dictionaries before serialization.

    Args:
        obj: The object to serialize

    Returns:
        JSON string representation of the object
    """
    return json.dumps(obj, default=default_serialization)


def dumpd(obj: Any) -> Any:  # noqa: ANN401
    """Convert an object to a JSON-serializable dict.

    Uses default_serialization for handling BaseModel instances.

    Args:
        obj: The object to convert

    Returns:
        Dict/list representation of the object that can be JSON serialized
    """
    return json.loads(dumps(obj))
