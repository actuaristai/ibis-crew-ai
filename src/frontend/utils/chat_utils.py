"""Chat utilities for saving and sanitizing chat messages."""

from pathlib import Path
from typing import Any

import yaml

SAVED_CHAT_PATH = str(Path.cwd()) + '/.saved_chats'


def clean_text(text: str) -> str:
    """Preprocess the input text by removing leading and trailing newlines."""
    if not text:
        return text

    text = text.removeprefix('\n')
    return text.removesuffix('\n')


def sanitize_messages(messages: list[dict[str, str | list[dict[str, str]]]],
                      ) -> list[dict[str, str | list[dict[str, str]]]]:
    """Preprocess and fix the content of messages."""
    for message in messages:
        if isinstance(message['content'], list):
            for part in message['content']:
                if part['type'] == 'text':
                    part['text'] = clean_text(part['text'])
        else:
            message['content'] = clean_text(message['content'])
    return messages


def save_chat(st: Any) -> None:  # noqa: ANN401
    """Save the current chat session to a YAML file."""
    Path(SAVED_CHAT_PATH).mkdir(parents=True, exist_ok=True)
    session_id = st.session_state['session_id']
    session = st.session_state.user_chats[session_id]
    messages = session.get('messages', [])
    if len(messages) > 0:
        session['messages'] = sanitize_messages(session['messages'])
        filename = f'{session_id}.yaml'
        with open(Path(SAVED_CHAT_PATH) / filename, 'w') as file:  # noqa: PTH123
            yaml.dump(
                [session],
                file,
                allow_unicode=True,
                default_flow_style=False,
                encoding='utf-8',
            )
        st.toast(f'Chat saved to path: ↓ {Path(SAVED_CHAT_PATH) / filename}')
