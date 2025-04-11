"""This module provides a class for managing local chat message history."""

import os
from datetime import datetime

import yaml
from langchain_core.chat_history import BaseChatMessageHistory

from frontend.utils.title_summary import chain_title


class LocalChatMessageHistory(BaseChatMessageHistory):
    """Manages local storage and retrieval of chat message history."""

    def __init__(self,
                 user_id: str,
                 session_id: str = 'default',
                 base_dir: str = '.streamlit_chats') -> None:
        """Initializes the LocalChatMessageHistory instance.

        Args:
            user_id (str): The unique identifier for the user.
            session_id (str): The unique identifier for the session.
            base_dir (str): The base directory for storing chat history files.
        """
        self.user_id = user_id
        self.session_id = session_id
        self.base_dir = base_dir
        self.user_dir = os.path.join(self.base_dir, self.user_id)  # noqa: PTH118
        self.session_file = os.path.join(self.user_dir, f'{session_id}.yaml')  # noqa: PTH118

        os.makedirs(self.user_dir, exist_ok=True)  # noqa: PTH103

    def get_session(self, session_id: str) -> None:
        """Updates the session ID and file path for the current session."""
        self.session_id = session_id
        self.session_file = os.path.join(self.user_dir, f'{session_id}.yaml')  # noqa: PTH118

    def get_all_conversations(self) -> dict[str, dict]:
        """Retrieves all conversations for the current user."""
        conversations = {}
        for filename in os.listdir(self.user_dir):  # noqa: PTH208
            if filename.endswith('.yaml'):
                file_path = os.path.join(self.user_dir, filename)  # noqa: PTH118
                with open(file_path) as f:  # noqa: PTH123
                    conversation = yaml.safe_load(f)
                    if not isinstance(conversation, list) or len(conversation) > 1:
                        msg = (f"""Invalid format in {file_path}.
                                YAML file can only contain one conversation with the following structure.
                                - messages:
                                    - content: [message text]
                                    - type: (human or ai)""")
                        raise ValueError(msg)
                    conversation = conversation[0]
                    if 'title' not in conversation:
                        conversation['title'] = filename
                conversations[filename[:-5]] = conversation
        return dict(sorted(conversations.items(), key=lambda x: x[1].get('update_time', '')))

    def upsert_session(self, session: dict) -> None:
        """Updates or inserts a session into the local storage."""
        session['update_time'] = datetime.now().isoformat()  # noqa: DTZ005
        with open(self.session_file, 'w') as f:  # noqa: PTH123
            yaml.dump([session],
                      f,
                      allow_unicode=True,
                      default_flow_style=False,
                      encoding='utf-8')

    def set_title(self, session: dict) -> None:
        """Set the title for the given session.

        This method generates a title for the session based on its messages.
        If the session has messages, it appends a special message to prompt
        for title creation, generates the title using a title chain, and
        updates the session with the new title.

        Args:
            session (dict): A dictionary containing session information,
                            including messages.

        Returns:
            None
        """
        if session['messages']:
            messages = session['messages'] + [{'type': 'human',
                                               'content': 'End of conversation - Create one single title'}]
            # Remove the tool calls from conversation
            messages = [msg for msg in messages if msg['type'] in ('ai', 'human') and isinstance(msg['content'], str)]

            response = chain_title.invoke(messages)
            title = (response.content.strip() if isinstance(response.content, str) else str(response.content))
            session['title'] = title
            self.upsert_session(session)

    def clear(self) -> None:
        """Removes the current session file if it exists."""
        if os.path.exists(self.session_file):  # noqa: PTH110
            os.remove(self.session_file)  # noqa: PTH107
