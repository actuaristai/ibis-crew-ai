"""Module for editing, refreshing, and deleting chat messages."""

from typing import Any


class MessageEditing:
    """Provides methods for editing, refreshing, and deleting chat messages."""

    @staticmethod
    def edit_message(st: Any, button_idx: int, message_type: str) -> None:  # noqa: ANN401
        """Edit a message in the chat history."""
        button_id = f'edit_box_{button_idx}'
        if message_type == 'human':
            messages = st.session_state.user_chats[st.session_state['session_id']]['messages']
            st.session_state.user_chats[st.session_state['session_id']]['messages'] = messages[:button_idx]
            st.session_state.modified_prompt = st.session_state[button_id]
        else:
            st.session_state.user_chats[st.session_state['session_id']]['messages'][button_idx]['content'] = \
                st.session_state[button_id]

    @staticmethod
    def refresh_message(st: Any, button_idx: int, content: str) -> None:  # noqa: ANN401
        """Refresh a message in the chat history."""
        messages = st.session_state.user_chats[st.session_state['session_id']]['messages']
        st.session_state.user_chats[st.session_state['session_id']]['messages'] = messages[:button_idx]
        st.session_state.modified_prompt = content

    @staticmethod
    def delete_message(st: Any, button_idx: int) -> None:  # noqa: ANN401
        """Delete a message from the chat history."""
        messages = st.session_state.user_chats[st.session_state['session_id']]['messages']
        st.session_state.user_chats[st.session_state['session_id']]['messages'] = messages[:button_idx]
