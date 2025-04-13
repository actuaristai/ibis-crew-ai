"""Load test for the chat stream API using Locust."""  # noqa: INP001

import json
import os
import time

from locust import HttpUser, between, task


class ChatStreamUser(HttpUser):
    """Simulates a user interacting with the chat stream API."""

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    @task
    def chat_stream(self) -> None:
        """Simulates a chat stream interaction."""
        headers = {'Content-Type': 'application/json'}
        if os.environ.get('_ID_TOKEN'):
            headers['Authorization'] = f"Bearer {os.environ['_ID_TOKEN']}"

        data = {'input': {'messages': [{'type': 'human',
                                        'content': 'Hello, AI!'},
                                       {'type': 'ai',
                                        'content': 'Hello!'},
                                       {'type': 'human',
                                        'content': 'Who are you?'}]},
                'config': {'metadata': {'user_id': 'test-user',
                                        'session_id': 'test-session'}}}

        start_time = time.time()

        with self.client.post('/stream_messages',
                              headers=headers,
                              json=data,
                              catch_response=True,
                              name='/stream_messages first message',
                              stream=True) as response:
            if response.status_code == 200:  # noqa: PLR2004
                events = []
                for line in response.iter_lines():
                    if line:
                        event = json.loads(line)
                        events.append(event)
                        for chunk in event:
                            if (isinstance(chunk, dict) and chunk.get('type') == 'constructor'):
                                if not chunk.get('kwargs', {}).get('content'):
                                    continue
                                response.success()
                                end_time = time.time()
                                total_time = end_time - start_time
                                self.environment.events.request.fire(request_type='POST',
                                                                     name='/stream_messages end',
                                                                     response_time=total_time * 1000,
                                                                     # Convert to milliseconds
                                                                     response_length=len(json.dumps(events)),
                                                                     response=response,
                                                                     context={})
                                return
                response.failure('No valid response content received')
            else:
                response.failure(f'Unexpected status code: {response.status_code}')
