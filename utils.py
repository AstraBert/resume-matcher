from llama_index.core.llms import ChatMessage
from typing import List


class ChatHistory:
    def __init__(self) -> None:
        self.message_history: List[ChatMessage] = []
    def add_to_history(self, content: str, role: str) -> None:
        history_piece = ChatMessage.from_str(content=content, role=role)
        self.message_history.append(history_piece)
    def get_history(self):
        return self.message_history
    




