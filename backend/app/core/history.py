"""Conversation history persistence utilities."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class StoredMessage:
    """Serializable chat message representation."""

    chat_id: str
    role: str
    content: str
    timestamp: str
    metadata: dict = field(default_factory=dict)


class ConversationStore:
    """
    File-backed message store for workshop conversations.

    Messages are written as JSON Lines (one message per line) allowing easy
    aggregation at the end of the workshop for quality and throughput metrics.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls, base_directory: Optional[Path | str] = None) -> "ConversationStore":
        """
        Create a store rooted at the workshop's standard output location.

        The default path resolves to ``outputs/advanced_rag/conversations`` within the
        project workspace unless a custom base directory is provided.
        """
        base = Path(base_directory) if base_directory else Path("outputs/advanced_rag/conversations")
        return cls(base)

    def _chat_path(self, chat_id: str) -> Path:
        return self.directory / f"{chat_id}.jsonl"

    def append(self, message: StoredMessage) -> None:
        """Persist a new message to the chat transcript."""
        path = self._chat_path(message.chat_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(message), ensure_ascii=False))
            handle.write("\n")

    def load(self, chat_id: str) -> List[StoredMessage]:
        """
        Load the full conversation for a chat id.

        Returns an empty list if the chat has not been recorded yet.
        """
        path = self._chat_path(chat_id)
        if not path.exists():
            return []

        messages: List[StoredMessage] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line.strip())
                messages.append(StoredMessage(**payload))
        return messages

    def list_chats(self) -> List[str]:
        """Return all chat identifiers that have been persisted."""
        return sorted(p.stem for p in self.directory.glob("*.jsonl"))

    def export(self, destination: Path | str) -> None:
        """
        Materialise all conversations into a single JSON Lines file.

        Each row contains the chat identifier and array of messages, which can be
        consumed by downstream analytics pipelines.
        """
        destination_path = Path(destination)
        payload = []
        for chat_id in self.list_chats():
            history = [asdict(message) for message in self.load(chat_id)]
            payload.append({"chat_id": chat_id, "messages": history})

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with destination_path.open("w", encoding="utf-8") as handle:
            for record in payload:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
