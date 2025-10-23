"""Utility for interacting with the Ollama-hosted LLM agent."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Iterable, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_ollama import ChatOllama

from .history import ConversationStore, StoredMessage


class LLMChatAgent:
    """
    Thin wrapper around LangChain's ChatOllama to standardise model usage.

    Workshop participants can focus on prompt composition in the service layer
    while relying on this helper for model invocation.
    """

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        timeout: Optional[float] = None,
        conversation_store: Optional[ConversationStore] = None,
    ) -> None:
        resolved_model = model_name or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        resolved_base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL",
            _default_ollama_url(),
        )

        self._client = ChatOllama(
            model=resolved_model,
            base_url=resolved_base_url,
            temperature=temperature,
            request_timeout=timeout,
        )
        self._store = conversation_store or ConversationStore.default()
        self._model_name = resolved_model

    def invoke(
        self,
        messages: Sequence[BaseMessage],
    ) -> str:
        """
        Invoke the model with a structured sequence of LangChain messages.

        Returns the assistant's textual reply.
        """
        response = self._client.invoke(messages)
        if isinstance(response, AIMessage):
            return response.content
        return str(response)

    def invoke_from_text(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[Iterable[tuple[str, str]]] = None,
    ) -> str:
        """
        Convenience helper that accepts raw strings for quick experimentation.

        - `system_prompt` seeds the conversation with contextual instructions.
        - `history` expects an iterable of (role, content) pairs where role is
          either 'user' or 'assistant'.
        """
        messages: list[BaseMessage] = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        if history:
            for role, content in history:
                if role == "assistant":
                    messages.append(AIMessage(content=content))
                else:
                    messages.append(HumanMessage(content=content))

        messages.append(HumanMessage(content=prompt))

        return self.invoke(messages)

    def chat(
        self,
        chat_id: str,
        *,
        prompt: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Invoke the model while persisting conversational context to the store.

        This is the primary method workshop participants will use inside
        `services.py` to combine RAG retrieval results with the LLM.
        """
        history_messages = self._build_history(chat_id, system_prompt=system_prompt)
        user_message = HumanMessage(content=prompt)
        history_messages.append(user_message)

        response_text = self.invoke(history_messages)

        timestamp = _timestamp()
        meta = metadata or {}

        self._store.append(
            StoredMessage(
                chat_id=chat_id,
                role="user",
                content=prompt,
                timestamp=timestamp,
                metadata=meta,
            )
        )
        self._store.append(
            StoredMessage(
                chat_id=chat_id,
                role="assistant",
                content=response_text,
                timestamp=_timestamp(),
                metadata={**meta, "model": self._model_name},
            )
        )

        return response_text

    def load_history(self, chat_id: str) -> list[StoredMessage]:
        """Return the persisted message history for the given chat."""
        return self._store.load(chat_id)

    def _build_history(
        self,
        chat_id: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> list[BaseMessage]:
        messages: list[BaseMessage] = []

        if system_prompt and not self._has_system_message(chat_id):
            messages.append(SystemMessage(content=system_prompt))

        stored_messages = self._store.load(chat_id)
        for stored in stored_messages:
            messages.append(_stored_to_message(stored))

        return messages

    def _has_system_message(self, chat_id: str) -> bool:
        for message in self._store.load(chat_id):
            if message.role == "system":
                return True
        return False


def _default_ollama_url() -> str:
    host = os.getenv("OLLAMA_HOST", "ollama")
    port = os.getenv("OLLAMA_PORT", "12434")
    # Some setups specify host with host:port. Respect it if present.
    if ":" in host:
        return f"http://{host}"
    return f"http://{host}:{port}"


def _stored_to_message(message: StoredMessage) -> BaseMessage:
    if message.role == "assistant":
        return AIMessage(content=message.content)
    if message.role == "system":
        return SystemMessage(content=message.content)
    return HumanMessage(content=message.content)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["LLMChatAgent"]
