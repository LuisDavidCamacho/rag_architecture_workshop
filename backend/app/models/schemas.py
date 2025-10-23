"""Pydantic schemas for Advanced RAG API endpoints."""

from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload for querying the conversational model."""

    query: str = Field(..., description="User prompt to send to the RAG system.")


class QueryResponse(BaseModel):
    """Response structure for chat interactions."""

    chat_id: UUID = Field(..., description="Identifier for the chat session.")
    response: str = Field(..., description="Model-generated reply.")


class EmbedRequest(BaseModel):
    """Payload for kicking off embedding generation."""

    documents: List[str] = Field(
        ...,
        description="Raw documents to chunk and embed. In practice this may map to file paths.",
    )
    chunk_size: Optional[int] = Field(
        default=512,
        description="Preferred chunk size used during preprocessing.",
    )
    overlap: Optional[int] = Field(
        default=50,
        description="Token overlap between chunks.",
    )


class EmbedResponse(BaseModel):
    """Response after generating embeddings."""

    embedded_documents: int = Field(..., description="Count of documents embedded.")
    message: str = Field(..., description="Status message for the embedding job.")

