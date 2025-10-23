"""API routes for Advanced RAG interactions."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException

from ..models.schemas import (
    EmbedRequest,
    EmbedResponse,
    QueryRequest,
    QueryResponse,
)
from ..services import services as rag_services

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["advanced-rag"])


@router.post("/query", response_model=QueryResponse)
def start_chat(payload: QueryRequest) -> QueryResponse:
    """Start a brand-new chat session with the Advanced RAG model."""
    try:
        chat_id, response = rag_services.start_new_chat(payload.query)
    except NotImplementedError as exc:
        logger.info("start_new_chat not implemented: %s", exc)
        raise HTTPException(
            status_code=501,
            detail="Advanced RAG start chat service not implemented yet.",
        ) from exc

    return QueryResponse(chat_id=chat_id, response=response)


@router.post("/query/{chat_id}", response_model=QueryResponse)
def continue_existing_chat(chat_id: UUID, payload: QueryRequest) -> QueryResponse:
    """Continue an existing chat session using the provided chat_id."""
    try:
        response = rag_services.continue_chat(chat_id=chat_id, user_query=payload.query)
    except NotImplementedError as exc:
        logger.info("continue_chat not implemented: %s", exc)
        raise HTTPException(
            status_code=501,
            detail="Advanced RAG continue chat service not implemented yet.",
        ) from exc

    return QueryResponse(chat_id=chat_id, response=response)


@router.post("/embed", response_model=EmbedResponse)
def embed_corpus(payload: EmbedRequest) -> EmbedResponse:
    """Trigger corpus chunking and embedding generation."""
    try:
        embedded_count = rag_services.embed_documents(
            documents=payload.documents,
            chunk_size=payload.chunk_size or 512,
            overlap=payload.overlap or 50,
        )
    except NotImplementedError as exc:
        logger.info("embed_documents not implemented: %s", exc)
        raise HTTPException(
            status_code=501,
            detail="Advanced RAG embedding service not implemented yet.",
        ) from exc

    return EmbedResponse(
        embedded_documents=embedded_count,
        message="Embedding job completed.",
    )

