"""Service layer skeleton for Advanced RAG operations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple
from uuid import UUID, uuid4

import numpy as np

from ..core import EmbeddingGenerator, GraphFileStore, LLMChatAgent
from ..core.chunking import iter_chunk_email_records


def start_new_chat(user_query: str) -> Tuple[UUID, str]:
    """
    Initialize a new conversational session for the Advanced RAG workflow.

    Returns the chat session UUID and the model response to the initial query.
    """
    chat_id = uuid4()
    answer = graph_rag_query(chat_id, user_query)
    return chat_id, answer


def continue_chat(chat_id: UUID, user_query: str) -> str:
    """
    Continue an existing chat session by appending a new user query.

    Returns the model-generated response tied to the provided chat_id.
    """
    return graph_rag_query(chat_id, user_query)


def embed_documents(
    filename: str,
    *,
    chunk_size: int,
    overlap: int,
    batch_size: int,
) -> int:
    """
    Generate embeddings for the specified CSV corpus located under data/corpus.

    This implementation is tailored for the Enron email dataset (emails.csv) shipped
    with the workshop materials.
    """
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "polars is required to process the email corpus. Add it via Poetry."
        ) from exc

    safe_name = Path(filename).name
    corpus_path = Path("data/corpus") / safe_name
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file '{safe_name}' was not found in data/corpus (looked for {corpus_path})."
        )

    output_dir = Path("outputs/advanced_rag/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "email_embeddings.jsonl"

    if embeddings_path.exists():
        print("*" * 20)
        print(
            f"Existing embeddings found at {embeddings_path}. "
            "Loading cached vectors instead of regenerating."
        )
        print("*" * 20)
        with embeddings_path.open("r", encoding="utf-8") as handle:
            cached_count = sum(1 for _ in handle)
        return cached_count

    dataframe = pl.read_csv(corpus_path)
    expected_columns = {"file", "message"}
    if not expected_columns.issubset(dataframe.columns):
        raise ValueError(
            f"Corpus file '{filename}' must contain columns {expected_columns}. "
            f"Found {set(dataframe.columns)} instead."
        )

    records = (
        (row["file"], row["message"])
        for row in dataframe.iter_rows(named=True)
    )
    chunk_records = iter_chunk_email_records(
        records,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    embedding_generator = EmbeddingGenerator(batch_size=batch_size)

    embedded_count = 0
    batch: List[Tuple[str, str, str]] = []
    start_time = perf_counter()

    def flush_batch(handle) -> None:
        nonlocal batch, embedded_count
        if not batch:
            return
        texts = [item[1] for item in batch]
        embeddings = embedding_generator.embed_batch(texts)
        if len(embeddings) != len(batch):
            raise RuntimeError("Embedding batch mismatch.")

        for (chunk_id, text, source_file), vector in zip(
            batch, embeddings, strict=True
        ):
            payload = {
                "chunk_id": chunk_id,
                "source_file": source_file,
                "embedding": vector,
                "text": text,
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")
            embedded_count += 1

        batch.clear()

    with embeddings_path.open("w", encoding="utf-8") as handle:
        for record in chunk_records:
            batch.append(record)
            if len(batch) >= embedding_generator.batch_size:
                flush_batch(handle)
        flush_batch(handle)

    elapsed = perf_counter() - start_time
    if embedded_count:
        print(
            f"Embedded {embedded_count} chunks to {embeddings_path} "
            f"in {elapsed:.2f}s (~{embedded_count / max(elapsed, 1e-6):.1f} chunks/s)."
        )

    return embedded_count


def build_graph_rag_index(filename: str, *, chunk_size: int, overlap: int) -> None:
    """Prepare graph-based artifacts for the specified corpus (Graph RAG)."""
    raise NotImplementedError("Graph RAG index construction not implemented yet.")


def graph_rag_query(chat_id: UUID, user_query: str) -> str:
    """Execute a Graph RAG retrieval + generation workflow."""
    query_embedding = embed_user_query(user_query)
    contexts = retrieve_relevant_chunks(query_embedding)
    graph_facts = graph_context_query(user_query)
    prompt = compose_query_prompt(
        user_query,
        contexts,
        graph_facts=graph_facts,
    )

    agent = LLMChatAgent()
    response = agent.chat(
        chat_id=str(chat_id),
        prompt=prompt,
        metadata={
            "mode": "graph_rag",
            "context_chunks": len(contexts),
            "graph_facts": len(graph_facts),
        },
    )
    return response


def embed_user_query(query: str) -> List[float]:
    """Embed a user query for similarity search."""
    generator = EmbeddingGenerator()
    vectors = generator.generate_from_texts(
        identifiers=["user-query"],
        documents=[query],
    )
    if not vectors:
        raise RuntimeError("Failed to embed user query.")
    _, vector = vectors[0]
    return vector


def retrieve_relevant_chunks(
    query_embedding: Sequence[float],
    *,
    top_k: int = 5,
    embedding_path: Path | None = None,
) -> List[Dict[str, str]]:
    """
    Retrieve the most similar chunks to the embedded query using cosine similarity.
    """
    path = embedding_path or Path("outputs/advanced_rag/embeddings/email_embeddings.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding store not found at {path}. Run the embed step first."
        )

    query_vec = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        raise ValueError("Query embedding has zero norm; cannot compute similarity.")

    results: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            chunk_vec = np.array(payload["embedding"], dtype=np.float32)
            denom = np.linalg.norm(chunk_vec) * query_norm
            score = float(np.dot(query_vec, chunk_vec) / denom) if denom else 0.0
            results.append(
                {
                    "chunk_id": payload["chunk_id"],
                    "source_file": payload.get("source_file", ""),
                    "text": payload.get("text", ""),
                    "score": f"{score:.4f}",
                }
            )

    results.sort(key=lambda item: float(item["score"]), reverse=True)
    return results[:top_k]


def compose_query_prompt(
    user_query: str,
    contexts: Sequence[Dict[str, str]],
    *,
    graph_facts: Sequence[Dict[str, str]] | None = None,
) -> str:
    """
    Compose a model prompt that combines the user question with retrieved context.
    """
    if not contexts:
        assembled = "No relevant context retrieved. Answer based on general knowledge."
    else:
        assembled = "\n\n".join(
            f"Context #{idx}\nSource: {ctx.get('source_file', 'unknown')}\n"
            f"Relevance: {ctx.get('score', '0.0')}\n{ctx.get('text', '').strip()}"
            for idx, ctx in enumerate(contexts, start=1)
        )

    if graph_facts is None:
        graph_facts = graph_context_query(user_query)

    graph_section = ""
    if graph_facts:
        graph_section = "\n\nGraph Insights:\n" + "\n".join(
            f"- {fact.get('summary', fact.get('entity', ''))}"
            for fact in graph_facts
        )

    prompt = (
        "You are an assistant that must answer questions using Enron email snippets.\n"
        "Use only the context provided. If the context does not contain the answer, "
        "state that you are unsure.\n\n"
        f"Question: {user_query}\n\n"
        f"Context:\n{assembled}"
        f"{graph_section}\n\n"
        "Answer:"
    )
    return prompt


def graph_context_query(user_query: str) -> List[Dict[str, str]]:
    """
    Query the graph artifacts (nodes/edges) and surface entities related to the query.
    """
    store = GraphFileStore()
    try:
        nodes = store.query_graph_file("graph/nodes.jsonl")
    except FileNotFoundError:
        return []

    if not isinstance(nodes, list):
        return []

    keywords = {token.lower() for token in re.findall(r"[A-Za-z0-9]{3,}", user_query)}
    if not keywords:
        return []

    matches: List[Dict[str, str]] = []
    for node in nodes:
        label = str(node.get("label") or node.get("id") or "")
        if not label:
            continue
        if not any(keyword in label.lower() for keyword in keywords):
            continue
        freq = node.get("frequency", 0)
        matches.append(
            {
                "entity": label,
                "summary": f"Entity '{label}' appears {freq} times in the email graph.",
                "frequency": freq,
            }
        )

    # Include relationships for matched entities when possible.
    try:
        edges = store.query_graph_file("graph/edges.jsonl")
    except FileNotFoundError:
        edges = []

    if isinstance(edges, list):
        for edge in edges:
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            if not source or not target:
                continue
            if not (
                any(keyword in source.lower() for keyword in keywords)
                or any(keyword in target.lower() for keyword in keywords)
            ):
                continue
            weight = edge.get("weight", 1)
            matches.append(
                {
                    "entity": f"{source} ↔ {target}",
                    "summary": (
                        f"Relationship between '{source}' and '{target}' "
                        f"occurs {weight} times in the graph."
                    ),
                    "frequency": weight,
                }
            )

    matches.sort(key=lambda item: item.get("frequency", 0), reverse=True)
    return matches[:5]
