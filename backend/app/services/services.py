"""Service layer skeleton for Advanced RAG operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple
from uuid import UUID, uuid4

import numpy as np

from ..core import EmbeddingGenerator, LLMChatAgent
from ..core.chunking import iter_chunk_email_records


@dataclass
class RagAnswer:
    chat_id: UUID
    answer: str


def answer_question(user_query: str, chat_id: UUID | None = None) -> RagAnswer:
    """
    Run the full retrieval + prompting workflow and return model-ready prompt.
    """
    query_embedding = embed_user_query(user_query)
    contexts = retrieve_relevant_chunks(query_embedding)
    prompt = compose_query_prompt(user_query, contexts)
    chat_identifier = chat_id or uuid4()

    agent = LLMChatAgent()
    response_text = agent.chat(
        chat_id=str(chat_identifier),
        prompt=prompt,
        metadata={
            "mode": "advanced_rag",
            "context_chunks": len(contexts),
        },
    )

    return RagAnswer(chat_id=str(chat_identifier), answer=response_text)


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

    dataframe = pl.read_csv(corpus_path)
    expected_columns = {"file", "message"}
    if not expected_columns.issubset(dataframe.columns):
        raise ValueError(
            f"Corpus file '{filename}' must contain columns {expected_columns}. "
            f"Found {set(dataframe.columns)} instead."
        )

    total_rows = dataframe.height
    if total_rows == 0:
        return 0

    sample_fraction = 0.01
    sample_seed = 42
    sample_n = max(1, int(total_rows * sample_fraction))
    print("*" * 20)
    print(
        f"sampling {sample_n} / {total_rows} rows (~{sample_fraction:.0%}) "
        f"from {filename} with seed {sample_seed}"
    )
    print("*" * 20)
    dataframe = dataframe.sample(
        n=sample_n,
        seed=sample_seed,
        with_replacement=False,
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

    print("*" * 20)
    print("loading embedding model")
    print("*" * 20)
    embedding_generator = EmbeddingGenerator(batch_size=batch_size)
    print("*" * 20)
    print("generating embeddings")
    print("*" * 20)

    output_dir = Path("outputs/advanced_rag/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "email_embeddings.jsonl"

    embedded_count = 0
    batch: List[Tuple[str, str, str]] = []
    start_time = perf_counter()

    def flush_batch(handle) -> None:
        nonlocal batch, embedded_count
        if not batch:
            return
        batch_ids = [record[0] for record in batch]
        batch_texts = [record[1] for record in batch]
        embeddings = embedding_generator.embed_batch(batch_texts)
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
        for chunk_record in chunk_records:
            batch.append(chunk_record)
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


def embed_user_query(query: str) -> List[float]:
    """
    Embed a user query so it can be compared against stored document vectors.
    """
    generator = EmbeddingGenerator()
    embeddings = generator.generate_from_texts(
        identifiers=["user-query"],
        documents=[query],
    )
    if not embeddings:
        raise RuntimeError("Failed to embed user query.")
    _, vector = embeddings[0]
    return vector


def retrieve_relevant_chunks(
    query_embedding: Sequence[float],
    *,
    top_k: int = 5,
    embedding_path: Path | None = None,
) -> List[Dict[str, str]]:
    """
    Retrieve the top-k chunks most similar to the embedded query using cosine similarity.
    """
    path = embedding_path or Path("outputs/advanced_rag/embeddings/email_embeddings.jsonl")
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding store not found at {path}. Run the embed stage first."
        )

    query_vec = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        raise ValueError("Query embedding has zero norm; cannot compute similarity.")

    candidates: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            chunk_vec = np.array(payload["embedding"], dtype=np.float32)
            denom = np.linalg.norm(chunk_vec) * query_norm
            score = float(np.dot(query_vec, chunk_vec) / denom) if denom else 0.0
            candidates.append(
                {
                    "chunk_id": payload["chunk_id"],
                    "source_file": payload.get("source_file", ""),
                    "text": payload.get("text", ""),
                    "score": f"{score:.4f}",
                }
            )

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return candidates[:top_k]


def compose_query_prompt(user_query: str, contexts: Sequence[Dict[str, str]]) -> str:
    """
    Compose the final prompt that will be sent to the model combining user query and retrieved snippets.
    """
    context_blocks = []
    for idx, context in enumerate(contexts, start=1):
        block = (
            f"Context #{idx}\n"
            f"Source: {context.get('source_file', 'unknown')}\n"
            f"Relevance: {context.get('score', '0.0')}\n"
            f"{context.get('text', '').strip()}"
        )
        context_blocks.append(block.strip())

    assembled_context = "\n\n".join(context_blocks) or "No relevant context retrieved."

    prompt = (
        "You are a helpful assistant with access to Enron email snippets.\n"
        "Answer the question using ONLY the provided context. If the context is insufficient, "
        "state that you are unsure.\n\n"
        f"Question: {user_query}\n\n"
        f"Context:\n{assembled_context}\n\n"
        "Final Answer:"
    )
    return prompt
