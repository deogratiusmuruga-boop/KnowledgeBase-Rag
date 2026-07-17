"""Convert FAISS retrieval results into structured, attributable evidence."""

from authority_mapping import classify_source_authority


def aggregate_evidence(chunks, distances, indices):
    """Build one evidence object for each valid FAISS retrieval result.

    Args:
        chunks: Chunk metadata in the same order used to construct the FAISS index.
        distances: FAISS similarity scores returned by ``index.search``.
        indices: FAISS result indices returned by ``index.search``.
    """
    if len(distances) != len(indices):
        raise ValueError("FAISS distances and indices must contain the same number of queries.")
    if len(indices) != 1:
        raise ValueError("Evidence aggregation expects retrieval results for one query.")

    evidence = []

    for similarity_score, chunk_index in zip(distances[0], indices[0]):
        chunk_index = int(chunk_index)
        if chunk_index == -1:
            continue
        if chunk_index < 0 or chunk_index >= len(chunks):
            raise IndexError(f"FAISS returned an invalid chunk index: {chunk_index}")

        chunk = chunks[chunk_index]
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {chunk_index} must be a dictionary.")

        chunk_id = chunk.get("chunk_id")
        source_document = chunk.get("source_document")
        text = chunk.get("text")
        if chunk_id is None or not isinstance(source_document, str) or not isinstance(text, str):
            raise ValueError(f"Chunk {chunk_index} is missing required evidence fields.")

        authority = classify_source_authority(source_document, text)
        evidence.append(
            {
                "chunk_id": chunk_id,
                "source_document": source_document,
                "text": text,
                "similarity_score": float(similarity_score),
                "document_category": authority.document_category,
                "authority_score": authority.authority_score,
            }
        )

    return evidence
