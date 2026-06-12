"""
Codebase index (RAG retrieval).

Turns the extracted facts into searchable text chunks (one per function/class),
embeds them with Gemini, and stores them in a FAISS vector index. Given a
question, search() returns the most relevant chunks. This is what lets the agent
work on codebases too large to fit entirely in a prompt.
"""

import faiss
import numpy as np

from doc_agent.core.llm import embed_texts


def _facts_to_chunks(facts: list[dict]) -> list[dict]:
    """Turn extracted facts into one searchable text chunk per function/class."""
    chunks = []
    for file in facts:
        module = file.get("file", "?")
        lang = file.get("language", "")
        header = f"File: {module}\n" + (f"Language: {lang}\n" if lang else "")
        for fn in file.get("functions", []):
            text = (header +
                    f"Function: {fn['signature']}\n"
                    f"{fn.get('docstring') or ''}")
            chunks.append({"id": f"{module}::{fn['name']}", "text": text})
        for cls in file.get("classes", []):
            method_names = ", ".join(m["name"] for m in cls.get("methods", []))
            text = (header +
                    f"Class: {cls['name']}\n"
                    f"{cls.get('docstring') or ''}\n"
                    f"Methods: {method_names}")
            chunks.append({"id": f"{module}::{cls['name']}", "text": text})
    return chunks


class CodebaseIndex:
    """A FAISS vector index over a codebase's extracted facts."""

    def __init__(self, facts: list[dict]):
        self.chunks = _facts_to_chunks(facts)
        self.index = None
        if not self.chunks:
            return
        vectors = embed_texts([c["text"] for c in self.chunks])
        matrix = np.array(vectors, dtype="float32")
        faiss.normalize_L2(matrix)                       # cosine similarity via inner product
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)

    def search(self, query: str, k: int = 4) -> list[dict]:
        """Return the k chunks most relevant to the query."""
        if self.index is None:
            return []
        q = np.array(embed_texts([query]), dtype="float32")
        faiss.normalize_L2(q)
        _scores, indices = self.index.search(q, min(k, len(self.chunks)))
        return [self.chunks[i] for i in indices[0]]