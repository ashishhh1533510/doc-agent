"""
The codebase-QA pipeline: RAG orchestration.

    extract facts (tool) -> embed + index with FAISS (rag) -> retrieve (RAG)
                                                            -> answer (agent)
"""

from doc_agent.tools.extractor import extract_from_directory
from doc_agent.retrieval.indexer import CodebaseIndex
from doc_agent.agents.qa import QAAgent
from doc_agent.tools.input_resolver import resolve_input


class CodebaseQA:
    """Builds a RAG index over a codebase, then answers questions about it."""

    def __init__(self, project_path, token=None):
        with resolve_input(project_path, token) as code_dir:
            facts = extract_from_directory(code_dir)    # TOOL: extract (clone alive here)
        self.index = CodebaseIndex(facts)               # RAG: embed + FAISS (in memory)
        self.qa = QAAgent()                             # AGENT                          # AGENT

    async def ask(self, question: str, k: int = 4) -> dict:
        retrieved = self.index.search(question, k=k)   # RAG retrieval
        answer = await self.qa.answer(question, retrieved)
        return {
            "answer": answer,
            "retrieved": [c["id"] for c in retrieved],  # which chunks RAG pulled
        }


# Run from the terminal: python -m doc_agent.orchestrators.qa doc_agent "your question"
if __name__ == "__main__":
    import asyncio
    import sys

    project = sys.argv[1] if len(sys.argv) > 1 else "doc_agent"
    question = sys.argv[2] if len(sys.argv) > 2 else "How does the reviewer work?"
    out = asyncio.run(CodebaseQA(project).ask(question))
    print("Q:", question)
    print("Retrieved chunks:", out["retrieved"])
    print("\nAnswer:\n", out["answer"])