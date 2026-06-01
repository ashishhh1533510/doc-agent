# Doc Agent: LLM-powered Documentation and Codebase QA

This project provides a system for automatically generating and reviewing codebase documentation (READMEs) and answering questions about codebases using LLM agents and Retrieval-Augmented Generation (RAG). It leverages a maker-checker loop for documentation generation, ensuring factuality based on extracted code structure.

## Overview

The `doc_agent` project is structured into several key modules, each responsible for a specific part of the documentation and QA pipelines:

*   **`doc_agent.agents`**: Contains the core LLM-driven agents responsible for writing, reviewing, and answering questions about codebases.
    *   **`qa`**: Implements the agent for answering questions about a codebase from retrieved facts.
    *   **`reviewer`**: Implements the agent that fact-checks generated READMEs against extracted facts, acting as the "checker" in the maker-checker loop.
    *   **`writer`**: Implements the agent that generates and revises READMEs from extracted codebase facts.
*   **`doc_agent.api`**: Exposes the core functionalities (documentation generation and codebase QA) via a FastAPI HTTP API.
*   **`doc_agent.core`**: Provides core utilities, specifically for interacting with the Gemini LLM for agent interactions and embeddings.
*   **`doc_agent.rag`**: Handles Retrieval-Augmented Generation (RAG) components, enabling agents to work with codebases too large to fit in a single prompt.
    *   **`indexer`**: Manages the creation and searching of a FAISS vector index built from extracted codebase facts.
*   **`doc_agent.tools`**: Contains deterministic utility tools that do not involve LLMs.
    *   **`extractor`**: Scans Python source code to extract its structure (functions, classes, signatures, docstrings) as "ground truth" facts.
    *   **`output`**: Handles saving generated markdown or JSON documentation to disk.
*   **`doc_agent.workflow`**: Orchestrates the entire documentation and QA pipelines.
    *   **`pipeline`**: Manages the sequential maker-checker flow for README generation: fact extraction -> draft writing -> review -> revision.
    *   **`qa`**: Manages the RAG-based codebase QA pipeline: fact extraction -> indexing -> retrieval -> answering.

## Usage / API Reference

This section details the main functions and classes available in the `doc_agent` project.

### `doc_agent.agents.qa`

QA agent (LLM-driven): answers questions about a codebase from retrieved facts.

#### `class QAAgent`

An LLM agent that answers codebase questions from retrieved context.

*   `__init__(self)`
*   `async answer(self, question: str, chunks: list[dict]) -> str`
    *   Answer the question using the retrieved chunks as context.

### `doc_agent.agents.reviewer`

Reviewer agent (LLM-driven): the "checker" in the maker-checker loop.

#### `class ReviewerAgent`

An LLM agent that fact-checks a README against extracted facts.

*   `__init__(self)`
*   `async review(self, facts, readme: str) -> dict`
    *   Check the README against the facts; return `{'approved': bool, 'issues': [...]}`.

### `doc_agent.agents.writer`

Writer agent (LLM-driven): turns extracted facts into a README.

#### `class WriterAgent`

An LLM agent that writes and revises READMEs from extracted facts.

*   `__init__(self)`
*   `async write(self, facts) -> str`
    *   Write a README from scratch using the facts.
*   `async revise(self, facts, draft: str, issues: list[str]) -> str`
    *   Rewrite the README, fixing the issues the reviewer raised.

### `doc_agent.api.app`

API layer: exposes the documentation and codebase-QA pipelines over HTTP (FastAPI).

*   `@app.get('/health')`
    *   `health()`
*   `@app.post('/generate-readme')`
    *   `async generate_readme_endpoint(req: GenerateRequest)`
*   `@app.post('/ask')`
    *   `async ask_endpoint(req: AskRequest)`

#### `class AskRequest(BaseModel)`

#### `class GenerateRequest(BaseModel)`

### `doc_agent.core.llm`

Model layer: the single place that connects to Gemini.

*   `build_agent(instructions: str, name: str)`
    *   Build a Gemini-backed chat agent with the given instructions and name.
*   `embed_texts(texts: list[str]) -> list[list[float]]`
    *   Return an embedding vector for each input text (used for RAG retrieval).

### `doc_agent.rag.indexer`

Codebase index (RAG retrieval).

#### `class CodebaseIndex`

A FAISS vector index over a codebase's extracted facts.

*   `__init__(self, facts: list[dict])`
*   `search(self, query: str, k: int = 4) -> list[dict]`
    *   Return the `k` chunks most relevant to the query.

### `doc_agent.tools.extractor`

Scan-and-extract tool for the documentation agent.

*   `extract_from_source(source: str, filename: str = '<unknown>') -> dict`
    *   Parse a string of Python source and return its documented structure.
*   `extract_from_file(path) -> dict`
    *   Read a single `.py` file from disk and extract its structure.
*   `extract_from_directory(path) -> list[dict]`
    *   Walk a directory tree and extract every `.py` file inside it.

### `doc_agent.tools.output`

Output tool: writes generated documentation to disk.

*   `save_markdown(path, content: str) -> str`
    *   Save markdown content to a `.md` file; return the absolute path written.
*   `save_json(path, data) -> str`
    *   Save a dict/list as pretty-printed JSON; return the absolute path written.

### `doc_agent.workflow.pipeline`

The documentation pipeline: orchestration.

#### `class DocumentationPipeline`

Runs the full extract -> write -> review -> revise loop.

*   `__init__(self, max_rounds: int = 2)`
*   `async run(self, project_path, output_path = None) -> dict`

### `doc_agent.workflow.qa`

The codebase-QA pipeline: RAG orchestration.

#### `class CodebaseQA`

Builds a RAG index over a codebase, then answers questions about it.

*   `__init__(self, project_path)`
*   `async ask(self, question: str, k: int = 4) -> dict`