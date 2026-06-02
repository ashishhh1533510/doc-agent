# Doc Agent

This project provides an automated system for generating documentation and answering questions about a codebase. It leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) techniques to extract information, write documentation, and provide context-aware answers.

## Overview

The Doc Agent project is structured into several key modules, each responsible for a specific aspect of the documentation and QA process:

*   **`doc_agent.agents`**: Contains various LLM-driven agents responsible for specific tasks like writing documentation (`writer`), reviewing it (`reviewer`), generating architecture diagrams (`diagrammer`), and answering questions (`qa`).
*   **`doc_agent.api`**: Exposes the core functionalities of the Doc Agent through a FastAPI web application, making it accessible via HTTP requests.
*   **`doc_agent.core`**: Provides the foundational components for interacting with LLMs, including functions to build agents and generate text embeddings.
*   **`doc_agent.rag`**: Implements the Retrieval-Augmented Generation (RAG) components, specifically an indexer for creating searchable vector indexes of codebase facts, and a search function.
*   **`doc_agent.tools`**: Offers utility functions for various tasks, including extracting code structure from Python source files (`extractor`), and formatting/saving generated documentation in different formats (`output`).
*   **`doc_agent.workflow`**: Orchestrates the execution of the documentation and QA pipelines, coordinating the use of agents and tools to achieve the desired outcomes.

## Usage / API Reference

### `doc_agent.agents.diagrammer`

#### Class `DiagrammerAgent`

Produces a high-level architecture diagram: the LLM models it, code renders it.

*   **`__init__(self)`**:
    Initializes the `DiagrammerAgent`.

*   **`async diagram(self, facts) -> str`**:
    Analyze the codebase into an architecture model, then render it to Mermaid.

### `doc_agent.agents.qa`

#### Class `QAAgent`

An LLM agent that answers codebase questions from retrieved context.

*   **`__init__(self)`**:
    Initializes the `QAAgent`.

*   **`async answer(self, question: str, chunks: list[dict]) -> str`**:
    Answer the question using the retrieved chunks as context.

### `doc_agent.agents.reviewer`

#### Class `ReviewerAgent`

An LLM agent that fact-checks a README against extracted facts.

*   **`__init__(self)`**:
    Initializes the `ReviewerAgent`.

*   **`async review(self, facts, readme: str) -> dict`**:
    Check the README against the facts; return `{'approved': bool, 'issues': [...]}`.

#### Function `_parse_verdict(text: str) -> dict`

Pull the JSON verdict out of the reply, tolerating ```json code fences.

### `doc_agent.agents.writer`

#### Class `WriterAgent`

An LLM agent that writes and revises READMEs from extracted facts.

*   **`__init__(self)`**:
    Initializes the `WriterAgent`.

*   **`async write(self, facts) -> str`**:
    Write a README from scratch using the facts.

*   **`async revise(self, facts, draft: str, issues: list[str]) -> str`**:
    Rewrite the README, fixing the issues the reviewer raised.

#### Function `_facts_block(facts)`

Render the extracted facts as a prompt block.

### `doc_agent.api.app`

#### Class `GenerateRequest`

(Inherits from `BaseModel`)

#### Class `AskRequest`

(Inherits from `BaseModel`)

#### Function `health()`

Endpoint for health checks.

#### Function `async generate_readme_endpoint(req: GenerateRequest) -> None`

Endpoint to generate a README for a project.

#### Function `async ask_endpoint(req: AskRequest) -> None`

Endpoint to ask questions about a project.

### `doc_agent.core.llm`

#### Function `build_agent(instructions: str, name: str)`

Build a Gemini-backed chat agent with the given instructions and name.

#### Function `embed_texts(texts: list[str]) -> list[list[float]]`

Return an embedding vector for each input text (used for RAG retrieval).

#### Function `async run_agent(agent, prompt: str, max_retries: int = 5, base_delay: float = 1.0) -> None`

Run the provided agent with `prompt` and return the reply text. Retries transient failures (e.g. 503/unavailable) with exponential backoff. Normalizes different agent return types by extracting `text` when available, otherwise falling back to `str(result)`.

### `doc_agent.rag.indexer`

#### Class `CodebaseIndex`

A FAISS vector index over a codebase's extracted facts.

*   **`__init__(self, facts: list[dict])`**:
    Initializes the `CodebaseIndex` with codebase facts.

*   **`search(self, query: str, k: int = 4) -> list[dict]`**:
    Return the k chunks most relevant to the query.

#### Function `_facts_to_chunks(facts: list[dict]) -> list[dict]`

Turn extracted facts into one searchable text chunk per function/class.

### `doc_agent.tools.extractor`

#### Function `_format_arguments(args: ast.arguments) -> str`

Turn a function's argument node into a readable parameter string.

#### Function `_describe_function(node)`

Extract the documentable facts from a single function node.

#### Function `_describe_class(node: ast.ClassDef) -> dict`

Extract the documentable facts from a class node.

#### Function `_extract_imports(tree: ast.Module) -> list[str]`

Collect the NON-standard-library modules this file imports, as dotted names.

#### Function `extract_from_source(source: str, filename: str = '<unknown>') -> dict`

Parse a string of Python source and return its documented structure.

#### Function `extract_from_file(path)`

Read a single .py file from disk and extract its structure.

#### Function `extract_from_directory(path) -> list[dict]`

Walk a directory tree and extract every .py file inside it.

### `doc_agent.tools.output`

#### Function `strip_code_fence(text: str) -> str`

Remove a wrapping ```...``` fence the model sometimes adds.

#### Function `to_json(data)`

Serialize extracted facts to a JSON string.

#### Function `to_yaml(data)`

Serialize extracted facts to a YAML string (spec/config-style docs).

#### Function `markdown_to_html(text: str) -> str`

Convert a markdown document into a standalone HTML page.

#### Function `save_text(path, content: str) -> str`

Write any text content to a file; return the absolute path written.

#### Function `save_json(path, data)`

Save a dict/list as pretty-printed JSON; return the absolute path written.

#### Function `render_architecture_mermaid(model: dict) -> str`

Deterministically render an architecture model (components, externals, edges) into a clean Mermaid flowchart.

### `doc_agent.workflow.pipeline`

#### Class `DocumentationPipeline`

Generates documentation in a chosen format from a codebase.

*   **`__init__(self, max_rounds: int = 2)`**:
    Initializes the `DocumentationPipeline`.

*   **`async _write_reviewed_markdown(self, facts)`**:
    Run the writer + maker-checker loop; return (`markdown`, `review_trace`).

*   **`async run(self, project_path, fmt: str = 'md', output_path = None) -> dict`**:
    Runs the documentation generation process for a given project path and format.

### `doc_agent.workflow.qa`

#### Class `CodebaseQA`

Builds a RAG index over a codebase, then answers questions about it.

*   **`__init__(self, project_path)`**:
    Initializes the `CodebaseQA` with the path to the codebase.

*   **`async ask(self, question: str, k: int = 4) -> dict`**:
    Asks a question about the codebase using the RAG index.