"""API layer: exposes the documentation and codebase-QA pipelines over HTTP (FastAPI)."""

from fastapi import FastAPI
from pydantic import BaseModel

from doc_agent.workflow.pipeline import DocumentationPipeline
from doc_agent.workflow.qa import CodebaseQA

app = FastAPI(title="Documentation Agent")
pipeline = DocumentationPipeline()

# Cache one RAG index per project path so we don't re-embed on every question.
_qa_cache: dict[str, CodebaseQA] = {}


class GenerateRequest(BaseModel):
    project_path: str
    format: str = "md"               # md | html | json | yaml | mermaid | png | svg | dot | drawio | drawio_xml
    output_path: str | None = None
    token: str | None = None         # optional, for private git repos


class AskRequest(BaseModel):
    project_path: str
    question: str
    token: str | None = None         # optional, for private git repos


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-readme")
async def generate_readme_endpoint(req: GenerateRequest):
    return await pipeline.run(
        req.project_path, fmt=req.format,
        output_path=req.output_path, token=req.token,
    )


@app.post("/ask")
async def ask_endpoint(req: AskRequest):
    if req.project_path not in _qa_cache:
        _qa_cache[req.project_path] = CodebaseQA(req.project_path, token=req.token)
    return await _qa_cache[req.project_path].ask(req.question)


# Export THIS app's OpenAPI (Swagger) spec to openapi.json:
#   python -m doc_agent.api.app
if __name__ == "__main__":
    from doc_agent.tools.output import save_json

    path = save_json("openapi.json", app.openapi())
    print(f"Saved OpenAPI (Swagger) spec to: {path}")