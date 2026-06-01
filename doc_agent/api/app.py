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
    output_path: str | None = None   # optional: also save the README to this .md file


class AskRequest(BaseModel):
    project_path: str
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-readme")
async def generate_readme_endpoint(req: GenerateRequest):
    return await pipeline.run(req.project_path, req.output_path)


@app.post("/ask")
async def ask_endpoint(req: AskRequest):
    if req.project_path not in _qa_cache:
        _qa_cache[req.project_path] = CodebaseQA(req.project_path)
    return await _qa_cache[req.project_path].ask(req.question)


# Export THIS app's OpenAPI (Swagger) spec to openapi.json:
#   python -m doc_agent.api.app
if __name__ == "__main__":
    from doc_agent.tools.output import save_json

    path = save_json("openapi.json", app.openapi())
    print(f"Saved OpenAPI (Swagger) spec to: {path}")