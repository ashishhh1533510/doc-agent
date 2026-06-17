"""API layer: exposes the documentation and codebase-QA pipelines over HTTP (FastAPI)."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from doc_agent.tools.extractor import UnsupportedLanguageError

from pydantic import BaseModel

from pathlib import Path
from fastapi.responses import FileResponse

from doc_agent.workflow.pipeline import DocumentationPipeline
from doc_agent.workflow.qa import CodebaseQA
from doc_agent.workflow.hld_pipeline import run_hld
from doc_agent.workflow.lld_pipeline import run_lld


app = FastAPI(title="Documentation Agent")
@app.exception_handler(UnsupportedLanguageError)
async def _unsupported_language_handler(request: Request, exc: UnsupportedLanguageError):
    """Turn 'no supported source' into a clean 400 instead of a 500."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})

pipeline = DocumentationPipeline()

# Cache one RAG index per project path so we don't re-embed on every question.
_qa_cache: dict[str, CodebaseQA] = {}
class JSONGenerateRequest(BaseModel):
    project_path: str
    output_path: str | None = None
    private_access_token: str | None = None


class YAMLGenerateRequest(BaseModel):
    project_path: str
    output_path: str | None = None
    private_access_token: str | None = None


class HTMLGenerateRequest(BaseModel):
    project_path: str
    output_path: str | None = None
    private_access_token: str | None = None

class MDGenerateRequest(BaseModel):
    project_path: str
    output_path: str | None = None
    private_access_token: str | None = None


class AskRequest(BaseModel):
    project_path: str
    question: str
    private_access_token: str | None = None

class HLDRequest(BaseModel):
    project_path: str
    output_type: str = "combined"   # combined | context | container
    output_path: str | None = None
    private_access_token: str | None = None


class LLDRequest(BaseModel):
    project_path: str
    diagram_type: str = "class"     # class | sequence | component | dependency
    output_path: str | None = None
    private_access_token: str | None = None



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate/readme/json")
async def generate_readme_json(req: JSONGenerateRequest):
    return await pipeline.run(
        req.project_path, fmt="json",
        output_path=req.output_path, token=req.private_access_token,
    )


@app.post("/generate/readme/yaml")
async def generate_readme_yaml(req: YAMLGenerateRequest):
    return await pipeline.run(
        req.project_path, fmt="yaml",
        output_path=req.output_path, token=req.private_access_token,
    )


@app.post("/generate/readme/html")
async def generate_readme_html(req: HTMLGenerateRequest):
    return await pipeline.run(
        req.project_path, fmt="html",
        output_path=req.output_path, token=req.private_access_token,
    )

@app.post("/generate/readme/md")
async def generate_readme_md(req: MDGenerateRequest):
    return await pipeline.run(
        req.project_path, fmt="md",
        output_path=req.output_path, token=req.private_access_token,
    )


# ===== HLD2 / LLD ENDPOINTS - Repo-specific C4 + LLD diagrams =====
@app.post("/generate/hld2/")
async def generate_hld2(req: HLDRequest):
    return await run_hld(
        req.project_path,
        output_type=req.output_type,
        output_path=req.output_path,
        token=req.private_access_token,
    )


@app.post("/generate/lld/")
async def generate_lld(req: LLDRequest):
    return await run_lld(
        req.project_path,
        diagram_type=req.diagram_type,
        output_path=req.output_path,
        token=req.private_access_token,
    )



@app.post("/ask")
async def ask_endpoint(req: AskRequest):
    if req.project_path not in _qa_cache:
        _qa_cache[req.project_path] = CodebaseQA(req.project_path, token=req.private_access_token)
    return await _qa_cache[req.project_path].ask(req.question)

@app.get("/")
def ui():
    return FileResponse(Path(__file__).parent / "index.html")

# Export THIS app's OpenAPI (Swagger) spec to openapi.json:
#   python -m doc_agent.api.app
if __name__ == "__main__":
    from doc_agent.tools.output import save_json

    path = save_json("openapi.json", app.openapi())
    print(f"Saved OpenAPI (Swagger) spec to: {path}")