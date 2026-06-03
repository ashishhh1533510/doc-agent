"""API layer: exposes the documentation and codebase-QA pipelines over HTTP (FastAPI)."""

from fastapi import FastAPI
from pydantic import BaseModel

from pathlib import Path
from fastapi.responses import FileResponse

from doc_agent.workflow.pipeline import DocumentationPipeline
from doc_agent.workflow.qa import CodebaseQA

app = FastAPI(title="Documentation Agent")
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


class DrawioXMLGenerateRequest(BaseModel):
    project_path: str
    output_path: str | None = None
    private_access_token: str | None = None


class DrawioGenerateRequest(BaseModel):
    project_path: str
    output_path: str | None = None
    private_access_token: str | None = None


class PNGGenerateRequest(BaseModel):
    project_path: str
    output_path: str | None = None
    private_access_token: str | None = None


class SVGGenerateRequest(BaseModel):
    project_path: str
    output_path: str | None = None
    private_access_token: str | None = None


class DotGenerateRequest(BaseModel):
    project_path: str
    output_path: str | None = None
    private_access_token: str | None = None


class MermaidGenerateRequest(BaseModel):
    project_path: str
    output_path: str | None = None
    private_access_token: str | None = None


class AskRequest(BaseModel):
    project_path: str
    question: str
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


# ===== HLD (High-Level Design) ENDPOINTS - Diagram Formats =====
@app.post("/generate/hld/drawio_xml")
async def generate_hld_drawio_xml(req: DrawioXMLGenerateRequest):
    return await pipeline.run(
        req.project_path, fmt="drawio_xml",
        output_path=req.output_path, token=req.private_access_token,
    )


@app.post("/generate/hld/drawio")
async def generate_hld_drawio(req: DrawioGenerateRequest):
    return await pipeline.run(
        req.project_path, fmt="drawio",
        output_path=req.output_path, token=req.private_access_token,
    )


@app.post("/generate/hld/png")
async def generate_hld_png(req: PNGGenerateRequest):
    return await pipeline.run(
        req.project_path, fmt="png",
        output_path=req.output_path, token=req.private_access_token,
    )


@app.post("/generate/hld/svg")
async def generate_hld_svg(req: SVGGenerateRequest):
    return await pipeline.run(
        req.project_path, fmt="svg",
        output_path=req.output_path, token=req.private_access_token,
    )


@app.post("/generate/hld/dot")
async def generate_hld_dot(req: DotGenerateRequest):
    return await pipeline.run(
        req.project_path, fmt="dot",
        output_path=req.output_path, token=req.private_access_token,
    )


@app.post("/generate/hld/mermaid")
async def generate_hld_mermaid(req: MermaidGenerateRequest):
    return await pipeline.run(
        req.project_path, fmt="mermaid",
        output_path=req.output_path, token=req.private_access_token,
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