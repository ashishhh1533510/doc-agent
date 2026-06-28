"""API layer: exposes the documentation and codebase-QA pipelines over HTTP (FastAPI)."""

import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from doc_agent.tools.extractor import UnsupportedLanguageError

from pydantic import BaseModel

from pathlib import Path
from fastapi.responses import FileResponse

from doc_agent.workflow.pipeline import DocumentationPipeline
# QA agent shelved — re-enable with the /ask endpoint + UI option below.
# from doc_agent.workflow.qa import CodebaseQA
from doc_agent.workflow.hld_pipeline import run_hld
from doc_agent.workflow.lld_pipeline import run_lld
from doc_agent.core.llm import start_run_metrics, summarize_run_metrics
from doc_agent.core import gcp_monitoring


async def _run_with_metrics(coro):
    """Run a diagram coroutine with a fresh per-request token collector and attach
    the resulting token_usage (plus the run's time window for later cross-check).

    start_run_metrics() seeds a ContextVar dict that every run_agent() in this
    request mutates in place; child tasks spawned by asyncio.gather inherit the
    same dict reference, so per-agent totals aggregate correctly across the run.
    """
    start_run_metrics()
    t0 = time.time()
    result = await coro
    t1 = time.time()
    if isinstance(result, dict):
        usage = summarize_run_metrics()
        usage["window"] = {"start": int(t0), "end": int(t1)}
        result["token_usage"] = usage
    return result


app = FastAPI(title="Documentation Agent")
@app.exception_handler(UnsupportedLanguageError)
async def _unsupported_language_handler(request: Request, exc: UnsupportedLanguageError):
    """Turn 'no supported source' into a clean 400 instead of a 500."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})

pipeline = DocumentationPipeline()

# QA agent shelved — cache disabled.
# Cache one RAG index per project path so we don't re-embed on every question.
# _qa_cache: dict[str, CodebaseQA] = {}
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


# QA agent shelved.
# class AskRequest(BaseModel):
#     project_path: str
#     question: str
#     private_access_token: str | None = None

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
    return await _run_with_metrics(pipeline.run(
        req.project_path, fmt="json",
        output_path=req.output_path, token=req.private_access_token,
    ))


@app.post("/generate/readme/yaml")
async def generate_readme_yaml(req: YAMLGenerateRequest):
    return await _run_with_metrics(pipeline.run(
        req.project_path, fmt="yaml",
        output_path=req.output_path, token=req.private_access_token,
    ))


@app.post("/generate/readme/html")
async def generate_readme_html(req: HTMLGenerateRequest):
    return await _run_with_metrics(pipeline.run(
        req.project_path, fmt="html",
        output_path=req.output_path, token=req.private_access_token,
    ))

@app.post("/generate/readme/md")
async def generate_readme_md(req: MDGenerateRequest):
    return await _run_with_metrics(pipeline.run(
        req.project_path, fmt="md",
        output_path=req.output_path, token=req.private_access_token,
    ))


# ===== HLD2 / LLD ENDPOINTS - Repo-specific C4 + LLD diagrams =====
@app.post("/generate/hld2/")
async def generate_hld2(req: HLDRequest):
    return await _run_with_metrics(run_hld(
        req.project_path,
        output_type=req.output_type,
        output_path=req.output_path,
        token=req.private_access_token,
    ))


@app.post("/generate/lld/")
async def generate_lld(req: LLDRequest):
    return await _run_with_metrics(run_lld(
        req.project_path,
        diagram_type=req.diagram_type,
        output_path=req.output_path,
        token=req.private_access_token,
    ))


class VerifyCallsRequest(BaseModel):
    start: int
    end: int


@app.post("/metrics/verify-calls")
def verify_calls(req: VerifyCallsRequest):
    """Cross-check our recorded LLM-call count against Google's own Cloud Monitoring
    request_count for the Gemini API over the run's window. Returns google_call_count
    = None when GCP isn't configured (the UI then hides the auto-reconciliation line).
    The window end is widened to absorb Cloud Monitoring's ~1-4 min ingestion lag."""
    count = gcp_monitoring.get_google_call_count(req.start, req.end + 300)
    return {"google_call_count": count, "configured": gcp_monitoring.is_configured()}



# QA agent shelved — re-enable this endpoint, AskRequest, the CodebaseQA import,
# _qa_cache, and the UI option to restore Q&A.
# @app.post("/ask")
# async def ask_endpoint(req: AskRequest):
#     if req.project_path not in _qa_cache:
#         _qa_cache[req.project_path] = CodebaseQA(req.project_path, token=req.private_access_token)
#     return await _qa_cache[req.project_path].ask(req.question)

@app.get("/")
def ui():
    return FileResponse(Path(__file__).parent / "index.html")

# Export THIS app's OpenAPI (Swagger) spec to openapi.json:
#   python -m doc_agent.api.app
if __name__ == "__main__":
    from doc_agent.tools.output import save_json

    path = save_json("openapi.json", app.openapi())
    print(f"Saved OpenAPI (Swagger) spec to: {path}")