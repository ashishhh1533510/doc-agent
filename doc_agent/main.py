"""API layer: exposes the documentation and codebase-QA pipelines over HTTP (FastAPI)."""

import time
import zlib
import base64

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from doc_agent.tools.extractor import UnsupportedLanguageError

from pydantic import BaseModel

from pathlib import Path
from fastapi.responses import FileResponse

from doc_agent.orchestrators.pipeline import DocumentationPipeline
# QA agent shelved — re-enable with the /ask endpoint + UI option below.
# from doc_agent.orchestrators.qa import CodebaseQA
from doc_agent.orchestrators.hld_pipeline import run_hld
from doc_agent.orchestrators.lld_pipeline import run_lld
from doc_agent.integrations.llm_provider import start_run_metrics, summarize_run_metrics
from doc_agent.observability import gcp_monitoring


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


def _diagram_image_url(code: str, fmt: str = "svg") -> str:
    """Build a Kroki render URL for a Mermaid diagram.

    The image is fetched by the *client* (the marketplace browser renders an
    <img src=...>), so this only constructs a string — it adds no network call,
    latency, or failure mode on our side. Kroki's GET API expects the diagram
    source zlib-compressed then url-safe-base64 encoded.
    """
    if not code:
        return ""
    packed = zlib.compress(code.encode("utf-8"), 9)
    encoded = base64.urlsafe_b64encode(packed).decode("ascii")
    return f"https://kroki.io/mermaid/{fmt}/{encoded}"


def _attach_image_urls(result):
    """Enrich a diagram result with rendered-image URLs alongside the Mermaid
    source, so the marketplace can show a picture AND offer the raw .mmd."""
    if not isinstance(result, dict):
        return result
    if isinstance(result.get("content"), str):
        result["image_url"] = _diagram_image_url(result["content"])
    diagrams = result.get("diagrams")
    if isinstance(diagrams, list):
        for d in diagrams:
            if isinstance(d, dict) and isinstance(d.get("content"), str):
                d["image_url"] = _diagram_image_url(d["content"])
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


# ===== UNIFIED DIAGRAM ENDPOINT — one endpoint for the Agentic Marketplace =====
# The marketplace binds an agent to a single endpoint, so this one entry point
# routes to HLD or LLD via `mode` and returns both the Mermaid source and a
# rendered-image URL.
class DiagramRequest(BaseModel):
    project_path: str
    # Single selector drives everything: "hld" (C4 Combined) or one of the LLD
    # types (class | sequence | component | dependency). `mode`/`output_type`
    # stay optional for backward compatibility / explicit control.
    diagram_type: str = "hld"
    mode: str = ""                  # optional override: hld | lld
    output_path: str | None = None
    private_access_token: str | None = None


_LLD_TYPES = {"class", "sequence", "component", "dependency"}


# Register both with- and without-trailing-slash so clients that strip the slash
# (e.g. the marketplace proxy) hit the handler directly instead of triggering a
# 307 redirect — some proxies downgrade the redirected POST to GET, which lands
# as "405 Method Not Allowed" on the POST-only route.
@app.post("/generate/diagram/")
@app.post("/generate/diagram", include_in_schema=False)
async def generate_diagram(req: DiagramRequest):
    token = (req.private_access_token or "").strip() or None
    sel = (req.diagram_type or "").strip().lower()
    mode = (req.mode or "").strip().lower()
    # Derive HLD vs LLD from the single selector unless mode is set explicitly.
    if mode not in ("hld", "lld"):
        mode = "lld" if sel in _LLD_TYPES else "hld"

    if mode == "lld":
        diagram_type = sel if sel in _LLD_TYPES else "class"
        result = await _run_with_metrics(run_lld(
            req.project_path, diagram_type=diagram_type,
            output_path=req.output_path, token=token,
        ))
    else:
        # HLD has a single view — C4 Combined.
        result = await _run_with_metrics(run_hld(
            req.project_path, output_type="combined",
            output_path=req.output_path, token=token,
        ))
    return _attach_image_urls(result)


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
#   python -m doc_agent.main
if __name__ == "__main__":
    from doc_agent.tools.output import save_json

    path = save_json("openapi.json", app.openapi())
    print(f"Saved OpenAPI (Swagger) spec to: {path}")