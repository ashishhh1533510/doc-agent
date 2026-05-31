"""API layer: exposes the documentation pipeline over HTTP (FastAPI)."""

from fastapi import FastAPI
from pydantic import BaseModel

from doc_agent.workflow.pipeline import DocumentationPipeline

app = FastAPI(title="Documentation Agent")
pipeline = DocumentationPipeline()


class GenerateRequest(BaseModel):
    project_path: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-readme")
async def generate_readme_endpoint(req: GenerateRequest):
    return await pipeline.run(req.project_path)