"""
The documentation pipeline: orchestration with format selection.

  md / html              -> prose docs via the writer + maker-checker review loop
  json / yaml            -> the extracted facts as a structured spec (no LLM)
  mermaid                -> an architecture diagram via the diagrammer agent
  png / svg              -> typed architecture image (diagrams library)
  dot / drawio / drawio_xml -> editable architecture exports

The pipeline picks the right producer for the requested format, then optionally
saves the result to disk via the output tool.

Inputs can be a local directory, a single .py file, or a git repo URL — the
input resolver normalizes all of them to a local directory before extraction.

LLM-bound formats receive a slimmed-down facts payload (slim_facts_for_llm) to
stay under the model's per-minute input-token limit on large repos. json/yaml
always use the full facts since they cost no LLM calls and need full detail.
"""

import json

from doc_agent.tools.extractor import extract_from_directory
from doc_agent.tools.output import (
    markdown_to_html, save_text, slim_facts_for_llm, strip_code_fence, to_json, to_yaml,
)
from doc_agent.agents.writer import WriterAgent
from doc_agent.agents.reviewer import ReviewerAgent
from doc_agent.agents.diagrammer import DiagrammerAgent
from doc_agent.tools.input_resolver import resolve_input


PROSE_FORMATS = {"md", "html"}
STRUCTURED_FORMATS = {"json", "yaml"}
DIAGRAM_FORMATS = {"mermaid"}
IMAGE_FORMATS = {"png", "svg"}
EDITABLE_FORMATS = {"drawio", "dot", "drawio_xml"}
SUPPORTED_FORMATS = (
    PROSE_FORMATS | STRUCTURED_FORMATS | DIAGRAM_FORMATS | IMAGE_FORMATS | EDITABLE_FORMATS
)


class DocumentationPipeline:
    """Generates documentation in a chosen format from a codebase."""

    def __init__(self, max_rounds: int = 2):
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()
        self.diagrammer = DiagrammerAgent()
        self.max_rounds = max_rounds

    async def _write_reviewed_markdown(self, facts):
        """Run the writer + maker-checker loop; return (markdown, review_trace)."""
        draft = await self.writer.write(facts)
        trace = []
        for round_num in range(1, self.max_rounds + 1):
            verdict = await self.reviewer.review(facts, draft)
            trace.append({
                "round": round_num,
                "approved": verdict["approved"],
                "issues": verdict["issues"],
            })
            if verdict["approved"]:
                break
            draft = await self.writer.revise(facts, draft, verdict["issues"])
        return strip_code_fence(draft), trace

    async def run(self, project_path, fmt: str = "md", output_path=None, token=None) -> dict:
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{fmt}'. Choose from: {sorted(SUPPORTED_FORMATS)}"
            )

        with resolve_input(project_path, token) as code_dir:
            facts = extract_from_directory(code_dir)
            llm_facts = slim_facts_for_llm(facts)   # trimmed payload for LLM calls
            result = {"format": fmt}

            # --- image diagrams (1 LLM call) -> slim payload ---
            if fmt in IMAGE_FORMATS:
                try:
                    saved = await self.diagrammer.diagram_typed(
                        llm_facts, output_path or f"architecture.{fmt}", fmt=fmt
                    )
                    result["saved_to"] = saved
                    if fmt == "png":
                        import base64
                        with open(saved, "rb") as f:
                            result["content"] = base64.b64encode(f.read()).decode("utf-8")
                        result["is_base64"] = True
                    else: # svg
                        with open(saved, "r", encoding="utf-8") as f:
                            result["content"] = f.read()
                except Exception as e:
                    import traceback
                    err_msg = f"Failed to generate technical architecture {fmt} diagram. Error: {str(e)}\n\n"
                    if "executable" in str(e).lower() or "not found" in str(e).lower() or "graphviz" in str(e).lower():
                        err_msg += (
                            "This usually means the Graphviz system library ('dot' executable) is not installed.\n"
                            "To fix this on Render, ensure you are deploying using the 'Docker' service environment "
                            "rather than the native 'Python' environment, so that the system dependencies in the "
                            "Dockerfile (graphviz) are installed."
                        )
                    else:
                        err_msg += traceback.format_exc()
                    result["error"] = err_msg
                    result["content"] = None
                return result

            if fmt == "drawio":
                try:
                    saved = await self.diagrammer.diagram_editable(
                        llm_facts, output_path or "architecture.svg"
                    )
                    result["saved_to"] = saved
                    with open(saved, "r", encoding="utf-8") as f:
                        result["content"] = f.read()
                except Exception as e:
                    import traceback
                    err_msg = f"Failed to generate drawio editable SVG. Error: {str(e)}\n\n"
                    if "executable" in str(e).lower() or "not found" in str(e).lower() or "graphviz" in str(e).lower():
                        err_msg += (
                            "This usually means the Graphviz system library ('dot' executable) is not installed.\n"
                            "To fix this on Render, ensure you are deploying using the 'Docker' service environment "
                            "rather than the native 'Python' environment, so that the system dependencies in the "
                            "Dockerfile (graphviz) are installed."
                        )
                    else:
                        err_msg += traceback.format_exc()
                    result["error"] = err_msg
                    result["content"] = None
                return result

            if fmt == "dot":
                try:
                    saved = await self.diagrammer.diagram_dot(
                        llm_facts, output_path or "architecture.gv"
                    )
                    result["saved_to"] = saved
                    with open(saved, "r", encoding="utf-8") as f:
                        result["content"] = f.read()
                except Exception as e:
                    import traceback
                    result["error"] = f"Failed to generate Graphviz DOT source: {str(e)}\n\n{traceback.format_exc()}"
                    result["content"] = None
                return result

            if fmt == "drawio_xml":
                try:
                    saved = await self.diagrammer.diagram_drawio(
                        llm_facts, output_path or "architecture.drawio"
                    )
                    result["saved_to"] = saved
                    with open(saved, "r", encoding="utf-8") as f:
                        result["content"] = f.read()
                except Exception as e:
                    import traceback
                    result["error"] = f"Failed to generate Draw.io XML: {str(e)}\n\n{traceback.format_exc()}"
                    result["content"] = None
                return result

            # --- structured spec (0 LLM calls) -> FULL facts ---
            content = None
            if fmt in STRUCTURED_FORMATS:
                try:
                    content = to_json(facts) if fmt == "json" else to_yaml(facts)
                except Exception as e:
                    import traceback
                    result["error"] = f"Failed to serialize facts: {str(e)}\n\n{traceback.format_exc()}"
            # --- mermaid diagram (1 LLM call) -> slim payload ---
            elif fmt in DIAGRAM_FORMATS:
                try:
                    content = strip_code_fence(await self.diagrammer.diagram(llm_facts))
                except Exception as e:
                    import traceback
                    result["error"] = f"Failed to generate Mermaid diagram: {str(e)}\n\n{traceback.format_exc()}"
            # --- prose: md or html (writer + reviewer) -> slim payload ---
            else:
                try:
                    markdown, trace = await self._write_reviewed_markdown(llm_facts)
                    result["review_trace"] = trace
                    content = markdown_to_html(markdown) if fmt == "html" else markdown
                except Exception as e:
                    import traceback
                    result["error"] = f"Failed to generate documentation: {str(e)}\n\n{traceback.format_exc()}"

            result["content"] = content
            if output_path and content:
                result["saved_to"] = save_text(output_path, content)
            return result


# Generate + save: python -m doc_agent.workflow.pipeline <project> <format> <output_file>
if __name__ == "__main__":
    import asyncio
    import sys

    project = sys.argv[1] if len(sys.argv) > 1 else "doc_agent"
    fmt = sys.argv[2] if len(sys.argv) > 2 else "md"
    output = sys.argv[3] if len(sys.argv) > 3 else None
    out = asyncio.run(DocumentationPipeline().run(project, fmt, output))
    print(out["content"])
    if out.get("saved_to"):
        print(f"\n\nSaved to: {out['saved_to']}")
    if out.get("review_trace"):
        print("\n--- REVIEW TRACE ---")
        print(json.dumps(out["review_trace"], indent=2))