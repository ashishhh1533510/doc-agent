from fastapi import FastAPI
from pydantic import BaseModel
from doc_agent.agent import build_agent

app = FastAPI(title="Documentation Agent")
agent = build_agent()

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    result = await agent.run(req.message)
    return {"reply": getattr(result, "text", str(result))}