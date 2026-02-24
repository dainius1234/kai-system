from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="KAI Advisor", version="0.1.0")

MODEL = os.getenv("KAI_MODEL", "deepseek-v4")
USE_GPU = os.getenv("USE_GPU", "false").lower() in ("1", "true", "yes")
DEVICE = "cuda" if USE_GPU else "cpu"

knowledge: List[str] = []
for root, dirs, files in os.walk("docs"):
    for fname in files:
        if fname.endswith(".md"):
            try:
                with open(os.path.join(root, fname), encoding="utf-8") as fh:
                    knowledge.append(fh.read())
            except Exception:
                pass

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "model": MODEL, "device": DEVICE, "knowledge_chunks": len(knowledge)}

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    model: str
    device: str

@app.post("/ask", response_model=QueryResponse)
async def ask(req: QueryRequest) -> QueryResponse:
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")
    answer = None
    for chunk in knowledge:
        if q.lower() in chunk.lower():
            answer = chunk.split(q, 1)[-1].strip()
            break
    if answer is None:
        answer = f"I heard: '{q}', but I'm just a simple KAI advisor stub."
    return QueryResponse(question=q, answer=answer, model=MODEL, device=DEVICE)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8090")))
