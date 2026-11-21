"""Long-running HTTP service for the agent: models are loaded and warmed up once, then every request reuses them.
This is the process a load test talks to.

    uvicorn sprout.server:app --host 0.0.0.0 --port 8080
    curl -s localhost:8080/chat -H 'content-type: application/json' \\
         -d '{"message": "why is the sky blue?", "age": "7-9", "session_id": "demo"}'

Configuration comes from the same environment variables as the CLI (VLLM_BASE_URL, VLLM_MODEL, LANCE_STORAGE_PATH,
SPROUT_ROUTER, SPROUT_TRACE). Traces go to logs/traces.jsonl unless SPROUT_TRACE says otherwise.
"""
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import AgentConfig
from .router import AlwaysRetrieve

STATE = {}


def build_agent(cfg: AgentConfig):
    from .agent import SproutAgent
    router = None if os.path.exists(cfg.router_path) else AlwaysRetrieve()
    return SproutAgent(cfg, router=router)


def warm_up(agent):
    """First calls load lazy state (embedding and reranker kernels, textstat's dictionary, the LLM server's caches)."""
    t0 = time.perf_counter()
    agent.respond("tell me a story about a cat", "7-9", session_id="_warmup", log=False)
    agent.respond("why is the sky blue?", "7-9", session_id="_warmup", log=False)
    agent.reset_session("_warmup")
    return round(time.perf_counter() - t0, 1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = AgentConfig()
    cfg.trace_path = cfg.trace_path or "logs/traces.jsonl"
    agent = build_agent(cfg)
    STATE.update(agent=agent, warmup_s=warm_up(agent), started=time.time())
    yield


app = FastAPI(title="Sprout", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    age: Literal["0-3", "4-6", "7-9", "10-13", "14-18"]
    session_id: str = "default"
    trace: bool = False


class ResetRequest(BaseModel):
    session_id: str


@app.post("/chat")
def chat(req: ChatRequest):        # a sync endpoint: FastAPI runs it in a thread pool, so requests overlap
    if not req.message.strip():
        raise HTTPException(400, "empty message")
    out = STATE["agent"].respond(req.message, req.age, session_id=req.session_id)
    if not req.trace:
        out = {k: v for k, v in out.items() if k in ("request_id", "answer", "route", "latency_ms")}
    return out


@app.post("/reset")
def reset(req: ResetRequest):
    STATE["agent"].reset_session(req.session_id)
    return {"ok": True}


@app.get("/health")
def health():
    return {"ok": "agent" in STATE, "warmup_s": STATE.get("warmup_s")}
