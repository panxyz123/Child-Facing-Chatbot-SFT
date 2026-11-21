"""Structured per-request traces: one JSON line per request, for debugging and for load-test analysis.

Children's messages are not written by default: a trace has the message's length and a hash, the route, guard and
retrieval decisions, token usage and per-node latency. Set log_text=True only for local debugging.
"""
import hashlib
import json
import os
import threading
import time
import uuid


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class TraceLogger:
    def __init__(self, path: str, log_text: bool = False):
        self.path, self.log_text = path, log_text
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.lock = threading.Lock()

    def write(self, record: dict):
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self.lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def build_record(state: dict, out: dict, session_id: str, log_text: bool) -> dict:
    rec = dict(
        request_id=out["request_id"], ts=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
        session=short_hash(session_id), age=state.get("age"),
        message_chars=len(state.get("message", "")), message_hash=short_hash(state.get("message", "")),
        route=out["route"], p_knowledge=out["p_knowledge"], guard=out["guard"], gate=out["gate"],
        sources=out["sources"], top_score=out.get("top_score"), fkgl=out["fkgl"], output_flag=out["output_flag"],
        regenerations=out["regenerations"], history_turns=out.get("history_turns"),
        prompt_tokens_est=out.get("prompt_tokens_est"), usage=out.get("usage"),
        node_ms={t["node"]: round(sum(x["ms"] for x in out["trace"] if x["node"] == t["node"]), 1) for t in out["trace"]},
        latency_ms=out["latency_ms"],
    )
    if log_text:
        rec.update(text=state.get("text"), answer=out["answer"])
    return rec


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]
