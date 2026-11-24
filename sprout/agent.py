"""Sprout agent: a LangGraph pipeline around the fine-tuned model.

  input_guard ──blocked / self-harm──────────────────────────────┐
      │ pass (personal details redacted)                          │
    router ──not a knowledge question───────────┐                 │
      │ knowledge                                │                 │
    retrieve (+ relevance gate)                  │                 │
      │                                          ▼                 │
      └────────────────────────────────────▶ generate ◀──┐         │
                                                 │       │ too hard: once more, simpler
                                            output_guard ┘         │
                                                 │ ok / unsafe     │
                                              finalize ◀───────────┘

Every node records its latency in the trace, so a response shows which path it took and where the time went.
"""
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from . import guards
from .config import AgentConfig, system_prompt
from .tracing import TraceLogger, build_record, new_request_id

MSG_OVERHEAD = 6     # chat-template tokens around each message (ChatML: <|im_start|>role, newline, <|im_end|>)
SENT_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")


def trim_to_sentence(text: str) -> str:
    """An answer cut at max_tokens ends mid-sentence: keep it up to the last complete sentence (unchanged when there
    is none, so a very long first sentence is still shown)."""
    ends = [m.end() for m in SENT_END.finditer(text)]
    return text[: ends[-1]].rstrip() if ends else text


class State(TypedDict, total=False):
    age: str
    message: str            # what the child typed
    text: str               # what the model sees (personal details redacted)
    history: List[Dict[str, str]]
    guard_category: Optional[str]
    guard_action: str
    p_knowledge: Optional[float]
    route: str              # blocked | chat | knowledge
    passages: List[Dict[str, Any]]
    gate: str               # not_needed | used | below_threshold | no_results
    simplify: bool
    regenerations: int
    answer: str
    truncated: bool         # the answer hit max_tokens and was trimmed to its last complete sentence
    guard_model_label: Optional[str]   # the output guard model's verdict when it blocked the answer
    usage: Optional[Dict[str, int]]
    prompt_tokens_est: int
    history_turns: int
    top_score: Optional[float]
    fkgl: Optional[float]
    output_flag: Optional[str]
    trace: List[Dict[str, Any]]


class SproutAgent:
    def __init__(self, config: AgentConfig = None, llm=None, retriever=None, router=None,
                 token_counter: Callable[[str], int] = None, guard_model=None):
        self.cfg = config or AgentConfig()
        c = self.cfg
        self.count_tokens = token_counter or self._default_counter()
        self.tracer = TraceLogger(c.trace_path, c.log_text) if c.trace_path else None
        self.lock = threading.Lock()
        if llm is None:
            from .llm import ChatLLM
            llm = ChatLLM(c.llm_base_url, c.llm_model, c.temperature, c.top_p, c.top_k, c.repetition_penalty, c.max_tokens)
        if retriever is None:
            from .retrieval import Retriever
            retriever = Retriever(c.db_path, c.table, c.retrieval_method, c.hybrid_alpha, c.rerank_pool,
                                  c.ann_nprobes, c.ann_refine)
        if router is None:
            from .router import KnowledgeRouter
            router = KnowledgeRouter(c.router_path)
        if guard_model is None and c.guard_url:
            guard_model = guards.ModelGuard(c.guard_url, c.guard_model, c.guard_controversial_unsafe,
                                            c.guard_fail_closed, c.guard_timeout, tokenizer=c.guard_tokenizer)
        self.llm, self.retriever, self.router, self.guard_model = llm, retriever, router, guard_model
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        self.graph = self._build()

    def _default_counter(self):
        """Token counts with the model's tokenizer; a words x 1.4 estimate if it can't be loaded."""
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.cfg.tokenizer)
            return lambda text: len(tok(text, add_special_tokens=False).input_ids)
        except Exception:
            return lambda text: int(len(text.split()) * 1.4) + 1

    # ------------------------------------------------------------------ nodes
    def _timed(self, name, fn):
        def node(state: State):
            t0 = time.perf_counter()
            out = fn(state)
            out["trace"] = state.get("trace", []) + [dict(node=name, ms=round(1000 * (time.perf_counter() - t0), 1))]
            return out
        return node

    def input_guard(self, s: State):
        r = guards.check_input(s["message"])
        out = dict(guard_category=r.category, guard_action=r.action, text=r.text)
        if not r.passed:
            out.update(route="blocked", answer=guards.safe_reply(r.category, s["age"], self.cfg.crisis_line))
        return out

    def route(self, s: State):
        p = self.router.p_knowledge(s["text"])
        return dict(p_knowledge=p, route="knowledge" if p >= self.cfg.knowledge_threshold else "chat",
                    gate="not_needed")

    def retrieve(self, s: State):
        hits = self.retriever.search(s["text"], self.cfg.top_k_docs)
        if not hits:
            return dict(passages=[], gate="no_results")
        t = self.cfg.gate_threshold
        if t is not None and hits[0].score < t:
            return dict(passages=[], gate="below_threshold", top_score=hits[0].score)
        return dict(passages=[dict(id=h.id, title=h.title, text=h.text, score=h.score) for h in hits], gate="used",
                    top_score=hits[0].score)

    def _system(self, s: State, passages):
        c = self.cfg
        system = system_prompt(s["age"])
        if passages:
            facts = "\n".join("- " + " ".join(p["text"].split()[: c.passage_words]) for p in passages)
            system += "\n\nFacts that may help (explain them in your own simple words):\n" + facts
        elif s.get("route") == "knowledge":
            system += "\nIf you are not sure about a fact, say so and suggest asking a grown-up."
        if s.get("simplify"):
            system += "\nUse short sentences and simple words."
        return system

    def build_messages(self, s: State):
        """Fit the prompt into the context window, leaving room for the answer. Priority: system prompt and the
        child's message, then retrieved passages (dropped from the last), then history (dropped from the oldest).
        Returns (messages, prompt_tokens_estimate, history_turns_used)."""
        c = self.cfg
        budget = c.context_tokens - c.max_tokens - 16
        n = lambda text: self.count_tokens(text) + MSG_OVERHEAD
        passages = list(s.get("passages", []))
        system = self._system(s, passages)
        used = n(system) + n(s["text"])
        while used > budget and passages:
            passages.pop()
            system = self._system(s, passages)
            used = n(system) + n(s["text"])
        turns = []
        for turn in reversed(s.get("history", [])[-c.history_turns:]):
            cost = n(turn["user"]) + n(turn["assistant"])
            if used + cost > budget:
                break
            turns.insert(0, turn)
            used += cost
        msgs = [dict(role="system", content=system)]
        for turn in turns:
            msgs += [dict(role="user", content=turn["user"]), dict(role="assistant", content=turn["assistant"])]
        return msgs + [dict(role="user", content=s["text"])], used, len(turns)

    def generate(self, s: State):
        msgs, est, turns = self.build_messages(s)
        if hasattr(self.llm, "chat"):
            answer, usage = self.llm.chat(msgs)
        else:
            answer, usage = self.llm(msgs), None
        truncated = bool(usage) and usage.get("finish_reason") == "length"
        if truncated:
            answer = trim_to_sentence(answer)
        if usage and s.get("usage"):        # a regeneration: add up the tokens of both calls
            usage = {k: (v + s["usage"].get(k, 0) if isinstance(v, (int, float)) else v) for k, v in usage.items()}
        return dict(answer=answer, usage=usage, prompt_tokens_est=est, history_turns=turns, truncated=truncated)

    def output_guard(self, s: State):
        c = self.cfg
        r = guards.check_output(s["answer"])
        if not r.passed:
            return dict(output_flag=r.category, answer=guards.safe_reply(r.category, s["age"], c.crisis_line))
        if self.guard_model is not None:            # second layer: the guard model reads message + answer
            m = self.guard_model.check(s["text"], s["answer"])
            if not m.passed:
                return dict(output_flag=m.category, guard_model_label=(m.matches or [None])[0],
                            answer=guards.safe_reply(m.category, s["age"], c.crisis_line))
        hard, g = guards.too_hard(s["answer"], s["age"], c.fkgl_band, c.fkgl_too_hard_margin, c.min_words_for_fkgl)
        n = s.get("regenerations", 0)
        if hard and n < c.max_regenerations:
            return dict(fkgl=g, output_flag="too_hard", simplify=True, regenerations=n + 1)
        return dict(fkgl=g, output_flag="too_hard_kept" if hard else None)

    def finalize(self, s: State):
        answer = s["answer"]
        if s.get("guard_action") == "redact":
            answer = answer + "\n\n" + guards.safe_reply("personal_info", s["age"])
        return dict(answer=answer)

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(State)
        for name, fn in [("input_guard", self.input_guard), ("router", self.route), ("retrieve", self.retrieve),
                         ("generate", self.generate), ("output_guard", self.output_guard), ("finalize", self.finalize)]:
            g.add_node(name, self._timed(name, fn))
        g.set_entry_point("input_guard")
        g.add_conditional_edges("input_guard", lambda s: "finalize" if s.get("route") == "blocked" else "router",
                                {"finalize": "finalize", "router": "router"})
        g.add_conditional_edges("router", lambda s: s["route"], {"knowledge": "retrieve", "chat": "generate"})
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", "output_guard")
        g.add_conditional_edges("output_guard", lambda s: "generate" if s.get("output_flag") == "too_hard" else "finalize",
                                {"generate": "generate", "finalize": "finalize"})
        g.add_edge("finalize", END)
        return g.compile()

    # ------------------------------------------------------------------ API
    def respond(self, message: str, age: str, session_id: str = "default", log: bool = True) -> Dict[str, Any]:
        if age not in self.cfg.fkgl_band:
            raise ValueError(f"age must be one of {list(self.cfg.fkgl_band)}")
        with self.lock:
            history = list(self.sessions.setdefault(session_id, []))
        t0 = time.perf_counter()
        s = self.graph.invoke(dict(age=age, message=message, history=history, regenerations=0, trace=[]))
        if s.get("route") != "blocked":      # memory keeps the redacted text, never blocked or self-harm turns
            with self.lock:
                h = self.sessions.setdefault(session_id, [])
                h.append(dict(user=s["text"], assistant=s["answer"]))
                del h[: -self.cfg.history_turns]
        out = dict(request_id=new_request_id(), answer=s["answer"], route=s.get("route"),
                   p_knowledge=s.get("p_knowledge"), guard=s.get("guard_category"), gate=s.get("gate"),
                   sources=[p["title"] for p in s.get("passages", [])], top_score=s.get("top_score"),
                   fkgl=s.get("fkgl"), output_flag=s.get("output_flag"), regenerations=s.get("regenerations", 0),
                   truncated=s.get("truncated", False), guard_model_label=s.get("guard_model_label"),
                   usage=s.get("usage"), prompt_tokens_est=s.get("prompt_tokens_est"),
                   history_turns=s.get("history_turns"), trace=s["trace"],
                   latency_ms=round(1000 * (time.perf_counter() - t0), 1))
        if log and self.tracer:
            self.tracer.write(build_record(s, out, session_id, self.cfg.log_text))
        return out

    def reset_session(self, session_id: str):
        """Forget a conversation (e.g. the parent changed the age, or asked to delete it)."""
        with self.lock:
            self.sessions.pop(session_id, None)
