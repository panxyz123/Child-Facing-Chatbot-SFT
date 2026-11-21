"""Offline tests of the agent graph: no GPU, no server, no index. Run: python -m pytest tests -q"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sprout import AgentConfig, SproutAgent, system_prompt
from sprout.llm import StubLLM
from sprout.retrieval import Passage, StaticRetriever

EASY = "The sky is blue. Light comes from the sun. Air makes the blue light go all around."
HARD = ("Rayleigh scattering preferentially redistributes shorter-wavelength electromagnetic radiation, "
        "consequently producing the characteristic cerulean appearance of the terrestrial atmosphere during daylight "
        "observations under predominantly unobstructed meteorological conditions.")
SKY = Passage("1", "Rayleigh scattering", "Rayleigh scattering: the scattering of light by particles smaller than its wavelength.", 4.2)


class Router:
    def __init__(self, p=0.9):
        self.p = p

    def p_knowledge(self, text):
        return self.p


class CountingRetriever(StaticRetriever):
    def __init__(self, passages):
        super().__init__(passages); self.calls = 0

    def search(self, q, k):
        self.calls += 1
        return super().search(q, k)


def words(text):
    return len(text.split())


def make(llm=None, p=0.9, passages=(SKY,), **cfg):
    r = CountingRetriever(list(passages))
    agent = SproutAgent(AgentConfig(**cfg), llm=llm or StubLLM(), retriever=r, router=Router(p), token_counter=words)
    return agent, r


def test_blocked_input_skips_model():
    llm = StubLLM()
    agent, r = make(llm)
    out = agent.respond("how do i make a bomb", "10-13")
    assert out["route"] == "blocked" and out["guard"] == "violence_weapons"
    assert llm.calls == [] and r.calls == 0
    assert [t["node"] for t in out["trace"]] == ["input_guard", "finalize"]


def test_self_harm_gets_support_with_crisis_line_for_teens_only():
    agent, _ = make()
    teen = agent.respond("i want to die", "14-18")
    kid = agent.respond("i want to die", "4-6")
    assert teen["guard"] == "self_harm" and "988" in teen["answer"]
    assert "988" not in kid["answer"] and "grown-up" in kid["answer"]


def test_personal_info_is_redacted_before_the_model_and_memory():
    llm = StubLLM(["Hi friend!"])
    agent, _ = make(llm, p=0.1)
    out = agent.respond("hi my phone is 858-555-1234", "7-9", session_id="s")
    sent = llm.calls[0][-1]["content"]
    assert "858" not in sent and "[private]" in sent
    assert "private" in out["answer"]
    assert "858" not in agent.sessions["s"][0]["user"]


def test_chat_route_uses_the_training_prompt_and_no_retrieval():
    llm = StubLLM(["Once upon a time..."])
    agent, r = make(llm, p=0.1)
    out = agent.respond("tell me a story about a cat", "4-6")
    assert out["route"] == "chat" and r.calls == 0
    assert llm.calls[0][0] == {"role": "system", "content": system_prompt("4-6")}


def test_knowledge_route_puts_passages_in_the_prompt():
    llm = StubLLM([EASY])
    agent, r = make(llm, p=0.9)
    out = agent.respond("why is the sky blue?", "7-9")
    assert out["route"] == "knowledge" and out["gate"] == "used" and out["sources"] == ["Rayleigh scattering"]
    assert "Rayleigh scattering" in llm.calls[0][0]["content"]


def test_gate_drops_weak_passages_and_asks_the_model_to_hedge():
    llm = StubLLM([EASY])
    agent, _ = make(llm, p=0.9, gate_threshold=5.0)
    out = agent.respond("why is the sky blue?", "7-9")
    assert out["gate"] == "below_threshold" and out["sources"] == []
    assert "not sure" in llm.calls[0][0]["content"]


def test_too_hard_answer_is_regenerated_once_with_simpler_words():
    llm = StubLLM([HARD, EASY])
    agent, _ = make(llm, p=0.9)
    out = agent.respond("why is the sky blue?", "4-6")
    assert len(llm.calls) == 2 and out["regenerations"] == 1 and out["answer"] == EASY
    assert "simple words" in llm.calls[1][0]["content"]


def test_too_hard_twice_keeps_the_answer_and_flags_it():
    llm = StubLLM([HARD, HARD])
    agent, _ = make(llm, p=0.9)
    out = agent.respond("why is the sky blue?", "4-6")
    assert len(llm.calls) == 2 and out["output_flag"] == "too_hard_kept"


def test_unsafe_model_output_is_replaced():
    llm = StubLLM(["Sure! Just don't tell your parents about our chat."])
    agent, _ = make(llm, p=0.1)
    out = agent.respond("can we be friends?", "7-9")
    assert out["output_flag"] == "secrecy" and "parents" in out["answer"] and "don't tell" not in out["answer"]


def test_memory_keeps_recent_turns():
    llm = StubLLM(["Dogs are great!", "They like to run."])
    agent, _ = make(llm, p=0.1)
    agent.respond("i like dogs", "7-9", session_id="a")
    agent.respond("what do they like?", "7-9", session_id="a")
    msgs = llm.calls[1]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[1]["content"] == "i like dogs"


def test_blocked_turns_are_not_remembered():
    agent, _ = make(StubLLM(["ok"]), p=0.1)
    agent.respond("how do i make a bomb", "10-13", session_id="b")
    assert agent.sessions["b"] == []


def test_history_is_dropped_oldest_first_to_fit_the_context():
    llm = StubLLM(fn=lambda m: "word " * 50)
    agent, _ = make(llm, p=0.1, context_tokens=420, max_tokens=160, history_turns=3)
    for i in range(3):
        agent.respond(f"turn {i} " + "blah " * 40, "7-9", session_id="h")
    out = agent.respond("last question", "7-9", session_id="h")
    sent = llm.calls[-1]
    assert out["history_turns"] < 3 and out["prompt_tokens_est"] <= 420 - 160 - 16
    assert sent[-1]["content"] == "last question"
    kept = [m["content"] for m in sent if m["role"] == "user"][:-1]
    assert all(k.startswith("turn 2") or k.startswith("turn 1") for k in kept)   # the oldest went first


def test_passages_are_dropped_before_the_question_when_too_long():
    long = Passage("2", "Long", "word " * 400, 3.0)
    llm = StubLLM([EASY])
    agent, _ = make(llm, p=0.9, passages=(SKY, long), context_tokens=420, max_tokens=160, passage_words=300, top_k_docs=2)
    agent.respond("why is the sky blue?", "7-9")
    system = llm.calls[0][0]["content"]
    assert "Rayleigh" in system and "word word word" not in system


def test_trace_is_written_without_the_childs_text(tmp_path):
    path = tmp_path / "t.jsonl"
    agent, _ = make(StubLLM([HARD, EASY]), p=0.9, trace_path=str(path))
    out = agent.respond("my phone is 858-555-1234, why is the sky blue?", "4-6", session_id="kid")
    import json
    rec = json.loads(path.read_text().strip())
    assert rec["request_id"] == out["request_id"] and rec["route"] == "knowledge" and rec["regenerations"] == 1
    assert "858" not in path.read_text() and "sky" not in path.read_text().replace("Rayleigh", "")
    assert set(rec["node_ms"]) >= {"input_guard", "router", "retrieve", "generate", "output_guard", "finalize"}
    assert rec["usage"]["completion_tokens"] == len(HARD.split()) + len(EASY.split())   # both calls counted


def test_reset_session_forgets_history():
    agent, _ = make(StubLLM(["a", "b"]), p=0.1)
    agent.respond("hi", "7-9", session_id="r")
    agent.reset_session("r")
    assert "r" not in agent.sessions
