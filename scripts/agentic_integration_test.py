"""Kai System Agentic Integration Test

Verifies that all four agentic frameworks can be imported and
basic objects instantiated without external API keys or services.
This is a structural / smoke test — it does NOT call any LLMs.

Frameworks tested:
  1. LangGraph  — graph construction
  2. AutoGen    — agent class instantiation
  3. CrewAI     — crew/task/agent wiring
  4. OpenAgents — agent runner class loading
"""
import sys
import os

# Ensure the workspace root is NOT first on sys.path so our local
# langgraph/ service directory does not shadow the installed package.
_ws = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys.path and os.path.abspath(sys.path[0]) == _ws:
    sys.path.pop(0)

passed = 0
failed = 0


def check(label: str, fn):
    """Run *fn* and report pass/fail."""
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {label}")
        passed += 1
    except Exception as exc:
        print(f"  [FAIL] {label}: {exc}")
        failed += 1


# ── 1. LangGraph ────────────────────────────────────────────────────
def test_langgraph():
    pytest = __import__("pytest")
    try:
        from langgraph.graph import StateGraph
    except ImportError:
        pytest.skip("langgraph.graph not installed")
        return
    from langgraph.graph import StateGraph
    from typing_extensions import TypedDict

    class St(TypedDict):
        value: str

    g = StateGraph(St)
    g.add_node("echo", lambda s: s)
    g.set_entry_point("echo")
    g.set_finish_point("echo")
    compiled = g.compile()
    assert compiled is not None, "StateGraph did not compile"


# ── 2. AutoGen ──────────────────────────────────────────────────────
def test_autogen():
    pytest = __import__("pytest")
    try:
        from autogen import AssistantAgent, UserProxyAgent
    except ImportError:
        pytest.skip("autogen not installed")
        return
    from autogen import AssistantAgent, UserProxyAgent

    # llm_config=False means no LLM calls — pure structural test
    assistant = AssistantAgent("kai-assistant", llm_config=False)
    user = UserProxyAgent("kai-user", llm_config=False, code_execution_config=False)
    assert assistant.name == "kai-assistant"
    assert user.name == "kai-user"


# ── 3. CrewAI ───────────────────────────────────────────────────────
def test_crewai():
    pytest = __import__("pytest")
    # CrewAI requires OPENAI_API_KEY even for object construction;
    # set a placeholder so the smoke test can verify wiring.
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder-not-real")
    try:
        from crewai import Agent, Task, Crew
    except ImportError:
        pytest.skip("crewai not installed")
        return
    from crewai import Agent, Task, Crew

    agent = Agent(role="planner", goal="plan tasks", backstory="test")
    task = Task(description="Plan a workflow.", expected_output="plan", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1


# ── 4. OpenAgents ───────────────────────────────────────────────────
def test_openagents():
    pytest = __import__("pytest")
    try:
        from openagents.agents.simple_agent import SimpleAutoAgent
    except ImportError:
        pytest.skip("openagents not installed")
        return
    from openagents.agents.simple_agent import SimpleAutoAgent
    assert SimpleAutoAgent is not None, "SimpleAutoAgent class not found"


# ── Run ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Kai System — Agentic Framework Smoke Test")
    print("=" * 48)
    check("LangGraph  — graph construction", test_langgraph)
    check("AutoGen    — agent instantiation", test_autogen)
    check("CrewAI     — crew/task wiring", test_crewai)
    check("OpenAgents — class loading", test_openagents)
    print("=" * 48)
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        print("WARNING: Some frameworks had issues — check API changes.")
        raise SystemExit(1)
    else:
        print("All agentic integration checks passed.")
        raise SystemExit(0)
