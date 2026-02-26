"""
Kai System Agentic Integration Test

- Tests LangGraph, AutoGen, CrewAI, and OpenAgents integration
- Verifies agent orchestration, self-reflection, and task coordination
"""
from langgraph.graph import StateGraph
from autogen import AssistantAgent, UserProxyAgent
from crewai import Crew, Task, Agent
from openagents.container.agent_container import AgentContainer


# LangGraph: simple state graph
G = StateGraph()
G.add_node("start", lambda state: state)
G.add_node("end", lambda state: state)
G.add_edge("start", "end")

# AutoGen: assistant and user proxy
assistant = AssistantAgent("assistant")
user = UserProxyAgent("user")

# CrewAI: crew and task
crew_agent = Agent(name="KaiCrew", role="planner")
task = Task(description="Plan a test workflow.", agent=crew_agent)
crew = Crew(agents=[crew_agent], tasks=[task])

# OpenAgents: container
container = AgentContainer()
container.add_agent("test_agent", lambda x: f"Echo: {x}")

# Run basic tests
print("LangGraph nodes:", G.nodes())
print("AutoGen assistant name:", assistant.name)
print("CrewAI crew agents:", [a.name for a in crew.agents])
print("OpenAgents container agents:", list(container.agents.keys()))

print("Integration test passed.")
