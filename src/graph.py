import functools
import typing as t

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.agent import AgentCreator
from src.agents.agent_node import agent_node
from src.agents.paper_agent.paper_agents import (
    arxiv_paper_search_agent,
    paper_team_leader_agent,
)
from src.agents.paper_agent.utils.paper_agent_utils import (
    get_recent_papers,
    get_user_question_part_contents,
    paper_index_extract,
    search_paper_by_arxiv_id,
)
from src.state import ArxivMultiAgentState

load_dotenv("../.env")


## 1. supervisor agent define
agent_creator = AgentCreator()
supervisor_agent = agent_creator.create_supervisor_agent()

## 2. agent node define
supervisor_agent_node = functools.partial(agent_node, agent=supervisor_agent, name="supervisor")
paper_team_leader_agent_node = functools.partial(
    agent_node, agent=paper_team_leader_agent, name="paper_team_leader"
)
arxiv_paper_search_agent_node = functools.partial(
    agent_node, agent=arxiv_paper_search_agent, name="arxiv_paper_searcher"
)
tool_node = ToolNode(
    tools=[
        search_paper_by_arxiv_id,
        paper_index_extract,
        get_recent_papers,
        get_user_question_part_contents,
    ]
)  # 추후 search tools도 적재


## 3. define router function
def router(state):
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    if "<FINISHED>" in last_message.content:
        return END
    return state["next_role"]


## 3. generate graph
workflow = StateGraph(ArxivMultiAgentState)
workflow.add_node("supervisor", supervisor_agent_node)
workflow.add_node("paper_team_leader", paper_team_leader_agent_node)
workflow.add_node("arxiv_paper_searcher", arxiv_paper_search_agent_node)
workflow.add_node("call_tool", tool_node)

workflow.add_conditional_edges(
    "supervisor", router, {"paper_team_leader": "paper_team_leader", END: END}
)
workflow.add_conditional_edges(
    "paper_team_leader",
    router,
    {"arxiv_paper_searcher": "arxiv_paper_searcher", "call_tool": "call_tool"},
)
workflow.add_conditional_edges("arxiv_paper_searcher", router, {"call_tool": "call_tool"})
workflow.add_edge("paper_team_leader", "supervisor")
workflow.add_edge("arxiv_paper_searcher", "paper_team_leader")

workflow.set_entry_point("supervisor")
graph = workflow.compile()


if __name__ == "__main__":
    from icecream import ic

    chat_logs = []
    events = graph.stream(
        {
            "messages": [
                # HumanMessage(content="최신 자연어처리 분야의 논문들의 정보를 보여줘")
                HumanMessage(content="search recent paper in nlp domain.")
            ]
        },
    )
    for s in events:
        print(f"\n\nevent: {s}")
        print("-----")
        chat_logs.append(s)
