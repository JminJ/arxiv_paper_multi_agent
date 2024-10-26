import functools 
import typing as t

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI

from src.common.prompts import PAPER_TEAM_MEMBER_DESC_PROMPT
from src.state import ArxivMultiAgentState
from src.agents.agent import AgentCreator
from src.agents.agent_node import agent_node
from src.agents.paper_agent.paper_agents import get_recent_papers, get_user_question_part_contents

from dotenv import load_dotenv
load_dotenv("../.env")



## 1. agent define
agent_creator = AgentCreator()

supervisor_agent = agent_creator.create_supervisor_agent()

paper_team_leader_agent = agent_creator.create_leader_agent(
    system_prompt="You are Paper Team's leader.",
    team_member_desc=PAPER_TEAM_MEMBER_DESC_PROMPT,
    next_roles=["arxiv_paper_searcher"],
    tools=[get_recent_papers, get_user_question_part_contents],
)
arxiv_paper_search_agent = agent_creator.create_chat_agent(
    system_prompt="You are Paper Team's member agent, 'arxiv_paper_search_agent'. You will do below jobs:\n1. search paper by arxiv paper id.n\2. download that paper\n3. extract paper's indexes.",
    next_roles=["paper_team_leader", "supervisor"]
)

## 2. agent node define
supervisor_agent_node = functools.partial(agent_node, agent=supervisor_agent, name="supervisor")
paper_team_leader_agent_node = functools.partial(agent_node, agent=paper_team_leader_agent, name="paper_team_leader")


## 3. generate graph
MEMBERS = [
    "paper_team_leader",
    # "search_team_leader",
    # "call_tool"
]
workflow = StateGraph(ArxivMultiAgentState)
workflow.add_node("supervisor", supervisor_agent_node)
workflow.add_node("paper_team_leader", paper_team_leader_agent_node)
workflow.add_node("paper_arxiv_paper_searcher", )
# workflow.add_node("call_tool", tool_node)

for member in MEMBERS:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
workflow.add_conditional_edges("paper_team_leader", lambda x: x["next"], {"arxiv_paper_searcher": "paper_arxiv_paper_searcher"})
workflow.add_edge("paper_arxiv_paper_searcher", "paper_team_leader")

conditional_map = {k: k for k in MEMBERS}
conditional_map["FINISH"] = END
print(f"conditional_map: {conditional_map}")
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)


workflow.set_entry_point("supervisor")
print(workflow._all_edges)
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
