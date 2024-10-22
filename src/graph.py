import functools 
import typing as t

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI

# from common.common import CHAT_MODEL
# from common.prompts import (
#     ALL_PAPER_TOOLS_EXPLAINS,
#     ALL_SEARCH_TOOLS_EXPLAINS,
#     PAPER_AGENT_SYSTEM_PROMPT,
#     SEARCH_AGENT_SYSTEM_PROMPT,
#     SUPERVISOR_AGENT_SYSTEM_PROMPT,
#     PAPER_AGENT_DESC,
#     SEARCH_AGENT_DESC
# )
from src.agents.agent import AgentCreator
from src.agents.paper_agent.utils.paper_agent_utils import arxiv_paper_search, get_recent_papers, get_user_question_part_contents
# from src.agents.paper_agent.paper_agent_nodes import paper_search_agent_node
from src.common.prompts import PAPER_TEAM_MEMBER_DESC_PROMPT
from src.state import ArxivMultiAgentState

from dotenv import load_dotenv
load_dotenv("../.env")


MEMBERS = [
    "paper_team_leader",
    # "search_team_leader",
    # "call_tool"
]

agent_creator = AgentCreator()
supervisor_agent = agent_creator.create_supervisor_agent()
paper_teamleader_agent = agent_creator.create_leader_agent(
    system_prompt="You are leader agent of 'paper_team'. collaborate with your team members and using your tools, answering to user's question.", 
    tools=[get_recent_papers, get_user_question_part_contents],
    team_member_desc=PAPER_TEAM_MEMBER_DESC_PROMPT,
    next_roles=["arxiv_paper_searcher"]
)
paper_arxiv_paper_search_agent = arxiv_paper_search

workflow = StateGraph(ArxivMultiAgentState)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("paper_team_leader", paper_teamleader_agent)
workflow.add_node("paper_arxiv_paper_searcher", paper_arxiv_paper_search_agent)
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

# workflow.add_conditional_edges(
#     "supervisor",
#     router,
#     {"continue": "supervisor", "Paper": "Paper", "Search": "Search", "call_tool": "call_tool", "__end__": END}
# )
# workflow.add_conditional_edges(
#     "Paper",
#     router,
#     {"continue": "supervisor", "call_tool": "call_tool", "__end__": END},
# )
# workflow.add_conditional_edges(
#     "Search",
#     router,
#     {"continue": "supervisor", "call_tool": "call_tool", "__end__": END},
# )

# workflow.add_conditional_edges(
#     "call_tool",
#     lambda x: x["sender"],
#     {
#         "Paper": "Paper",
#         "Search": "Search"
#     },
# )

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

    # chat_logs에 메세지를 넣는 방법을 생각해야 할듯
    # ic(chat_logs)