import functools 
import typing as t

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI

from common.common import CHAT_MODEL
from common.prompts import (
    ALL_PAPER_TOOLS_EXPLAINS,
    ALL_SEARCH_TOOLS_EXPLAINS,
    PAPER_AGENT_SYSTEM_PROMPT,
    SEARCH_AGENT_SYSTEM_PROMPT,
    SUPERVISOR_AGENT_SYSTEM_PROMPT,
    PAPER_AGENT_DESC,
    SEARCH_AGENT_DESC
)
from agents.agent import create_agent, agent_node, create_supervisor_agent
from tools import (
    arxiv_paper_search_tool,
    paper_index_read_tool,
    extract_index_contents_tool,
    get_recent_papers_tool,   
    duckduckgo_search_tool
)
from state import ArxivMultiAgentState

from dotenv import load_dotenv
load_dotenv("../.env")


llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.7)

paper_agent = create_agent(
    llm,
    [arxiv_paper_search_tool, paper_index_read_tool, extract_index_contents_tool, get_recent_papers_tool],
    system_prompt=PAPER_AGENT_SYSTEM_PROMPT,
    tools_explain=ALL_PAPER_TOOLS_EXPLAINS
)
paper_node = functools.partial(agent_node, agent=paper_agent, name="Paper")

search_agent = create_agent(
    llm,
    [duckduckgo_search_tool],
    system_prompt=SEARCH_AGENT_SYSTEM_PROMPT,
    tools_explain=ALL_SEARCH_TOOLS_EXPLAINS
)
search_node = functools.partial(agent_node, agent=search_agent, name="Search")

# run_tools = [
#     arxiv_paper_search_tool,
#     paper_index_read_tool,
#     extract_index_contents_tool,
#     get_recent_papers_tool, 
#     duckduckgo_search_tool
# ]
# tool_node = ToolNode(run_tools)

MEMBERS = [
    "Paper",
    "Search",
    # "call_tool"
]

supervisor_agent = create_supervisor_agent(
    llm,
    system_prompt=SUPERVISOR_AGENT_SYSTEM_PROMPT,
    agent_desc=[PAPER_AGENT_DESC, SEARCH_AGENT_DESC],
    next_roles=MEMBERS + ["FINISH"]
)
supervisor_node = functools.partial(agent_node, agent=supervisor_agent, name="supervisor")

# def router(state) -> t.Literal["call_tool", "__end__", "continue"]:
#     messages = state["messages"]
#     last_message = messages[-1]
#     if last_message.tool_calls:
#         return "call_tool"
#     if "FINAL ANSWER" in last_message.content:
#         return "__end__"
#     if "Patent" in last_message.content:
#         return "Paper"
#     if "Search" in last_message.content:
#         return "Search"
#     return "continue"


workflow = StateGraph(ArxivMultiAgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("Paper", paper_agent)
workflow.add_node("Search", search_node)
# workflow.add_node("call_tool", tool_node)

for member in MEMBERS:
    # We want our workers to ALWAYS "report back" to the supervisor when done
    workflow.add_edge(member, "supervisor")
# workflow.add_edge("supervisor", "supervisor")
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