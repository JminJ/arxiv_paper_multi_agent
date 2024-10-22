import typing as t
from icecream import ic

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import JsonOutputToolsParser
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ToolMessage
from langchain_openai import ChatOpenAI
from parser.supservisor_result_parser import parsing_supervisor_result


def create_agent(llm: ChatOpenAI, tools: t.List[Runnable], system_prompt:str, tools_explain:str) -> AgentExecutor:
    """Create a function-calling agnet and add it to the graph.

     Args:
        llm (ChatOpenAI): llm agent that base of agent.
        tools (t.List[Runnable]): tools that used in agent.
        system_prompt (str): system prompt of this agent.
        tools_explain (str): explainations of this agent's tools.
    """ 
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " every input and results should be korean."
                " You have access to the following tools: {tool_names}.\n\nbelow, tool's explains.'n{tools_explain}\n\n[system prompt]\n{system_prompt}",
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
    )
    prompt = prompt.partial(system_prompt=system_prompt)
    prompt = prompt.partial(tools_explain=tools_explain)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))

    # agent를 커스텀코드로 작성할지 고민 -> 매우 고민 파서가 그냥 내 결과값 다 날려버림(output 엉망)
    agent = create_openai_functions_agent(llm=llm, prompt=prompt, tools=tools)

    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    ic(type(agent))
    result = agent.invoke(state)
    # why?
    ic(state)
    # ic(agent)
    ic(name)
    ic(result)

    # if len(result) == 0:
    #     return {"messages": [HumanMessage(content="", name=name)], "next": "supervisor"}
    # return {"messages": [HumanMessage(content=result["output"], name=name)], "next": "supervisor"}
    if len(result) == 0:
        return {"messages": [AIMessage(content="", name=name)], "next": "supervisor"}
    return {"messages": [AIMessage(content=result["output"], name=name)], "next": "supervisor"}


def create_supervisor_agent(llm: ChatOpenAI, system_prompt:str, agent_desc:t.List[t.Dict], next_roles:t.List[str]) -> RunnableLambda:
    """Create a supervisor agent. It choose propered agent in graph.

    Args:
        llm (ChatOpenAI): a base llm of supervisor agent.
        system_prompt (str): a system prompt for base llm.
        agent_desc (t.List[t.Dict]): descriptions of each agents.
        next_roles (t.List[str]): next role names.

    Returns:
        RunnableLambda
    """
    system_prompt += f"Below, I'll give you descriptions: {str(agent_desc)}"
    agent_desc_prompt = f"Given the conversation logs above, who should act next? Or should we FINISH? Select one of: {next_roles}"
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
            ),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                agent_desc_prompt
            )
        ]
    )
    ic(prompt)
    supervisor_next_role_func = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": next_roles},
                    ],
                }
            },
            "required": ["next"],
        },
    }
    prompt = prompt.partial(system_prompt=system_prompt)
    return (
        prompt 
        | llm.bind_functions(functions=[supervisor_next_role_func], function_call="route")
        # | parsing_supervisor_result
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    from tools import *
    from common.prompts import ALL_PAPER_TOOLS_EXPLAINS, PAPER_AGENT_SYSTEM_PROMPT, SUPERVISOR_AGENT_SYSTEM_PROMPT, PAPER_AGENT_DESC, SEARCH_AGENT_DESC
    from langchain_core.messages import HumanMessage
    from icecream import ic

    load_dotenv("/Users/jeongminju/Documents/GITHUB/arxiv_paper_multi_agent/.env")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

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

    ic(supervisor_agent.invoke({"messages": [HumanMessage(content="자연어처리 분야 최신논문들을 줘")]}))
    # ic(paper_agent.invoke({"messages": [HumanMessage(content="최신 자연어처리 분야의 논문들을 보여줘")]}))

    