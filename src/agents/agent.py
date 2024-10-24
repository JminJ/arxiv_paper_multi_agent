import typing as t

from icecream import ic
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers import JsonOutputToolsParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI

from src.parser.supservisor_result_parser import parsing_supervisor_result


class AgentCreator:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
        )

    def create_supervisor_agent(self):
        """Create supervisor agent."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "## 1. SYSTEM PROMPT\n"
                    "You are the supervisor agent of 'arxiv paper agent system'.\n"
                    "User will asking about special paper or any informations. You have to answer them collaborate with 'paper_team' and 'web_search_team'.\n"
                    "Each of team can below works.\n"
                    "### 1.1 paper_team\nThe paper_team handling task about paper. special tasks are below.\n"
                    "1. search paper using arxiv paper id and download it and extract indexes from downloaded paper pdf\n2. get recent paper information that user want to search domain\n3. get index content user want to be explained from downloaded paper pdf.\n</paper_team>"
                    "If you need to collaborate with special team, then return team's name. Follow 'TEAM CALL FORMAT' part.\n"
                    "## 2. TEAM CALL FORMAT\n"
                    "<next_team>paper_team</next_team>\n if you call a team, don't write any annotations and write only this format message.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        supervisor_agent = prompt | self.llm

        return supervisor_agent

    def create_leader_agent(
        self,
        system_prompt: str,
        tools: t.Sequence[BaseTool],
        team_member_desc: t.List[t.Dict],
        next_roles: t.List[str],
    ) -> Runnable:
        """Create a team leader agent.

        Args:
            system_prompt (str): agent system prompt.
            tools (t.Sequence[BaseTool]): tools object that binding to llm.
            team_member_desc (t.List[t.Dict]): team member's descriptions.
            next_roles (t.List[str]): next role names.

        Returns:
            Runnable: Agent object.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "## TIPS\nin this part, the tips for agent object\n - tool and agent is different. so, when you move to next agent, don't generate tool call part, only generate next agent calling message following '5. HOW TO MOVE TO AGENTS?' part."
                    "## 1. DEFAULT SYSTEM PROMPT\nUse the provided tools to progress towards answering the question. If you can't answer user question, it's ok, move to other agents or your team member agent. when generate chat message, follow '1.1 CHAT RULE' part.\n"
                    "### 1.1 CHAT RULE\n- every answer should korean.\n"
                    "## 2. LEADER SYSTEM PROMPT\n{prompt}\n"
                    "## 3. TEAM MEMBER AGENTS\nthis is your team member agent's descriptions.\n{team_member_desc}\n"
                    "## 4. NEXT ROLES\nthis is agent list that you can move to(include team member agents): {next_roles}\n"
                    "## 5. HOW TO MOVE TO AGENTS?\nTo move to team member agent or other, generate only(not permit any annotations) message following this exact format:\n<next_agent>agent_name</next_agent>\n",
                ),
                MessagesPlaceholder(variable_name="messages"),
                # MessagesPlaceholder(variable_name="state_infos"), # graph state 정보 반영
            ]
        )
        prompt = (
            prompt.partial(prompt=system_prompt)
            .partial(team_member_desc=team_member_desc)
            .partial(next_roles=str(next_roles))
        )
        llm_with_tools = self.llm.bind_tools(tools=tools)

        agent = prompt | llm_with_tools
        return agent

    def create_chat_agent(
        self, tools: t.Sequence[BaseTool], system_prompt: str, tools_explain: str = None
    ) -> Runnable:
        """Create a function-calling agnet and add it to the graph.

        Args:
            llm (ChatOpenAI): llm agent that base of agent.
            tools (t.Sequence[BaseTool]): tools that used in agent.
            system_prompt (str): system prompt of this agent.
            tools_explain (str): explainations of this agent's tools. default is None.

        Returns:
            Runnable: Agent Object.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "## Common SYSTEM PROMPT\n"
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK, another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any of the other assistants have the final answer or deliverable,"
                    " prefix your response with FINAL ANSWER so the team knows to stop."
                    " every input and results should be korean.\n"
                    "## SYSTEM PROMPT\n{system_prompt}\n"
                    "## TOOLS\nYou have access to the following tools: {tool_names}"
                    "\n\nbelow, tool's explains.\n{tools_explain}",
                ),
                MessagesPlaceholder(variable_name="messages"),
                # MessagesPlaceholder(variable_name="state_infos"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        prompt = prompt.partial(system_prompt=system_prompt)

        # agent를 커스텀코드로 작성할지 고민 -> 매우 고민 파서가 그냥 내 결과값 다 날려버림(output 엉망)
        if bool(tools):
            llm_with_tools = self.llm.bind(functions=[convert_to_openai_function(t) for t in tools])
            if tools_explain is not None:
                prompt = prompt.partial(tools_explain=tools_explain)
            else:
                prompt = prompt.partial(
                    tools_explain=", ".join([tool.description for tool in tools])
                )
            prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        else:
            prompt = prompt.partial(tools_explain="")
            prompt = prompt.partial(tool_names="")
            llm_with_tools = self.llm

        agent = prompt | llm_with_tools

        # agent = create_openai_functions_agent(llm=self.llm, prompt=prompt, tools=tools)
        return agent
