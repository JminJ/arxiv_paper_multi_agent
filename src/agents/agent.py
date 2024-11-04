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
                    "You are the supervisor agent of 'arxiv paper agent system'. Every answers to user, be kindly, detaily.\n"
                    "User will asking about special paper or any informations. You have to answer them collaborate with 'paper_team_leader' and 'web_search_team_leader'.\n"
                    "Each of team can below works.\n"
                    "### 1.1 paper_team\nThe paper_team handling task about paper. special tasks are below.\n"
                    "a. search paper using arxiv paper id and download it and extract indexes from downloaded paper pdf\nb. get recent paper information that user want to search domain\nc. get index content user want to be explained from downloaded paper pdf.\n</paper_team>"
                    "### 1.2 search_team\nThe search_team can search special information from internet. special tasks are below.\n"
                    "a. search special subject from internet.\n</search_team>"
                    "If you need to collaborate with special team, then return team leader's name. Follow 'TEAM CALL FORMAT' part.\n"
                    "## 2. TEAM CALL FORMAT\n"
                    "<next_agent>paper_team_leader</next_agent>\n if you call a team leader, don't write any annotations and write only this format message."
                    "## 3. JUDGE END POINT OF CHAT"
                    "if you think that the user's question answerd perfectly, or just you can answer to user input(don't need to move another agent) like just greeting, then you should write end mark '<FINISHED>' to message.",
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
                    "## 5. HOW TO MOVE TO AGENTS?\nTo move to team member agent or other, generate only(not permit any annotations) message following this exact format:\n<next_agent>agent_name</next_agent>\n"
                    "## JUDGE END POINT OF CHAT\n"
                    "if you think that the user's question answerd perfectly, or just you can answer to user input(don't need to move another agent) like just greeting, then you should write end mark '<FINISHED>' to message.",
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
        self, tools: t.Sequence[BaseTool], system_prompt: str, next_roles: str
    ) -> Runnable:
        """Create a function-calling agnet and add it to the graph.

        Args:
            llm (ChatOpenAI): llm agent that base of agent.
            tools (t.Sequence[BaseTool]): tools that used in agent.
            system_prompt (str): system prompt of this agent.
            next_roles (t.List[str]): next role names.

        Returns:
            Runnable: Agent Object.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "## TIPS\nin this part, the tips for agent object\n - tool and agent is different. so, when you move to next agent, don't generate tool call part, only generate next agent calling message following '5. HOW TO MOVE TO AGENTS?' part."
                    "## 1. DEFAULT SYSTEM PROMPT\nUse the provided tools to progress towards answering the question. If you can't answer user question, it's ok, move to other agents. when generate chat message, follow '1.1 CHAT RULE' part.\n"
                    "### 1.1 CHAT RULE\n- every answer should korean.\n"
                    "## 2. SYSTEM PROMPT\n{prompt}\n"
                    "## 3. NEXT ROLES\nthis is agent list that you can move to: {next_roles}\n"
                    "## 4. HOW TO MOVE TO AGENTS?\nTo move to team member agent or other, generate only(not permit any annotations) message following this exact format:\n<next_agent>agent_name</next_agent>\n"
                    "## JUDGE END POINT OF CHAT\n"
                    "if you think that the user's question answerd perfectly, or just you can answer to user input(don't need to move another agent) like just greeting, then you should write end mark '<FINISHED>' to message.",
                ),
                MessagesPlaceholder(variable_name="messages"),
                # MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        prompt = prompt.partial(prompt=system_prompt).partial(next_roles=str(next_roles))
        llm_with_tools = self.llm.bind_tools(tools=tools)

        agent = prompt | llm_with_tools

        return agent
