import os
import typing as t
import fitz
from icecream import ic

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import ArxivLoader

from src.agents.agent import AgentCreator
from src.agents.paper_agent.utils.paper_agent_utils import get_user_question_part_contents
from src.common.common import CHAT_MODEL, RECENT_PAPER_SEARCH_RSS_FORMAT, PDF_DOWNLOAD_DIR
from src.common.prompts import (
    EXTRACT_ARXIV_PAPER_ID_PROMPT,
    MAKE_MARKDOWN_FORMAT_RECENT_PAPER_SUMMARY_PROMPT,
    EXTRACT_RECENT_PAPER_TYPE_PROMPT,
)


# 수정 필요할 듯(24.10.21)
agent_creator = AgentCreator()

# 1. paper searcher
paper_search_agent = agent_creator.create_chat_agent(
    tools=[],
    system_prompt=(
        "You are paper searcher. Your main job is below.\n"
        "\t1. extract arxiv paper id from user message.\n"
        "\t2. search found id paper from arxiv.\n"
        "\t3. download paper pdf, extract paper indexes.\n"
        "\t4. explain paper refer to paper summaries.\n"
    ),
)

# 2. paper handler
paper_handler_agent = agent_creator.create_chat_agent(
    tools=[],
    system_prompt=(
        "You are paper handler agent. Your main job is below.\n"
        "\t1. download paper pdf by arxiv paper id.\n"
        "\t2. extract paper {index name:page} informations from paper pdf.\n"
    )
)

# 3. paper team leader
## test를 위해 일단 chat agent로 생성
paper_team_leader = agent_creator.create_chat_agent(
    tools=[get_user_question_part_contents],
    system_prompt=(
        "You are paper explainer. Your main job is below.\n"
        "\t1. extract paper index name that user want to answer from user messages.\n"
        "\t2. get target index part from paper pdf using your tool.\n"
        "\t3. answer for user question following paper page content."
    )
)


if __name__ == "__main__":
    from langchain_core.messages import HumanMessage, AIMessage
    # print(paper_search_agent)
    # result = paper_search_agent.invoke(input={"messages": [HumanMessage(content="2402.09353 논문을 검색해줘")], "agent_scratchpad": []})
    # print(result)

    result = paper_team_leader.invoke(input={"messages": [HumanMessage(content="3. Pattern Analysis of LoRA and FT 부분을 설명해주세요")], "agent_scratchpad": []})
    print(result)