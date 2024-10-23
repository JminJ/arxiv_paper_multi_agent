import os
import typing as t

import fitz
from icecream import ic
from langchain_community.document_loaders import ArxivLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.agents.agent import AgentCreator
from src.agents.paper_agent.utils.paper_agent_utils import (
    get_recent_papers,
    get_user_question_part_contents,
)
from src.common.common import (
    CHAT_MODEL,
    PDF_DOWNLOAD_DIR,
    RECENT_PAPER_SEARCH_RSS_FORMAT,
)
from src.common.prompts import (
    EXTRACT_ARXIV_PAPER_ID_PROMPT,
    EXTRACT_RECENT_PAPER_TYPE_PROMPT,
    MAKE_MARKDOWN_FORMAT_RECENT_PAPER_SUMMARY_PROMPT,
    PAPER_TEAM_MEMBER_DESC_PROMPT,
)

agent_creator = AgentCreator()

# 1. supervisor
supervisor_agent = agent_creator.create_supervisor_agent()
# 2. paper team leader
paper_team_leader_agent = agent_creator.create_leader_agent(
    system_prompt="You are Paper Team's leader.",
    team_member_desc=PAPER_TEAM_MEMBER_DESC_PROMPT,
    next_roles=["arxiv_paper_searcher"],
    tools=[get_recent_papers, get_user_question_part_contents],
)