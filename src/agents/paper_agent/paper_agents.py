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
    get_recent_upload_papers,
    get_user_question_part_contents,
    paper_index_extract,
    search_paper_by_arxiv_id,
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

paper_team_leader_agent = agent_creator.create_leader_agent(
    system_prompt="You are Paper Team's leader.",
    team_member_desc=PAPER_TEAM_MEMBER_DESC_PROMPT,
    next_roles=["arxiv_paper_searcher", "supervisor"],
    # tools=[get_recent_papers, get_user_question_part_contents],
    tools=[get_recent_upload_papers, get_user_question_part_contents],
)
arxiv_paper_search_agent = agent_creator.create_chat_agent(
    tools=[search_paper_by_arxiv_id, paper_index_extract],
    system_prompt="You are Paper Team's member agent, 'arxiv_paper_search_agent'. You will do below jobs by your tools:\n- search paper by arxiv paper id.\n- download paper pdf\n- extract paper's indexes.",
    next_roles=["paper_team_leader"],
)
