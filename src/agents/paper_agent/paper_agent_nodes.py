import os
from langchain_community.document_loaders import ArxivLoader
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.utils.paper_pdf_handler import paper_pdf_download, paper_pdf_load
from src.utils.get_paper_page_indexes import ExtractPaperIndexes
from src.agents.paper_agent.utils.paper_agent_utils import arxiv_paper_search, paper_index_read, get_recent_papers
from src.agents.paper_agent.paper_agents import paper_search_agent
from src.common.common import CHAT_MODEL, RECENT_PAPER_SEARCH_RSS_FORMAT, PDF_DOWNLOAD_DIR
from src.common.prompts import (
    EXTRACT_ARXIV_PAPER_ID_PROMPT,
    MAKE_MARKDOWN_FORMAT_RECENT_PAPER_SUMMARY_PROMPT,
    EXTRACT_RECENT_PAPER_TYPE_PROMPT,
)

# paper search agent node
def paper_search_agent_node(state, config):
    last_message = state["messages"][-1]
    paper_infos, paper_indexes, target_paper_path = arxiv_paper_search.invoke(input=last_message)
    print(f"paper_infos: {paper_infos}")

    # paper information explain
    explainer = ChatOpenAI(model=CHAT_MODEL)
    explain_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", "You are paper explainer. Refer to given paper summary, introducing paper to User. Result should be korean.\n"
            ),
            (
                "user", "{paper_infos}"
            )
        ]
    )
    explain_chain = explain_prompt | explainer

    result = explain_chain.invoke(input={"paper_infos": paper_infos})
    return {"messages": [result], "paper_indexes": paper_indexes, "target_paper_path": target_paper_path}


if __name__ == "__main__":
    from icecream import ic
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [
                # HumanMessage(content="최신 자연어처리 분야의 논문들의 정보를 보여줘")
                HumanMessage(content="2402.09353 논문을 검색해줘")
            ],
        "agent_scratchpad": [],
        # "pdf_object": "",
        # "paper_indexes": {},
        # "next": None
    }

    result = paper_search_agent_node(state=state, config=None)
    ic(result)


    