import typing as t
from icecream import ic

from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import ArxivLoader

from common import CHAT_MODEL
from prompts import (
    EXTRACT_ARXIV_PAPER_ID_PROMPT,
    RECENT_PAPER_SEARCH_RSS_FORMAT
)
from utils.get_rss_url_values import get_parse_rss_url

@chain
def arxiv_search_chain(user_input:str)->t.List[t.Dict]:
    """arxiv paper id로 paper를 검색합니다.

    Args:
        user_input (str): 사용자 입력

    Returns:
        t.List[t.Dict]: paper info list
    """
    client = ChatOpenAI(model=CHAT_MODEL, temperature=0.0)

    # 1. extract arxiv paper id from user_input.
    extract_paper_id_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EXTRACT_ARXIV_PAPER_ID_PROMPT[0]),
            ("user", EXTRACT_ARXIV_PAPER_ID_PROMPT[1])
        ]
    )

    extract_paper_id_chain = extract_paper_id_prompt | client
    paper_id = eval(extract_paper_id_chain.invoke(input={"user_input": user_input}).content)
    ic(paper_id)

    # 2. search arxiv paper
    paper_infos = [ArxivLoader(query=id, doc_content_chars_max=1).load()[0] for id in paper_id]
    ic(paper_infos)

    return paper_infos


@chain
def get_recent_patents(user_input:str):
    client = ChatOpenAI(model=CHAT_MODEL, temperature=0.0)
    
    # 1. 사용자 입력에서 분야를 추출
    extract_paper_type = ChatPromptTemplate.from_messages(
        [
            ("system", None),
            ("user", None)
        ]
    )

    extract_paper_type_chain = extract_paper_type | client
    paper_type = extract_paper_type_chain.invoke(input={"user_input": user_input})
    rss_url = RECENT_PAPER_SEARCH_RSS_FORMAT.format(paper_type)

    rss_values = get_parse_rss_url(rss_url)
    




if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/Users/jeongminju/Documents/GITHUB/langgraph_multi_agent/.env")

    user_input = "2404.19756 논문 내용을 정리해줘."
    arxiv_chain_result = arxiv_search_chain.invoke(input={"user_input": user_input})
    ic(arxiv_chain_result[0].metadata)
