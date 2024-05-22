import typing as t
from icecream import ic

from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import ArxivLoader
from langchain_community.tools import DuckDuckGoSearchRun

from common import CHAT_MODEL, RECENT_PAPER_SEARCH_RSS_FORMAT
from prompts import (
    EXTRACT_ARXIV_PAPER_ID_PROMPT,
    MAKE_MARKDOWN_FORMAT_RECENT_PAPER_SUMMARY_PROMPT,
    EXTRACT_RECENT_PAPER_TYPE_PROMPT,
    POLISH_UP_PROMPTS
)
from utils.get_rss_url_values import get_processed_entries_from_rss_url


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
def get_recent_papers(user_input:str)->str:
    """사용자가 원하는 분야의 최신 논문들의 summary를 제공합니다.

    Args:
        user_input (str): 사용자 입력

    Returns:
        str: 최신 논문들을 정리한 markdown format text
    """
    client = ChatOpenAI(model=CHAT_MODEL, temperature=0.0)
    
    # 1. 사용자 입력에서 분야를 추출
    extract_paper_type_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EXTRACT_RECENT_PAPER_TYPE_PROMPT[0]),
            ("user", EXTRACT_RECENT_PAPER_TYPE_PROMPT[1])
        ]
    )
    extract_paper_type_chain = extract_paper_type_prompt | client
    paper_type = extract_paper_type_chain.invoke(input={"user_input": user_input}).content
    ic(paper_type)
    rss_url = RECENT_PAPER_SEARCH_RSS_FORMAT.format(paper_type)

    # 2. rss feed entries를 추출
    rss_entries = get_processed_entries_from_rss_url(rss_url)

    # 3. rss feed 기반 markdown을 생성(llm 사용)
    markdown_generate_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", MAKE_MARKDOWN_FORMAT_RECENT_PAPER_SUMMARY_PROMPT[0]),
            ("user", MAKE_MARKDOWN_FORMAT_RECENT_PAPER_SUMMARY_PROMPT[1])
        ]
    )
    markdown_generate_chain = markdown_generate_prompt | client
    recent_papers_markdown = markdown_generate_chain.invoke(input={"rss_entries": rss_entries}).content

    return recent_papers_markdown
    
    
@chain
def duckduckgo_search(user_input:str)->str:
    """internet search를 duckduckgo search를 사용해 진행합니다. 이후 검색 결과들을 llm을 통해 정리하고 반환합니다. 

    Args:
        user_input (str): 사용자 입력

    Returns:
        str: llm을 통해 정리된 인터넷 검색 결과
    """
    ddg_search = DuckDuckGoSearchRun()
    internet_search_result = ddg_search.invoke(user_input)
    ic(internet_search_result)

    polish_up_prompt = ChatPromptTemplate.from_messages(
        [
            # MessagesPlaceholder(variable_name="messages"),
            ("user", str(POLISH_UP_PROMPTS[0]))
        ]
    )
    polish_up_chain = polish_up_prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    polish_up_result = polish_up_chain.invoke(input={"internet_search_result": internet_search_result})

    return polish_up_result


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv("/Users/jeongminju/Documents/GITHUB/arxiv_paper_multi_agent/.env")

    # user_input = "2404.19756 논문 내용을 정리해줘."
    # arxiv_chain_result = arxiv_search_chain.invoke(input={"user_input": user_input})
    # ic(arxiv_chain_result[0].metadata)

    user_input = "자연어처리 분야의 최신 특허들을 나열"
    recent_paper_markdown = get_recent_papers.invoke(input={"user_input": user_input})

    ic(recent_paper_markdown)
