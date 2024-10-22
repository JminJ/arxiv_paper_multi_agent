import os
import typing as t
import fitz
from icecream import ic

from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import ArxivLoader

from src.common.common import CHAT_MODEL, RECENT_PAPER_SEARCH_RSS_FORMAT, PDF_DOWNLOAD_DIR
from src.common.prompts import (
    EXTRACT_ARXIV_PAPER_ID_PROMPT,
    MAKE_MARKDOWN_FORMAT_RECENT_PAPER_SUMMARY_PROMPT,
    EXTRACT_RECENT_PAPER_TYPE_PROMPT,
)
from src.utils.get_rss_url_values import get_processed_entries_from_rss_url
from src.utils.get_paper_page_indexes import ExtractPaperIndexes


# 
@chain
def arxiv_paper_search(user_input:str)->t.List[t.Dict]:
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
    paper_infos = [ArxivLoader(query=id, doc_content_chars_max=1, load_all_available_meta=True).load()[0] for id in paper_id]
    ic(paper_infos)

    return paper_infos

# 
@chain
def paper_index_read(arxiv_pdf_name:str)->t.List[str]:
    """논문 내 목차 정보를 수집하고 반환합니다.

    Args:
        arxiv_pdf_name (str): 목차 정보를 수집할 대상 논문 pdf 이름

    Returns:
        t.List[str]: 수집된 목차 리스트
    """
    extract_paper_indexer = ExtractPaperIndexes(using_llm_name=CHAT_MODEL)
    loaded_pdf = fitz.open(os.path.join(PDF_DOWNLOAD_DIR, arxiv_pdf_name))
    arxiv_pdf_indexes = extract_paper_indexer.run_extract_all_indexes(paper_pages=loaded_pdf)

    del loaded_pdf
    del extract_paper_indexer

    return arxiv_pdf_indexes

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
    ic(recent_papers_markdown)
    
    return recent_papers_markdown


if __name__ == "__main__":
    # result = arxiv_paper_search.invoke(input="2402.09353 논문을 찾아줘")
    # print(result)
    # print(type(result))

    result = get_recent_papers.invoke(input="nlp최신 논문 보여주세요")
    print(result)