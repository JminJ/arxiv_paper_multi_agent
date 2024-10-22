import os
import typing as t
from typing_extensions import Annotated
import fitz
from icecream import ic

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import ArxivLoader
from langgraph.prebuilt import InjectedState

from src.common.common import CHAT_MODEL, RECENT_PAPER_SEARCH_RSS_FORMAT, PDF_DOWNLOAD_DIR
from src.common.prompts import (
    EXTRACT_ARXIV_PAPER_ID_PROMPT,
    MAKE_MARKDOWN_FORMAT_RECENT_PAPER_SUMMARY_PROMPT,
    EXTRACT_RECENT_PAPER_TYPE_PROMPT,
)
from src.utils.get_paper_page_indexes import ExtractPaperIndexes
from src.utils.get_rss_url_values import get_processed_entries_from_rss_url
from src.utils.paper_pdf_handler import paper_pdf_download, paper_pdf_load


# @tool
# def search_paper_from_arxiv(arxiv_paper_ids:t.List[str])->t.List[Document]:
#     """search paper from arxiv by arxiv paper id.

#     Args:
#         arxiv_paper_ids (t.List[str]): arxiv paper ids that user want to search.

#     Returns:
#         t.List[Document]: searched paper objects.
#     """
#     paper_infos = [ArxivLoader(query=id, doc_content_chars_max=1, load_all_available_meta=True).load()[0] for id in arxiv_paper_ids]

#     return paper_infos

# @tool
# def paper_index_read(pdf_object:pymupdf.Document)->t.List[str]:
#     """논문 내 목차 정보를 수집하고 반환합니다.

#     Args:
#         pdf_object (pymupdf.Document): 논문 목차 추출 대상 논문. 

#     Returns:
#         t.Dict[str, int]: 수집된 목차 dict. 목차명(str), 페이지(int)로 구성.
#     """
#     extract_paper_indexer = ExtractPaperIndexes(using_llm_name=CHAT_MODEL)
#     arxiv_pdf_indexes = extract_paper_indexer.run_extract_all_indexes(paper_pages=pdf_object)

#     del extract_paper_indexer

#     return arxiv_pdf_indexes

# @tool
# def paper_pdf_read_set(arxiv_pdf_name:str)->pymupdf.Document:
#     """논문을 다운로드 받고, pdf object를 반환합니다.

#     Args:
#         arxiv_pdf_name (str): 목차 정보를 수집할 대상 논문 pdf 이름

#     Returns:
#         pymupdf.Document: 로드된 논문 object.
#     """
#     loaded_pdf = fitz.open(os.path.join(PDF_DOWNLOAD_DIR, arxiv_pdf_name))

#     return loaded_pdf

# @tool
# def extract_index_contents(target_pdf_indexes:t.List[str], pdf_indexes:t.Dict[str, int], all_pdf_text_content:pymupdf.Document)->t.List[str]:
#     """pdf 내의 index에 해당하는 내용들을 추출합니다.

#     Args:
#         target_pdf_indexes (t.List[str]): 논문 전체 pdf_index들
#         pdf_indexes (t.List[str]): 논문 본문 추출을 원하는 pdf_index들
#         all_pdf_text_content (str): 특허 문서 전체 내용

#     Returns:
#         t.List[str]: 추출된 index의 내용들
#     """
#     each_index_content_dict = {}
#     for idx in range(len(pdf_indexes)):
#         temp_index = pdf_indexes[idx]
#         if temp_index not in target_pdf_indexes.keys():
#             continue
#         temp_index_start_index = all_pdf_text_content.find(temp_index)

#         temp_index_content = all_pdf_text_content[temp_index_start_index:]
#         if idx < len(pdf_indexes)-1:
#             # pdf_indexes의 길이와 idx가 같을 시 pass
#             next_index_start_index = temp_index_content.find(pdf_indexes[idx+1])
#             temp_index_content = temp_index_content[:next_index_start_index]

#         each_index_content_dict[temp_index] = temp_index_content
    
#     return each_index_content_dict


@chain
def arxiv_paper_search(user_input:str)->t.Tuple[t.List[Document], t.Dict[str, int], str]:
    """arxiv paper id로 paper를 검색합니다.

    Args:
        user_input (str): 사용자 입력

    Returns:
        t.Tuple[t.List[Document], t.Dict[str, int], str]: paper info list, paper indexes, paper path
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

    # 3. download arxiv paper pdf
    paper_save_paths = []
    for paper_info in paper_infos:
        http_pdf_path = paper_info.metadata["links"][1]
        arxiv_pdf_name = ".".join([http_pdf_path.split("/")[-1][:10], "pdf"])
        download_path = os.path.join(PDF_DOWNLOAD_DIR, arxiv_pdf_name)

        if not os.path.isfile(download_path):
            paper_pdf_download(http_pdf_path=http_pdf_path, pdf_download_path=download_path)
        paper_save_paths.append(download_path)

    # 4. extract paper index
    index_extractor = ExtractPaperIndexes(using_llm_name=CHAT_MODEL)
    paper_index_dict = index_extractor.run_extract_all_indexes(paper_pdf_load(paper_save_paths[0]))

    return paper_infos, paper_index_dict, paper_save_paths[0]

@tool
def get_recent_papers(user_input:str)->str:
    """사용자가 원하는 분야의 최신 논문들의 summary를 제공합니다. 특정 논문 검색은 수행하지 못합니다.

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

@tool
def get_user_question_part_contents(target_index_name: str, state: Annotated[dict, InjectedState])->t.List[str]:
    """find target index from state.paper_indexes, return page content that contain target index part.

    Args:
        target_index_name (str): paper index name that user answered. it can have index number. ex) 1. Intruduction
        state (Annotated[dict, InjectedState]): temp graph's state.

    Returns:
        t.List[str]: page content that have index part.
    """
    paper_indexes = state["paper_indexes"]
    target_index_page_number = paper_indexes.get(target_index_name, None)
    if target_index_page_number is None:
        pass # return error message 
    
    # extract index part pages
    target_paper_pdf = paper_pdf_load(state["target_paper_path"])
    if target_index_page_number == 0:
        target_page_numbers = [target_index_page_number, target_index_page_number + 1]
    else:
        target_page_numbers = [target_index_page_number - 1, target_index_page_number, target_index_page_number + 1]
    
    result_paper_pages = []
    for page_number in target_page_numbers:
        page_content = target_paper_pdf.load_page(page_number).get_text("text")
        result_paper_pages.append(page_content)

    return result_paper_pages


if __name__ == "__main__":
    paper_information, paper_indexes, paper_save_path = arxiv_paper_search.invoke(input="2402.09353 논문을 찾아줘")
    print(f"\n\n{paper_information}")
    print(f"\n{paper_indexes}")
    # print(type(result))

    # result = get_recent_papers.invoke(input="nlp최신 논문 보여주세요")
    # print(result)