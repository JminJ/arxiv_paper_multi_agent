import typing as t
import pymupdf

import tiktoken
import openai
from icecream import ic


class ExtractPaperIndexes:
    def __init__(self, using_llm_name:str, extract_page_range:int=3):
        self.tokenizer = tiktoken.encoding_for_model(using_llm_name)
        self.using_llm_name = using_llm_name
        self.llm_client = openai.OpenAI()
        self.extract_page_range = extract_page_range

    def _count_pages_tokens(self, page_contents:str)->int:
        """논문 page들의 token 길이를 count

        Args:
            page_contents (str): 검사 대상 논문 pages

        Returns:
            int: 검사 완료 token 길이
        """
        tok_result = self.tokenizer.encode(page_contents)
        return len(tok_result)

    def _get_target_range_pages(self, paper_pages:pymupdf.Document, start_page:int, extract_page_range:int)->str:
        """범위에 해당되는 논문의 page들을 추출

        Args:
            paper_pages (pymupdf.Document): 논문 전체 페이지
            start_page (int): 시작 페이지
            extract_page_range (int): 추출 범위

        Returns:
            str: 범위 내 논문 페이지 텍스트
        """
        target_text_list = [page.get_text() for page in paper_pages[start_page:(start_page+extract_page_range)]]
        target_text = "\n\n\n".join(target_text_list)
        
        ## 추후 사용되는 llm에 따라 max_token len 정보를 가져와 재귀로 추출 예정, token 길이에 따른 수정된 page range범위도 반환되어야 함.
        # if self._count_pages_tokens(page_contents=target_text):
        #     target_text, extract_page_range = self._get_target_range_pages(paper_pages, start_page, (extract_page_range-1)) 
        return target_text, extract_page_range
    
    def extract_page_indexes_using_llm(self, range_page_texts:str)->str:
        """주어지는 범위 텍스트 내에서 llm을 사용해 목차들을 추출

        Args:
            range_page_texts (str): 범위 페이지 텍스트
            
        Returns:
            str: 추출된 목차들
        """
        llm_result = self.llm_client.chat.completions.create(
            model=self.using_llm_name,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": """주어지는 논문 페이지 내에서 목차들을 추출하세요. 결과 텍스트는 아래 포맷을 따르세요:\n["목차1", "목차2", "목차3", ...]"""
                },
                {
                    "role": "user",
                    "content": f"given paper page:\n{range_page_texts}"
                }
            ]
        ).choices[0].message.content

        return llm_result

    def run_extract_all_indexes(self, paper_pages:pymupdf.Document)->t.List[str]:
        """pdf read 결과 내에서 논문 목차들을 추출

        Args:
            paper_pages (pymupdf.Document): 논문 전체 페이지
            
        Returns:
            t.List[str]: 추출된 논문 목차들의 list
        """
        start_page = 0
        paper_indexes = []
        while start_page < len(paper_pages):
            temp_range_page_texts, temp_extract_range = self._get_target_range_pages(paper_pages, start_page, self.extract_page_range)
            start_page += temp_extract_range
            # temp_range_page_texts 내에서 목차들을 추출
            temp_range_indexes = eval(self.extract_page_indexes_using_llm(range_page_texts=temp_range_page_texts))
            paper_pages.extend(temp_range_indexes)

        return paper_indexes
        

