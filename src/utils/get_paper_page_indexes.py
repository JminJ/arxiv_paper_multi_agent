import typing as t
import fitz
import ast

import tiktoken
import openai


class ExtractPaperIndexes:
    def __init__(self, using_llm_name:str, extract_page_range:int=1):
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

    def _get_target_range_pages(self, paper_pages:fitz.Document, start_page:int, extract_page_range:int)->str:
        """범위에 해당되는 논문의 page들을 추출

        Args:
            paper_pages (fitz.Document): 논문 전체 페이지
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
                    "content": """주어지는 논문 페이지 내에서 목차들을 추출하세요. 목차의 번호와 제목이 '\\n'으로 끊어진 경우, 무조건 번호와 제목을 논문에 기입된 그대로 추출하세요. \n목차에는 다음과 같은 내용은 포함되지 않습니다:1) Figure, 2) Equation, 3) Table.\n특히 '\\n'기호 주변 문자들에 집중하세요, 일반적으로 목차는 줄바꿈을 통해 표기합니다. 또한 Abstract는 존재 시, 포함하세요.\n결과 텍스트는 아래 포맷을 따르세요:\n["목차1", "목차2", "목차3", ...] 현재 페이지 내에 목차 정보가 존재하지 않을 시, []를 반환하세요(주석 텍스트를 적는 것을 금자합니다!)."""
                },
                {
                    "role": "user",
                    "content": f"given paper page:\n{range_page_texts}"
                },
                {
                    "role": "assistant",
                    "content": "알겠습니다. 주신 논문 페이지에서 목차를 추출하고 리턴 포맷에 맞춰 결과를 드리도록 하겠습니다."
                }
            ]
        ).choices[0].message.content

        return llm_result

    def run_extract_all_indexes(self, paper_pages:fitz.Document)->t.Dict[str, int]:
        """pdf read 결과 내에서 논문 목차들을 추출

        Args:
            paper_pages (fitz.Document): 논문 전체 페이지
            
        Returns:
            t.Dict[int, str]: 추출된 논문 목차들. Dict 내에는 page: index로 구성.
        """
        start_page = 0
        paper_indexes = {}
        while start_page < len(paper_pages):
            temp_range_page_texts, temp_extract_range = self._get_target_range_pages(paper_pages, start_page, self.extract_page_range)
            # temp_range_page_texts 내에서 목차들을 추출
            temp_range_indexes = ast.literal_eval(self.extract_page_indexes_using_llm(range_page_texts=temp_range_page_texts))
            for index in temp_range_indexes:
                paper_indexes[index] = start_page
            start_page += temp_extract_range

        return paper_indexes
        

if __name__ == "__main__":
    import os
    import fitz
    from icecream import ic

    paper_indexer = ExtractPaperIndexes(using_llm_name="gpt-4o", extract_page_range=1)
    loaded_pdf = fitz.open("/Users/jeongminju/Documents/GITHUB/arxiv_paper_multi_agent/pdfs/2402.09353.pdf")
    paper_bookmarks = paper_indexer.run_extract_all_indexes(paper_pages=loaded_pdf)

    print(paper_bookmarks)


