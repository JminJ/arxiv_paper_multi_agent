from icecream import ic

from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

from common.prompts import POLISH_UP_PROMPTS

    
@chain
def duckduckgo_search(search_query_text:str)->str:
    """internet search를 duckduckgo search를 사용해 진행합니다. 이후 검색 결과들을 llm을 통해 정리하고 반환합니다. 

    Args:
        search_query_text (str): 사용자가 원하는 검색 대상

    Returns:
        str: llm을 통해 정리된 인터넷 검색 결과
    """
    ddg_search = DuckDuckGoSearchRun()
    internet_search_result = ddg_search.invoke(search_query_text)

    polish_up_prompt = ChatPromptTemplate.from_messages(
        [
            # MessagesPlaceholder(variable_name="messages"),
            ("user", str(POLISH_UP_PROMPTS[0]))
        ]
    )
    polish_up_chain = polish_up_prompt | ChatOpenAI(temperature=0) | StrOutputParser()
    polish_up_result = polish_up_chain.invoke(input={"internet_search_result": internet_search_result})

    return polish_up_result
