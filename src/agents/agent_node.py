import re
import typing as t

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable


def _re_search_next_agent(text: str) -> t.Union[str, bool]:
    """search next agent format from agent result message.

    Args:
        text (str): agent result message

    Returns:
        t.Union[str, bool]: if found, return agent name, not found, return False
    """
    re_pattern = r"<next_agent>(.*?)</next_agent>"
    search_result = re.findall(re_pattern, text)

    if bool(search_result):
        return search_result[-1]
    else:
        return False


def _re_search_paper_index(text: str) -> t.Union[str, bool]:
    """search paper index format from agent result message.

    Args:
        text (str): agent result message
    Returns:
        t.Union[str, bool]: if found, return agent name, not found, return False
    """
    re_pattern = r"<paper_indexes>(.*?)</paper_indexes>"
    search_result = re.findall(re_pattern, text)

    if bool(search_result):
        return search_result[-1]
    else:
        return False


def agent_node(state, agent: Runnable) -> t.Dict[str, object]:
    temp_agent_result = {}
    agent_result = agent.invoke(state)

    # 1. next agent 검색
    next_agent_search_result = _re_search_next_agent(text=agent_result.content)
    if next_agent_search_result:
        temp_agent_result["next_role"] = next_agent_search_result

    # 2. paper index 검색
    paper_index_search_result = _re_search_paper_index(text=agent_result.content)
    if next_agent_search_result:
        temp_agent_result["paper_indexes"] = paper_index_search_result

    # 3. messages 핸들링
    if isinstance(agent_result, ToolMessage):
        result_messages = [agent_result]
    else:
        result_messages = [AIMessage(agent_result.content)]  # 추후 기능 추가 등 고려(10.23)
    temp_agent_result["messages"] = result_messages

    return temp_agent_result
