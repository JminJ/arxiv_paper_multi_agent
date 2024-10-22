from langchain_core.tools import Tool

from common.prompts import (
    ARXIV_PAPER_SEARCH_TOOL_DESC,
    PAPER_INDEX_READ_TOOL_DESC,
    EXTRACT_INDEX_CONTENTS_TOOL_DESC,
    GET_RECENT_PAPERS_TOOL_DESC,
    DUCKDUCKGO_SEARCH_TOOL_DESC
)
from agents.paper_agent.paper_agents_regacy import (
    arxiv_paper_search,
    paper_index_read,
    extract_index_contents,
    get_recent_papers
)
from agents.search_agent.search_agent_utils import duckduckgo_search


arxiv_paper_search_tool = Tool.from_function(
    func=arxiv_paper_search.invoke,
    name="arxiv_paper_search_tool",
    description=ARXIV_PAPER_SEARCH_TOOL_DESC,
    return_direct=False
)
paper_index_read_tool = Tool.from_function(
    func=paper_index_read.invoke,
    name="paper_index_read_tool",
    description=PAPER_INDEX_READ_TOOL_DESC,
    return_direct=False
)
extract_index_contents_tool = Tool.from_function(
    func=extract_index_contents.invoke,
    name='extract_index_contents_tool',
    description=EXTRACT_INDEX_CONTENTS_TOOL_DESC,
    return_direct=False
)
get_recent_papers_tool = Tool.from_function(
    func=get_recent_papers.invoke,
    name='get_recent_papers_tool',
    description=GET_RECENT_PAPERS_TOOL_DESC,
    # return_direct=False
    return_direct=True
)

duckduckgo_search_tool = Tool.from_function(
    func=duckduckgo_search.invoke,
    name="duckduckgo_search_tool",
    description=DUCKDUCKGO_SEARCH_TOOL_DESC,
    return_direct=False
)

if __name__ == "__main__":
    from icecream import ic

    ic(arxiv_paper_search_tool)