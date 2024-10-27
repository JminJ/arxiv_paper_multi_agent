import operator
import typing as t

from langchain_core.messages import BaseMessage


class ArxivMultiAgentState(t.TypedDict):
    messages: t.Annotated[t.Sequence[BaseMessage], operator.add]
    next_role: str
    sender: str
    target_paper_path: str
    paper_indexes: t.Dict
