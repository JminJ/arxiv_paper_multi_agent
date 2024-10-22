import typing as t
import operator
from langchain_core.messages import BaseMessage


class ArxivMultiAgentState(t.TypedDict):
    messages: t.Annotated[t.Sequence[BaseMessage], operator.add]
    next: str
    target_paper_path: str
    paper_indexes: t.Dict
