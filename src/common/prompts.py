EXTRACT_ARXIV_PAPER_ID_PROMPT = [
    """
Extract arxiv paper id from user input.

arxiv paper id has below formats:
1. 2401.15884
2. 240115884

Return only extracted ["id1", ...] always. Also, if user give wrong id in input message, return ["WRONG_ID"].
Must wrap string id/s always.

<example>
user_input: 2401.15884번 논문을 읽고 설명해줘.
Answer: ["2401.15884"]
</example>
""",
    "extract arxiv paper id from below user input msg\nuser_input: {user_input}\nAnswer: "
]


EXTRACT_RECENT_PAPER_TYPE_PROMPT = [
    """
Extract the arxiv rss paper type the user wants through user input.

Below, I'll give you example of rss paper types:
• cs.AI: 인공지능 (Artificial Intelligence)
• cs.CL: 계산 및 언어 (Computation and Language, NLP)
• cs.CC: 계산 이론 (Computational Complexity)
• cs.CE: 계산 및 엔지니어링 (Computational Engineering, Finance, and Science)
• cs.CG: 컴퓨터 그래픽스 (Computational Geometry)
• cs.GT: 게임 이론 (Computer Science and Game Theory)
• cs.CV: 컴퓨터 비전 및 패턴 인식 (Computer Vision and Pattern Recognition)
• cs.CY: 컴퓨터 및 사회 (Computers and Society)
• cs.CR: 암호학 및 보안 (Cryptography and Security)
• cs.DS: 데이터 구조 및 알고리즘 (Data Structures and Algorithms)
• cs.DB: 데이터베이스 (Databases)
• cs.DL: 디지털 라이브러리 (Digital Libraries)
• cs.DM: 이산 수학 (Discrete Mathematics)
• cs.DC: 분산, 병렬 및 클러스터 컴퓨팅 (Distributed, Parallel, and Cluster Computing)
• cs.ET: 신뢰성 (Emerging Technologies)
• cs.FL: 형식 언어 및 자동 이론 (Formal Languages and Automata Theory)
• cs.GL: 일반 문서 (General Literature)
• cs.GR: 그래픽스 (Graphics)
• cs.AR: 하드웨어 아키텍처 (Hardware Architecture)
• cs.HC: 인간-컴퓨터 상호작용 (Human-Computer Interaction)
• cs.IR: 정보 검색 (Information Retrieval)
• cs.IT: 정보 이론 (Information Theory)
• cs.LG: 기계 학습 (Machine Learning)
• cs.LO: 논리 (Logic in Computer Science)
• cs.MS: 멀티미디어 (Multimedia)
• cs.MA: 멀티에이전트 시스템 (Multiagent Systems)
• cs.NI: 네트워크 및 인터넷 아키텍처 (Networking and Internet Architecture)
• cs.NE: 신경망 및 유전적 알고리즘 (Neural and Evolutionary Computing)
• cs.NA: 수치 해석 (Numerical Analysis)
• cs.OS: 운영 체제 (Operating Systems)
• cs.OH: 기타 (Other)
• cs.PF: 성능 (Performance)
• cs.PL: 프로그래밍 언어 (Programming Languages)
• cs.RO: 로보틱스 (Robotics)
• cs.SE: 소프트웨어 공학 (Software Engineering)
• cs.SD: 소리 (Sound)
• cs.SC: 과학 컴퓨팅 (Scientific Computing)
• cs.SI: 소셜 및 정보 네트워크 (Social and Information Networks)
• cs.SY: 시스템 및 제어 (Systems and Control)

Must return only rss paper type to user.
<Example>
user_input: 컴퓨터 비전 최신 논문들을 보고싶어
Answer: cs.CV
</Example>
""",
    "extract arxiv paper type from user input:\nuser_input: {user_input}\nAnswer: "
]


MAKE_MARKDOWN_FORMAT_RECENT_PAPER_SUMMARY_PROMPT = [
    """You are world class markdown maker. Please, make markdown format text using arxiv rss feed entries. 

Each entry is Dict and has below keys:
    1. title
    2. paper_id
    3. authors
    4. tags
    5. summary

Please, make each entries to markdown follow below instructions:
    1. entry keys will be headings of markdown.
    2. if the values ​​of each key require conversion of \n to space, please apply.
    3. title should be head2(##), others should be head3(###).
    4. summary content must be translated to korean, and summrized 3~4 line.
""",
    "please, make markdown text using below datas:\n{rss_entries}"
]


EXPLAIN_MATHEMATIC_PROMPT = [
    """You are the world class math explainer. Please provide explaination of user question.
The cases of user question can be below list:
    1. request that explain formulas that in given full paper.
    2. request that explain formulas that user given.
    3. request that explain the concept in the field of mathematics.

Basically you have to follow this procedure:
    1. explains formula in detail. 
    2. give a list of concepts are used in the formula. 
    3. explains concepts in above step's result.

If user want to explain mathematic concept, explain that with easy examples.
""",
    "answer to me:\n{user_input}\nAnswer: "
]


SUMMARY_PATENT_SUMMARY_CONTENT_PROMPT = [
    """
You're the world class summarizer. Please summarize the original outline of the paper in 2 to 3 lines.
""",
    "summarize below text\ntext: {}\nAnswer: "
]


POLISH_UP_PROMPTS = [
    """
Polishing up bellow internet search results. Result should be korean:

<internet search result>
{internet_search_result}
</internet search result>

Answer:
"""
]


######### Tool Prompts
ARXIV_PAPER_SEARCH_TOOL_DESC = """
TOOL NAME)
    arvix_paper_search_tool
DESC)
    This tool is searching arxiv papers by user given arxiv paper ids.
ARGS)
    user_input (str): user input text. you will extract arxiv paper ids from this input text.
RETURN)
    t.List[t.Dict]: paper infos.
"""
PAPER_INDEX_READ_TOOL_DESC = """
TOOL NAME)
    paper_index_read_tool
DESC)
    This tool is reading indexes(목차) in target paper pdf file.
ARGS)
    arxiv_pdf_name (str): target pdf file name for reading indexes.
RETURN)
    t.List[str]: list of paper pdf indexes.
"""
EXTRACT_INDEX_CONTENTS_TOOL_DESC = """
TOOL NAME)
    extract_index_contents_tool
DESC)
    This tool is extracting content text of each index.
ARGS)
    target_pdf_indexes (t.List[str]): all indexes in target paper pdf.
    pdf_indexes (t.List[str]): indexes user want to extract content text.
    all_pdf_text_content (str): all paper pdf content text.
RETURN)
    t.List[str]: extracted content text list.
"""
GET_RECENT_PAPERS_TOOL_DESC = """
TOOL NAME)
    get_recent_papers
DESC)
    This tool get recent paper infos in domain that user want to search. you will extract domain from user input text and search recent papers in that domain.
ARGS)
    user_input (str): user input text.
RETURN)
    str: markdown format text contain recent paper informations.
"""

ALL_PAPER_TOOLS_EXPLAINS = "\n\n".join(
    [
        ARXIV_PAPER_SEARCH_TOOL_DESC,
        PAPER_INDEX_READ_TOOL_DESC,
        EXTRACT_INDEX_CONTENTS_TOOL_DESC,
        GET_RECENT_PAPERS_TOOL_DESC 
    ]
)


DUCKDUCKGO_SEARCH_TOOL_DESC = """
TOOL NAME)
    duckduckgo_search_tool
DESC)
    This tool is used for searching informations from internet(using duckduckgo search api).
ARGS)
    search_query_text (str): query text for searching.
RETURN)
    str: search result analyzed by llm.
"""

ALL_SEARCH_TOOLS_EXPLAINS = "\n\n".join(
    [
        DUCKDUCKGO_SEARCH_TOOL_DESC
    ]
)


######### Agent Prompts
PAPER_AGENT_SYSTEM_PROMPT = "You are the agent who dealing operations about Arxiv Paper."
PAPER_AGENT_DESC = f"""
AGENT NAME)
    Paper
AGENT DESC)
    This agent is handling jobs about paper. It can below operations. 
        - search arxiv paper by arxiv id.
        - reading indexes from paper pdf.
        - extract content texts of target indexes.
        - get recent paper informations in special domain.
"""
# {ALL_PAPER_TOOLS_EXPLAINS}

SEARCH_AGENT_SYSTEM_PROMPT = "You are the agent who searching query and analyzing results."
SEARCH_AGENT_DESC = f"""
AGENT NAME)
    Search
AGENT DESC)
    This agent is searching special query from internet. It can below operations. 
        - search user query using duckduckgosearch api.
"""
# {ALL_PAPER_TOOLS_EXPLAINS}

SUPERVISOR_AGENT_SYSTEM_PROMPT = """You are master supervisor in this system. 
You will receive an input message from the user and need to parse it and pass it to the appropriate agent. 
In some case, user input question is not matched any agent. Then, you should answer to user.
If the agent's answer fits the user's question, select next role to FINISH for stop task."""

######### Agent Prompts v2
PAPER_SEARCH_AGENT_SYSTEM_PROMPT = "you are paper searcher from arxiv by arxiv paper id."
PAPER_TEAM_MEMBER_DESC_PROMPT = """
AGENT NAME)
    arxiv_paper_searcher
AGENT DESC)
    this member agent is searching paper from arxiv using arxiv paper id. after searching, download that paper pdf, and extract indexes from it.
"""