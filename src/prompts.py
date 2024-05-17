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