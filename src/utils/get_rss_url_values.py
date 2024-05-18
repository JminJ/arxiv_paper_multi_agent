import typing as t
import urllib
import feedparser
from openai import OpenAI

import sys 
sys.path.append("/Users/jeongminju/Documents/GITHUB/arxiv_paper_multi_agent/src")
from prompts import SUMMARY_PATENT_SUMMARY_CONTENT_PROMPT
from common import CHAT_SEED, CHAT_MODEL


def _summarize_patent_summary_content(org_summary_text:str) -> str:
    client = OpenAI()

    summary_result = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": SUMMARY_PATENT_SUMMARY_CONTENT_PROMPT[0]
            },
            {
                "role": "user",
                "content": SUMMARY_PATENT_SUMMARY_CONTENT_PROMPT[1].format(org_summary_text)
            }
        ],
        temperature=0.0,
        seed=CHAT_SEED
    )

    return summary_result.choices[0].message.content


def _extract_using_values_from_entry(temp_entry:t.Dict) -> t.Dict:
    """entry 값 내에서 필요한 값들을 추출하고 정제해 새로운 dict를 리턴.
        - title
        - author
        - tags
        - paper_id
        - summary => 단일 특허마다 llm을 통한 summary 적용.

    Args:
        temp_entry (t.Dict): 현재 entry

    Returns:
        t.Dict
    """
    new_using_values = {}

    new_using_values["title"] = temp_entry["title"].strip()

    new_using_values["authors"] = ", ".join([author["name"].strip() for author in temp_entry["authors"]])
    
    new_using_values["tags"] = str(", ".join([tag["term"].strip() for tag in temp_entry["tags"]]))

    paper_id = str(temp_entry["link"].split("/")[-1][:10])

    new_using_values["paper_id"] = paper_id

    # new_using_values["summary"] = _summarize_patent_summary_content(org_summary_text=temp_entry["summary"])
    new_using_values["summary"] = temp_entry["summary"]

    return new_using_values 


def _change_rss_url_type(org_rss_url:str)->str:
    """rss feed의 entry가 비어있을 경우, http/https를 변경한 rss_url을 리턴

    Args:
        org_rss_url (str): entry가 비어있던 url

    Returns:
        str: http/https를 변환한 rss_url
    """
    splited_rss_url = org_rss_url.split("://")
    url_type = splited_rss_url[0]

    if url_type == "http":
        new_rss_url = "://".join(["https", splited_rss_url[1]])
    else:
        new_rss_url = "://".join(["http", splited_rss_url[1]])

    return new_rss_url


def get_processed_entries_from_rss_url(rss_url:str) -> t.List[t.Dict]:
    """arxiv rss url 내 값을 가져와 entries 정제 후 리턴

    Args:
        rss_url (str): target rss url

    Returns:
        t.List[t.Dict]: rss url 내 feed entries
    """
    def get_rss_feed_and_parse(rss_url:str)->t.List[t.Dict]:
        # 1. rss url 내 값 read
        response = urllib.request.urlopen(rss_url)
        rss_feed = response.read()
        # 2. feed parsing
        parsing_result = feedparser.parse(rss_feed).entries
        return parsing_result
    
    parsing_result = get_rss_feed_and_parse(rss_url)
    if not bool(parsing_result):
        print("[get_processed_entries_from_rss_url] change http type and re-search")
        new_rss_url = _change_rss_url_type(org_rss_url=rss_url)
        parsing_result = get_rss_feed_and_parse(new_rss_url)

    # 3. using key값 정제 및 추출
    processed_parsing_result = [_extract_using_values_from_entry(result) for result in parsing_result]

    return processed_parsing_result