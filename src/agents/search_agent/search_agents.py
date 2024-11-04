from langchain_community.tools import DuckDuckGoSearchRun
from src.agents.agent import AgentCreator

agent_creator = AgentCreator()

search_team_leader_agent = agent_creator.create_leader_agent(
    system_prompt="You are Search(WEB) Team's leader.",
    team_member_desc="",
    next_roles=["supervisor"],
    tools=[DuckDuckGoSearchRun()],
)

