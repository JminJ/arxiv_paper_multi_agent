import typing as t
import  functools

from src.agents.paper_agent.paper_agents import supervisor_agent, paper_team_leader_agent
from src.agents.agent_node import agent_node


## supervisor
supervisor_agent_node = functools.partial(agent_node, agent=supervisor_agent)

## paper team
paper_team_leader_agent_node = functools.partial(agent_node, agent=paper_team_leader_agent)

## rearch team
