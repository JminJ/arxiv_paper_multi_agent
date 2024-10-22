from langchain_core.runnables import Runnable

def agent_node(state, agent: Runnable):
    agent_result = agent.invoke(state)

    print(agent_result)
    

